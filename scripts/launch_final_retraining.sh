#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/opt/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export VRWKV_T_MAX="${VRWKV_T_MAX:-256}"

RUN_ID="${RUN_ID:-rwkv2_mlp2_final_20260830_v1}"
LOADERS="${LOADERS:-/data/loaders32.pkl}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/TrainingLogs/final_retraining_20260830}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/TrainingCheckpoints/final_retraining_20260830}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/data/rwkv_medsam2_model_comparisons/final_retraining_20260830/${RUN_ID}}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/data/TrainingCheckpoints/teacher/oxford_refine_20260805_195125/checkpoints/best_val.pth}"
UOFT_RESUME_DIR="${UOFT_RESUME_DIR:-/data/rwkv_medsam2_model_comparisons/retrain_aux/20260805-195118/strict_video_console_20260826_110739/strict_video_protocol_20260826_110739_attempt001}"

GPU_BASE="${GPU_BASE:-0}"
GPU_NODISTILL="${GPU_NODISTILL:-1}"
GPU_DISTILL="${GPU_DISTILL:-2}"
GPU_BENCHMARK="${GPU_BENCHMARK:-3}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6007}"
EPOCHS="${EPOCHS:-20}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_SAMPLES="${VAL_SAMPLES:-150}"
VAL_FREQUENCY="${VAL_FREQUENCY:-5}"
TRAINING_STALE_SECONDS="${TRAINING_STALE_SECONDS:-3600}"
CANDIDATE_ARCHITECTURE="rwkv2_mlp2"
RESUME_CAMPAIGN="${RESUME_CAMPAIGN:-0}"
ADOPT_BASE_PID="${ADOPT_BASE_PID:-0}"
ADOPT_NODISTILL_PID="${ADOPT_NODISTILL_PID:-0}"
ADOPT_DISTILL_PID="${ADOPT_DISTILL_PID:-0}"

LAUNCH_DIR="$EXPERIMENT_ROOT/launcher/$RUN_ID"
INITIAL_DIR="$CHECKPOINT_ROOT/initialization/$RUN_ID"
BASE_RUN_DIR="$EXPERIMENT_ROOT/base/$RUN_ID"
NODISTILL_RUN_DIR="$EXPERIMENT_ROOT/nodistill/$RUN_ID"
DISTILL_RUN_DIR="$EXPERIMENT_ROOT/distill/$RUN_ID"

for required in "$LOADERS" "$TEACHER_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 2
    fi
done
if [[ ! -d "$UOFT_RESUME_DIR" ]]; then
    echo "Missing resumable UofT benchmark directory: $UOFT_RESUME_DIR" >&2
    exit 2
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "Four visible GPUs are required." >&2
    exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "$GPU_BASE" "$GPU_NODISTILL" "$GPU_DISTILL" "$GPU_BENCHMARK"; do
    if [[ -n "${SEEN_GPUS[$gpu]+present}" ]]; then
        echo "Each workload must have a distinct GPU; duplicate allocation: $gpu" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu]=1
done
if pgrep -f "tensorboard.*--port(=| )${TENSORBOARD_PORT}" >/dev/null; then
    echo "TensorBoard port $TENSORBOARD_PORT is already in use." >&2
    exit 2
fi
if [[ "$RESUME_CAMPAIGN" != "1" ]]; then
    for path in     "$LAUNCH_DIR" "$INITIAL_DIR" "$BASE_RUN_DIR"     "$NODISTILL_RUN_DIR" "$DISTILL_RUN_DIR" "$BENCHMARK_ROOT"; do
        if [[ -e "$path" ]]; then
            echo "Refusing to reuse campaign path: $path" >&2
            exit 2
        fi
    done
fi

mkdir -p "$LAUNCH_DIR" "$INITIAL_DIR" "$BENCHMARK_ROOT"

HIERA_INITIAL="$INITIAL_DIR/hiera_initial.pth"
RWKV_INITIAL="$INITIAL_DIR/rwkv_initial.pth"
if [[ -f "$HIERA_INITIAL" && -f "$RWKV_INITIAL" && "$RESUME_CAMPAIGN" == "1" ]]; then
    echo "Reusing matched initializations from $INITIAL_DIR."
else
    echo "Preparing matched random-encoder initializations..."
    CUDA_VISIBLE_DEVICES="$GPU_BASE"     python3 -u scripts/prepare_hiera_rwkv_predictor_matched_initialization.py     --output-dir "$INITIAL_DIR"     --efficient-vcr-architecture "$CANDIDATE_ARCHITECTURE"     >"$LAUNCH_DIR/initialization.log" 2>&1
    if [[ ! -f "$HIERA_INITIAL" || ! -f "$RWKV_INITIAL" ]]; then
        echo "Matched initialization failed; see $LAUNCH_DIR/initialization.log." >&2
        exit 1
    fi
fi

TENSORBOARD_PORT="$TENSORBOARD_PORT"     nohup scripts/start_tensorboard.sh "$EXPERIMENT_ROOT" "$BENCHMARK_ROOT"     >"$LAUNCH_DIR/tensorboard.log" 2>&1 &
TB_PID=$!
printf '%s\n' "$TB_PID" >"$LAUNCH_DIR/tensorboard.pid"
sleep 2
if ! kill -0 "$TB_PID" 2>/dev/null; then
    echo "TensorBoard failed; see $LAUNCH_DIR/tensorboard.log." >&2
    exit 1
fi

COMMON_TRAIN=(
    python3 -u scripts/train_headless.py
    --loaders "$LOADERS"
    --experiment-root "$EXPERIMENT_ROOT"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --run-id "$RUN_ID"
    --epochs "$EPOCHS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --val-samples "$VAL_SAMPLES"
    --val-frequency "$VAL_FREQUENCY"
    --monitor-every-steps 10
    --recovery-minutes 30
    --optimized
    --matched-run
    --match-common-sam2-settings
    --ignore-pretrained-image-encoder
    --wkv-t-max 256
)
BASE_ADOPT_ARGS=()
NODISTILL_ADOPT_ARGS=()
DISTILL_ADOPT_ARGS=()
if [[ "$ADOPT_BASE_PID" -gt 0 ]]; then
    BASE_ADOPT_ARGS=(--adopt-pid "$ADOPT_BASE_PID")
fi
if [[ "$ADOPT_NODISTILL_PID" -gt 0 ]]; then
    NODISTILL_ADOPT_ARGS=(--adopt-pid "$ADOPT_NODISTILL_PID")
fi
if [[ "$ADOPT_DISTILL_PID" -gt 0 ]]; then
    DISTILL_ADOPT_ARGS=(--adopt-pid "$ADOPT_DISTILL_PID")
fi

BASE_TRAIN_RESUME_ARGS=()
NODISTILL_TRAIN_RESUME_ARGS=()
DISTILL_TRAIN_RESUME_ARGS=()
if [[ "$RESUME_CAMPAIGN" == "1" ]]; then
    BASE_RECOVERY="$CHECKPOINT_ROOT/base/$RUN_ID/recovery_latest.pth"
    if [[ -f "$BASE_RECOVERY" ]]; then
        BASE_TRAIN_RESUME_ARGS=(--resume "$BASE_RECOVERY")
    fi
    NODISTILL_RECOVERY="$CHECKPOINT_ROOT/nodistill/$RUN_ID/recovery_latest.pth"
    if [[ -f "$NODISTILL_RECOVERY" ]]; then
        NODISTILL_TRAIN_RESUME_ARGS=(--resume "$NODISTILL_RECOVERY")
    fi
    DISTILL_RECOVERY="$CHECKPOINT_ROOT/distill/$RUN_ID/recovery_latest.pth"
    if [[ -f "$DISTILL_RECOVERY" ]]; then
        DISTILL_TRAIN_RESUME_ARGS=(--resume "$DISTILL_RECOVERY")
    fi
fi

CUDA_VISIBLE_DEVICES="$GPU_BASE"     python3 -u scripts/supervise_training.py     --mode base --run-id "$RUN_ID"     --experiment-root "$EXPERIMENT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"     "${BASE_ADOPT_ARGS[@]}"     --stale-seconds "$TRAINING_STALE_SECONDS" --     "${COMMON_TRAIN[@]}"     --mode base     --initial-model-state "$HIERA_INITIAL"     "${BASE_TRAIN_RESUME_ARGS[@]}"     >"$LAUNCH_DIR/base_console.log" 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES="$GPU_NODISTILL"     python3 -u scripts/supervise_training.py     --mode nodistill --run-id "$RUN_ID"     --experiment-root "$EXPERIMENT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"     "${NODISTILL_ADOPT_ARGS[@]}"     --stale-seconds "$TRAINING_STALE_SECONDS" --     "${COMMON_TRAIN[@]}"     --mode nodistill     --efficient-vcr-backbone     --efficient-vcr-architecture "$CANDIDATE_ARCHITECTURE"     --initial-model-state "$RWKV_INITIAL"     "${NODISTILL_TRAIN_RESUME_ARGS[@]}"     >"$LAUNCH_DIR/nodistill_console.log" 2>&1 &
PID_NODISTILL=$!

CUDA_VISIBLE_DEVICES="$GPU_DISTILL"     python3 -u scripts/supervise_training.py     --mode distill --run-id "$RUN_ID"     --experiment-root "$EXPERIMENT_ROOT" --checkpoint-root "$CHECKPOINT_ROOT"     "${DISTILL_ADOPT_ARGS[@]}"     --stale-seconds "$TRAINING_STALE_SECONDS" --     "${COMMON_TRAIN[@]}"     --mode distill     --teacher-checkpoint "$TEACHER_CHECKPOINT"     --efficient-vcr-backbone     --efficient-vcr-architecture "$CANDIDATE_ARCHITECTURE"     --initial-model-state "$RWKV_INITIAL"     "${DISTILL_TRAIN_RESUME_ARGS[@]}"     >"$LAUNCH_DIR/distill_console.log" 2>&1 &
PID_DISTILL=$!

CUDA_VISIBLE_DEVICES="$GPU_BENCHMARK"     python3 -u scripts/run_strict_video_benchmark_console.py     --resume-dir "$UOFT_RESUME_DIR"     --output-base "$BENCHMARK_ROOT"     --models uoft_medsam2     --progress-interval-sec 60     --status-update-every-sec 60     >"$LAUNCH_DIR/uoft_benchmark_console.log" 2>&1 &
PID_UOFT=$!

printf '%s\n' "$PID_BASE" >"$LAUNCH_DIR/base.pid"
printf '%s\n' "$PID_NODISTILL" >"$LAUNCH_DIR/nodistill.pid"
printf '%s\n' "$PID_DISTILL" >"$LAUNCH_DIR/distill.pid"
printf '%s\n' "$PID_UOFT" >"$LAUNCH_DIR/uoft.pid"

PIDS=("$PID_BASE" "$PID_NODISTILL" "$PID_DISTILL" "$PID_UOFT")
stop_jobs() {
    trap - INT TERM HUP
    echo "Stopping campaign workers..."
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_jobs INT TERM HUP

echo "Started final four-GPU campaign: $RUN_ID"
echo "  GPU $GPU_BASE: Hiera Base + optimized predictor"
echo "  GPU $GPU_NODISTILL: RWKV2-MLP2 non-distilled + optimized predictor"
echo "  GPU $GPU_DISTILL: RWKV2-MLP2 distilled + optimized predictor"
echo "  GPU $GPU_BENCHMARK: resumed UofT strict benchmark"
echo "  Training: $EPOCHS full epochs, global batch $GLOBAL_BATCH_SIZE, workers $NUM_WORKERS"
echo "  Validation: $VAL_SAMPLES examples every $VAL_FREQUENCY epochs"
echo "  TensorBoard: http://127.0.0.1:$TENSORBOARD_PORT"
echo "  Launcher files: $LAUNCH_DIR"

wait_job() {
    local name="$1"
    local pid="$2"
    local rc=0
    set +e
    wait "$pid"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "$name exited with status $rc; see $LAUNCH_DIR/${name}_console.log." >&2
    fi
    return "$rc"
}

phase_one_rc=0
wait_job base "$PID_BASE" || phase_one_rc=1
wait_job nodistill "$PID_NODISTILL" || phase_one_rc=1
wait_job distill "$PID_DISTILL" || phase_one_rc=1
wait_job uoft_benchmark "$PID_UOFT" || phase_one_rc=1
if [[ "$phase_one_rc" -ne 0 ]]; then
    echo "At least one phase-one workload failed; final native benchmarks were not started." >&2
    exit 1
fi

campaign_checkpoint() {
    local mode="$1"
    local checkpoint_dir="$CHECKPOINT_ROOT/$mode/$RUN_ID"
    local filename
    for filename in best.pth latest.pth; do
        if [[ -f "$checkpoint_dir/$filename" ]]; then
            printf '%s\n' "$checkpoint_dir/$filename"
            return 0
        fi
    done
    echo "No completed checkpoint for $mode under $checkpoint_dir." >&2
    return 1
}

latest_benchmark_resume() {
    local latest_file
    latest_file="$(
        find "$BENCHMARK_ROOT" -type f -name latest_resume_dir.txt -printf '%T@ %p\n'         | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
    if [[ -z "$latest_file" || ! -f "$latest_file" ]]; then
        echo "Could not find a benchmark resume pointer under $BENCHMARK_ROOT." >&2
        return 1
    fi
    sed -n '1p' "$latest_file"
}

BASE_CHECKPOINT="$(campaign_checkpoint base)"
NODISTILL_CHECKPOINT="$(campaign_checkpoint nodistill)"
DISTILL_CHECKPOINT="$(campaign_checkpoint distill)"
CURRENT_RESUME="$(latest_benchmark_resume)"

run_final_benchmark() {
    local model_name="$1"
    local checkpoint_path="$2"
    local log_name="$3"
    echo "Benchmarking $model_name from $checkpoint_path..."
    CUDA_VISIBLE_DEVICES="$GPU_BENCHMARK"         python3 -u scripts/run_strict_video_benchmark_console.py         --resume-dir "$CURRENT_RESUME"         --output-base "$BENCHMARK_ROOT"         --models "$model_name"         --checkpoint-override "$model_name=$checkpoint_path"         --progress-interval-sec 60         --status-update-every-sec 60         >"$LAUNCH_DIR/${log_name}_console.log" 2>&1
    CURRENT_RESUME="$(latest_benchmark_resume)"
}

run_final_benchmark sam2_1_final_base "$BASE_CHECKPOINT" final_base_benchmark
run_final_benchmark rwkv_medsam2_final_nodistill "$NODISTILL_CHECKPOINT" final_nodistill_benchmark
run_final_benchmark rwkv_medsam2_final_distill "$DISTILL_CHECKPOINT" final_distill_benchmark

printf '%s\n' "$CURRENT_RESUME" >"$LAUNCH_DIR/final_resume_dir.txt"
echo "Final training and benchmark campaign complete."
echo "Final resume directory: $CURRENT_RESUME"
