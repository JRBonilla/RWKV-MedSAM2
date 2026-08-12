#!/usr/bin/env bash
set -euo pipefail


cuda_home="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${cuda_home}/nvvm/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/opt/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${cuda_home}/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
experiment_root="${RWKV_EXPERIMENT_ROOT:-/data/TrainingLogs/retrain_20260805}"
checkpoint_root="${RWKV_CHECKPOINT_ROOT:-/data/TrainingCheckpoints/retrain_20260805}"
run_id="${RWKV_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
teacher_root="/data/TrainingCheckpoints/teacher"
benchmark_root="${RWKV_BENCHMARK_ROOT:-/data/rwkv_medsam2_model_comparisons/retrain_aux/${run_id}}"
benchmark_resume="${RWKV_BENCHMARK_RESUME_DIR:-${benchmark_root}/resume_seed}"
launcher_logs="${experiment_root}/launcher"
mkdir -p "${launcher_logs}" "${benchmark_resume}"
benchmark_pid=""
benchmark_rc=0

stop_benchmark() {
  local pid="${benchmark_pid:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  set +e
  wait "${pid}"
  benchmark_rc=$?
  set -e
  benchmark_pid=""
}

latest_benchmark_resume() {
  local latest_file=""
  latest_file="$(
    find "${benchmark_root}" -type f -name latest_resume_dir.txt -printf '%T@ %p\n' 2>/dev/null \
      | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "${latest_file}" && -f "${latest_file}" ]]; then
    sed -n '1p' "${latest_file}"
  else
    printf '%s\n' "${benchmark_resume}"
  fi
}

campaign_checkpoint() {
  local mode="$1"
  local checkpoint_dir="${checkpoint_root}/${mode}/${run_id}"
  local filename=""
  for filename in best.pth latest.pth; do
    if [[ -f "${checkpoint_dir}/${filename}" ]]; then
      printf '%s\n' "${checkpoint_dir}/${filename}"
      return 0
    fi
  done
  echo "No completed ${mode} checkpoint found under ${checkpoint_dir}." >&2
  return 1
}

wait_for_job() {
  local job_name="$1"
  local pid="$2"
  local rc=0
  set +e
  wait "${pid}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "${job_name} exited with status ${rc}; see ${launcher_logs}/${job_name}.log." >&2
    return "${rc}"
  fi
}

run_external_benchmark() {
  local resume_dir=""
  resume_dir="$(latest_benchmark_resume)"
  echo "Resuming UofT/Oxford benchmark from ${resume_dir}."
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_strict_video_benchmark_console.py \
    --resume-dir "${resume_dir}" --output-base "${benchmark_root}" \
    --models uoft_medsam2 oxford_medical_sam2 \
    --progress-interval-sec 60 --status-update-every-sec 60 \
    >>"${launcher_logs}/benchmark.log" 2>&1
}

run_native_benchmark() {
  local model_name="$1"
  local checkpoint_path="$2"
  local resume_dir=""
  resume_dir="$(latest_benchmark_resume)"
  echo "Benchmarking ${model_name} from ${checkpoint_path}."
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_strict_video_benchmark_console.py \
    --resume-dir "${resume_dir}" --output-base "${benchmark_root}" \
    --models "${model_name}" \
    --checkpoint-override "${model_name}=${checkpoint_path}" \
    --progress-interval-sec 60 --status-update-every-sec 60 \
    >>"${launcher_logs}/benchmark.log" 2>&1
}

trap stop_benchmark EXIT


if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
  echo "Four visible GPUs are required for the staged launcher." >&2
  exit 1
fi

TENSORBOARD_PORT=6007 scripts/start_tensorboard.sh "${experiment_root}" "${benchmark_root}" \
  >"${launcher_logs}/tensorboard.log" 2>&1 &
tensorboard_pid=$!

training_stale_seconds="${RWKV_TRAINING_STALE_SECONDS:-3600}"

CUDA_VISIBLE_DEVICES=2 python3 scripts/supervise_training.py \
  --mode base --run-id "${run_id}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --stale-seconds "${training_stale_seconds}" -- \
  python3 scripts/train_headless.py --mode base --run-id "${run_id}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --global-batch-size 4 --val-samples 150 --val-frequency 5 \
  >"${launcher_logs}/base.log" 2>&1 &
base_pid=$!

CUDA_VISIBLE_DEVICES=3 python3 scripts/supervise_training.py \
  --mode nodistill --run-id "${run_id}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --stale-seconds "${training_stale_seconds}" -- \
  python3 scripts/train_headless.py --mode nodistill --run-id "${run_id}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --global-batch-size 4 --val-samples 150 --val-frequency 5 \
  >"${launcher_logs}/nodistill.log" 2>&1 &
nodistill_pid=$!

CUDA_VISIBLE_DEVICES=1 python3 scripts/run_strict_video_benchmark_console.py \
  --resume-dir "${benchmark_resume}" \
  --output-base "${benchmark_root}" \
  --models uoft_medsam2 oxford_medical_sam2 \
  --progress-interval-sec 60 \
  --status-update-every-sec 60 \
  >"${launcher_logs}/benchmark.log" 2>&1 &
benchmark_pid=$!

teacher_marker="/tmp/rwkv-teacher-${run_id}.marker"
touch "${teacher_marker}"
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert \
  --to notebook --execute notebooks/Oxford-Refine_v9_launch.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output "teacher-${run_id}.ipynb" \
  --output-dir "${launcher_logs}" \
  >"${launcher_logs}/teacher.log" 2>&1
teacher_checkpoint="$(find "${teacher_root}" -type f -name best_val.pth -newer "${teacher_marker}" -print | sort | tail -n 1)"
rm -f "${teacher_marker}"
if [[ -z "${teacher_checkpoint}" ]]; then
  echo "Teacher completed without producing a new best_val.pth." >&2
  exit 1
fi

# GPU 1 becomes a distillation rank next. Ask the benchmark supervisor to
# checkpoint its JSONL and terminate its worker before torchrun initializes.
stop_benchmark
if [[ "${benchmark_rc}" -ne 0 && "${benchmark_rc}" -ne 130 ]]; then
  echo "Benchmark supervisor exited with status ${benchmark_rc}; see ${launcher_logs}/benchmark.log." >&2
fi

CUDA_VISIBLE_DEVICES=0,1 python3 scripts/supervise_training.py \
  --mode distill --run-id "${run_id}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --stale-seconds "${training_stale_seconds}" -- \
  torchrun --standalone --nproc-per-node=2 scripts/train_headless.py \
  --mode distill --run-id "${run_id}" --teacher-checkpoint "${teacher_checkpoint}" \
  --experiment-root "${experiment_root}" --checkpoint-root "${checkpoint_root}" \
  --global-batch-size 4 --val-samples 150 --val-frequency 5 \
  >"${launcher_logs}/distill.log" 2>&1 &
distill_pid=$!

echo "TensorBoard PID: ${tensorboard_pid}"
echo "TensorBoard URL through the configured tunnel: http://127.0.0.1:6007"
echo "Base PID: ${base_pid}"
echo "No-distill PID: ${nodistill_pid}"
echo "Benchmark output root: ${benchmark_root}"
echo "Distill PID: ${distill_pid}"
echo "Run ID: ${run_id}"

wait_for_job distill "${distill_pid}"
distill_checkpoint="$(campaign_checkpoint distill)"

# GPU 0 is free once the two-rank student finishes. Complete the interrupted
# external-model work first, then benchmark each newly trained campaign model.
run_external_benchmark
run_native_benchmark rwkv_medsam2_distill "${distill_checkpoint}"

wait_for_job nodistill "${nodistill_pid}"
nodistill_checkpoint="$(campaign_checkpoint nodistill)"
run_native_benchmark rwkv_medsam2_nodistill "${nodistill_checkpoint}"

wait_for_job base "${base_pid}"
base_checkpoint="$(campaign_checkpoint base)"
run_native_benchmark sam2_1_base "${base_checkpoint}"

echo "Training and all campaign benchmarks completed."
echo "Benchmark output root: ${benchmark_root}"
