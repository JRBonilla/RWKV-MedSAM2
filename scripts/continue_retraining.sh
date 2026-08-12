#!/usr/bin/env bash
set -euo pipefail

run_id="${RWKV_RUN_ID:?RWKV_RUN_ID is required}"
original_launcher_pid="${RWKV_ORIGINAL_LAUNCHER_PID:?RWKV_ORIGINAL_LAUNCHER_PID is required}"
experiment_root="${RWKV_EXPERIMENT_ROOT:-/data/TrainingLogs/retrain_20260805}"
checkpoint_root="${RWKV_CHECKPOINT_ROOT:-/data/TrainingCheckpoints/retrain_20260805}"
benchmark_root="${RWKV_BENCHMARK_ROOT:-/data/rwkv_medsam2_model_comparisons/retrain_aux/${run_id}}"
launcher_logs="${experiment_root}/launcher"
mkdir -p "${launcher_logs}"

log() {
  printf '[campaign-handoff %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

pid_is_running() {
  local pid="$1"
  local state=""
  [[ -r "/proc/${pid}/stat" ]] || return 1
  state="$(sed -n 's/.*) \([^ ]\).*/\1/p' "/proc/${pid}/stat")"
  [[ "${state}" != "Z" ]]
}

wait_for_original_launcher() {
  while pid_is_running "${original_launcher_pid}"; do
    sleep 60
  done
  log "Original launcher PID ${original_launcher_pid} has exited; taking over remaining stages."
}

wait_for_mode() {
  local mode="$1"
  local status_path="${experiment_root}/${mode}/${run_id}/status.json"
  while true; do
    if [[ -f "${status_path}" ]] && grep -q '"state": "complete"' "${status_path}"; then
      log "${mode} training is complete."
      return 0
    fi
    if [[ -f "${experiment_root}/${mode}/${run_id}/supervisor_status.json" ]] \
      && grep -q '"state": "failed"' "${experiment_root}/${mode}/${run_id}/supervisor_status.json"; then
      log "${mode} supervisor failed; see ${experiment_root}/${mode}/${run_id}/supervisor_status.json"
      return 1
    fi
    sleep 60
  done
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
    printf '%s\n' "${benchmark_root}/resume_seed"
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
  log "No completed ${mode} checkpoint found under ${checkpoint_dir}."
  return 1
}

run_benchmark() {
  local model_name="$1"
  local checkpoint_path="${2:-}"
  local resume_dir=""
  resume_dir="$(latest_benchmark_resume)"
  log "Benchmarking ${model_name} from resume directory ${resume_dir}."
  local command=(
    python3 scripts/run_strict_video_benchmark_console.py
    --resume-dir "${resume_dir}"
    --output-base "${benchmark_root}"
    --models "${model_name}"
    --progress-interval-sec 60
    --status-update-every-sec 60
  )
  if [[ -n "${checkpoint_path}" ]]; then
    command+=(--checkpoint-override "${model_name}=${checkpoint_path}")
  fi
  CUDA_VISIBLE_DEVICES=0 "${command[@]}" >>"${launcher_logs}/benchmark.log" 2>&1
}

wait_for_original_launcher

# These calls are deliberately idempotent. If the old launcher completed a
# stage, resume_completed_rows skips it; if it exited early, work continues
# from the newest latest_resume_dir.txt instead of returning to resume_seed.
wait_for_mode distill
run_benchmark uoft_medsam2
run_benchmark oxford_medical_sam2
run_benchmark rwkv_medsam2_distill "$(campaign_checkpoint distill)"

wait_for_mode nodistill
run_benchmark rwkv_medsam2_nodistill "$(campaign_checkpoint nodistill)"

wait_for_mode base
run_benchmark sam2_1_base "$(campaign_checkpoint base)"

log "Recovered training campaign and all benchmarks completed."
