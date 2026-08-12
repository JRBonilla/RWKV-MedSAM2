#!/usr/bin/env bash
set -euo pipefail

experiment_root="${1:-/data/TrainingLogs/retrain_20260805}"
benchmark_root="${2:-${BENCHMARK_TENSORBOARD_ROOT:-/data/rwkv_medsam2_model_comparisons/retrain_aux}}"
teacher_root="${TEACHER_TENSORBOARD_ROOT:-/data/TrainingCheckpoints/teacher}"
port="${TENSORBOARD_PORT:-6007}"

exec tensorboard \
  --logdir_spec="experiments:${experiment_root},teacher:${teacher_root},benchmarks:${benchmark_root}" \
  --host=0.0.0.0 \
  --port="${port}" \
  --reload_interval=15
