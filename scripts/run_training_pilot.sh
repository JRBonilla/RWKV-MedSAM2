#!/usr/bin/env bash
set -euo pipefail

pilot_root="${RWKV_PILOT_ROOT:-/tmp/rwkv-training-pilot}"
pilot_id="${RWKV_PILOT_ID:-pilot-$(date -u +%Y%m%d-%H%M%S)}"
gpu_ids="${RWKV_PILOT_GPUS:-0,1}"
pilot_steps="${RWKV_PILOT_STEPS:-4}"
pilot_val_samples="${RWKV_PILOT_VAL_SAMPLES:-2}"

mkdir -p "${pilot_root}/logs" "${pilot_root}/checkpoints"

CUDA_VISIBLE_DEVICES="${gpu_ids}" torchrun \
  --standalone \
  --nproc-per-node=2 \
  scripts/train_headless.py \
  --mode distill \
  --pilot \
  --run-id "${pilot_id}" \
  --experiment-root "${pilot_root}/logs" \
  --checkpoint-root "${pilot_root}/checkpoints" \
  --epochs 1 \
  --global-batch-size 4 \
  --num-workers 0 \
  --max-steps-per-epoch "${pilot_steps}" \
  --val-samples "${pilot_val_samples}" \
  --val-frequency 1 \
  --monitor-every-steps 1 \
  --recovery-minutes 0
