#!/usr/bin/env bash
# Explicit commands only. Creating/installing this file launches nothing.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
bundle_root=$(cd -- "$repo_root/.." && pwd)
gamma_data_root=${GAMMA_DATA_ROOT:-"$HOME/gamma-data"}
if (( $# == 0 )); then
    printf '%s\n' 'Supply a performance subcommand and its arguments; see docs/evaluation/performance-runner.md.'
    exit 2
fi
test -d "$gamma_data_root"
test -f "$bundle_root/manifest.json"
cd "$repo_root"
# Both GPUs are visible so pre-trial checks can detect competing compute on
# either card. The runner itself uses only the explicitly selected device.
sudo docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" --shm-size=1g \
    --mount "type=bind,src=$repo_root,dst=/workspace" \
    --mount "type=bind,src=$bundle_root,dst=/bundle,readonly" \
    --mount "type=bind,src=$gamma_data_root,dst=/gamma-data,readonly" \
    rwkv-gamma:torch271-cu118 \
    bash -c 'source deploy/gamma/env.sh && exec python -m rwkv_medsam2.evaluation.performance "$@"' \
    gamma-performance "$@"
