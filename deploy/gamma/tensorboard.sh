#!/usr/bin/env bash
set -euo pipefail
campaign_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
primary_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"][:12])' "$campaign_root/plans/positive.json")
stress_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"][:12])' "$campaign_root/plans/full_scan.json")
logspec="Primary:/workspace/artifacts/gamma/performance-campaign/positive-$primary_id/tensorboard-overview,Full-scan-stress:/workspace/artifacts/gamma/performance-campaign/full_scan-$stress_id/tensorboard-overview"
sudo docker run --rm -p 127.0.0.1:6008:6008 --user "$(id -u):$(id -g)" -e USER=gamma -e LOGNAME=gamma \
 --mount "type=bind,src=$campaign_root/repo,dst=/workspace,readonly" \
 --entrypoint tensorboard rwkv-gamma:torch271-cu118 \
 --logdir_spec "$logspec" --host 0.0.0.0 --port 6008 --load_fast=false
