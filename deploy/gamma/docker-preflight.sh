#!/usr/bin/env bash
# Run on Gamma from the unpacked repository. Only checks the environment.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
mkdir -p artifacts/gamma
sudo docker build -t rwkv-gamma:torch271-cu118 deploy/gamma
sudo docker image inspect rwkv-gamma:torch271-cu118 > artifacts/gamma/docker-image.json
sudo docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --shm-size=1g \
    --mount "type=bind,src=$repo_root,dst=/workspace" \
    rwkv-gamma:torch271-cu118
