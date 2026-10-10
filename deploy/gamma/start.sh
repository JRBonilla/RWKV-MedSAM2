#!/usr/bin/env bash
set -euo pipefail
# Distributed at the root of a separate, self-contained source snapshot.
campaign_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bundle_root=${GAMMA_BUNDLE_ROOT:-"$HOME/gamma-eval"}
data_root=${GAMMA_DATA_ROOT:-"$HOME/gamma-data"}
test -f "$bundle_root/manifest.json" || { echo "Missing $bundle_root/manifest.json"; exit 1; }
test -d "$data_root" || { echo "Missing $data_root"; exit 1; }
test -d "$campaign_root/repo/rwkv_medsam2" || { echo 'Extract the complete campaign package first.'; exit 1; }
python3 - "$bundle_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
entry = json.loads((root / 'manifest.json').read_text())['checkpoints'].get('corrected_hiera')
if not entry or not (root / entry['bundle_path']).is_file():
    raise SystemExit('Corrected Hiera is missing. Install gamma-models-update-20261006 before this campaign.')
PY
sudo docker run --rm --gpus all --user "$(id -u):$(id -g)" --shm-size=1g \
  --mount "type=bind,src=$campaign_root/repo,dst=/workspace" \
  --mount "type=bind,src=$campaign_root,dst=/campaign,readonly" \
  --mount "type=bind,src=$bundle_root,dst=/bundle,readonly" \
  --mount "type=bind,src=$data_root,dst=/gamma-data,readonly" \
  rwkv-gamma:torch271-cu118 \
  bash -c 'source deploy/gamma/env.sh && exec python deploy/gamma/performance-campaign.py "$@"' \
  gamma-campaign "$@"
