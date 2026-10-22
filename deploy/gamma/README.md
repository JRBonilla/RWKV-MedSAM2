# Gamma Deployment

The target Gamma environment has two GTX 1070 cards (8 GiB each), 16 GiB RAM,
and Docker GPU passthrough. Its pinned image uses Python 3.10, PyTorch 2.7.1 with
CUDA 11.8, and native sm_61 WKV kernels. `env.sh` disables Torch compilation and
limits CPU threads.

From a repository checkout on Gamma, build and check the image with:

```bash
bash deploy/gamma/docker-preflight.sh
```

This checks runtime compatibility; it does not run an accuracy or performance
campaign. Preparation and measurement must also use matching NumPy, Pillow, and
SimpleITK versions. The launcher uses the image tag `rwkv-gamma:torch271-cu118`.

## Final Fixed Campaign

Prepare matching positive-frame and full-scan accuracy manifests using the same
evaluator source snapshot that will be packaged. The model list is explicit;
include all five models for the full comparison:

```bash
MODELS=(corrected_hiera rwkv_medsam2_final_nodistill rwkv_medsam2_final_distill oxford_medical_sam2 uoft_medsam2)
python -m rwkv_medsam2.evaluation.performance plan \
  --manifest POSITIVE_FROZEN --models "${MODELS[@]}" \
  --data-root STAGED_DATA --source-root /data --output PRIMARY.json
python -m rwkv_medsam2.evaluation.performance plan \
  --manifest FULL_FROZEN --models "${MODELS[@]}" \
  --data-root STAGED_DATA --source-root /data --output FULL.json
python deploy/gamma/finalize-performance-benchmark.py \
  --primary-plan PRIMARY.json --full-scan-plan FULL.json --output FINAL_PLANS
python deploy/gamma/package.py \
  --plans FINAL_PLANS --checkpoints CHECKPOINTS.json \
  --output artifacts/gamma-performance.tar.gz
```

The standard planner excludes unresolved source identities. The finalizer keeps
the primary parent panel and selects three distinct full-scan images at
short/middle/long lengths for every modality/sequence-kind group in the full-scan
parent. Each group needs at least three distinct images. Panel size is determined
by the supplied plans; neither a fixed target count nor a runtime estimate is
enforced by the code. Stress results are reported separately.

The finalizer sets three warmups, two opposite-order timing blocks with three
repetitions each, and one fresh-process memory worker per case/model/prompt.
Encoder timing is performed for the first prompt mode of each target. All models
listed in each plan receive that plan's cases and prompts. Legacy Hiera is
excluded, and no adaptive panel expansion is scheduled.

The packager verifies plan/source identity and includes local model source,
`dripp/dripp/defaults.ini`, finalized plans, checkpoint specifications, and launch
scripts. It copies eligible source/configuration files; it does not include
checkpoint weight files, external model repositories, or datasets. Keep those in
the separate Gamma bundle and data directory:

```text
~/gamma-eval/manifest.json
~/gamma-eval/checkpoints/
~/gamma-eval/external/oxford/
~/gamma-eval/external/uoft/
~/gamma-data/
```

The bundle manifest must contain checkpoint paths and pinned hashes for every
selected model, including corrected Hiera. `GAMMA_BUNDLE_ROOT` and
`GAMMA_DATA_ROOT` override the launcher's bundle and data roots.

Transfer the archive and checksum to Gamma. From the directory containing both:

```bash
sha256sum -c gamma-performance.tar.gz.sha256
tar -xzf gamma-performance.tar.gz
bash gamma-performance/start.sh
```

The extracted launcher runs the primary campaign and full-scan stress campaign
sequentially on GPU 1, then writes both reports. It requires two visible GPUs
with no active compute processes. The container exposes both cards so checks can
detect competing work; measurements use `cuda:1`.

Rerun the same command to resume validated completed trials with the same source,
plans, checkpoints, environment, and hardware identity. A hard kill may leave
`.writer.claim`: check its owner and active processes before removing it. GPU
boundary telemetry is saved; transient interference is not continuously
monitored. Use a persistent terminal such as tmux for long campaigns.

Both evaluator source hashes and the package file manifest are checked. Keep an
existing extracted package intact, including its packaged README. Python
docstring or formatting edits require matching newly frozen plans, or the
original source snapshot; they do not require retraining model weights.

## Progress

The terminal reports stage, model, dataset, frame count, and measurement job.
Within the extracted package, `repo/artifacts/gamma/performance-campaign/`
contains raw trials, logs, JSON status, reports, and TensorBoard events.
Checkpoints, datasets, and results remain outside Git.

In another Gamma terminal, from the directory containing the extracted package:

```bash
bash gamma-performance/tensorboard.sh
```

Open `http://localhost:6008` in a browser on Gamma, or arrange your own port
forwarding. The script binds the port to Gamma's loopback interface. Select
Primary or Full-scan-stress, then Text → Progress or Scalars → Completion percent.

For individual module commands, `deploy/gamma/docker-performance.sh` runs the
performance CLI from a repository checkout. It expects `manifest.json` in the
checkout's parent directory and accepts `GAMMA_DATA_ROOT` for data; it does not
use `GAMMA_BUNDLE_ROOT`. These paths differ from the extracted campaign launcher.
See `python -m rwkv_medsam2.evaluation.performance --help` for subcommands.
