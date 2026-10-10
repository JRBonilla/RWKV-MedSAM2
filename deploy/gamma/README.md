# Gamma deployment

Gamma uses two GTX 1070 cards (8 GiB each), 16 GiB RAM and Docker GPU passthrough.
The pinned image uses Python 3.10, PyTorch 2.7.1/CUDA 11.8 and native sm_61 WKV
kernels. `env.sh` disables Inductor/Triton compilation and limits CPU threads.

Build and check the environment with `bash deploy/gamma/docker-preflight.sh`.
This checks runtime compatibility, not model accuracy or benchmark coverage.

## Final fixed campaign

Create full primary/full-scan parent plans with the performance `plan` command
using matching frozen accuracy manifests and staged data. Keep the plan seed,
case/target hashes and source uncertainty. Then finalize once:

```bash
python deploy/gamma/finalize-performance-benchmark.py --primary-plan PRIMARY.json --full-scan-plan FULL.json --output FINAL_PLANS
python deploy/gamma/package.py --plans FINAL_PLANS --checkpoints CHECKPOINTS.json --output artifacts/gamma-performance.tar.gz
```

The packager checks source/plan identity and includes `dripp/defaults.ini` and
required model source. It does not include checkpoints, external repositories or
datasets. These remain in the existing Gamma bundle: `~/gamma-eval/manifest.json`,
`checkpoints/`, `external/oxford`, `external/uoft`; data is under `~/gamma-data`.
The bundle manifest must map all five current model names to pinned hashes.
`GAMMA_BUNDLE_ROOT` and `GAMMA_DATA_ROOT` override these host paths.

Transfer the archive and checksum, verify, extract, and run in a tmux terminal:

```bash
sha256sum -c gamma-performance.tar.gz.sha256
tar -xzf gamma-performance.tar.gz
bash gamma-performance/start.sh
```

One command runs primary measurements and the fixed full-scan stress panel
sequentially on GPU 1, then writes both reports. Six timing repetitions, three
warmups and one memory worker per case/model/prompt are frozen in the plans.
All five models receive the same cases and prompts. Legacy Hiera is excluded.
The current panel's pilot-derived estimate is roughly 4–6 days, with long-input
runtime uncertainty; this is not a guaranteed deadline or a universal sample-size
claim. No adaptive expansion is scheduled.

Rerun the same command to resume validated completed trials. A hard kill may
leave `.writer.claim`: check its owner and active processes before removing it.
Do not run another GPU benchmark concurrently. GPU boundary telemetry is saved;
transient interference is not continuously monitored.

## Progress

The terminal reports stage/model/dataset/frame count and measurement job.
`repo/artifacts/gamma/performance-campaign/` contains raw trials, logs, status,
reports and TensorBoard events. Checkpoints, data and results stay outside Git.

In another Gamma terminal run `bash gamma-performance/tensorboard.sh`, then open
http://localhost:6008 **inside Gamma's RDP desktop browser**. Select Primary or
Full-scan-stress, then Text → Progress or Scalars → Completion percent. The
SSH-only Guacamole session does not forward that port to your own computer.

`docker-performance.sh` remains available for explicit individual CLI commands.
