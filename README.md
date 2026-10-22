# RWKV-MedSAM2

RWKV-MedSAM2 explores replacing SAM2's image encoder with an RWKV-based encoder
for medical segmentation: predicting masks around structures in images, scans,
and videos, guided by box, point, or mask prompts. Scans are processed as slice
sequences, and videos as frame sequences.

SAM2 extracts image features, interprets prompts, and uses memory to carry
information across a sequence. RWKV provides recurrent feature mixing as an
alternative to attention in the image encoder. This project supplies image
features with **VCR**, a hybrid convolutional/RWKV backbone, while retaining
SAM2's prompt and memory prediction pipeline. The aim is to compare segmentation
quality and computational cost against **Hiera**, SAM2's original hierarchical
transformer encoder.

The workflow is to standardize datasets with DRIPP, build training loaders,
train a model variant, and evaluate its accuracy and performance separately.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `rwkv_medsam2/` | The medical segmentation code: VCR, dataset sampling, training, and evaluation. |
| `ext/` | Local SAM2 and vision RWKV implementations, including CUDA kernels and model configurations. |
| `dripp/` | Turns different raw dataset layouts into grouped, normalized images and masks. Its Tk debugger helps inspect grouping and outputs. |
| `scripts/` | Command-line entry points for loader preparation, training, and repeatable study runs. |
| `notebooks/` | Interactive experiments and historical training/benchmark workflows. |
| `deploy/gamma/` | A separate environment for measuring models on the smaller GTX 1070 system. |

## Environment

Use Python 3.10 or newer. Model training requires a CUDA-capable PyTorch
installation and a CUDA toolchain for the RWKV extensions. Datasets, pretrained
initialization weights, teacher weights, and final checkpoints are supplied
separately.

The root [Dockerfile](Dockerfile) keeps the dependencies and CUDA builds together,
including PyTorch 2.7.1/CUDA 12.8 and CUDA-enabled OpenCV.
[Docker Compose](docker-compose.yml) targets four A100 GPUs and NFS storage.
Copy [.env.example](.env.example) to `.env` and configure all four GPU IDs,
storage paths, and service settings for your host:

```bash
cp .env.example .env
# Edit .env for your host before launching the container.
docker compose build training
docker compose up -d training
docker compose exec training bash
```

The smaller Gamma evaluation environment has its own Dockerfile and dependency
pins; follow the [Gamma deployment guide](deploy/gamma/README.md).

## Prepare Data

Medical datasets use different file formats, folder layouts, and class labels.
DRIPP groups matching images and masks, normalizes intensities, and filters mask
components so training can read a common layout. Follow the
[DRIPP guide](dripp/README.md) for indexing, CT profile preparation, and
preprocessing.

Training loaders turn those processed groups into prompted examples and sampled
sequences. [scripts/build_loaders.py](scripts/build_loaders.py) builds and saves
them using
[ext/sam2/configs/sam2.1/sam2.1_vcr.yaml](ext/sam2/configs/sam2.1/sam2.1_vcr.yaml)
and writes `/data/loaders32.pkl`. Set its `dripp.output_dir` and
`dripp.tasks_file` paths to your processed data and task catalog before building
loaders. The script also assumes the checkout is at `/RWKV-MedSAM2`.

## Train

Run from the repository root in the configured CUDA environment:

```bash
python scripts/train_headless.py --help
python scripts/train_headless.py --mode nodistill \
  --loaders /data/loaders32.pkl \
  --experiment-root /data/TrainingLogs/my_run \
  --checkpoint-root /data/TrainingCheckpoints/my_run \
  --run-id example
```

The training modes separate the encoder comparison from the effect of teacher
supervision:

| Mode | What it trains |
| --- | --- |
| `nodistill` | RWKV/VCR using segmentation labels, without a distillation loss. |
| `distill` | RWKV/VCR with an additional loss that learns from a teacher model's mask predictions. |
| `base` | SAM2 with the Hiera encoder, providing the comparison baseline. |

The RWKV modes use `sam2.1_vcr.yaml` or `sam2.1_vcr_nodistill.yaml`; `base` uses
`sam2.1_hiera_t512.yaml` for the model. Check initialization paths in these
configs before running. `distill` also needs a teacher checkpoint, which can be
overridden with `--teacher-checkpoint`.

Use `torchrun --standalone --nproc_per_node=N scripts/train_headless.py ...` for
multiple GPUs; the global batch size must divide evenly across processes.
`--resume CHECKPOINT` restores training state. For the default variant, logs and
checkpoints go under the supplied roots in `{mode}/{run-id}`.

The launch scripts under `scripts/` contain study-specific paths and settings;
review them before using them for a different run.

## Evaluate And Test

Accuracy evaluation measures how closely predicted masks match the targets.
Performance evaluation measures prediction time, encoder time, and memory use.
They share inputs and prompts so the comparison can examine both segmentation
quality and computational cost. Frozen cohorts record the exact cases, prompts,
and source hashes used in a run.

The [evaluation guide](docs/evaluation/README.md) explains these workflows and
checkpoint requirements. The
[evaluation code overview](rwkv_medsam2/evaluation/README.md) maps the modules and
tests.

In an environment with the runtime dependencies installed, install the test and
analysis dependencies and run:

```bash
python -m pip install pytest -r rwkv_medsam2/evaluation/requirements.txt
python -m pytest dripp/tests rwkv_medsam2/tests \
  rwkv_medsam2/evaluation/accuracy/tests \
  rwkv_medsam2/evaluation/performance/tests deploy/gamma/tests \
  deploy/gamma/test_compile_policy.py
```

Frozen evaluators hash Python source, including documentation and formatting.
See the guide before reusing a manifest after source edits; retraining is not
required for documentation changes.
