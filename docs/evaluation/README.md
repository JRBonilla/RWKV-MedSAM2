# Accuracy And Performance Evaluation

The shared prediction path compares corrected Hiera, distilled and non-distilled
RWKV, Oxford, and UofT models with pinned checkpoint hashes, identical 512×512
inputs, frozen prompts, FP32 inference, and math attention. External model code
and weights are runtime assets; the historical Hiera checkpoint is excluded.

[shared-protocol.json](shared-protocol.json) records the protocol specification.
It retains its historical draft status and is not a complete launch configuration:
its corrected Hiera checkpoint entry is unset. Supply a separate checkpoint JSON
with final paths and SHA256 hashes for the model names you run:

```text
corrected_hiera
rwkv_medsam2_final_nodistill
rwkv_medsam2_final_distill
oxford_medical_sam2
uoft_medsam2
```

Run commands from the repository root in an environment with the project and
model dependencies installed. Install the additional analysis dependencies with:

```bash
python -m pip install -r rwkv_medsam2/evaluation/requirements.txt
```

## Accuracy

`python -m rwkv_medsam2.evaluation.accuracy --help` lists cohort preparation,
inference, reporting, and boundary analysis commands.

- Independent 2D images use boxes and fresh prediction state. Anatomical 3D
  target-positive frame sequences are primary; complete stored scans are a
  separate robustness condition. CAMUS is video. Stored inputs may already be
  cropped and resized.
- The default sequence policy selects `min(P-1, max(1, ceil(0.25*P)))` support
  frames from `P >= 2` positive frames, leaving at least one positive query.
  Box and mask conditions are separate. Query Dice/IoU and all-frame assisted
  scores are reported separately; supplied masks may be echoed on support frames.
- Predictions and per-frame diagnostics are archived. NSD, HD95, and ASSD can run
  in background CPU workers with `run --surface-workers 1`, or later with the
  `surfaces` command. The direct `run` command defaults to zero background workers.
  Unverified physical spacing is reported in pixels/voxels rather than millimetres.
- Reports balance targets, cases, sources, tasks, and datasets. Where reviewed
  overlap metadata is supplied, reporting categories distinguish exact-image/scan,
  same-scan/video, and same-patient training overlap from unused reserved-pool
  membership. Unresolved source identities and external pretraining remain
  explicit limitations. Categories reuse predictions.

For your own reviewed case records, use the module commands. The
[example JSONL](reviewed-cases.example.jsonl) shows the expected input structure;
replace its paths and review its provenance fields for your data:

```bash
python -m rwkv_medsam2.evaluation.accuracy prepare \
  --cases REVIEWED.jsonl --data-root /data --output FROZEN
python -m rwkv_medsam2.evaluation.accuracy derive-full-scan \
  --manifest FROZEN --output FULL_FROZEN
python -m rwkv_medsam2.evaluation.accuracy run \
  --manifest FROZEN --checkpoints CHECKPOINTS.json \
  --model corrected_hiera --device cuda:0 --data-root /data --output RESULTS
python -m rwkv_medsam2.evaluation.accuracy summarize \
  --manifest FROZEN --runs RESULTS --models corrected_hiera --output REPORT
```

### Categorized Study Campaign

The repository also contains a study-specific controller:

```bash
python scripts/prepare_categorized_accuracy.py --inventory INVENTORY.jsonl --audit-dir AUDIT_DIR --output COHORT
python scripts/run_categorized_accuracy.py --cohort COHORT --checkpoints CHECKPOINTS.json --output RESULTS
python scripts/monitor_categorized_accuracy.py --output RESULTS --cohort COHORT
```

To resume the same campaign, rerun the controller with `--resume` and the same
`--cohort`, `--checkpoints`, and `--output`. It verifies completed job identities
and boundary-metric coverage before reusing their completion markers; unfinished
shards resume through the accuracy runner. The controller checks all frozen
manifests before scheduling models and checks runtime source before launching
workers. Restore the campaign's exact source snapshot if these checks reject a
source change; retain newer edits separately and keep existing manifests intact.

This preparer consumes reviewed inventory, local research grouping metadata,
and hashed overlap-audit evidence outside Git. It requires the study's 56-dataset
coverage and assumes `/data` paths. It does not execute audit scripts or rewrite
historical training caches. The controller validates pilot reports, then runs
four disjoint GPU shards per model and condition; it requires four CUDA devices.
The direct module CLI supports explicit shard counts for other environments.

JSON status and TensorBoard preserve progress. Final categorized reports include
main/supplementary views and with/without-PAIP summaries. Changing reporting
categories does not require another prediction pass.

## Performance

`python -m rwkv_medsam2.evaluation.performance --help` lists panel selection,
serial measurement, and reporting commands.

The standard `plan` command selects eligible, source-verified targets whose image
and mask files exist under the staged data root. It excludes unresolved source
identities. Within each dataset/task/dimensionality group, it selects up to 20
2D sources, or up to five sources in each short/medium/long sequence stratum,
plus the longest available sequence as a stress target. Counts depend on the
frozen cohort and staged inputs; complete dataset coverage is not guaranteed.

The standard schedule uses three warmups per measurement worker, two
opposite-order timing blocks with five repetitions each, and three separate
memory workers per case/model/prompt. The
[Gamma finalizer](../../deploy/gamma/README.md) creates a different fixed schedule:
three repetitions per timing block and one memory worker, with a separate
short/middle/long full-scan stress panel. Stress panel size depends on the
modality/sequence-kind groups in the supplied parent plan.

A direct plan and run can be created with:

```bash
python -m rwkv_medsam2.evaluation.performance plan \
  --manifest FROZEN --models corrected_hiera rwkv_medsam2_final_nodistill \
  --data-root STAGED_DATA --source-root /data --output PLAN.json
python -m rwkv_medsam2.evaluation.performance run \
  --plan PLAN.json --checkpoints CHECKPOINTS.json \
  --data-root STAGED_DATA --source-root /data --device cuda:1 --output TIMINGS
python -m rwkv_medsam2.evaluation.performance summarize \
  --runs TIMINGS --output TIMING_REPORT
```

The performance runner checks for competing GPU work. Its default device is
`cuda:1`; choose a device explicitly when using a different machine.

Complete prediction timing includes transfers and returned masks but excludes
input decoding, model loading, metrics, and logging. Encoder CUDA-event timing
excludes transfers and decoder/propagation. Memory workers record baseline,
peak, and delta allocated/reserved GPU memory plus CPU RSS; CPU lifetime peaks
include loading and preparation. Every case/model/prompt first passes
same-machine binary prediction parity against the accuracy path. This does not
certify cross-machine bitwise parity or predict another GPU's performance.

## Integrity And Generated Files

Frozen manifests bind data, prompts, protocol, input-library versions, and raw
evaluation Python source hashes. Docstrings, comments, and formatting change those
hashes even when executable behavior is identical. Existing manifests require
the source snapshot they were frozen with; prepare new cohorts and plans to use
updated evaluator source. Broader run identities also hash model code and other
runtime assets. Do not rewrite source during a campaign or replace old identity
hashes to make changed code appear compatible.

Documentation changes do not alter trained weights or require retraining.
Existing result files remain associated with their original source identity.
README files are excluded from evaluator source fingerprints, so edits to these
guides do not change them.

Runs reject invalid inputs, identity mismatches, and failed predictions rather
than silently shortening inputs or dropping cases. `artifacts/` and `reports/`
hold generated manifests, audit evidence, archives, and run outputs outside Git.
Regression tests live under each evaluator's `tests/` directory.
