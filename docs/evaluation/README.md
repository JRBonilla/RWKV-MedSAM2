# Accuracy and performance evaluation

The shared prediction path compares corrected Hiera, RWKV distilled/non-distilled,
Oxford and UofT with strict checkpoint hashes, identical 512×512 inputs, frozen
prompts, FP32 and math attention. External model code and weights are supplied as
runtime assets; legacy Hiera is excluded. See [shared-protocol.json](shared-protocol.json)
for the frozen protocol specification. Its historical draft status is retained
because changing this file changes the protocol identity.

## Accuracy

`python -m rwkv_medsam2.evaluation.accuracy --help` exposes preparation, inference,
reporting and boundary analysis. Install the optional analysis dependencies from
`rwkv_medsam2/evaluation/requirements.txt` in the evaluation environment.

- Independent 2D images use boxes and fresh state. Anatomical 3D positive-frame
  sequences are primary; complete stored scans are a separate robustness pass.
  CAMUS is video, not anatomical depth. Stored scans may already be cropped.
- Sequence support prompts cover 25% of positive frames, with positive queries
  retained; box and mask conditions are separate. Query Dice/IoU and all-frame
  assisted Dice/IoU are reported separately. Prompt frames may echo supplied
  masks by the common protocol; assisted scores are labelled accordingly.
- Predictions, per-frame diagnostics, HD95, NSD and ASSD are recorded. Surface
  metrics run in bounded CPU workers while GPU inference proceeds. Unverified
  legacy spacing is labelled in pixels/voxels, not millimetres.
- All test data is evaluated. Audited exact-image/scan, same-scan/video and
  same-patient training overlaps are separate reporting categories. Unused
  reserved-pool membership is not called training exposure. Uncertain source
  histories and external pretraining remain explicit limitations.
- Aggregation balances targets, cases, sources, tasks and datasets. Shared source
  aliases share bootstrap draws. Reporting categories reuse the same predictions.

Operational entry points:

```bash
python scripts/prepare_categorized_accuracy.py --inventory INVENTORY.jsonl --audit-dir AUDIT_DIR --output COHORT
python scripts/run_categorized_accuracy.py --cohort COHORT --checkpoints CHECKPOINTS.json --output RESULTS
python scripts/monitor_categorized_accuracy.py --help
```

The cohort preparer consumes reviewed inventory, research grouping metadata and
hashed overlap-audit evidence. These study inputs live outside Git; it does not
execute an audit script from a reports folder. Historical training caches are
never silently repaired. The controller validates its small launch check before
running four disjoint GPU shards per model. TensorBoard and JSON status preserve
progress; final reports include main/supplementary views and with/without-PAIP
summaries. No model is rerun just to obtain another reporting category.

## Performance

`python -m rwkv_medsam2.evaluation.performance --help` exposes panel selection,
serial measurement and reports. [Gamma deployment](../../deploy/gamma/README.md)
provides pinned CUDA 11.8 setup and the final campaign launcher.

The fixed primary panel covers all staged datasets/tasks and available length
strata; uncertain patient IDs are retained and labelled for timing coverage.
The final selector adds twelve short/middle/long complete-input stress cases
across CT, MRI, ultrasound volumes and ultrasound video, reported separately.
Sample counts depend on frozen input availability; the current study has 921
primary targets on 842 image files and twelve stress inputs.

Three warmups precede each worker. Two opposite-order blocks have three timing
repetitions each. One dedicated fresh-process memory worker records baseline,
peak and delta allocated/reserved GPU memory plus CPU RSS; CPU lifetime peaks
include loading and preparation. Complete prediction includes transfers and
returned masks but excludes decoding, loading, metrics and logging. Encoder
CUDA-event timing excludes transfers and decoder/propagation. Each case/model/
prompt first passes same-machine binary prediction parity against accuracy.
This does not certify cross-machine bitwise parity or A100 performance.

## Integrity and generated files

Frozen manifests bind data, prompts and evaluation Python source hashes. Do not
rewrite source under a live campaign or relabel old manifests after code changes.
Campaigns fail on errors rather than silently shortening inputs or dropping cases.
`artifacts/` and `reports/` hold generated manifests, audit evidence, transfer
archives, timing experiments and run outputs; they are not committed. The
regression tests under each evaluator protect protocol and reporting behavior.
