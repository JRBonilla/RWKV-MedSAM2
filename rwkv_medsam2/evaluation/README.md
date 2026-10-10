# Evaluation code

The two evaluators have separate packages and tests:

```text
evaluation/
  accuracy/
    runner.py          # Cohort freezing and accuracy inference
    records.py         # Prediction archives and accuracy diagnostics
    report.py          # Accuracy summaries and TensorBoard
    surfaces.py        # NSD, HD95 and ASSD analysis
    inputs.py          # Bounded verified image/mask reuse
    surface_workers.py # Background CPU analysis and verified result reuse
    tests/
  performance/
    runner.py          # Serial Gamma campaign and measurement workers
    data.py            # Timing panel and trial schedule
    core.py            # Prepared prediction path and parameter counts
    report.py          # Performance summaries and TensorBoard
    tests/
  shared/
    data.py            # Frozen inputs, prompts, hashes and metric primitives
    models.py          # Model loading and common runtime controls
    prediction.py      # Frame encoding/tracking used by both evaluators
```

Each evaluator has `__init__.py` and `__main__.py`. Existing CLI commands are
unchanged: `python -m rwkv_medsam2.evaluation.accuracy` and
`python -m rwkv_medsam2.evaluation.performance`, followed by the subcommand.
Importing either package does not launch work.

Performance deliberately uses accuracy's frozen-cohort reader, untimed reference
evaluation, and averaging helpers rather than duplicating their definitions.

Source fingerprinting covers Python implementation files recursively, with paths
relative to this directory so matching module basenames cannot collide. Tests
are excluded from that engine fingerprint. This folder reorganization changes
the fingerprint; existing frozen manifests/plans require regeneration before
use. It does not change the prediction, prompt, timing or scoring policies.

See the [evaluation guide](../../docs/evaluation/README.md) and
[Gamma deployment guide](../../deploy/gamma/README.md).
