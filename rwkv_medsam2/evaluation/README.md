# Evaluation Code

The evaluators have separate packages and tests:

```text
evaluation/
  accuracy/
    runner.py          # Cohort freezing and accuracy inference
    cohorts.py         # Reviewed provenance and overlap-audit inputs
    categorized.py     # Reports grouped by training overlap and provenance
    records.py         # Prediction archives and accuracy diagnostics
    report.py          # Accuracy summaries and TensorBoard
    surfaces.py        # NSD, HD95, and ASSD analysis
    inputs.py          # Bounded verified image/mask reuse
    surface_workers.py # Background CPU analysis and verified result reuse
    tests/
  performance/
    runner.py          # Serial campaigns and measurement workers
    data.py            # Timing panel selection and frozen trial schedules
    core.py            # Prepared prediction path and parameter counts
    report.py          # Performance summaries and TensorBoard
    tests/
  shared/
    data.py            # Frozen inputs, prompts, hashes, and metric primitives
    models.py          # Pinned model loading and common runtime controls
    prediction.py      # Frame encoding/tracking used by both evaluators
```

Each evaluator has `__init__.py` and `__main__.py`. Run
`python -m rwkv_medsam2.evaluation.accuracy --help` or
`python -m rwkv_medsam2.evaluation.performance --help` for its subcommands.
Importing either package does not launch evaluation.

Performance uses accuracy's frozen-cohort reader, untimed reference evaluation,
and averaging helpers. Its `plan` command selects locally available,
source-verified targets and creates the standard schedule. The Gamma finalizer
then creates a separate schedule with fewer repetitions and a fixed full-scan
stress panel; see the deployment guide for that workflow.

Frozen manifests and plans record SHA256 hashes of evaluation Python files
recursively, with paths relative to this directory. Files under `tests/` are
excluded from that engine fingerprint. Hashes use source bytes, so docstrings,
comments, and formatting changes also change the fingerprint. Use the matching
source snapshot for an existing cohort, or prepare new manifests and plans for
the updated source. Never substitute new hashes into old measurement records.

Run identities also include a broader source snapshot of model and preprocessing
code, configurations, CUDA sources, and available external model repositories.
README edits in this checkout do not change either source fingerprint. Existing
checkpoint weights and saved results are not modified by documentation edits.

See the [evaluation guide](../../docs/evaluation/README.md) and
[Gamma deployment guide](../../deploy/gamma/README.md).
