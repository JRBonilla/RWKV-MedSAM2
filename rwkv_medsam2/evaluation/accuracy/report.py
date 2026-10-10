"""Coverage-aware aggregation and paired source-cluster bootstrap; no model imports."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from ..shared.data import (
    case_kind,
    SEED,
    atomic_json,
    digest,
    file_hash,
    validate_result,
)


def mean(values):
    """
    Compute an arithmetic mean while handling an empty iterable.

    Args:
        values (Iterable[float]): Numeric values to summarize.

    Returns:
        float | None: Mean value, or None when no values are supplied.
    """
    values = list(values)

    return sum(values) / len(values) if values else None


def source_means(rows, metric):
    """
    Equal targets -> cases -> sources. Rows already belong to one dimension/mode.

    Args:
        rows (Iterable[dict]): Result records to aggregate.
        metric (str): Metric key to extract from each score record.

    Returns:
        dict[tuple, float]: Per-source scores keyed by dataset, task, and source.
    """
    cases = defaultdict(list)
    for row in rows:
        c = row["case"]
        cases[
            (
                c["dataset"],
                c["task"],
                c.get("global_source_id", c["source_id"]),
                c["case_id"],
            )
        ].append(row["primary"][metric])

    sources = defaultdict(list)
    for (dataset, task, source, _), values in cases.items():
        sources[(dataset, task, source)].append(mean(values))

    return {key: mean(values) for key, values in sources.items()}


def hierarchical(values, weights=None):
    """
    Average source scores within tasks, datasets, and the overall macro score.

    Args:
        values (dict[tuple, float]): Source scores keyed by dataset, task, and source
            identity.
        weights (dict | None): Optional sampling weights for the aggregation.

    Returns:
        dict: Macro, dataset, and task scores with optional source resampling weights.
    """
    tasks = defaultdict(list)
    for (dataset, task, source), value in values.items():
        weight = 1 if weights is None else weights.get((dataset, source), 0)
        tasks[(dataset, task)].append((value, weight))

    task_scores = {}
    for key, pairs in tasks.items():
        denominator = sum(w for _, w in pairs)
        if not denominator:
            raise ValueError("Bootstrap removed an entire task")

        task_scores[key] = sum(v * w for v, w in pairs) / denominator

    # Average within each task first so datasets with more tasks or sources
    # do not receive extra weight in the final macro score.
    datasets = defaultdict(list)
    for (dataset, _), value in task_scores.items():
        datasets[dataset].append(value)

    dataset_scores = {key: mean(values) for key, values in datasets.items()}
    return {
        "macro": mean(dataset_scores.values()),
        "datasets": dataset_scores,
        "tasks": {
            dataset + "/" + task: value
            for (dataset, task), value in task_scores.items()
        },
    }


def paired_interval(left, right, metric, replicates):
    """
    Resample paired source clusters; retain cross-task source correlations.

    Strata are dataset x source task-membership. This keeps each frozen task
    represented in every replicate, even for sources appearing in several tasks.

    Args:
        left (list[dict]): Case results from the first model in the paired comparison.
        right (list[dict]): Matching case results from the second model.
        metric (str): Metric key to extract from each score record.
        replicates (int): Number of bootstrap resampling replicates.

    Returns:
        dict: Estimated difference, 95 percent interval, and bootstrap provenance.
    """
    import numpy as np

    differences = []
    right_by_key = {r["unit_key"]: r for r in right}

    for row in left:
        other = right_by_key[row["unit_key"]]
        differences.append(
            dict(
                case=row["case"],
                primary={metric: row["primary"][metric] - other["primary"][metric]},
            )
        )

    values = source_means(differences, metric)
    estimate = hierarchical(values)["macro"]
    # A shared scan across dataset names receives one bootstrap draw.
    membership = defaultdict(set)

    for dataset, task, source in values:
        cluster = source if source.startswith("global:") else dataset + ":" + source
        membership[cluster].add((dataset, task))

    strata = defaultdict(list)
    for cluster, members in membership.items():
        strata[tuple(sorted(members))].append(cluster)

    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(replicates):
        cluster_weights = {}
        for members, clusters in sorted(strata.items()):
            clusters = sorted(clusters)
            draws = rng.integers(0, len(clusters), size=len(clusters))
            cluster_weights.update(Counter(clusters[int(i)] for i in draws))

        weights = {
            (dataset, source): cluster_weights.get(
                source if source.startswith("global:") else dataset + ":" + source, 0
            )
            for dataset, task, source in values
        }
        samples.append(hierarchical(values, weights)["macro"])

    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(
        difference=estimate,
        ci95=[float(low), float(high)],
        replicates=replicates,
        seed=SEED,
        source_clusters=len(membership),
        single_source_strata=sum(len(v) == 1 for v in strata.values()),
        method="paired source-cluster bootstrap, stratified by joint dataset/task membership; cross-dataset source aliases shared",
        limitation="Does not estimate training-seed variability; unresolved source IDs limit independence claims",
    )


def summarize(args):
    """
    Validate saved runs and write coverage-aware accuracy reports.

    Args:
        args (argparse.Namespace): Parsed command options, including input paths and report
            or trial settings.

    Returns:
        None.
    """
    from .runner import read_frozen
    from .records import distribution, verify_archive

    meta, cases = read_frozen(args.manifest)

    if args.bootstrap < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Duplicate model names")

    expected = {
        c["unit_id"] + ":" + mode: (c, mode)
        for c in cases
        if c["eligible"]
        for mode in c["modes"]
    }
    manifest_sha = file_hash(args.manifest / "manifest.json")
    groups = sorted({(case_kind(c), mode) for c, mode in expected.values()})
    results, coverage, input_hashes = {}, {}, {}
    result_paths = {}
    shared_source, shared_environment = None, None

    for model in args.models:
        rows = {}
        jobs = list((args.runs / model).glob("shard-*/job.json"))
        identity_id, shard_count, shards = None, None, set()

        for path in sorted(jobs):
            job = json.loads(path.read_text())
            input_hashes[str(path)] = file_hash(path)
            identity = job["identity"]

            if job["run_id"] != digest(identity) or identity["model"] != model:
                raise ValueError("Invalid model/run identity in job")
            if identity["manifest_sha256"] != manifest_sha:
                raise ValueError("Cannot merge a different frozen cohort")
            if identity_id is not None and identity_id != job["run_id"]:
                raise ValueError(
                    "Cannot merge differing model/checkpoint/code/environment identities"
                )

            identity_id = job["run_id"]
            if shard_count is not None and shard_count != job["shards"]:
                raise ValueError("Mixed shard layouts")

            shard_count = job["shards"]
            if job["shard"] in shards:
                raise ValueError("Duplicate shard")

            shards.add(job["shard"])
            intended = {
                key
                for key, (c, _) in expected.items()
                if int(digest([c["dataset"], c["case_id"]]), 16) % job["shards"]
                == job["shard"]
            }
            if set(job["expected"]) != intended or len(job["expected"]) != len(
                intended
            ):
                raise ValueError("Shard intended coverage differs from manifest")
            if shared_source is not None and shared_source != identity["source_sha256"]:
                raise ValueError(
                    "Models used different source snapshots; repeat under one frozen implementation"
                )

            shared_source = identity["source_sha256"]
            if (
                shared_environment is not None
                and shared_environment != identity["environment"]
            ):
                raise ValueError("Models used different library environments")

            shared_environment = identity["environment"]
            for result_path in sorted((path.parent / "cases").glob("*.json")):
                row = json.loads(result_path.read_text())
                key = row["unit_key"]
                if (
                    key not in intended
                    or key in rows
                    or row["run_id"] != identity_id
                    or row["status"] != "complete"
                ):
                    raise ValueError(
                        "Duplicate, unexpected, incomplete, or mismatched result"
                    )

                c, mode = expected[key]
                validate_result(row, c, mode, identity_id)
                verify_archive(result_path, row)

                if (
                    row["case_sha256"] != digest(c)
                    or row["model"] != model
                    or row["mode"] != mode
                ):
                    raise ValueError("Result metadata differs from the frozen case")

                expected_query = (
                    1
                    if c["dim"] == 2
                    else len(c["frame_ids"]) - len(c["support_frames"])
                )
                if (
                    row["frames"] != len(c["frame_ids"])
                    or row["query_frames"] != expected_query
                ):
                    raise ValueError("Result has incomplete frame coverage")

                rows[key] = row
                result_paths[(model, key)] = result_path
                input_hashes[str(result_path)] = file_hash(result_path)

        results[model] = rows
        coverage[model] = dict(
            expected=len(expected),
            completed=len(rows),
            missing=len(expected) - len(rows),
            missing_keys=sorted(set(expected) - set(rows)),
            found_shards=sorted(shards),
            intended_shards=shard_count,
            full_cohort_complete=len(rows) == len(expected),
            run_id=identity_id,
            by_dataset={
                dataset: {
                    "expected": sum(
                        c["dataset"] == dataset for c, _ in expected.values()
                    ),
                    "completed": sum(
                        r["case"]["dataset"] == dataset for r in rows.values()
                    ),
                    "excluded_targets": sum(
                        c["dataset"] == dataset and not c["eligible"] for c in cases
                    ),
                }
                for dataset in sorted({c["dataset"] for c in cases})
            },
        )

    # Refuse overwrite: report and TensorBoard aggregates represent one immutable snapshot.
    args.output.mkdir(parents=True, exist_ok=False)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(args.output / "tensorboard"))
    summary = {
        "profile": meta["profile"],
        "sequence_condition": meta["sequence_condition"],
        "study_scope": meta["scope"],
        "manifest_sha256": manifest_sha,
        "coverage": coverage,
        "excluded_cases": [
            dict(
                unit_id=c["unit_id"], dataset=c["dataset"], reason=c["exclusion_reason"]
            )
            for c in cases
            if not c["eligible"]
        ],
        "metrics_grid": meta["metric_grid"],
        "models": {},
        "paired": {},
        "boundary_metrics": "Run the separate surfaces command on saved predictions; never mix physical and pixel units",
        "scientific_protocol_certified": False,
    }
    table = []

    try:
        for model, all_rows in results.items():
            summary["models"][model] = {}
            writer.add_scalar(f"coverage/{model}/completed", len(all_rows), 0)
            writer.add_scalar(f"coverage/{model}/expected", len(expected), 0)

            for dim, mode in groups:
                rows = [
                    r
                    for r in all_rows.values()
                    if case_kind(r["case"]) == dim and r["mode"] == mode
                ]
                intended = sum(
                    case_kind(c) == dim and m == mode for c, m in expected.values()
                )
                key = f'{meta["sequence_condition"]}/{dim}/{mode}'
                item = {
                    "completed": len(rows),
                    "expected": intended,
                    "scope": (
                        "full_frozen_cohort"
                        if len(rows) == intended
                        else "partial_completed_only_not_final"
                    ),
                    "source_identity_unresolved_conditions": sum(
                        r["case"]["source_identity_status"] != "verified" for r in rows
                    ),
                }
                item["modalities"] = {
                    modality: {
                        metric: hierarchical(
                            source_means(
                                [r for r in rows if r["case"]["modality"] == modality],
                                metric,
                            )
                        )
                        for metric in ("dice", "iou")
                    }
                    for modality in sorted({r["case"]["modality"] for r in rows})
                }
                item["diagnostics"] = {
                    metric: distribution(
                        r["diagnostics"]["query"][metric] for r in rows
                    )
                    for metric in (
                        "precision",
                        "recall",
                        "specificity",
                        "relative_size_error",
                    )
                }
                item["descriptive_low_dice_counts"] = {
                    str(t): sum(r["primary"]["dice"] < t for r in rows)
                    for t in (0.1, 0.5, 0.8)
                }
                item["low_dice_cutoffs_are_not_clinical_failure_definitions"] = True
                item["worst_cases"] = [
                    dict(
                        unit_key=r["unit_key"],
                        dice=r["primary"]["dice"],
                        prediction_archive=r["prediction_archive"],
                    )
                    for r in sorted(
                        rows, key=lambda r: (r["primary"]["dice"], r["unit_key"])
                    )[:20]
                ]
                item["size_and_length_diagnostics"] = {
                    "target_pixels": distribution(
                        r["diagnostics"]["query"]["target_pixels"] for r in rows
                    ),
                    "frames": distribution(r["frames"] for r in rows),
                }
                item["target_scopes"] = dict(
                    Counter(r["case"]["target_scope"] for r in rows)
                )

                for metric in ("dice", "iou"):
                    aggregate = hierarchical(source_means(rows, metric))
                    item[metric] = aggregate
                    item[metric + "_target_distribution"] = distribution(
                        r["primary"][metric] for r in rows
                    )

                    if rows:
                        import numpy as np

                        writer.add_histogram(
                            f"distributions/{model}/{key}/{metric}",
                            np.asarray([r["primary"][metric] for r in rows]),
                            0,
                        )

                    item["target_weighted_" + metric] = mean(
                        r["primary"][metric] for r in rows
                    )
                    item["positive_query_overlap_" + metric] = hierarchical(
                        source_means(
                            [
                                dict(
                                    case=r["case"], primary=r["positive_query_overlap"]
                                )
                                for r in rows
                            ],
                            metric,
                        )
                    )
                    item["all_frame_assisted_" + metric] = hierarchical(
                        source_means(
                            [
                                dict(case=r["case"], primary=r["all_frame_assisted"])
                                for r in rows
                            ],
                            metric,
                        )
                    )

                    for label, scores in (
                        (metric, aggregate),
                        (
                            "all_frame_assisted_" + metric,
                            item["all_frame_assisted_" + metric],
                        ),
                    ):
                        if scores["macro"] is not None:
                            writer.add_scalar(
                                f'{item["scope"]}/{model}/{key}/{label}',
                                scores["macro"],
                                0,
                            )
                        for dataset, value in scores["datasets"].items():
                            writer.add_scalar(
                                f"dataset/{model}/{key}/{dataset}/{label}", value, 0
                            )
                            table.append(
                                dict(
                                    model=model,
                                    sequence_condition=meta["sequence_condition"],
                                    dimension=(
                                        "video" if dim == "video" else int(dim[0])
                                    ),
                                    mode=mode,
                                    dataset=dataset,
                                    metric=label,
                                    score=value,
                                    scope=item["scope"],
                                )
                            )

                summary["models"][model][key] = item

        comparisons = [
            ("rwkv_medsam2_final_nodistill", "corrected_hiera"),
            ("rwkv_medsam2_final_distill", "rwkv_medsam2_final_nodistill"),
        ]
        for rwkv in ("rwkv_medsam2_final_nodistill", "rwkv_medsam2_final_distill"):
            comparisons.extend(
                (rwkv, ref) for ref in ("oxford_medical_sam2", "uoft_medsam2")
            )
        for left, right in comparisons:
            if left not in results or right not in results:
                continue

            common = set(results[left]) & set(results[right])
            comparison = {}
            for dim, mode in groups:
                keys = sorted(
                    k
                    for k in common
                    if case_kind(expected[k][0]) == dim and expected[k][1] == mode
                )
                intended = sum(
                    case_kind(c) == dim and m == mode for c, m in expected.values()
                )
                item = dict(
                    jointly_completed=len(keys),
                    intended=intended,
                    scope=(
                        "full_frozen_cohort"
                        if len(keys) == intended
                        else "joint_completed_subset_not_final"
                    ),
                )

                if keys:
                    a, b = [
                        [results[model][key] for key in keys] for model in (left, right)
                    ]
                    for metric in ("dice", "iou"):
                        item[metric] = paired_interval(a, b, metric, args.bootstrap)
                        writer.add_scalar(
                            f"paired/{left}_minus_{right}/{dim}/{mode}/{metric}",
                            item[metric]["difference"],
                            0,
                        )

                comparison[f"{dim}/{mode}"] = item

            summary["paired"][left + "_minus_" + right] = comparison

        from .categorized import build_reports

        build_reports(results, cases, result_paths, writer, args.output, args.bootstrap)
        summary["aggregate_scope"] = (
            "all_data_including_overlap; use categorized-summary.json for main results"
        )
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "input-result-hashes.json", input_hashes)

        with (args.output / "dataset-scores.csv").open("w", newline="") as stream:
            fields = [
                "model",
                "sequence_condition",
                "dimension",
                "mode",
                "dataset",
                "metric",
                "score",
                "scope",
            ]
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            output.writerows(table)
        with (args.output / "per-case.jsonl").open("w") as stream:
            for model in args.models:
                for key in sorted(results[model]):
                    stream.write(json.dumps(results[model][key], sort_keys=True) + "\n")

        writer.add_text("coverage/status", json.dumps(coverage, indent=2), 0)
    finally:
        writer.close()

    print(
        json.dumps(
            {
                "report": str(args.output),
                "coverage": {
                    model: {k: v for k, v in item.items() if k != "missing_keys"}
                    for model, item in coverage.items()
                },
            }
        ),
        flush=True,
    )


class RunningMacro:
    """
    Incremental equal-weight hierarchy over completed targets; O(depth) updates.
    """

    def __init__(self):
        """
        Initialize the incremental dataset, task, source, and case hierarchy.

        Args:
            None.

        Returns:
            None.
        """
        self.root = {"children": {}, "sum": 0.0}

    def update(self, row, metric, score_key="primary"):
        """
        Insert a target score and update the equal-weight hierarchy.

        Args:
            row (dict): Saved evaluation result or diagnostic record.
            metric (str): Metric key to extract from each score record.
            score_key (str): Result field containing the score dictionary to aggregate.

        Returns:
            float: Updated overall macro score.
        """
        c = row["case"]
        path = [
            c["dataset"],
            c["task"],
            c.get("global_source_id", c["source_id"]),
            c["case_id"],
            c["target_id"],
        ]

        def insert(node, keys):
            """
            Update cached sums along the hierarchy for the current score.

            Args:
                node (dict): Current node in the incremental aggregation tree.
                keys (tuple): Remaining hierarchy keys for the current target score.

            Returns:
                None.
            """
            key = keys[0]
            if len(keys) == 1:
                old = node["children"].get(key, 0.0)
                new = row[score_key][metric]
                node["children"][key] = new
            else:
                child = node["children"].setdefault(key, {"children": {}, "sum": 0.0})
                old = (
                    child["sum"] / len(child["children"]) if child["children"] else 0.0
                )
                insert(child, keys[1:])
                new = child["sum"] / len(child["children"])

            node["sum"] += new - old

        insert(self.root, path)
        return self.root["sum"] / len(self.root["children"])
