"""Reports reuse existing predictions; categories never change inference."""

from collections import Counter
import csv
import json

from ..shared.data import case_kind, file_hash
from .cohorts import report_views, LIMITATION


def build_reports(results, cases, result_paths, writer, output, bootstrap):
    """
    Write category-specific coverage, diagnostics, and paired model comparisons.

    Args:
        results (dict): Saved case results grouped by model.
        cases (list[dict]): Frozen evaluation cases.
        result_paths (dict): Saved result paths grouped by model and condition.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for evaluation
            summaries.
        output (pathlib.Path): Destination directory for saved evaluation artifacts.
        bootstrap (int): Number of paired bootstrap resampling replicates.

    Returns:
        dict: Summaries keyed by reporting view, PAIP policy, dimension, and prompt mode.
    """
    from .report import source_means, hierarchical, paired_interval

    expected = [(c, mode) for c in cases if c["eligible"] for mode in c["modes"]]
    views = sorted({v for c in cases for v in report_views(c)})
    conditions = sorted({(case_kind(c), mode) for c, mode in expected})
    summary = {
        "limitation": LIMITATION,
        "views_overlap": True,
        "views": {},
        "paired": {},
    }
    table, surface_cache = [], {}

    for model, rows in results.items():
        for key, row in rows.items():
            path = result_paths[(model, key)]
            surface = path.parent.parent / "surfaces/cases" / path.name
            if surface.exists():
                s = json.loads(surface.read_text())
                if (
                    s["result_sha256"] != file_hash(path)
                    or s["case_sha256"] != row["case_sha256"]
                    or s["run_id"] != row["run_id"]
                    or s["unit_key"] != key
                ):
                    raise ValueError("Boundary result differs from prediction")

                surface_cache[(model, key)] = s
    for view in views:
        for without_paip in (False, True):
            name = view + ("/without_PAIP" if without_paip else "/with_PAIP")

            def included(c):
                """
                Check whether a case belongs in the current reporting view.

                Args:
                    c (dict): Evaluation case metadata.

                Returns:
                    bool: Whether the case passes view membership and the PAIP filter.
                """
                return view in report_views(c) and (
                    not without_paip or c["dataset"] != "PAIP2019"
                )

            intended = [(c, mode) for c, mode in expected if included(c)]
            view_results = summary["views"][name] = dict(
                expected_conditions=len(intended),
                datasets=sorted({c["dataset"] for c, _ in intended}),
                models={},
            )
            for model, all_rows in results.items():
                model_summary = view_results["models"][model] = {}
                for kind, mode in conditions:
                    rows = [
                        r
                        for r in all_rows.values()
                        if included(r["case"])
                        and case_kind(r["case"]) == kind
                        and r["mode"] == mode
                    ]
                    count = sum(case_kind(c) == kind and m == mode for c, m in intended)
                    if not count:
                        continue

                    status = (
                        "complete"
                        if len(rows) == count
                        else "partial_completed_only_not_final"
                    )
                    key = f"{kind}/{mode}"
                    item = model_summary[key] = dict(
                        completed=len(rows), expected=count, scope=status
                    )

                    for score_key, prefix in [
                        ("primary", ""),
                        ("all_frame_assisted", "all_frame_assisted_"),
                    ]:
                        for metric in ("dice", "iou"):
                            label = prefix + metric

                            def aggregate(subset):
                                """
                                Compute the current metric's hierarchical mean for a result subset.

                                Args:
                                    subset (list[dict]): Result records included in the current reporting group.

                                Returns:
                                    dict: Macro, dataset, and task scores for the selected records.
                                """
                                return hierarchical(
                                    source_means(
                                        [
                                            dict(case=r["case"], primary=r[score_key])
                                            for r in subset
                                        ],
                                        metric,
                                    )
                                )

                            scores = item[label] = aggregate(rows)
                            modalities = {
                                m: aggregate(
                                    [r for r in rows if r["case"]["modality"] == m]
                                )["macro"]
                                for m in sorted({r["case"]["modality"] for r in rows})
                            }
                            item.setdefault("modalities", {})[label] = modalities

                            for level, mapping in [
                                ("macro", {"all": scores["macro"]}),
                                ("dataset", scores["datasets"]),
                                ("task", scores["tasks"]),
                                ("modality", modalities),
                            ]:
                                for group, value in mapping.items():
                                    if value is None:
                                        continue

                                    table.append(
                                        dict(
                                            view=name,
                                            model=model,
                                            condition=key,
                                            level=level,
                                            group=group,
                                            metric=label,
                                            score=value,
                                            scope=status,
                                        )
                                    )
                                    writer.add_scalar(
                                        f"categorized/{name}/{model}/{key}/{level}/{group}/{label}",
                                        value,
                                        0,
                                    )

                    surfaces = [
                        surface_cache[(model, r["unit_key"])]
                        for r in rows
                        if (model, r["unit_key"]) in surface_cache
                    ]
                    item["boundary_completed"] = len(surfaces)
                    item["boundaries"] = {}

                    for unit_scope in sorted(
                        {(s["units"], s["scope"]) for s in surfaces}
                    ):
                        subset = [
                            s
                            for s in surfaces
                            if (s["units"], s["scope"]) == unit_scope
                        ]
                        boundary = item["boundaries"]["/".join(unit_scope)] = dict(
                            completed=len(subset),
                            status_counts=dict(Counter(s["status"] for s in subset)),
                        )
                        for metric in ("hd95", "assd", "nsd_1.0", "nsd_2.0", "nsd_3.0"):
                            values = [
                                (
                                    s,
                                    (
                                        s["nsd"][metric[4:]]
                                        if metric.startswith("nsd_")
                                        else s[metric]
                                    ),
                                )
                                for s in subset
                            ]
                            valid = [
                                (s, value) for s, value in values if value is not None
                            ]
                            scores = hierarchical(
                                source_means(
                                    [
                                        dict(case=s["case"], primary={metric: value})
                                        for s, value in valid
                                    ],
                                    metric,
                                )
                            )
                            boundary[metric] = dict(
                                defined=len(valid),
                                undefined=len(values) - len(valid),
                                **scores,
                            )

                            if scores["macro"] is not None:
                                writer.add_scalar(
                                    f'categorized_boundary/{name}/{model}/{key}/{"/".join(unit_scope)}/{metric}',
                                    scores["macro"],
                                    0,
                                )
                                table.append(
                                    dict(
                                        view=name,
                                        model=model,
                                        condition=key,
                                        level="boundary_macro",
                                        group="/".join(unit_scope),
                                        metric=metric,
                                        score=scores["macro"],
                                        scope="defined_only; see undefined counts in JSON",
                                    )
                                )

            pairs = [
                ("rwkv_medsam2_final_nodistill", "corrected_hiera"),
                ("rwkv_medsam2_final_distill", "rwkv_medsam2_final_nodistill"),
            ]
            pairs += [
                (m, ref)
                for m in ("rwkv_medsam2_final_nodistill", "rwkv_medsam2_final_distill")
                for ref in ("oxford_medical_sam2", "uoft_medsam2")
            ]
            for left, right in pairs:
                if left not in results or right not in results:
                    continue
                for kind, mode in conditions:
                    keys = sorted(
                        k
                        for k in results[left].keys() & results[right].keys()
                        if included(results[left][k]["case"])
                        and case_kind(results[left][k]["case"]) == kind
                        and results[left][k]["mode"] == mode
                    )
                    count = sum(case_kind(c) == kind and m == mode for c, m in intended)
                    if not keys:
                        continue

                    a, b = [[results[m][k] for k in keys] for m in (left, right)]
                    paired = dict(
                        completed=len(keys),
                        expected=count,
                        scope=(
                            "complete"
                            if len(keys) == count
                            else "partial_paired_intersection"
                        ),
                    )
                    for metric in ("dice", "iou"):
                        paired[metric] = paired_interval(a, b, metric, bootstrap)

                    summary["paired"][
                        f"{name}/{left}_minus_{right}/{kind}/{mode}"
                    ] = paired

    (output / "categorized-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    with (output / "categorized-scores.csv").open("w", newline="") as handle:
        out = csv.DictWriter(
            handle,
            fieldnames=[
                "view",
                "model",
                "condition",
                "level",
                "group",
                "metric",
                "score",
                "scope",
            ],
        )
        out.writeheader()
        out.writerows(table)

    return summary
