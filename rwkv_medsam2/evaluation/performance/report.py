"""Summaries of raw performance trials; never launches a model or benchmark."""

from collections import defaultdict
import csv
import json

from ..shared.data import atomic_json, digest, file_hash
from ..accuracy.report import hierarchical, paired_interval, source_means
from ..accuracy.records import distribution
from .data import read_plan


def summarize(args):
    """
    Validate completed trials and report paired timing and memory measurements.

    Args:
        args (argparse.Namespace): Parsed command options, including input paths and report
            or trial settings.

    Returns:
        None.
    """
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    if args.bootstrap < 1:
        raise ValueError("Positive bootstrap count required")

    plan = read_plan(args.runs / "plan.json")
    campaign = json.loads((args.runs / "campaign.json").read_text())
    if campaign["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Campaign/plan mismatch")

    expected = {
        t["trial_id"]: t
        for t in plan["trials"]
        if t["trial_id"] in campaign["expected"]
    }
    if set(expected) != set(campaign["expected"]):
        raise ValueError("Unknown campaign trial")

    cases = {c["unit_id"]: c for c in plan["cases"]}
    hashes, records, identities, parameters = {}, {}, {}, {}
    shared_runtime = None

    for path in sorted((args.runs / "trials").glob("*.json")):
        row = json.loads(path.read_text())
        t = row["trial"]
        if (
            row["status"] != "complete"
            or row["plan_sha256"] != plan["plan_sha256"]
            or expected.get(t["trial_id"]) != t
        ):
            raise ValueError("Result does not match the expected trial")
        if row["case_sha256"] != digest(cases[t["unit_id"]]):
            raise ValueError("Result case changed")
        if t["trial_id"] in records:
            raise ValueError("Duplicate trial result")

        identity_path = args.runs / "identities" / (row["identity_sha256"] + ".json")
        identity = json.loads(identity_path.read_text())
        if (
            digest(identity) != row["identity_sha256"]
            or identity["plan_sha256"] != plan["plan_sha256"]
        ):
            raise ValueError("Invalid measurement identity")

        runtime = {
            k: identity[k]
            for k in (
                "source_sha256",
                "environment",
                "device",
                "controls",
                "visible_gpu_identity",
                "platform",
            )
        }
        if shared_runtime is not None and runtime != shared_runtime:
            raise ValueError(
                "Models were measured with different code, environment, settings or hardware"
            )

        shared_runtime = runtime
        prior = identities.setdefault(t["model"], row["identity_sha256"])
        if prior != row["identity_sha256"]:
            raise ValueError(
                "Model trials mix checkpoint/source/environment/device identities"
            )
        if t["kind"] != "verify" and len(row["repetitions"]) != t["repetitions"]:
            raise ValueError("Incomplete repetition record")
        if parameters.setdefault(t["model"], row["parameters"]) != row["parameters"]:
            raise ValueError("Parameter counts changed between trials")

        hashes[str(path)] = file_hash(path)
        records[t["trial_id"]] = row

    args.output.mkdir(parents=True, exist_ok=False)
    grouped, expected_groups = defaultdict(list), defaultdict(list)
    for t in expected.values():
        key = (t["model"], t["unit_id"], t["mode"], t["kind"])
        expected_groups[key].append(t["trial_id"])
        if t["trial_id"] in records:
            grouped[key].append(records[t["trial_id"]])

    items, incomplete = [], []
    for key, trial_ids in expected_groups.items():
        model, unit, mode, kind = key
        if kind == "verify":
            continue

        rows = grouped[key]
        if len(rows) != len(trial_ids):
            incomplete.append(
                dict(
                    model=model,
                    unit_id=unit,
                    mode=mode,
                    kind=kind,
                    completed=len(rows),
                    expected=len(trial_ids),
                )
            )
            continue

        verify_trials = expected_groups[(model, unit, mode, "verify")]
        if not verify_trials or any(
            t not in records or records[t]["verification"]["status"] != "passed"
            for t in verify_trials
        ):
            raise ValueError("Measurement lacks completed parity verification")

        raw = [r for row in rows for r in row["repetitions"]]
        values = {}
        if kind == "prediction":
            times = [r["seconds"] for r in raw]
            median = float(np.median(times))
            if not np.isfinite(times).all() or min(times) <= 0:
                raise ValueError("Invalid latency")

            n = rows[0]["frames"]
            values.update(
                latency_seconds=median,
                milliseconds_per_frame=1000 * median / n,
                frames_per_second=n / median,
            )
            spreads = {"latency_seconds": distribution(times)}
        elif kind == "encoder":
            times = [r["milliseconds_per_frame"] for r in raw]
            if not np.isfinite(times).all() or min(times) <= 0:
                raise ValueError("Invalid encoder latency")

            values["encoder_ms_per_frame"] = float(np.median(times))
            spreads = {"encoder_ms_per_frame": distribution(times)}
        else:
            spreads = {}
            for scope in (
                "baseline_gpu_bytes",
                "peak_gpu_bytes",
                "delta_peak_gpu_bytes",
            ):
                for allocator in ("allocated", "reserved"):
                    metric = scope + "_" + allocator
                    samples = [r[scope][allocator] for r in raw]
                    values[metric] = float(np.median(samples))
                    spreads[metric] = distribution(samples)

            values["max_peak_gpu_bytes_allocated"] = max(
                r["peak_gpu_bytes"]["allocated"] for r in raw
            )
            values["max_peak_gpu_bytes_reserved"] = max(
                r["peak_gpu_bytes"]["reserved"] for r in raw
            )
            values["process_lifetime_peak_rss_bytes"] = max(
                r["cpu_after"]["lifetime_peak_bytes"] for r in raw
            )

        item = dict(
            model=model,
            unit_key=unit + ":" + mode,
            mode=mode,
            kind=kind,
            case=rows[0]["case"],
            panel=plan["panel_labels"][unit],
            frames=rows[0]["frames"],
            sequence_condition=rows[0]["sequence_condition"],
            primary=values,
            distributions=spreads,
            trials=trial_ids,
            repetitions=len(raw),
        )
        items.append(item)

    summary = dict(
        plan_sha256=plan["plan_sha256"],
        scientific_protocol_certified=False,
        pilot=bool(campaign["pilot_unit"]),
        coverage=dict(
            expected_trials=len(expected),
            completed_trials=len(records),
            missing_trials=sorted(set(expected) - set(records)),
            incomplete_case_measurements=incomplete,
        ),
        parameters=parameters,
        telemetry_review_required=True,
        cross_machine_parity="not_certified",
        cpu_peak_scope="process lifetime including preparation/load/warmup",
        sampling=plan["coverage"],
        models={},
        paired={},
    )
    table = []
    writer = SummaryWriter(str(args.output / "tensorboard"))

    try:
        # Keep stress-only results out of main averages; report both roles explicitly.
        groups = defaultdict(list)
        for index, item in enumerate(items):
            c = item["case"]
            for metric, value in item["primary"].items():
                writer.add_scalar(
                    f'case/{item["model"]}/{item["sequence_condition"]}/{c["dim"]}d/{item["mode"]}/{metric}',
                    value,
                    index,
                )

            roles = []
            if item["panel"]["main"]:
                roles += ["main/all", "main/" + item["panel"]["length_bin"]]
            if item["panel"]["stress"]:
                roles += ["stress"]
            for role in roles:
                key = (
                    item["model"],
                    item["sequence_condition"],
                    c["dim"],
                    item["mode"],
                    item["kind"],
                    role,
                )
                groups[key].append(item)
        for key, rows in groups.items():
            label = "/".join(map(str, key))
            model, condition, dim, mode, kind, role = key
            aggregate = {}

            for metric in rows[0]["primary"]:
                score = hierarchical(source_means(rows, metric))
                aggregate[metric] = dict(
                    **score,
                    distribution=distribution(r["primary"][metric] for r in rows),
                )
                writer.add_scalar(f"completed_only/{label}/{metric}", score["macro"], 0)

                for dataset, value in score["datasets"].items():
                    table.append(
                        dict(
                            model=model,
                            sequence_condition=condition,
                            dimension=dim,
                            mode=mode,
                            kind=kind,
                            panel=role,
                            dataset=dataset,
                            metric=metric,
                            score=value,
                            scope="completed_measurements_pending_validation",
                        )
                    )
                    writer.add_scalar(f"dataset/{label}/{dataset}/{metric}", value, 0)

            summary["models"][label] = dict(
                completed_cases=len(rows), metrics=aggregate
            )

        # CIs use case-level summaries: repetitions are not additional patients.
        comparisons = [
            (m, "corrected_hiera") for m in plan["models"] if m != "corrected_hiera"
        ]
        comparisons += [("rwkv_medsam2_final_distill", "rwkv_medsam2_final_nodistill")]
        for left, right in comparisons:
            for key, left_rows in groups.items():
                model, condition, dim, mode, kind, role = key
                if model != left or not role.startswith("main/"):
                    continue

                right_rows = groups.get((right, condition, dim, mode, kind, role), [])
                other = {r["unit_key"]: r for r in right_rows}
                a = [r for r in left_rows if r["unit_key"] in other]

                if not a:
                    continue

                b = [other[r["unit_key"]] for r in a]
                label = "/".join(
                    map(str, (left + "_vs_" + right, condition, dim, mode, kind, role))
                )
                result = dict(
                    joint_cases=len(a),
                    scope="joint_completed_cases_pending_validation",
                    metrics={},
                )

                for metric in a[0]["primary"]:
                    if metric not in (
                        "latency_seconds",
                        "encoder_ms_per_frame",
                        "peak_gpu_bytes_allocated",
                        "peak_gpu_bytes_reserved",
                    ):
                        continue
                    if kind in ("prediction", "encoder"):
                        ratios = [
                            dict(
                                case=x["case"],
                                unit_key=x["unit_key"],
                                primary={
                                    metric: y["primary"][metric] / x["primary"][metric]
                                },
                            )
                            for x, y in zip(a, b)
                        ]
                        zeros = [dict(r, primary={metric: 0.0}) for r in ratios]
                        score = paired_interval(ratios, zeros, metric, args.bootstrap)
                        score["speed_ratio_reference_over_model"] = score.pop(
                            "difference"
                        )
                        score["interpretation"] = (
                            "above 1 means the left model is faster"
                        )
                    else:
                        score = paired_interval(a, b, metric, args.bootstrap)
                        score["interpretation"] = (
                            "left minus right bytes; below 0 means less memory"
                        )

                    result["metrics"][metric] = score

                summary["paired"][label] = result

        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "input-result-hashes.json", hashes)
        with (args.output / "per-case.jsonl").open("w") as stream:
            for item in items:
                stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        with (args.output / "dataset-scores.csv").open("w", newline="") as stream:
            fields = [
                "model",
                "sequence_condition",
                "dimension",
                "mode",
                "kind",
                "panel",
                "dataset",
                "metric",
                "score",
                "scope",
            ]
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            output.writerows(table)

        writer.add_text("coverage", json.dumps(summary["coverage"], indent=2), 0)
        writer.add_text(
            "status",
            "Unvalidated implementation; boundary telemetry and cross-machine parity require review.",
            0,
        )
    finally:
        writer.close()

    print(
        json.dumps(dict(report=str(args.output), completed_cases=len(items))),
        flush=True,
    )
