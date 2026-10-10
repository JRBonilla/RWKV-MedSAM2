"""Freeze a portable timing panel and a serial, balanced trial schedule."""

from collections import defaultdict
import json
from pathlib import Path

from ..shared.data import atomic_json, digest, file_hash, sequence_plan

SCHEMA = "gamma-performance-v1"


def local_case(case, data_root, source_root):
    """
    Map dataset paths from the declared source root to the local data root.

    Args:
        case (dict): Evaluation case metadata, including input paths and sequence settings.
        data_root (str | pathlib.Path): Root directory for relative dataset image and mask
            paths.
        source_root (str | pathlib.Path): Original root used by the frozen dataset paths.

    Returns:
        dict: Case copy with resolved local image and mask paths.
    """
    out = dict(case)
    for key in ("image", "mask"):
        path = Path(case[key])
        if path.is_absolute():
            path = path.relative_to(source_root)
        if ".." in path.parts:
            raise ValueError("Dataset path must stay within the declared root")

        out[key] = str((Path(data_root) / path).resolve())

    return out


def read_plan(path):
    """
    Read a performance plan and validate its identity and frame membership.

    Args:
        path (str | pathlib.Path): Frozen performance-plan JSON file to verify and read.

    Returns:
        dict: Verified performance panel and trial schedule.
    """
    plan = json.loads(Path(path).read_text())
    identity = dict(plan)
    recorded = identity.pop("plan_sha256")

    if plan["schema"] != SCHEMA or digest(identity) != recorded:
        raise ValueError("Performance plan identity changed")
    if len({c["unit_id"] for c in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Duplicate panel target")
    for case in plan["cases"]:
        sequence_plan(case)

    return plan


def make_plan(args):
    # No imports of torch/model code and no GPU access during metadata selection.
    """
    Select a portable metadata-balanced panel and freeze its serial trial schedule.

    Args:
        args (argparse.Namespace): Parsed command options, including input paths and report
            or trial settings.

    Returns:
        None.
    """
    from ..accuracy.runner import read_frozen, engine_files
    from ..shared.models import MODELS
    import numpy as np

    meta, cases = read_frozen(args.manifest)

    if (
        not args.models
        or len(set(args.models)) != len(args.models)
        or not set(args.models).issubset(MODELS)
    ):
        raise ValueError("Select distinct known models")
    if not 0 < args.host_cache_gib <= 6:
        raise ValueError(
            "Prepared input budget must be positive and at most 6 GiB on Gamma"
        )

    eligible = [
        c for c in cases if c["eligible"] and c["source_identity_status"] == "verified"
    ]
    groups = defaultdict(list)
    for c in eligible:
        groups[(c["dataset"], c["task"], c["dim"])].append(c)

    selected, labels, coverage = {}, {}, []
    for group, cohort in sorted(groups.items()):
        available = []
        for c in cohort:
            mapped = local_case(c, args.data_root, args.source_root)
            if all(Path(mapped[k]).is_file() for k in ("image", "mask")):
                available.append(c)

        # Bin cutpoints come from the full eligible group, not just transferred data.
        lengths = [len(c["frame_ids"]) for c in cohort]
        cuts = (
            [float(x) for x in np.quantile(lengths, [1 / 3, 2 / 3])]
            if group[2] == 3
            else []
        )

        def stratum(c):
            """
            Assign a case to a dimensionality or frozen sequence-length stratum.

            Args:
                c (dict): Evaluation case metadata.

            Returns:
                str: The 2d, short, medium, or long stratum label.
            """
            n = len(c["frame_ids"])

            return (
                "2d"
                if not cuts
                else ("short" if n <= cuts[0] else "medium" if n <= cuts[1] else "long")
            )

        # One metadata-selected target per source prevents patient duplication.
        source_choices = {}
        for c in sorted(available, key=lambda c: digest([42, c["unit_id"]])):
            source_choices.setdefault(c["source_id"], c)

        bins = defaultdict(list)
        for c in source_choices.values():
            bins[stratum(c)].append(c)

        counts = {}
        for label in ("2d",) if group[2] == 2 else ("short", "medium", "long"):
            candidates = sorted(
                bins[label], key=lambda c: digest([42, c["source_id"], c["unit_id"]])
            )
            limit = 20 if group[2] == 2 else 5
            take = candidates[:limit]
            counts[label] = dict(
                available_sources=len(candidates),
                selected=len(take),
                requested_max=limit,
            )

            for c in take:
                selected[c["unit_id"]] = c
                labels[c["unit_id"]] = dict(main=True, stress=False, length_bin=label)

        stress = None
        if group[2] == 3 and available:
            c = min(
                available,
                key=lambda c: (-len(c["frame_ids"]), digest([42, c["unit_id"]])),
            )
            stress = c["unit_id"]
            selected[stress] = c
            labels.setdefault(
                stress, dict(main=False, stress=False, length_bin=stratum(c))
            )["stress"] = True

        coverage.append(
            dict(
                dataset=group[0],
                task=group[1],
                dim=group[2],
                eligible_targets=len(cohort),
                available_targets=len(available),
                cutpoints=cuts,
                strata=counts,
                stress_unit_id=stress,
                stress_is_full_cohort_longest=bool(
                    stress and len(selected[stress]["frame_ids"]) == max(lengths)
                ),
            )
        )
    if not selected:
        raise ValueError("No eligible, source-verified, locally available cases")

    panel = sorted(
        selected.values(), key=lambda c: (c["dataset"], c["task"], c["unit_id"])
    )
    conditions = [(c, mode) for c in panel for mode in c["modes"]]
    trials = []

    def add(c, mode, model, kind, block, repetitions):
        """
        Append a trial with a stable identity to the schedule.

        Args:
            c (dict): Evaluation case metadata.
            mode (str): Frozen prompt mode, either box or mask.
            model (str): Pinned model identifier assigned to this trial.
            kind (str): Trial type or NVIDIA query category.
            block (int): Measurement block index used to balance model ordering.
            repetitions (int): Number of repetitions in the trial.

        Returns:
            None.
        """
        item = dict(
            unit_id=c["unit_id"],
            mode=mode,
            model=model,
            kind=kind,
            block=block,
            repetitions=repetitions,
        )
        item["trial_id"] = digest(item)[:24]
        trials.append(item)

    # Each worker is a fresh process. Verification precedes all measurements.
    for index, (c, mode) in enumerate(conditions):
        offset = index % len(args.models)
        order = args.models[offset:] + args.models[:offset]
        for model in order:
            add(c, mode, model, "verify", 0, 1)
        for block in range(2):
            for model in order if block == 0 else list(reversed(order)):
                add(c, mode, model, "prediction", block, 5)
        # Encoder compute is independent of prompt type; measure once per target.
        if mode == c["modes"][0]:
            for block in range(2):
                for model in order if block == 0 else list(reversed(order)):
                    add(c, mode, model, "encoder", block, 5)
        for block in range(3):
            for model in order if block % 2 == 0 else list(reversed(order)):
                add(c, mode, model, "memory", block, 1)

    plan = dict(
        schema=SCHEMA,
        status="pilot_required_not_certified",
        scientific_protocol_certified=False,
        accuracy_manifest_sha256=file_hash(args.manifest / "manifest.json"),
        accuracy_protocol_sha256=meta["protocol_sha256"],
        sequence_condition=meta["sequence_condition"],
        preparation_environment=meta["preparation_environment"],
        engine_files=engine_files(),
        seed=42,
        models=args.models,
        cases=panel,
        panel_labels=labels,
        coverage=coverage,
        excluded_unresolved_sources=sum(
            c["eligible"] and c["source_identity_status"] != "verified" for c in cases
        ),
        warmups=3,
        host_cache_bytes=int(args.host_cache_gib * 1024**3),
        trials=trials,
        cross_machine_parity="pending; local adapter equality does not certify A100/Gamma equality",
        selection="metadata only; source-deduplicated; equal lengths stay in the same bin",
    )
    plan["plan_sha256"] = digest(plan)
    if args.output.exists():
        raise FileExistsError(args.output)

    atomic_json(args.output, plan)
    print(
        json.dumps(
            dict(
                plan=str(args.output),
                cases=len(panel),
                trials=len(trials),
                status=plan["status"],
            )
        ),
        flush=True,
    )
