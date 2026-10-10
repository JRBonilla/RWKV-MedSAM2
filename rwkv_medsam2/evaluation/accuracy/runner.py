"""Accuracy study CLI. Nothing runs until a subcommand is explicitly invoked."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import traceback

from ..shared.data import (
    PROFILE,
    SEED,
    add_counts,
    array_hash,
    atomic_json,
    canonical_frame,
    canonical_mask,
    decode_mask,
    digest,
    file_hash,
    freeze_case,
    case_kind,
    jsonl,
    overlaps,
    read_case,
    validate_result,
    validate_protocol,
    condition_case,
    sequence_plan,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "docs/evaluation/shared-protocol.json"


def engine_files():
    """
    Hash evaluation source files while excluding tests.

    Args:
        None.

    Returns:
        dict[str, str]: Relative Python source paths mapped to SHA256 digests.
    """
    root = Path(__file__).resolve().parents[1]

    return {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted(root.rglob("*.py"))
        if "tests" not in p.relative_to(root).parts
    }


def prepare(args):
    """
    Future explicit CPU job; do not call as part of source installation.

    Args:
        args (argparse.Namespace): Parsed options for the selected accuracy command.

    Returns:
        None.
    """
    from ..shared.models import environment

    protocol = json.loads(args.protocol.read_text())

    if (
        protocol["protocol_id"]
        != "encoder_replacement_positive_primary_fullscan_robustness_v3_draft"
    ):
        raise ValueError(
            "This implementation needs review before using another protocol version"
        )

    validate_protocol(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", protocol)
    seen, clusters, case_inputs = set(), {}, {}
    counts = Counter()
    errors = []
    manifest = args.output / "cases.jsonl"

    with manifest.open("w") as stream:
        for index, case in enumerate(jsonl(args.cases)):
            try:
                frozen = freeze_case(
                    case,
                    args.data_root,
                    protocol["protocol_id"],
                    args.support_policy,
                    args.prompt_seed,
                )
                if frozen["unit_id"] in seen:
                    raise ValueError("Duplicate dataset/task/case/target identity")

                seen.add(frozen["unit_id"])
                cluster_key = (case["dataset"], case["case_id"])
                if (
                    cluster_key in clusters
                    and clusters[cluster_key] != case["source_id"]
                ):
                    raise ValueError(
                        "Targets of the same case have different source clusters"
                    )

                clusters[cluster_key] = case["source_id"]
                image_identity = (case["dim"], frozen["file_sha256"]["image"])
                if (
                    cluster_key in case_inputs
                    and case_inputs[cluster_key] != image_identity
                ):
                    raise ValueError(
                        "One case_id refers to different images or dimensions"
                    )

                case_inputs[cluster_key] = image_identity
                counts["records"] += 1
                counts["eligible" if frozen["eligible"] else "excluded"] += 1

                if frozen["eligible"]:
                    counts["conditions"] += len(frozen["modes"])
                    counts["prediction_frames_all_conditions"] += len(
                        frozen["frame_ids"]
                    ) * len(frozen["modes"])

                stream.write(json.dumps(frozen, sort_keys=True, allow_nan=False) + "\n")
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "case_id": case.get("case_id"),
                        "target_id": case.get("target_id"),
                        "error": str(exc),
                    }
                )

    atomic_json(args.output / "preparation-errors.json", errors)
    if errors or not counts["eligible"]:
        raise ValueError(
            f"Freeze incomplete: {len(errors)} errors; no usable manifest metadata was released"
        )

    atomic_json(
        args.output / "manifest.json",
        {
            "schema": 3,
            "profile": PROFILE,
            "seed": SEED,
            "sequence_condition": "positive_frames",
            "support_policy": args.support_policy,
            "prompt_seed": args.prompt_seed,
            "scope": (
                "primary"
                if args.support_policy == "quarter_positive" and args.prompt_seed == 42
                else "separate_sensitivity_study"
            ),
            "storage_per_model_uncompressed": {
                "packed_masks_bytes": counts["prediction_frames_all_conditions"]
                * 512
                * 512
                // 8,
                "optional_fp32_logits_bytes": counts["prediction_frames_all_conditions"]
                * 512
                * 512
                * 4,
            },
            "protocol_sha256": file_hash(args.output / "protocol.json"),
            "cases_sha256": file_hash(manifest),
            "reviewed_cases_sha256": file_hash(args.cases),
            "engine_files": engine_files(),
            "preparation_environment": environment(),
            "counts": dict(counts),
            "metric_grid": "canonical_512x512_per_slice",
            "box_coordinates": "inclusive min/max pixel indices; SAM point labels 2 and 3",
            "support_mask_encoding": "numpy.packbits C order, zlib, base64; 512x512",
            "scientific_protocol_certified": False,
        },
    )
    print(json.dumps({"frozen": str(args.output), **counts}), flush=True)


def derive_full_scan(args):
    """
    Derive the robustness cohort from frozen primary records, without new prompts.

    Args:
        args (argparse.Namespace): Parsed options for the selected accuracy command.

    Returns:
        None.
    """
    meta, cases = read_frozen(args.manifest)
    if meta["sequence_condition"] != "positive_frames":
        raise ValueError("Derive robustness from the positive-frame primary manifest")

    derived = [condition_case(case, "full_scan") for case in cases]
    counts = Counter()
    for case in derived:
        counts["records"] += 1
        counts["eligible" if case["eligible"] else "excluded"] += 1
        if case["eligible"]:
            counts["conditions"] += len(case["modes"])
            counts["prediction_frames_all_conditions"] += len(case["frame_ids"]) * len(
                case["modes"]
            )
    if not counts["eligible"]:
        raise ValueError("No eligible 3D targets in the primary cohort")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(
        (args.manifest / "protocol.json").read_bytes()
    )
    with (args.output / "cases.jsonl").open("w") as stream:
        for case in derived:
            stream.write(json.dumps(case, sort_keys=True, allow_nan=False) + "\n")

    meta = dict(
        meta,
        sequence_condition="full_scan",
        scope="secondary_full_scan_robustness",
        paired_primary_manifest_sha256=file_hash(args.manifest / "manifest.json"),
        cases_sha256=file_hash(args.output / "cases.jsonl"),
        counts=dict(counts),
        storage_per_model_uncompressed={
            "packed_masks_bytes": counts["prediction_frames_all_conditions"]
            * 512
            * 512
            // 8,
            "optional_fp32_logits_bytes": counts["prediction_frames_all_conditions"]
            * 512
            * 512
            * 4,
        },
    )
    atomic_json(args.output / "manifest.json", meta)
    print(json.dumps({"frozen_robustness": str(args.output), **counts}), flush=True)


def draft_cases(args):
    """
    Convert legacy inventory into UNREVIEWED candidates, never a runnable cohort.

    Args:
        args (argparse.Namespace): Parsed options for the selected accuracy command.

    Returns:
        None.
    """
    seen = set()
    with args.output.open("x") as stream:
        for item in jsonl(args.inventory):
            m = item["metadata"]
            sequence = m["sequence"]
            if len(sequence) != 1:
                raise ValueError(
                    "Legacy multi-file sequences require separate provenance review"
                )

            image, mask = sequence[0]
            case_id = digest([m["dataset"], image])[:24]
            for task in m["tasks"]:
                key = (m["dataset"], task, image, mask)
                if key in seen:
                    continue

                seen.add(key)
                row = dict(
                    case_id=case_id,
                    target_id=digest(mask)[:24],
                    dataset=m["dataset"],
                    task=task,
                    source_id=str(m.get("source_identifier") or case_id),
                    source_identity_status="unresolved",
                    source_note="REVIEW: verify original patient/scan/slide grouping",
                    modality=m["modality"],
                    image=image,
                    mask=mask,
                    dim=m["dim"],
                    label=None,
                    target_scope="filtered_component",
                    target_name=m.get("class"),
                    split="test",
                    provenance_reviewed=False,
                    preprocessing_note="REVIEW: confirm input scaling and target meaning",
                    legacy_index=item.get("legacy_index"),
                    legacy_identifier=m.get("identifier"),
                )
                if m["dim"] == 3:
                    row.update(
                        axis_zyx=None,
                        reverse_slices=None,
                        anatomical_volume=False,
                        geometry_note="REVIEW: validate anatomical slice axis and direction",
                    )
                    if m["dataset"] == "CAMUS":
                        row.update(
                            sequence_kind="video",
                            temporal_axis_note="REVIEW: confirm stored axis represents forward video frames",
                            geometry_note="REVIEW: confirm spatial image/mask grids and temporal axis",
                        )

                stream.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(seen)} UNREVIEWED candidates to {args.output}", flush=True)


def read_frozen(folder):
    """
    Read and validate a frozen manifest and its case identities.

    Args:
        folder (pathlib.Path): Directory containing evaluation inputs or outputs.

    Returns:
        tuple[dict, list[dict]]: Verified manifest metadata and frozen case records.
    """
    meta = json.loads((folder / "manifest.json").read_text())
    if meta["profile"] != PROFILE or meta["engine_files"] != engine_files():
        raise ValueError(
            "Evaluation implementation differs from the frozen cohort; create a new cohort/run"
        )
    for filename, key in [
        ("cases.jsonl", "cases_sha256"),
        ("protocol.json", "protocol_sha256"),
    ]:
        if file_hash(folder / filename) != meta[key]:
            raise ValueError(f"Frozen file changed: {filename}")

    cases = list(jsonl(folder / "cases.jsonl"))
    if len(cases) != meta["counts"]["records"]:
        raise ValueError("Frozen case count mismatch")
    if meta["sequence_condition"] not in ("positive_frames", "full_scan"):
        raise ValueError("Unknown manifest sequence condition")
    for case in cases:
        if case["eligible"]:
            sequence_plan(case)
            expected_condition = (
                "independent_2d" if case["dim"] == 2 else meta["sequence_condition"]
            )
            if case["sequence_condition"] != expected_condition:
                raise ValueError("Mixed inference conditions within one manifest")

            common = {
                k: v
                for k, v in case.items()
                if k not in ("sequence_condition", "frame_ids", "shared_case_sha256")
            }
            if digest(common) != case["shared_case_sha256"]:
                raise ValueError("Frozen common-case identity changed")

    return meta, cases


@contextmanager
def output_lock(folder):
    """
    Atomic directory ownership works without the NFS advisory-lock service.

    Args:
        folder (pathlib.Path): Directory containing evaluation inputs or outputs.

    Returns:
        Iterator[None]: Context-manager scope that owns the output directory until exit.
    """
    import socket
    import uuid

    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / ".writer.claim"
    token = uuid.uuid4().hex
    owner = dict(host=socket.gethostname(), pid=os.getpid(), token=token)

    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another writer or an interrupted process owns {claim}; "
            "check owner.json and process liveness before removing a stale claim"
        ) from exc
    try:
        atomic_json(claim / "owner.json", owner)
        yield
    finally:
        # Only remove the claim this process created.
        owner_path = claim / "owner.json"
        if (
            owner_path.exists()
            and json.loads(owner_path.read_text()).get("token") == token
        ):
            owner_path.unlink()
            claim.rmdir()


def evaluate_case(
    model, case, mode, data_root, device, on_progress, archive, input_cache=None
):
    """
    Evaluate one frozen case and archive all predicted frames.

    Args:
        model (SAM2Base): SAM2 model configured for the fixed accuracy protocol.
        case (dict): Evaluation case metadata, including input paths and sequence settings.
        mode (str): Frozen prompt mode, either box or mask.
        data_root (str | pathlib.Path): Root directory for relative dataset image and mask
            paths.
        device (str | torch.device): Device used for model computation.
        on_progress (Callable[[int, int], None]): Callback receiving the completed and total
            frame counts.
        archive (PredictionArchive): Archive sink receiving each predicted frame and its
            logits.
        input_cache (InputCache | None): Optional verified input cache reused across
            evaluation cases.

    Returns:
        tuple[dict, np.ndarray | None]: Accuracy diagnostics and the first visualization
            panel when available.
    """
    from ..shared.prediction import predict_frame
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from ..shared.data import confusion
    from .records import (
        confusion_diagnostics,
        probability_record,
        pool_probability,
        slice_summary,
    )

    # Independent RNG state makes resume/shard order irrelevant to stochastic ops.
    import random

    case_seed = int(digest([SEED, case["unit_id"], mode])[:8], 16)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    torch.cuda.manual_seed_all(case_seed)

    if input_cache is None:
        for key, expected in case["file_sha256"].items():
            path = (
                Path(case[key])
                if Path(case[key]).is_absolute()
                else data_root / case[key]
            )
            if file_hash(path) != expected:
                raise ValueError(f'Frozen {key} changed for {case["unit_id"]}')

        frames, masks, transform = read_case(case, data_root)
    else:
        frames, masks, transform = input_cache.read_case(case, data_root)
    if transform != case["transform"]:
        raise ValueError("Input geometry, frame count, or intensity transform changed")

    frame_ids, sequence_index, order = sequence_plan(case)
    n = len(frame_ids)
    support = set(case["support_frames"])
    memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    metrics, observed, diagnostics = {}, [], {}
    expected_input = []
    prediction_hash = __import__("hashlib").sha256()
    first_panel = None

    def observe(_, inputs):
        """
        Record and validate the image encoder's actual FP32 input.

        Args:
            _ (object): Unused callback argument supplied by the caller.
            inputs (tuple[torch.Tensor, ...]): Image-encoder positional inputs observed by the
                hook.

        Returns:
            None.
        """
        value = inputs[0]
        if value.dtype != torch.float32:
            raise ValueError("Encoder received non-FP32 input")

        observed.append(array_hash(value.detach().cpu().numpy()))

    hook = model.image_encoder.register_forward_pre_hook(observe)
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ), sdpa_kernel(SDPBackend.MATH):
            for step, t in enumerate(order, 1):
                frame = canonical_frame(frames[t], transform["intensity_limits"])
                target = canonical_mask(masks[t])
                frame_hash = array_hash(frame)

                if (
                    frame_hash != case["encoder_input_sha256"][t]
                    or array_hash(target) != case["target_sha256"][t]
                ):
                    raise ValueError(f"Canonical frame/target mismatch at slice {t}")

                expected_input.append(frame_hash)
                points, supplied = None, None
                if t in support:
                    prompt = case["prompts"][str(t)]
                    saved_mask = decode_mask(prompt["mask_packbits_zlib_base64"])
                    if array_hash(saved_mask) != prompt[
                        "mask_sha256"
                    ] or not np.array_equal(saved_mask, target):
                        raise ValueError("Saved support mask does not match target")
                    if mode == "box":
                        points = {
                            "point_coords": torch.tensor(
                                [prompt["box_xy"]], dtype=torch.float32, device=device
                            ),
                            "point_labels": torch.tensor(
                                [[2, 3]], dtype=torch.int32, device=device
                            ),
                        }
                    else:
                        supplied = torch.from_numpy(saved_mask.astype(np.float32))[
                            None, None
                        ].to(device)

                out = predict_frame(
                    model,
                    torch.from_numpy(frame).to(device),
                    points,
                    supplied,
                    memory,
                    sequence_index[t],
                    t in support,
                    n,
                )
                logits, stored = out["pred_masks_high_res"], out["maskmem_features"]
                if (
                    tuple(logits.shape) != (1, 1, 512, 512)
                    or logits.dtype != torch.float32
                ):
                    raise ValueError("Invalid prediction shape/dtype")
                if (
                    stored is None
                    or stored.dtype != torch.float32
                    or not torch.isfinite(stored).all()
                ):
                    raise ValueError("Invalid FP32 temporal memory")
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite prediction")

                pred = (logits[0, 0] >= 0).cpu().numpy()
                if mode == "mask" and t in support and not np.array_equal(pred, target):
                    raise ValueError("Mask support was not retained exactly")

                metrics[t] = confusion(pred, target)
                z = logits[0, 0].detach().cpu().numpy()
                archive.frame(t, pred, z)
                scores = {}

                for name in ("iou_predictions", "object_score_logits"):
                    value = out.get(name)
                    if value is not None and value.numel() <= 32:
                        scores[name] = value.detach().float().cpu().reshape(-1).tolist()

                diagnostics[t] = dict(
                    probability=probability_record(z, target),
                    sequence_index=sequence_index[t],
                    **{
                        (
                            "distance_to_nearest_support_frames"
                            if case_kind(case) == "video"
                            else "distance_to_nearest_support_slices"
                        ): min(abs(t - u) for u in support)
                    },
                    distance_to_nearest_support_sequence_steps=min(
                        abs(sequence_index[t] - sequence_index[u]) for u in support
                    ),
                    decoder_scores=scores,
                    **confusion_diagnostics(metrics[t]),
                )
                prediction_hash.update(t.to_bytes(8, "little"))
                prediction_hash.update(np.packbits(pred).tobytes())
                memory[
                    "cond_frame_outputs" if t in support else "non_cond_frame_outputs"
                ][sequence_index[t]] = out

                if first_panel is None and (case["dim"] == 2 or t not in support):
                    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
                    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
                    rgb = np.clip(frame[0] * std + mean, 0, 1)
                    first_panel = np.concatenate(
                        [
                            rgb,
                            np.repeat(target[None], 3, 0),
                            np.repeat(pred[None], 3, 0),
                        ],
                        axis=2,
                    )
                if step == 1 or step % 32 == 0 or step == n:
                    on_progress(step, n)

                del out, logits, stored, points, supplied
        if sorted(metrics) != frame_ids or observed != expected_input:
            raise ValueError("Missing/repeated slices or encoder input parity failure")

        query = (
            frame_ids
            if case["dim"] == 2
            else [t for t in frame_ids if t not in support]
        )
        primary_counts = add_counts(metrics[t] for t in query)
        all_counts = add_counts(metrics.values())
        result = dict(
            primary={**primary_counts, **overlaps(primary_counts)},
            all_frame_assisted={**all_counts, **overlaps(all_counts)},
            per_frame=[
                dict(
                    frame=t,
                    sequence_index=sequence_index[t],
                    support=t in support,
                    **metrics[t],
                )
                for t in frame_ids
            ],
            sequence_condition=case["sequence_condition"],
            frame_ids=frame_ids,
            stored_frames=len(frames),
            shared_case_sha256=case["shared_case_sha256"],
            frames=n,
            query_frames=len(query),
            support_frames=sorted(support),
            positive_frames=case["positive_frames"],
            encoder_inputs_verified=True,
            prediction_sha256=prediction_hash.hexdigest(),
        )
        positive_ids = set(case["positive_frame_ids"])
        positive_queries = [t for t in query if t in positive_ids]
        positive_counts = add_counts(metrics[t] for t in positive_queries)
        result["positive_query_overlap"] = {
            **positive_counts,
            **overlaps(positive_counts),
        }
        result["diagnostics"] = dict(
            query=confusion_diagnostics(primary_counts),
            probability_query=pool_probability(
                diagnostics[t]["probability"] for t in query
            ),
            slices=slice_summary(metrics[t] for t in query),
            support_fraction_positive=len(support) / case["positive_frames"],
            support_fraction_all=len(support) / len(frames),
            support_fraction_input=len(support) / n,
            per_frame=[dict(frame=t, **diagnostics[t]) for t in frame_ids],
        )

        return result, first_panel
    finally:
        hook.remove()
        memory.clear()


def run(args):
    """
    Run or resume one model's frozen accuracy conditions and surface analysis.

    Args:
        args (argparse.Namespace): Parsed options for the selected accuracy command.

    Returns:
        None.
    """
    from .records import PredictionArchive, verify_archive
    from ..shared.models import (
        CONTROLS,
        checkpoint_identity,
        configure_runtime,
        environment,
        initialization_assets,
        load_model,
        source_snapshot,
    )
    from .inputs import InputCache
    from .surface_workers import SurfacePool, analysis_identity

    if (
        args.surface_workers < 0
        or args.surface_cache_gib < 0
        or args.input_cache_gib < 0
    ):
        raise ValueError("Worker/cache settings must be nonnegative")
    if args.surface_workers:
        analysis_identity()

    meta, cases = read_frozen(args.manifest)
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")

    selected = [
        c
        for c in cases
        if c["eligible"]
        and int(digest([c["dataset"], c["case_id"]]), 16) % args.shards == args.shard
    ]
    expected = [(c, mode) for c in selected for mode in c["modes"]]
    if not expected:
        raise ValueError("This shard has no eligible cases")

    spec = json.loads(args.checkpoints.read_text())
    checkpoint = checkpoint_identity(args.model, spec)
    snapshot = source_snapshot()
    env = environment()

    # Input preprocessing versions must match the freeze, even on another machine.
    for package in ("numpy", "Pillow", "SimpleITK"):
        if (
            env["packages"][package]
            != meta["preparation_environment"]["packages"][package]
        ):
            raise ValueError(f"{package} differs from frozen input preparation")

    identity = dict(
        profile=PROFILE,
        manifest_sha256=file_hash(args.manifest / "manifest.json"),
        cases_sha256=meta["cases_sha256"],
        protocol_sha256=meta["protocol_sha256"],
        model=args.model,
        checkpoint=checkpoint,
        initialization_assets=initialization_assets(args.model),
        source_sha256=snapshot["sha256"],
        environment=env,
        controls=CONTROLS,
        dynamic_multimask_via_stability=False,
        dtype="float32",
        attention_backend="math",
        frame_batch=1,
        state_policy="all_frame_GPU",
        seed=SEED,
        sequence_condition=meta["sequence_condition"],
        recording_schema=3,
        save_logits=args.save_logits,
    )
    run_id = digest(identity)
    folder = args.output / args.model / f"shard-{args.shard:04d}-of-{args.shards:04d}"

    with output_lock(folder):
        job = dict(
            run_id=run_id,
            identity=identity,
            shard=args.shard,
            shards=args.shards,
            expected=[c["unit_id"] + ":" + mode for c, mode in expected],
        )
        job_path = folder / "job.json"
        if job_path.exists() and json.loads(job_path.read_text()) != job:
            raise ValueError(
                "Existing output has another checkpoint, cohort, code, environment, or shard layout"
            )

        atomic_json(job_path, job)
        (folder / "complete.json").unlink(missing_ok=True)
        atomic_json(folder / "source-snapshot.json", snapshot)
        torch = configure_runtime(args.device)
        atomic_json(
            folder / "runtime.json",
            dict(
                device=args.device,
                gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),
                torch_cuda=torch.version.cuda,
                torch_version=torch.__version__,
                tf32=False,
                autocast=False,
                compile=False,
                attention_backend="math",
                cpu_threads=2,
                input_cache_gib=args.input_cache_gib,
                surface_workers=args.surface_workers,
                surface_cache_gib_per_worker=args.surface_cache_gib,
            ),
        )
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            str(args.output / "tensorboard" / args.model / folder.name), flush_secs=15
        )
        writer.add_text("protocol/run", json.dumps(job, indent=2), 0)
        writer.add_text(
            "protocol/scope",
            "Accuracy only; shared-A100 runtime is not an efficiency benchmark.",
            0,
        )
        from .report import RunningMacro

        running = {}
        model = None
        completed = 0
        started = time.monotonic()
        input_cache = InputCache(int(args.input_cache_gib * 1024**3))
        surface_pool = None
        surfaces_completed = 0

        def surface_progress(items):
            """
            Record completed surface jobs and log analysis progress.

            Args:
                items (list[dict]): Completed surface-job records to incorporate into reporting.

            Returns:
                None.
            """
            nonlocal surfaces_completed
            for item in items:
                surfaces_completed += 1
                r = item["result"]
                tag = f'{r["sequence_condition"]}/{case_kind(r["case"])}/{r["mode"]}/{r["case"]["dataset"]}/{r["case"]["task"]}/{r["units"]}'

                for metric in ("hd95", "assd"):
                    if r[metric] is not None:
                        writer.add_scalar(
                            f"boundary/case/{tag}/{metric}",
                            r[metric],
                            surfaces_completed,
                        )
                for tolerance, value in r["nsd"].items():
                    writer.add_scalar(
                        f"boundary/case/{tag}/nsd_{tolerance}",
                        value,
                        surfaces_completed,
                    )

                writer.add_scalar(
                    "boundary/progress/completed",
                    surfaces_completed,
                    surfaces_completed,
                )
                writer.add_scalar(
                    "boundary/progress/fraction",
                    surfaces_completed / len(expected),
                    surfaces_completed,
                )
                writer.add_scalar(
                    "boundary/operations/case_seconds",
                    item["seconds"],
                    surfaces_completed,
                )
                atomic_json(
                    folder / "surfaces/progress.json",
                    dict(
                        completed=surfaces_completed,
                        expected=len(expected),
                        last_unit_key=r["unit_key"],
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            surface_completed=surfaces_completed,
                            surface_expected=len(expected),
                            unit_key=r["unit_key"],
                            reused=item["reused"],
                        )
                    ),
                    flush=True,
                )
            if items:
                writer.flush()

        try:
            if args.surface_workers:
                surface_pool = SurfacePool(
                    args.surface_workers, int(args.surface_cache_gib * 1024**3)
                )
            for index, (case, mode) in enumerate(expected):
                if surface_pool:
                    surface_progress(surface_pool.collect())

                key = case["unit_id"] + ":" + mode
                destination = (
                    folder / "cases" / (case["unit_id"] + "-" + mode + ".json")
                )
                if destination.exists():
                    result = json.loads(destination.read_text())
                    validate_result(result, case, mode, run_id)
                    verify_archive(destination, result)
                else:
                    if model is None:
                        model = load_model(args.model, checkpoint, args.device)
                        settings = {key: getattr(model, key) for key in CONTROLS}
                        settings.update(
                            {
                                key: getattr(model, key, None)
                                for key in (
                                    "num_maskmem",
                                    "max_cond_frames_in_attn",
                                    "memory_temporal_stride_for_eval",
                                    "directly_add_no_mem_embed",
                                    "image_size",
                                )
                            }
                        )
                        settings["dynamic_multimask_via_stability"] = (
                            model.sam_mask_decoder.dynamic_multimask_via_stability
                        )
                        atomic_json(folder / "effective-model-settings.json", settings)

                    def progress(step, total):
                        """
                        Log progress through the current case's frame sequence.

                        Args:
                            step (int): Number of frames already processed.
                            total (int): Total number of frames in the current sequence.

                        Returns:
                            None.
                        """
                        writer.add_scalar(
                            "progress/current_case_fraction", step / total, index
                        )
                        writer.flush()
                        print(
                            json.dumps(
                                {
                                    "model": args.model,
                                    "case": case["case_id"],
                                    "mode": mode,
                                    "frames_complete": step,
                                    "frames_total": total,
                                }
                            ),
                            flush=True,
                        )

                    archive = PredictionArchive(
                        folder
                        / "predictions"
                        / (case["unit_id"] + "-" + mode + ".zip"),
                        dict(
                            run_id=run_id,
                            unit_key=key,
                            case_sha256=digest(case),
                            sequence_condition=case["sequence_condition"],
                            frame_ids=case["frame_ids"],
                        ),
                        args.save_logits,
                    )
                    try:
                        scores, panel = evaluate_case(
                            model,
                            case,
                            mode,
                            args.data_root,
                            args.device,
                            progress,
                            archive,
                            input_cache=input_cache,
                        )
                        result = dict(
                            run_id=run_id,
                            status="complete",
                            unit_key=key,
                            model=args.model,
                            mode=mode,
                            case={
                                k: case[k]
                                for k in (
                                    "unit_id",
                                    "case_id",
                                    "target_id",
                                    "dataset",
                                    "task",
                                    "source_id",
                                    "source_identity_status",
                                    "target_scope",
                                    "modality",
                                    "dim",
                                )
                            },
                            case_sha256=digest(case),
                            **scores,
                        )
                        for field in (
                            "reporting",
                            "global_source_id",
                            "task_label_provenance",
                        ):
                            if field in case:
                                result["case"][field] = case[field]
                        if case_kind(case) == "video":
                            result["case"]["sequence_kind"] = "video"

                        validate_result(result, case, mode, run_id)
                        result["prediction_archive"] = archive.commit()
                        atomic_json(destination, result)

                        # A fixed first-N manifest selection, never chosen by score.
                        if index < args.qualitative_cases and panel is not None:
                            writer.add_image(
                                f"qualitative/{index:04d}_{mode}_input_truth_prediction",
                                panel,
                                0,
                            )
                    except Exception:
                        archive.abort()
                        failure = dict(
                            run_id=run_id,
                            unit_key=key,
                            status="failed",
                            traceback=traceback.format_exc(),
                        )
                        atomic_json(
                            folder
                            / "failures"
                            / (case["unit_id"] + "-" + mode + ".json"),
                            failure,
                        )
                        writer.add_scalar(
                            "coverage/failures_this_attempt", 1, index + 1
                        )
                        writer.add_text(
                            "failures/latest", failure["traceback"], index + 1
                        )
                        raise  # Includes OOM: never shorten a scan, reduce prompts, or silently skip it.
                if surface_pool:
                    while len(surface_pool.pending) >= surface_pool.capacity:
                        surface_progress(surface_pool.collect(block=True))

                    surface_pool.submit(
                        result,
                        destination,
                        case,
                        args.data_root,
                        folder / "surfaces/cases" / destination.name,
                    )

                completed += 1
                group = f'{case["sequence_condition"]}/{case_kind(case)}/{mode}/{case["dataset"]}/{case["task"]}'
                for metric in ("dice", "iou"):
                    for score_key, label in (
                        ("primary", metric),
                        ("all_frame_assisted", "all_frame_assisted_" + metric),
                    ):
                        writer.add_scalar(
                            f"case/{group}/{label}",
                            result[score_key][metric],
                            index + 1,
                        )
                        macro_key = (case_kind(case), mode, label)
                        tracker = running.setdefault(macro_key, RunningMacro())
                        value = tracker.update(result, metric, score_key=score_key)
                        writer.add_scalar(
                            f'running_completed_only/{case["sequence_condition"]}/{case_kind(case)}/{mode}/{label}',
                            value,
                            index + 1,
                        )
                        from .cohorts import report_views

                        for view in report_views(case):
                            variants = [view] + (
                                [view + "/without_PAIP"]
                                if case["dataset"] != "PAIP2019"
                                else []
                            )
                            for variant in variants:
                                tracker = running.setdefault(
                                    (variant, *macro_key), RunningMacro()
                                )
                                score = tracker.update(
                                    result, metric, score_key=score_key
                                )
                                writer.add_scalar(
                                    f'categorized_running/{variant}/{case["sequence_condition"]}/{case_kind(case)}/{mode}/{label}',
                                    score,
                                    index + 1,
                                )
                for metric, value in result["diagnostics"]["query"].items():
                    if value is not None:
                        writer.add_scalar(
                            f"diagnostics/{group}/{metric}", value, index + 1
                        )
                for metric in ("brier", "log_loss", "ece20"):
                    writer.add_scalar(
                        f"calibration/{group}/{metric}",
                        result["diagnostics"]["probability_query"][metric],
                        index + 1,
                    )

                writer.add_scalar("coverage/completed_conditions", completed, index + 1)
                writer.add_scalar(
                    "coverage/expected_conditions", len(expected), index + 1
                )
                writer.add_scalar(
                    "progress/fraction_complete", completed / len(expected), index + 1
                )
                writer.add_scalar(
                    "operations/elapsed_seconds_not_benchmark",
                    time.monotonic() - started,
                    index + 1,
                )
                writer.flush()

            atomic_json(
                folder / "predictions-complete.json",
                dict(
                    run_id=run_id,
                    completed=completed,
                    expected=len(expected),
                    input_cache_stats=input_cache.stats,
                ),
            )
            if surface_pool:
                while surface_pool.pending:
                    surface_progress(surface_pool.collect(block=True))

            atomic_json(
                folder / "complete.json",
                dict(
                    run_id=run_id,
                    completed=completed,
                    expected=len(expected),
                    surfaces_requested=bool(surface_pool),
                    surfaces_completed=surfaces_completed,
                ),
            )
        except Exception:
            failure = dict(
                run_id=run_id,
                status="failed",
                completed=completed,
                traceback=traceback.format_exc(),
            )
            atomic_json(folder / "last-failed-attempt.json", failure)
            writer.add_text("failures/attempt", failure["traceback"], completed)
            writer.add_scalar("coverage/attempt_failed", 1, completed)
            raise
        finally:
            if surface_pool:
                surface_pool.close()

            writer.close()

    print(
        json.dumps(
            {"status": "complete", "folder": str(folder), "conditions": completed}
        ),
        flush=True,
    )


def main():
    """
    Parse arguments and dispatch the requested accuracy command.

    Args:
        None.

    Returns:
        None.
    """
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    derive = commands.add_parser(
        "derive-full-scan",
        help="Create a paired 3D robustness manifest with the same targets/prompts",
    )
    derive.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen positive-frame primary manifest",
    )
    derive.add_argument("--output", type=Path, required=True)
    derive.set_defaults(function=derive_full_scan)
    draft = commands.add_parser(
        "draft-cases",
        help="Convert inventory to candidates requiring provenance review",
    )
    draft.add_argument("--inventory", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    draft.set_defaults(function=draft_cases)
    freeze = commands.add_parser(
        "prepare", help="Explicit CPU-only job: freeze reviewed test cases and prompts"
    )
    freeze.add_argument("--cases", type=Path, required=True)
    freeze.add_argument("--data-root", type=Path, default=Path("/data"))
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--support-policy",
        choices=["quarter_positive", "single_middle_positive"],
        default="quarter_positive",
    )
    freeze.add_argument("--prompt-seed", type=int, choices=[42, 137, 2026], default=42)
    freeze.set_defaults(function=prepare)
    execute = commands.add_parser(
        "run", help="Run ONE model in this process, with mandatory TensorBoard logging"
    )
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--checkpoints", type=Path, default=DEFAULT_PROTOCOL)
    execute.add_argument("--model", required=True)
    execute.add_argument(
        "--device",
        required=True,
        help="Explicit CUDA device; no automatic GPU selection",
    )
    execute.add_argument("--data-root", type=Path, default=Path("/data"))
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--shard", type=int, default=0)
    execute.add_argument("--shards", type=int, default=1)
    execute.add_argument(
        "--save-logits",
        action="store_true",
        help="Also archive exact FP32 logits; about 1 MiB/frame before compression",
    )
    execute.add_argument("--qualitative-cases", type=int, default=4)
    execute.add_argument(
        "--input-cache-gib",
        type=float,
        default=2.0,
        help="Read-only image/mask cache per GPU process",
    )
    execute.add_argument(
        "--surface-workers",
        type=int,
        default=0,
        help="CPU workers per GPU process; 1 enables overlapping HD95/NSD/ASSD",
    )
    execute.add_argument(
        "--surface-cache-gib",
        type=float,
        default=0.5,
        help="Mask cache per background CPU worker; not total working-memory limit",
    )
    execute.set_defaults(function=run)
    report = commands.add_parser(
        "summarize", help="Merge coverage and paired accuracy statistics; CPU only"
    )
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--models", nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap", type=int, default=2000)

    def summarize(args):
        """
        Dispatch saved-result aggregation from the command line.

        Args:
            args (argparse.Namespace): Parsed options for the selected accuracy command.

        Returns:
            None.
        """
        from .report import summarize as make_report

        make_report(args)

    report.set_defaults(function=summarize)
    surface = commands.add_parser(
        "surfaces",
        help="CPU-only boundary analysis from archived predictions; no model run",
    )
    surface.add_argument("--manifest", type=Path, required=True)
    surface.add_argument("--runs", type=Path, required=True)
    surface.add_argument("--data-root", type=Path, default=Path("/data"))
    surface.add_argument("--output", type=Path, required=True)
    surface.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CPU surface workers; reuses verified inline results",
    )
    surface.add_argument(
        "--cache-gib",
        type=float,
        default=0.5,
        help="Mask cache per CPU worker; temporary metric arrays require extra RAM",
    )

    def surfaces(args):
        """
        Dispatch prediction-archive surface analysis from the command line.

        Args:
            args (argparse.Namespace): Parsed options for the selected accuracy command.

        Returns:
            None.
        """
        from .surfaces import surfaces as analyze_surfaces

        analyze_surfaces(args)

    surface.set_defaults(function=surfaces)
    args = p.parse_args()
    # Model construction changes cwd; make user paths absolute first.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())

    args.function(args)


if __name__ == "__main__":
    main()
