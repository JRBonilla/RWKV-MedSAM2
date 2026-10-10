"""Explicit CPU-only surface analysis of saved masks on the stored annotation grid."""

from __future__ import annotations
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import zipfile

from ..shared.data import (
    case_kind,
    add_counts,
    atomic_json,
    confusion,
    digest,
    file_hash,
    overlaps,
    validate_result,
)
from .records import distribution, verify_archive


def boundary_metrics(pred, target, spacing, tolerances=(1.0, 2.0, 3.0)):
    """
    Compute robust Hausdorff, average surface distance, and normalized surface Dice.

    Args:
        pred (np.ndarray): Binary prediction mask.
        target (np.ndarray): Binary ground-truth mask aligned with the predictions.
        spacing (Sequence[float]): Spatial pixel or voxel spacing in array-axis order.
        tolerances (tuple[float, ...]): Boundary-distance tolerances for normalized surface
            Dice.

    Returns:
        dict: Boundary metrics and defined/empty-mask status for the requested tolerances.
    """
    import numpy as np
    from surface_distance import metrics

    spacing = np.asarray(spacing, dtype=float)

    if (
        len(spacing) != target.ndim
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("Invalid metric spacing")
    if pred.shape != target.shape:
        raise ValueError("Surface prediction/annotation grids differ")

    a, b = bool(pred.any()), bool(target.any())
    if not a or not b:
        return dict(
            status="both_empty" if a == b else "one_empty",
            hd95=0.0 if a == b else None,
            assd=0.0 if a == b else None,
            nsd={str(t): 1.0 if a == b else 0.0 for t in tolerances},
        )

    distances = metrics.compute_surface_distances(
        target.astype(bool), pred.astype(bool), spacing
    )
    gt_area, pred_area = distances["surfel_areas_gt"], distances["surfel_areas_pred"]
    total_area = gt_area.sum() + pred_area.sum()
    assd = (
        np.sum(distances["distances_gt_to_pred"] * gt_area)
        + np.sum(distances["distances_pred_to_gt"] * pred_area)
    ) / total_area

    return dict(
        status="defined",
        hd95=float(metrics.compute_robust_hausdorff(distances, 95)),
        assd=float(assd),
        nsd={
            str(t): float(metrics.compute_surface_dice_at_tolerance(distances, t))
            for t in tolerances
        },
    )


def stored_grid(case, transform):
    """
    Determine valid spatial spacing and units for the stored target grid.

    Args:
        case (dict): Evaluation case metadata, including input paths and sequence settings.
        transform (dict): Frozen geometry, intensity limits, and stored frame count.

    Returns:
        tuple[list[float], str]: Spatial spacing and units, excluding a video's temporal
            spacing.
    """
    import numpy as np

    if case["dim"] == 2 or case_kind(case) == "video":
        # Video spacing is spatial H/W only: never interpret frame time as depth.
        if case.get("spacing_verified_mm") is True:
            spacing = case.get("spacing_hw_mm")
            if spacing is None:
                raise ValueError("Verified 2D spacing requires spacing_hw_mm")

            return list(spacing), "mm"

        return [1.0, 1.0], "stored_pixels"

    geometry = transform["geometry"]
    if case.get("spacing_verified_mm") is not True:
        return [1.0, 1.0, 1.0], "stored_voxels"

    direction = np.asarray(geometry["direction"]).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sheared geometry requires explicit physical-grid resampling")

    axis = case["axis_zyx"]
    order = [axis] + [i for i in range(3) if i != axis]
    zyx = list(reversed(geometry["spacing_xyz"]))

    return [zyx[i] for i in order], "mm"


def analyze(row, result_path, case, data_root, cache=None):
    """
    Compute boundary metrics from a verified prediction archive and target masks.

    Args:
        row (dict): Saved evaluation result or diagnostic record.
        result_path (str | pathlib.Path): Path to the saved case-result JSON file.
        case (dict): Evaluation case metadata, including input paths and sequence settings.
        data_root (str | pathlib.Path): Root directory for relative dataset image and mask
            paths.
        cache (InputCache | None): Optional verified input cache reused for target reads.

    Returns:
        dict: Case-level boundary metrics, scope, units, and provenance.
    """
    import numpy as np
    from PIL import Image

    archive = verify_archive(result_path, row)
    from .inputs import InputCache

    cache = cache if cache is not None else InputCache(0)
    target, transform = cache.read_target(case, data_root)
    spacing, units = stored_grid(case, transform)
    frame_ids = row["frame_ids"]
    support = set(case["support_frames"])
    query = (
        frame_ids if case["dim"] == 2 else [t for t in frame_ids if t not in support]
    )
    base = dict(
        model=row["model"],
        unit_key=row["unit_key"],
        case=row["case"],
        mode=row["mode"],
        sequence_condition=row["sequence_condition"],
        run_id=row["run_id"],
        result_sha256=file_hash(result_path),
        archive_sha256=row["prediction_archive"]["sha256"],
        case_sha256=digest(case),
        shared_case_sha256=case["shared_case_sha256"],
        frame_ids=frame_ids,
        query_frame_ids=query,
        grid="stored annotation grid; canonical binary predictions resized nearest-neighbor",
        hd95_convention="max of directed 95th surfel-area-weighted percentiles",
        assd_convention="pooled bidirectional surfel-area-weighted mean",
    )

    def restore(stream, t):
        """
        Restore one packed prediction mask to the stored target grid.

        Args:
            stream (zipfile.ZipFile): Open prediction archive containing packed frame masks.
            t (int): Stored frame index.

        Returns:
            np.ndarray: Boolean prediction resized with nearest-neighbor interpolation.
        """
        packed = np.frombuffer(stream.read(f"masks/{t:06d}.bin"), dtype=np.uint8)
        if packed.size != 512 * 512 // 8:
            raise ValueError("Invalid packed prediction length")

        mask = np.unpackbits(packed).reshape(512, 512)
        return np.asarray(
            Image.fromarray(mask).resize(
                (target.shape[2], target.shape[1]), Image.Resampling.NEAREST
            ),
            dtype=bool,
        )

    video = case_kind(case) == "video"
    if video or (case["dim"] == 3 and row["sequence_condition"] == "positive_frames"):
        # Discontinuous slice selections do not define a physically contiguous volume.
        # Compute real 2D query-slice surfaces, then average within this case.
        slice_metrics, counts = [], {}
        with zipfile.ZipFile(archive) as stream:
            for t in frame_ids:
                prediction = restore(stream, t)
                counts[t] = confusion(prediction, target[t])
                if t not in support:
                    slice_metrics.append(
                        dict(
                            frame=t,
                            **boundary_metrics(
                                prediction, target[t], spacing if video else spacing[1:]
                            ),
                        )
                    )

        bad = sum(r["status"] == "one_empty" for r in slice_metrics)
        metric_values = dict(
            hd95=(
                None
                if bad
                else sum(r["hd95"] for r in slice_metrics) / len(slice_metrics)
            ),
            assd=(
                None
                if bad
                else sum(r["assd"] for r in slice_metrics) / len(slice_metrics)
            ),
            nsd={
                str(t): sum(r["nsd"][str(t)] for r in slice_metrics)
                / len(slice_metrics)
                for t in (1.0, 2.0, 3.0)
            },
        )
        all_counts, query_counts = add_counts(counts.values()), add_counts(
            counts[t] for t in query
        )

        return dict(
            base,
            scope=(
                "mean_query_video_frame_surfaces"
                if video
                else "mean_positive_query_slice_surfaces"
            ),
            units="stored_pixels" if units == "stored_voxels" else units,
            spacing=spacing if video else spacing[1:],
            status="one_empty" if bad else "defined",
            undefined_surface_slices=bad,
            per_slice=slice_metrics,
            overlap_on_stored_grid={**all_counts, **overlaps(all_counts)},
            query_overlap_on_stored_grid={**query_counts, **overlaps(query_counts)},
            **metric_values,
        )

    pred = np.empty(target.shape, dtype=bool)
    with zipfile.ZipFile(archive) as stream:
        for t in frame_ids:
            pred[t] = restore(stream, t)
    if case["dim"] == 2:
        pred, target = pred[0], target[0]
    elif frame_ids != list(range(len(target))):
        raise ValueError("Volumetric surfaces require every stored slice")

    metrics = boundary_metrics(pred, target, spacing)
    counts = confusion(pred, target)
    result = dict(
        base,
        units=units,
        spacing=spacing,
        scope=(
            "independent_2d" if case["dim"] == 2 else "all_frame_assisted_full_scan_3d"
        ),
        overlap_on_stored_grid={**counts, **overlaps(counts)},
        **metrics,
    )

    if case["dim"] == 3:
        counts = confusion(pred[query], target[query])
        result["query_overlap_on_stored_grid"] = {**counts, **overlaps(counts)}

    return result


def surfaces(args):
    """
    Analyze saved prediction surfaces and write coverage-aware summaries.

    Args:
        args (argparse.Namespace): Parsed command options, including input paths and report
            or trial settings.

    Returns:
        None.
    """
    from .runner import read_frozen
    from .report import source_means, hierarchical
    from surface_distance import metrics as backend
    from torch.utils.tensorboard import SummaryWriter
    from .surface_workers import SurfacePool, analysis_identity

    workers, cache_gib = getattr(args, "workers", 1), getattr(args, "cache_gib", 0.5)

    if workers < 1 or cache_gib < 0:
        raise ValueError("Require positive workers and nonnegative cache size")

    meta, cases = read_frozen(args.manifest)
    by_id = {c["unit_id"]: c for c in cases if c["eligible"]}
    manifest_hash = file_hash(args.manifest / "manifest.json")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        manifest_sha256=manifest_hash,
        sequence_condition=meta["sequence_condition"],
        backend_version=importlib.metadata.version("surface-distance"),
        backend_source_sha256=file_hash(backend.__file__),
        tolerances=[1.0, 2.0, 3.0],
        scientific_protocol_certified=False,
        analysis_identity=analysis_identity(),
        workers=workers,
        cache_gib_per_worker=cache_gib,
    )
    atomic_json(args.output / "provenance.json", provenance)
    groups, seen, model_ids = defaultdict(list), set(), {}
    writer = SummaryWriter(str(args.output / "tensorboard"))
    pool = None
    completed = 0
    reused = 0

    def collect(items):
        """
        Group completed surface results and record analysis progress.

        Args:
            items (list[dict]): Completed surface-job records to incorporate into reporting.

        Returns:
            None.
        """
        nonlocal completed, reused
        for item in items:
            result = item["result"]
            group = (
                result["model"],
                result["sequence_condition"],
                case_kind(result["case"]),
                result["mode"],
                result["units"],
            )
            groups[group].append(result)
            completed += 1
            reused += int(item["reused"])
            writer.add_scalar("boundary/progress/completed", completed, completed)
            writer.add_scalar("boundary/progress/reused", reused, completed)

    try:
        pool = SurfacePool(workers, int(cache_gib * 1024**3))
        for job_path in sorted(args.runs.glob("*/shard-*/job.json")):
            job = json.loads(job_path.read_text())
            identity = job["identity"]
            if (
                job["run_id"] != digest(identity)
                or identity["manifest_sha256"] != manifest_hash
            ):
                raise ValueError("Surface analysis job/cohort identity mismatch")

            model = identity["model"]
            if model in model_ids and model_ids[model] != job["run_id"]:
                raise ValueError("Mixed runs for the same model")

            model_ids[model] = job["run_id"]
            job_expected = set(job["expected"])
            for path in sorted((job_path.parent / "cases").glob("*.json")):
                row = json.loads(path.read_text())
                case = by_id[row["case"]["unit_id"]]
                validate_result(row, case, row["mode"], job["run_id"])

                if row["unit_key"] not in job_expected or row["model"] != model:
                    raise ValueError("Unexpected surface input result")

                key = (model, row["unit_key"])
                if key in seen:
                    raise ValueError("Duplicate surface input result")

                seen.add(key)
                collect(pool.collect())
                while len(pool.pending) >= pool.capacity:
                    collect(pool.collect(block=True))

                pool.submit(
                    row,
                    path,
                    case,
                    args.data_root,
                    args.output / model / path.name,
                    job_path.parent / "surfaces/cases" / path.name,
                )
        while pool.pending:
            collect(pool.collect(block=True))

        summary = {}
        for group, rows in groups.items():
            rows.sort(key=lambda r: r["unit_key"])
            name = "/".join(map(str, group))
            item = dict(
                completed=len(rows),
                one_empty=sum(r["status"] == "one_empty" for r in rows),
                scope=rows[0]["scope"],
                metrics={},
            )

            for metric in ("hd95", "assd", "nsd_1", "nsd_2", "nsd_3"):
                get = lambda r: (
                    r["nsd"][metric[-1] + ".0"]
                    if metric.startswith("nsd")
                    else r[metric]
                )
                defined = [
                    dict(case=r["case"], primary={metric: get(r)})
                    for r in rows
                    if get(r) is not None
                ]
                aggregate = hierarchical(source_means(defined, metric))
                item["metrics"][metric] = dict(
                    distribution=distribution(get(r) for r in rows),
                    undefined=len(rows) - len(defined),
                    conditional_on_defined=True,
                    **aggregate,
                )

                if aggregate["macro"] is not None:
                    writer.add_scalar(
                        f"boundary/{name}/{metric}", aggregate["macro"], 0
                    )

            summary[name] = item

        expected = sum(len(c["modes"]) for c in by_id.values())
        atomic_json(
            args.output / "summary.json",
            dict(
                groups=summary,
                coverage={
                    m: dict(
                        expected=expected,
                        completed=sum(model == m for model, _ in seen),
                    )
                    for m in model_ids
                },
                sequence_condition=meta["sequence_condition"],
                units_are_never_pooled=True,
                reused_inline_results=reused,
                full_3d_includes_support_annotations=meta["sequence_condition"]
                == "full_scan"
                and any(case_kind(c) == "3d" for c in by_id.values()),
            ),
        )
    finally:
        if pool:
            pool.close()

        writer.close()
