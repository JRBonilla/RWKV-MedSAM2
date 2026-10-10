"""Lossless prediction archives and supplementary accuracy diagnostics."""

from __future__ import annotations
import io
import json
import os
from pathlib import Path
import zipfile

from ..shared.data import file_hash


class PredictionArchive:
    """
    Store lossless per-frame binary predictions and optional logits atomically.
    """

    def __init__(self, path, identity, save_logits=False):
        """
        Open a temporary prediction archive with frozen identity metadata.

        Args:
            path (str | pathlib.Path): Final destination of the atomically committed prediction
                ZIP archive.
            identity (dict): Frozen run, case, sequence, and frame identity stored in the
                archive.
            save_logits (bool): Whether to store float32 logits alongside packed masks.

        Returns:
            None.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        self.stream = zipfile.ZipFile(
            self.temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        )
        self.save_logits = save_logits
        self.stream.writestr(
            "identity.json",
            json.dumps(
                dict(
                    identity,
                    shape_hw=[512, 512],
                    encoding="numpy.packbits, C order, big bit order",
                    save_logits=save_logits,
                ),
                sort_keys=True,
            ),
        )

    def frame(self, t, prediction, logits):
        """
        Append a packed binary mask and optional float32 logits for one frame.

        Args:
            t (int): Stored frame index.
            prediction (np.ndarray): Binary predicted mask aligned with the target.
            logits (np.ndarray): Predicted binary-mask logits.

        Returns:
            None.
        """
        import numpy as np

        self.stream.writestr(
            f"masks/{t:06d}.bin", np.packbits(prediction.reshape(-1)).tobytes()
        )

        if self.save_logits:
            payload = io.BytesIO()
            np.save(payload, np.asarray(logits, dtype=np.float32), allow_pickle=False)
            self.stream.writestr(f"logits/{t:06d}.npy", payload.getvalue())

    def commit(self):
        """
        Close and atomically publish the completed prediction archive.

        Args:
            None.

        Returns:
            dict: Archive filename, SHA256 digest, size, and saved-payload flags.
        """
        self.stream.close()
        os.replace(self.temporary, self.path)

        return dict(
            filename=self.path.name,
            sha256=file_hash(self.path),
            bytes=self.path.stat().st_size,
            logits_saved=self.save_logits,
            binary_predictions_saved=True,
        )

    def abort(self):
        """
        Close the archive and remove its temporary file.

        Args:
            None.

        Returns:
            None.
        """
        self.stream.close()
        self.temporary.unlink(missing_ok=True)


def verify_archive(result_path, row):
    """
    Validate archive content, identity, and complete frame coverage.

    Args:
        result_path (str | pathlib.Path): Path to the saved case-result JSON file.
        row (dict): Saved evaluation result or diagnostic record.

    Returns:
        pathlib.Path: Path to the verified prediction archive.
    """
    info = row["prediction_archive"]
    path = Path(result_path).parent.parent / "predictions" / info["filename"]

    if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
        raise ValueError(f"Prediction archive changed: {path}")
    with zipfile.ZipFile(path) as stream:
        identity = json.loads(stream.read("identity.json"))
        for key in (
            "run_id",
            "unit_key",
            "case_sha256",
            "sequence_condition",
            "frame_ids",
        ):
            if identity[key] != row[key]:
                raise ValueError("Prediction archive identity mismatch")

        expected = {f"masks/{t:06d}.bin" for t in row["frame_ids"]}
        if set(n for n in stream.namelist() if n.startswith("masks/")) != expected:
            raise ValueError("Prediction archive has incomplete frame coverage")

        expected_logits = (
            {f"logits/{t:06d}.npy" for t in row["frame_ids"]}
            if info["logits_saved"]
            else set()
        )
        if (
            set(n for n in stream.namelist() if n.startswith("logits/"))
            != expected_logits
        ):
            raise ValueError("Logit archive has incomplete frame coverage")

    return path


def confusion_diagnostics(c):
    """
    Compute classification and size-error diagnostics from confusion counts.

    Args:
        c (dict[str, int]): Pixel confusion counts keyed by tp, fp, fn, and tn.

    Returns:
        dict: Pixel metrics and size errors, with None for undefined ratios.
    """
    tp, fp, fn, tn = [c[k] for k in ("tp", "fp", "fn", "tn")]
    divide = lambda a, b: a / b if b else None

    return dict(
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        specificity=divide(tn, tn + fp),
        pixel_accuracy=divide(tp + tn, tp + fp + fn + tn),
        false_positive_rate=divide(fp, fp + tn),
        false_negative_rate=divide(fn, fn + tp),
        target_pixels=tp + fn,
        predicted_pixels=tp + fp,
        signed_size_error_pixels=fp - fn,
        absolute_size_error_pixels=abs(fp - fn),
        relative_size_error=divide(fp - fn, tp + fn),
    )


def probability_record(logits, target):
    """
    Raw pixel sums + bins permit pooled calibration summaries without rerunning.

    Args:
        logits (np.ndarray): Predicted binary-mask logits.
        target (np.ndarray): Binary ground-truth mask aligned with the predictions.

    Returns:
        dict: Pixel counts, Brier/log-loss sums, and twenty calibration-bin statistics.
    """
    import numpy as np

    z = np.asarray(logits, np.float64).reshape(-1)
    y = np.asarray(target, np.float64).reshape(-1)
    p = np.exp(-np.logaddexp(0.0, -z))
    bins = np.minimum((p * 20).astype(np.int64), 19)
    squared = (p - y) ** 2
    loss = np.logaddexp(0.0, z) - y * z

    return dict(
        pixel_count=len(y),
        brier_sum=float(squared.sum()),
        log_loss_sum=float(loss.sum()),
        foreground_count=int(y.sum()),
        foreground_brier_sum=float(squared[y == 1].sum()),
        background_brier_sum=float(squared[y == 0].sum()),
        bin_count=np.bincount(bins, minlength=20).tolist(),
        bin_probability_sum=np.bincount(bins, weights=p, minlength=20).tolist(),
        bin_foreground_sum=np.bincount(bins, weights=y, minlength=20).tolist(),
    )


def pool_probability(records):
    """
    Combine probability records into pixel-weighted calibration metrics.

    Args:
        records (Iterable[dict]): Per-frame or per-case diagnostic records.

    Returns:
        dict: Pooled calibration metrics and bin statistics.
    """
    import numpy as np

    records = list(records)
    n = sum(r["pixel_count"] for r in records)
    counts = np.sum([r["bin_count"] for r in records], axis=0)
    probability = np.sum([r["bin_probability_sum"] for r in records], axis=0)
    foreground = np.sum([r["bin_foreground_sum"] for r in records], axis=0)
    positive = sum(r["foreground_count"] for r in records)

    return dict(
        pixel_count=n,
        brier=sum(r["brier_sum"] for r in records) / n,
        log_loss=sum(r["log_loss_sum"] for r in records) / n,
        ece20=float(np.abs(probability - foreground).sum() / n),
        foreground_brier=(
            sum(r["foreground_brier_sum"] for r in records) / positive
            if positive
            else None
        ),
        background_brier=(
            sum(r["background_brier_sum"] for r in records) / (n - positive)
            if n > positive
            else None
        ),
        bin_count=counts.tolist(),
        bin_probability_sum=probability.tolist(),
        bin_foreground_sum=foreground.tolist(),
        bin_edges=np.linspace(0, 1, 21).tolist(),
        interpretation="pixel-weighted foreground-probability calibration; not clinical uncertainty",
    )


def slice_summary(frames):
    """
    Summarize positive and empty query-slice prediction errors.

    Args:
        frames (Iterable[dict]): Per-frame prediction records.

    Returns:
        dict: Slice counts, false-positive diagnostics, and mean positive-slice Dice.
    """
    from ..shared.data import overlaps

    frames = list(frames)
    positive = [r for r in frames if r["tp"] + r["fn"] > 0]
    empty = [r for r in frames if r["tp"] + r["fn"] == 0]

    return dict(
        positive_query_slices=len(positive),
        empty_query_slices=len(empty),
        empty_query_slices_with_false_positives=sum(r["fp"] > 0 for r in empty),
        empty_query_false_positive_pixels=sum(r["fp"] for r in empty),
        missed_positive_query_slices=sum(r["tp"] == 0 for r in positive),
        positive_slice_mean_dice=(
            sum(overlaps(r)["dice"] for r in positive) / len(positive)
            if positive
            else None
        ),
    )


def distribution(values):
    """
    Summarize defined values with moments and quantiles.

    Args:
        values (Iterable[float]): Numeric values to summarize.

    Returns:
        dict: Defined count and descriptive statistics, or only a zero count when empty.
    """
    import numpy as np

    x = np.asarray([v for v in values if v is not None], dtype=np.float64)

    if not len(x):
        return {"defined": 0}

    return dict(
        defined=len(x),
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if len(x) > 1 else None,
        min=float(x.min()),
        p05=float(np.quantile(x, 0.05)),
        q25=float(np.quantile(x, 0.25)),
        median=float(np.median(x)),
        q75=float(np.quantile(x, 0.75)),
        max=float(x.max()),
    )
