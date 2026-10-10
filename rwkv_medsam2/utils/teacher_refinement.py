"""Correctness helpers for Oxford Medical-SAM2 teacher refinement.

The notebook imports these small, dependency-light functions so their input
normalization, 3D window selection, split checks, and checkpoint scoring can be
covered by ordinary unit tests.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM2_IMAGE_STD = (0.229, 0.224, 0.225)


def teacher_prefetch_memory_requirement(
    *,
    batch_size: int,
    sequence_length: int,
    height: int,
    width: int,
    num_workers: int,
    prefetch_factor: int,
    image_channels: int = 3,
    mask_channels: int = 1,
    image_itemsize: int = 1,
    mask_itemsize: int = 1,
    headroom: float = 1.25,
) -> Mapping[str, int]:
    """
    Estimate shared memory needed by a teacher DataLoader prefetch queue.

    Args:
        batch_size (int): Number of samples or objects in the batch.
        sequence_length (int): Number of frames in each prefetched sequence.
        height (int): Frame height in pixels.
        width (int): Frame width in pixels.
        num_workers (int): Number of DataLoader worker processes.
        prefetch_factor (int): Number of batches prefetched by each worker.
        image_channels (int): Number of image channels transported by the loader.
        mask_channels (int): Number of mask channels transported by the loader.
        image_itemsize (int): Bytes per transported image element.
        mask_itemsize (int): Bytes per transported mask element.
        headroom (float): Safety multiplier applied to the estimated prefetch footprint.

    Returns:
        Mapping[str, int]: Batch bytes, queue slots, and required bytes including headroom.
    """
    positive = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "height": height,
        "width": width,
        "image_channels": image_channels,
        "mask_channels": mask_channels,
        "image_itemsize": image_itemsize,
        "mask_itemsize": mask_itemsize,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be nonnegative, got {num_workers}")
    if int(num_workers) > 0 and int(prefetch_factor) <= 0:
        raise ValueError(
            f"prefetch_factor must be positive when workers are enabled, got {prefetch_factor}"
        )
    if float(headroom) < 1.0:
        raise ValueError(f"headroom must be at least 1.0, got {headroom}")

    pixels = int(batch_size) * int(sequence_length) * int(height) * int(width)
    batch_bytes = pixels * (
        int(image_channels) * int(image_itemsize)
        + int(mask_channels) * int(mask_itemsize)
    )
    queue_slots = 1 + int(num_workers) * int(prefetch_factor) if num_workers else 1
    required_bytes = math.ceil(batch_bytes * queue_slots * float(headroom))

    return {
        "batch_bytes": batch_bytes,
        "queue_slots": queue_slots,
        "required_bytes": required_bytes,
    }


def available_shared_memory_bytes(path: str = "/dev/shm") -> int:
    """
    Return currently available bytes in the shared-memory filesystem.

    Args:
        path (str): Shared-memory filesystem directory whose available capacity is queried.

    Returns:
        int: Bytes available to unprivileged processes at the supplied path.
    """
    stats = os.statvfs(path)

    return int(stats.f_bavail * stats.f_frsize)


def assert_teacher_shared_memory_capacity(
    *,
    available_bytes: Optional[int] = None,
    shm_path: str = "/dev/shm",
    **loader_shape,
) -> Mapping[str, int]:
    """
    Fail early when a teacher loader configuration is unsafe for /dev/shm.

    Args:
        available_bytes (Optional[int]): Explicit shared-memory capacity, or None to query
            the filesystem.
        shm_path (str): Shared-memory filesystem path whose capacity is checked.
        **loader_shape (dict): Keyword arguments accepted by
            teacher_prefetch_memory_requirement, including batch_size, sequence_length,
            height, width, num_workers, and prefetch_factor.

    Returns:
        Mapping[str, int]: Estimated memory requirement; raises RuntimeError when capacity
            is insufficient.
    """
    requirement = dict(teacher_prefetch_memory_requirement(**loader_shape))
    available = (
        available_shared_memory_bytes(shm_path)
        if available_bytes is None
        else int(available_bytes)
    )

    if available < 0:
        raise ValueError(f"available_bytes must be nonnegative, got {available}")
    if available < requirement["required_bytes"]:
        required_mib = requirement["required_bytes"] / (1024**2)
        available_mib = available / (1024**2)
        raise RuntimeError(
            "Unsafe teacher DataLoader shared-memory configuration: "
            f"estimated requirement is {required_mib:.1f} MiB but only "
            f"{available_mib:.1f} MiB is available. Reduce workers or prefetching; "
            "do not switch worker transport back to float tensors."
        )

    requirement["available_bytes"] = available
    return requirement


def normalize_sam2_images(images_255: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB images in ``[0, 255]`` to SAM2's normalized input space.

    Args:
        images_255 (torch.Tensor): RGB image tensor with intensities on the [0, 255] scale.

    Returns:
        torch.Tensor: Float32 ImageNet-normalized RGB tensor with the input shape.
    """
    if images_255.ndim < 4 or images_255.shape[-3] != 3:
        raise ValueError(
            f"Expected RGB images with channel dimension 3, got {tuple(images_255.shape)}"
        )

    images = images_255.float() / 255.0
    shape = [1] * images.ndim
    shape[-3] = 3
    mean = images.new_tensor(SAM2_IMAGE_MEAN).view(shape)
    std = images.new_tensor(SAM2_IMAGE_STD).view(shape)

    return (images - mean) / std


def highest_resolution_axis(volume_shape: Sequence[int]) -> int:
    """
    Return the slicing axis whose slices have the greatest pixel area.

    Args:
        volume_shape (Sequence[int]): Three-dimensional volume shape in array-axis order.

    Returns:
        int: Selected axis index, with the first axis winning ties.
    """
    if len(volume_shape) != 3:
        raise ValueError(f"Expected a 3D volume shape, got {tuple(volume_shape)}")

    d, h, w = (int(value) for value in volume_shape)
    if min(d, h, w) <= 0:
        raise ValueError(f"Volume dimensions must be positive, got {(d, h, w)}")

    return int(np.asarray((h * w, d * w, d * h), dtype=np.int64).argmax())


def select_contiguous_foreground_window(
    mask_volume: np.ndarray,
    seq_len: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[int, np.ndarray]:
    """
    Select a contiguous foreground-rich window on the highest-resolution axis.

    Windows are ranked first by their number of nonempty slices and then by
    total foreground area. Validation callers omit ``rng`` for deterministic
    center-nearest tie breaking; training callers may provide an RNG to sample
    among equally ranked windows.

    Args:
        mask_volume (np.ndarray): Three-dimensional mask used to select a foreground-rich
            window.
        seq_len (int): Required number of consecutive slices in the selected window.
        rng (Optional[np.random.RandomState]): Optional random generator for training-time
            tie breaking.

    Returns:
        Tuple[int, np.ndarray]: Slice axis and consecutive selected slice indices.
    """
    mask = np.asarray(mask_volume)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    axis = highest_resolution_axis(mask.shape)
    length = int(mask.shape[axis])
    if length < seq_len:
        raise ValueError(f"Axis {axis} is too short: need {seq_len}, have {length}")

    reduce_axes = tuple(i for i in range(3) if i != axis)
    areas = np.count_nonzero(mask, axis=reduce_axes).astype(np.int64, copy=False)
    if int(areas.sum()) == 0:
        raise ValueError("3D mask has no foreground")

    foreground = (areas > 0).astype(np.int64)
    # Prefix sums score every contiguous window without rescanning its slices.
    count_cs = np.concatenate(([0], np.cumsum(foreground)))
    area_cs = np.concatenate(([0], np.cumsum(areas, dtype=np.int64)))
    window_counts = count_cs[seq_len:] - count_cs[:-seq_len]
    window_areas = area_cs[seq_len:] - area_cs[:-seq_len]

    best_count = int(window_counts.max())
    candidates = np.flatnonzero(window_counts == best_count)
    best_area = int(window_areas[candidates].max())
    candidates = candidates[window_areas[candidates] == best_area]

    if rng is not None and candidates.size > 1:
        start = int(rng.choice(candidates))
    else:
        peak = int(areas.argmax())
        centered_start = max(0, min(length - seq_len, peak - seq_len // 2))
        start = int(candidates[np.abs(candidates - centered_start).argmin()])

    return axis, np.arange(start, start + seq_len, dtype=np.int64)


def extract_contiguous_volume_window(
    image_volume: np.ndarray,
    mask_volume: np.ndarray,
    seq_len: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Extract matching ``[T,H,W]`` image and mask windows.

    Args:
        image_volume (np.ndarray): Three-dimensional image aligned with mask_volume.
        mask_volume (np.ndarray): Three-dimensional mask used to select a foreground-rich
            window.
        seq_len (int): Required number of consecutive slices in the selected window.
        rng (Optional[np.random.RandomState]): Optional random generator for training-time
            tie breaking.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, np.ndarray]: Image window, mask window, slice
            axis, and selected indices.
    """
    image = np.asarray(image_volume)
    mask = np.asarray(mask_volume)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")

    axis, indices = select_contiguous_foreground_window(mask, seq_len, rng=rng)
    images = np.moveaxis(np.take(image, indices, axis=axis), axis, 0)
    masks = np.moveaxis(np.take(mask, indices, axis=axis), axis, 0)

    return images, masks, axis, indices


def scale_intensity_to_u8(array: np.ndarray) -> np.ndarray:
    """
    Robustly window a scalar medical image/volume into uint8.

    Args:
        array (np.ndarray): Input image, mask, or volume array.

    Returns:
        np.ndarray: Uint8 image or volume with nonfinite values mapped to valid byte
            intensities.
    """
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("Image contains no finite intensities")

    lo, hi = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Could not determine a finite intensity window")
    if hi <= lo:
        hi = lo + 1.0

    scaled = (np.clip(values, lo, hi) - lo) * (255.0 / (hi - lo))
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def source_paths(sequences: Iterable[Mapping]) -> set[str]:
    """
    Collect normalized source image paths from formatted sequence records.

    Args:
        sequences (Iterable[Mapping]): Formatted dataset records containing source
            identities and image paths.

    Returns:
        set[str]: Absolute, case-normalized source image paths.
    """
    paths = set()
    for record in sequences:
        sequence = record.get("sequence") or []
        if sequence:
            paths.add(os.path.normcase(os.path.abspath(str(sequence[0][0]))))

    return paths


def source_keys(sequences: Iterable[Mapping]) -> set[str]:
    """
    Collect case/group identities, falling back to normalized image paths.

    Args:
        sequences (Iterable[Mapping]): Formatted dataset records containing source
            identities and image paths.

    Returns:
        set[str]: Dataset-qualified case identities, falling back to normalized image paths.
    """
    keys = set()
    for record in sequences:
        source = record.get("source_identifier") or record.get("identifier")
        if source is None:
            sequence = record.get("sequence") or []
            if sequence:
                source = os.path.normcase(os.path.abspath(str(sequence[0][0])))
        if source is not None:
            # Identifiers are only guaranteed to be unique inside their DRIPP
            # dataset/subdataset. Namespace them to avoid false collisions such
            # as two unrelated datasets both using identifier "0001".
            dataset = str(record.get("dataset") or "")
            subdataset = str(record.get("subdataset") or "")
            keys.add(f"{dataset}\x1f{subdataset}\x1f{source}")

    return keys


def assert_disjoint_sources(
    train_sequences: Iterable[Mapping], val_sequences: Iterable[Mapping]
) -> None:
    """
    Raise when train and validation share an image or source-group identity.

    Args:
        train_sequences (Iterable[Mapping]): Formatted training sequence records.
        val_sequences (Iterable[Mapping]): Formatted validation sequence records.

    Returns:
        None.
    """
    train_sequences = list(train_sequences)
    val_sequences = list(val_sequences)
    image_overlap = source_paths(train_sequences) & source_paths(val_sequences)
    group_overlap = source_keys(train_sequences) & source_keys(val_sequences)

    if image_overlap or group_overlap:
        examples = sorted(image_overlap or group_overlap)[:3]
        raise RuntimeError(
            "Train/validation source leakage detected "
            f"({len(image_overlap)} shared images, {len(group_overlap)} shared source groups). "
            "Rebuild /data/loaders32.pkl with the current formatted-dataset cache version. "
            f"Examples: {examples}"
        )


def checkpoint_selection_score(metrics: Mapping[str, float]) -> float:
    """
    Macro-average fixed-threshold 2D/3D Dice for checkpoint selection.

    Args:
        metrics (Mapping[str, float]): Validation metrics keyed by fixed-threshold Dice
            names.

    Returns:
        float: Balanced checkpoint-selection score for the available dimensions.
    """
    values = []
    for key in ("dice2d@0.5", "dice3d@0.5"):
        value = metrics.get(key)
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            values.append(float(value))
    if not values:
        raise ValueError("Validation did not produce fixed-threshold 2D or 3D Dice")

    return float(np.mean(values))
