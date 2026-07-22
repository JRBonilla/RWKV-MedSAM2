#!/usr/bin/env python3
"""Export Sunnybrook Cardiac MRI annotations as phase-separated 3D volumes.

The manual contours contain two cardiac phases at most anatomical positions.  A
single DICOM series therefore cannot be treated as one ordinary z-stack: doing
so interleaves the phases and breaks 3D connectivity.  This exporter assigns
frames to phase 00/01, orders each phase spatially, and saves overlapping
cardiac regions as separate binary NIfTI masks.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image


DEFAULT_ROOT = Path("/data/research/Cardiac MRI/Merged_Cardiac_MRI")
DEFAULT_OUTPUT = Path("/data/research/Cardiac MRI/Fixed_Niftis_v2")
CLASS_NAMES = ("bloodpool", "myocardium", "epicardium", "endocardium")
POSITION_PRECISION = 3
CANONICAL_ORIENTATION = "LPS"


@dataclass(frozen=True)
class DicomFrame:
    path: Path
    position: float
    trigger_time: float
    instance_number: int


def _frame_metadata(path: Path) -> DicomFrame:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    normal = np.cross(orientation[:3], orientation[3:])
    position = float(np.dot(normal, np.asarray(ds.ImagePositionPatient, dtype=np.float64)))
    return DicomFrame(
        path=path,
        position=position,
        trigger_time=float(getattr(ds, "TriggerTime", 0.0)),
        instance_number=int(getattr(ds, "InstanceNumber", 0)),
    )


def split_cardiac_phases(frames: Iterable[DicomFrame]) -> list[list[DicomFrame]]:
    """Assign annotated frames to two trigger-time phases and sort each in z."""
    by_position: dict[float, list[DicomFrame]] = defaultdict(list)
    for frame in frames:
        by_position[round(frame.position, POSITION_PRECISION)].append(frame)
    if not by_position:
        raise ValueError("No annotated DICOM frames were found")
    if any(len(items) > 2 for items in by_position.values()):
        raise ValueError("Expected no more than two annotated cardiac phases per position")

    paired = [
        sorted(items, key=lambda item: (item.trigger_time, item.instance_number))
        for items in by_position.values()
        if len(items) == 2
    ]
    if not paired:
        raise ValueError("Could not identify two cardiac phases from the annotated frames")

    phase_centers = [float(np.median([pair[index].trigger_time for pair in paired])) for index in range(2)]
    phases: list[list[DicomFrame]] = [[], []]
    for items in by_position.values():
        ordered = sorted(items, key=lambda item: (item.trigger_time, item.instance_number))
        if len(ordered) == 2:
            phases[0].append(ordered[0])
            phases[1].append(ordered[1])
        else:
            frame = ordered[0]
            phase_index = int(np.argmin([abs(frame.trigger_time - center) for center in phase_centers]))
            phases[phase_index].append(frame)

    for phase in phases:
        phase.sort(key=lambda item: (item.position, item.instance_number))
        if len(phase) < 2:
            raise ValueError("A cardiac phase contains fewer than two spatial slices")
    return phases


def _read_dicom_volume(frames: list[DicomFrame]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(frame.path) for frame in frames])
    return reader.Execute()


def _canonicalize_orientation(image: sitk.Image) -> sitk.Image:
    """Use local axis-aligned training coordinates without permuting voxel data.

    Sunnybrook short-axis DICOM directions are highly oblique.  Some NIfTI
    viewers reduce those directions to duplicate anatomical axes (for example
    SLL), while anatomical reorientation can move the slice stack away from
    array axis zero.  DRIPP requires [slices, height, width], so the exported
    image and masks share a standardized local LPS header instead.
    """
    oriented = sitk.Image(image)
    oriented.SetDirection((
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ))
    oriented.SetOrigin((0.0, 0.0, 0.0))
    return oriented


def _binary_png(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(Image.open(path))
    if array.shape != shape:
        raise ValueError(f"Mask shape {array.shape} does not match DICOM shape {shape}: {path}")
    return array != 0


def _mask_paths(mask_dir: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(mask_dir.glob("*.png")):
        base, separator, class_name = path.stem.rpartition("-")
        if separator and class_name in CLASS_NAMES:
            grouped[base][class_name] = path
    return grouped


def _phase_masks(
    frames: list[DicomFrame], grouped_masks: dict[str, dict[str, Path]], shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    for frame in frames:
        source = grouped_masks.get(frame.path.stem, {})
        endocardium = _binary_png(source.get("endocardium"), shape)
        epicardium = _binary_png(source.get("epicardium"), shape)
        bloodpool_source = _binary_png(source.get("bloodpool"), shape)
        myocardium_source = _binary_png(source.get("myocardium"), shape)

        # Prefer the already-derived source masks, while keeping deterministic
        # fallbacks for older merged folders that only contain contour fills.
        bloodpool = bloodpool_source if bloodpool_source.any() else endocardium
        myocardium = myocardium_source if myocardium_source.any() else np.logical_and(epicardium, ~endocardium)
        values = {
            "bloodpool": bloodpool,
            "myocardium": myocardium,
            "epicardium": epicardium,
            "endocardium": endocardium,
        }
        for class_name, mask in values.items():
            result[class_name].append(mask.astype(np.uint8, copy=False))
    return {name: np.stack(slices, axis=0) for name, slices in result.items()}


def _atomic_write_image(image: sitk.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.nii.gz")
    try:
        sitk.WriteImage(image, str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_study(study_dir: Path, output_root: Path, overwrite: bool = False) -> list[dict]:
    image_dir = study_dir / "Images"
    mask_dir = study_dir / "Masks"
    frames = [_frame_metadata(path) for path in sorted(image_dir.glob("*.dcm"))]
    phases = split_cardiac_phases(frames)
    grouped_masks = _mask_paths(mask_dir)
    exported = []

    for phase_index, phase_frames in enumerate(phases):
        identifier = f"{study_dir.name}_phase{phase_index:02d}"
        phase_dir = output_root / identifier
        destinations = [phase_dir / f"{identifier}_image.nii.gz"] + [
            phase_dir / f"{identifier}_{name}_mask.nii.gz" for name in CLASS_NAMES
        ]
        if not overwrite and any(path.exists() for path in destinations):
            raise FileExistsError(f"Output already exists for {identifier}; pass --overwrite to replace it")

        source_image = _read_dicom_volume(phase_frames)
        source_array = sitk.GetArrayViewFromImage(source_image)
        if source_array.ndim != 3 or source_array.shape[0] != len(phase_frames):
            raise ValueError(f"Unexpected image shape for {identifier}: {source_array.shape}")
        masks = _phase_masks(phase_frames, grouped_masks, tuple(source_array.shape[1:]))

        image = _canonicalize_orientation(source_image)
        _atomic_write_image(image, destinations[0])
        foreground = {}
        for class_name, destination in zip(CLASS_NAMES, destinations[1:]):
            mask_image = sitk.GetImageFromArray(masks[class_name])
            mask_image.CopyInformation(source_image)
            mask_image = _canonicalize_orientation(mask_image)
            if (
                mask_image.GetSize() != image.GetSize()
                or mask_image.GetSpacing() != image.GetSpacing()
                or mask_image.GetOrigin() != image.GetOrigin()
                or mask_image.GetDirection() != image.GetDirection()
            ):
                raise ValueError(f"Canonical mask geometry mismatch for {identifier}/{class_name}")
            _atomic_write_image(mask_image, destination)
            foreground[class_name] = int(masks[class_name].sum())

        exported.append({
            "identifier": identifier,
            "source_study": study_dir.name,
            "phase_index": phase_index,
            "slice_count": len(phase_frames),
            "trigger_time_median": float(np.median([frame.trigger_time for frame in phase_frames])),
            "foreground_voxels": foreground,
        })
    return exported


def export_all(root: Path, output: Path, study: str | None = None, overwrite: bool = False) -> list[dict]:
    studies = [root / study] if study else sorted(path for path in root.iterdir() if path.is_dir())
    records = []
    for study_dir in studies:
        if not (study_dir / "Images").is_dir() or not (study_dir / "Masks").is_dir():
            continue
        study_records = export_study(study_dir, output, overwrite=overwrite)
        records.extend(study_records)
        print(f"Exported {study_dir.name}: " + ", ".join(
            f"{record['identifier']} ({record['slice_count']} slices)" for record in study_records
        ))

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "export_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "studies": records}, indent=2) + "\n")
    os.replace(temporary, summary_path)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Merged Cardiac MRI input directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Phase-separated NIfTI output directory")
    parser.add_argument("--study", help="Export one study identifier for validation")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace existing output files")
    args = parser.parse_args()
    records = export_all(args.root, args.output, study=args.study, overwrite=args.overwrite)
    print(f"Exported {len(records)} phase volumes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
