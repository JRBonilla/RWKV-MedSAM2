"""Validated CT intensity profiles shared by DRIPP and its debugger."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 2


class CTProfileError(ValueError):
    """Raised when a CT profile file is missing, incompatible, or invalid."""


@dataclass(frozen=True)
class CTProfile:
    """Fixed window and global z-score parameters for one CT subdataset."""

    name: str
    intensity_space: str
    window_lower: float
    window_upper: float
    mean: float
    std: float
    count: int
    source_image_count: int
    failed_image_count: int

    @property
    def window_width(self) -> float:
        return self.window_upper - self.window_lower

    @property
    def window_level(self) -> float:
        return (self.window_upper + self.window_lower) / 2.0


def _finite_number(value, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CTProfileError(f"CT profile field {field!r} must be numeric") from exc
    if not math.isfinite(number):
        raise CTProfileError(f"CT profile field {field!r} must be finite")
    return number


def _parse_profile(name: str, raw: Mapping) -> CTProfile:
    if raw.get("stats_basis") != "full_after_window":
        raise CTProfileError(f"CT profile {name!r} was not computed from the full windowed data")
    if raw.get("window_stats_compatible") is not True:
        raise CTProfileError(f"CT profile {name!r} does not contain window-compatible statistics")

    lower = _finite_number(raw.get("window_lower"), "window_lower")
    upper = _finite_number(raw.get("window_upper"), "window_upper")
    mean = _finite_number(raw.get("mean"), "mean")
    std = _finite_number(raw.get("std"), "std")
    if upper <= lower:
        raise CTProfileError(f"CT profile {name!r} has an empty or reversed window")
    if std <= 0:
        raise CTProfileError(f"CT profile {name!r} has a non-positive standard deviation")

    try:
        count = int(raw.get("count"))
        source_count = int(raw.get("source_image_count"))
        failed_count = int(raw.get("failed_image_count"))
    except (TypeError, ValueError) as exc:
        raise CTProfileError(f"CT profile {name!r} has invalid count fields") from exc
    if count <= 0 or source_count <= 0 or failed_count < 0 or failed_count > source_count:
        raise CTProfileError(f"CT profile {name!r} has inconsistent count fields")

    intensity_space = str(raw.get("intensity_space", "")).strip().lower()
    if intensity_space not in {"hu", "display"}:
        raise CTProfileError(f"CT profile {name!r} has unsupported intensity_space={intensity_space!r}")

    return CTProfile(
        name=name,
        intensity_space=intensity_space,
        window_lower=lower,
        window_upper=upper,
        mean=mean,
        std=std,
        count=count,
        source_image_count=source_count,
        failed_image_count=failed_count,
    )


def load_ct_profiles(path, expected_dataset: str | None = None) -> dict[str, CTProfile]:
    """Load and fully validate a schema-v2 dataset CT profile file."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise CTProfileError(f"CT profile file does not exist: {profile_path}")
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CTProfileError(f"Could not read CT profile file {profile_path}: {exc}") from exc

    if document.get("schema_version") != SCHEMA_VERSION:
        raise CTProfileError(
            f"CT profile file {profile_path} is legacy or incompatible; expected schema_version={SCHEMA_VERSION}"
        )
    dataset_name = str(document.get("dataset_name", "")).strip()
    if not dataset_name:
        raise CTProfileError(f"CT profile file {profile_path} has no dataset_name")
    if expected_dataset is not None and dataset_name != expected_dataset:
        raise CTProfileError(
            f"CT profile dataset mismatch: expected {expected_dataset!r}, found {dataset_name!r}"
        )
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise CTProfileError(f"CT profile file {profile_path} has no completed profiles")
    return {str(name): _parse_profile(str(name), raw) for name, raw in raw_profiles.items()}


def select_ct_profile(profiles: Mapping[str, CTProfile], subdataset_name: str | None) -> CTProfile:
    """Select an exact subdataset profile; unnamed groups may explicitly use ``default``."""

    requested = str(subdataset_name).strip() if subdataset_name is not None else ""
    key = requested or "default"
    if key not in profiles:
        available = ", ".join(sorted(profiles)) or "none"
        raise CTProfileError(
            f"No exact CT profile for subdataset {key!r}; available profiles: {available}"
        )
    return profiles[key]
