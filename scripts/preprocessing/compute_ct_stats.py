#!/usr/bin/env python3
"""Build crash-resistant, subdataset-specific CT window/z-score profiles.

The inexpensive sampling phase chooses one fixed window.  A single full pass
then calculates population mean and standard deviation after that window using
mergeable Welford accumulators.  Existing legacy statistics are never read or
overwritten; the default output is ``CTStats_v2``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


LOGGER = logging.getLogger("compute_ct_stats")
SCHEMA_VERSION = 2
DEFAULT_PERCENTILES = (0.5, 99.5)
DEFAULT_DISPLAY_DATASETS = ("Mosmeddata",)


@dataclass
class WelfordAccumulator:
    """Mergeable population mean/variance accumulator."""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return
        partial = WelfordAccumulator(
            count=int(array.size),
            mean=float(array.mean(dtype=np.float64)),
            M2=float(array.var(dtype=np.float64) * array.size),
        )
        self.merge(partial)

    def merge(self, other: "WelfordAccumulator") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.M2 = other.count, other.mean, other.M2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.M2 += other.M2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    @property
    def std(self) -> float:
        return math.sqrt(self.M2 / self.count) if self.count else 0.0

    @classmethod
    def from_dict(cls, value: dict) -> "WelfordAccumulator":
        return cls(int(value.get("count", 0)), float(value.get("mean", 0.0)), float(value.get("M2", 0.0)))


@dataclass(frozen=True)
class ProfileJob:
    dataset_name: str
    subdataset_name: str
    paths: tuple[str, ...]
    intensity_space: str
    total_source_image_count: int = 0


def atomic_write_json(path: Path, document: dict) -> None:
    """Durably replace a JSON file without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_image(path: str) -> np.ndarray:
    """Load one source image, memory-mapping NumPy arrays where possible."""

    if path.lower().endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r"))
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)


def configure_worker() -> None:
    """Prevent every process from creating its own SimpleITK thread pool."""

    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except AttributeError:
        pass


def valid_mask(image: np.ndarray, intensity_space: str, valid_hu_min: float) -> np.ndarray:
    mask = np.isfinite(image)
    if intensity_space == "hu":
        mask &= image >= valid_hu_min
    return mask


def accumulate_file(
    path: str,
    intensity_space: str,
    window_lower: float,
    window_upper: float,
    valid_hu_min: float,
) -> WelfordAccumulator:
    """Accumulate one image; a valid background-only image returns an empty result."""

    image = load_image(path)
    valid = valid_mask(image, intensity_space, valid_hu_min)
    foreground = valid & (image > window_lower)
    accumulator = WelfordAccumulator()
    if not np.any(foreground):
        return accumulator
    values = np.clip(image[foreground], window_lower, window_upper)
    accumulator.update(values)
    return accumulator


def process_batch(
    batch_id: int,
    paths: Sequence[str],
    intensity_space: str,
    window_lower: float,
    window_upper: float,
    valid_hu_min: float,
) -> dict:
    """Worker entry point. Only compact statistics and errors cross processes."""

    accumulator = WelfordAccumulator()
    failures = []
    successes = 0
    empty_foreground = 0
    for path in paths:
        try:
            file_accumulator = accumulate_file(
                path, intensity_space, window_lower, window_upper, valid_hu_min
            )
            if file_accumulator.count == 0:
                empty_foreground += 1
            accumulator.merge(file_accumulator)
            successes += 1
        except Exception as exc:  # isolate malformed/unreadable inputs
            failures.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "batch_id": batch_id,
        "file_count": len(paths),
        "success_file_count": successes,
        "empty_foreground_file_count": empty_foreground,
        "accumulator": asdict(accumulator),
        "failures": failures,
    }


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_modality_list(value: str) -> set[str]:
    return {item.strip().lower() for item in (value or "").split(",") if item.strip()}


def read_ct_dataset_names(csv_path: Path) -> tuple[list[str], dict[str, str]]:
    names = []
    roots = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Dataset Name") or "").strip()
            if name and "ct" in normalized_modality_list(row.get("Modality", "")):
                names.append(name)
                roots[name] = (row.get("Root Folder") or "").strip()
    return sorted(set(names), key=str.casefold), roots


def resolve_source_path(path: str, data_root: Path, dataset_name: str, root_folder: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or re.match(r"^[A-Za-z]:[/\\]", path):
        return os.path.normpath(path)
    base = data_root / dataset_name
    if root_folder:
        base /= root_folder
    return os.path.normpath(str(base / candidate))


def collect_jobs(args) -> list[ProfileJob]:
    dataset_names, roots = read_ct_dataset_names(args.metadata_csv)
    requested_datasets = {value.casefold() for value in args.dataset}
    requested_subdatasets = {value.casefold() for value in args.subdataset}
    display_datasets = {value.casefold() for value in args.display_scaled_dataset}
    jobs = []

    for dataset_name in dataset_names:
        if requested_datasets and dataset_name.casefold() not in requested_datasets:
            continue
        groups_path = args.groups_dir / f"{dataset_name}_groups.json"
        if not groups_path.is_file():
            LOGGER.warning("CT dataset %s has no groups file at %s", dataset_name, groups_path)
            continue
        groups = read_json(groups_path, {}).get("subdatasets", [])
        for subdataset in groups:
            if str(subdataset.get("modality", "")).strip().lower() != "ct":
                continue
            subdataset_name = str(subdataset.get("name") or "default").strip()
            if requested_subdatasets and subdataset_name.casefold() not in requested_subdatasets:
                continue
            paths = set()
            for split in ("train", "test"):
                for group in subdataset.get(split, []) or []:
                    for record in group.get("images", []) or []:
                        source = record.get("path")
                        if source:
                            paths.add(resolve_source_path(source, args.data_root, dataset_name, roots.get(dataset_name, "")))
            if not paths:
                LOGGER.warning("No source images for %s/%s", dataset_name, subdataset_name)
                continue
            intensity_space = "display" if dataset_name.casefold() in display_datasets else "hu"
            ordered_paths = tuple(sorted(paths, key=str.casefold))
            selected_paths = ordered_paths
            if args.smoke_test:
                selected_paths = tuple(evenly_spaced_sample(ordered_paths, args.smoke_files))
            jobs.append(ProfileJob(
                dataset_name, subdataset_name, selected_paths, intensity_space,
                total_source_image_count=len(ordered_paths),
            ))

    if requested_datasets:
        found = {job.dataset_name.casefold() for job in jobs}
        missing = sorted(requested_datasets - found)
        if missing:
            raise ValueError(f"Requested CT datasets were not found in CSV/groups: {', '.join(missing)}")
    return jobs


def evenly_spaced_sample(paths: Sequence[str], maximum: int) -> list[str]:
    if maximum <= 0 or len(paths) <= maximum:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=maximum, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def estimate_window(job: ProfileJob, args) -> tuple[float, float, dict]:
    if job.intensity_space == "display":
        return 0.0, 255.0, {
            "sample_file_count": 0,
            "sample_voxel_count": 0,
            "sample_failures": [],
            "fixed_display_window": True,
        }

    sample_paths = evenly_spaced_sample(job.paths, args.sample_files)
    rng = np.random.default_rng(args.seed)
    per_file_limit = max(1, args.sample_voxels // max(1, len(sample_paths)))
    samples = []
    failures = []
    bar = tqdm(
        sample_paths, desc=f"  sampling {job.dataset_name}/{job.subdataset_name}",
        unit="img", leave=False, position=1,
    )
    for path in bar:
        try:
            image = load_image(path)
            values = np.asarray(image[valid_mask(image, "hu", args.valid_hu_min)]).reshape(-1)
            if values.size == 0:
                raise ValueError("no valid HU voxels")
            if values.size > per_file_limit:
                indices = rng.choice(values.size, size=per_file_limit, replace=False)
                values = values[indices]
            samples.append(values.astype(np.float64, copy=False))
        except Exception as exc:
            failures.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
    if not samples:
        raise RuntimeError(f"Window sampling failed for every image in {job.dataset_name}/{job.subdataset_name}")
    combined = np.concatenate(samples)
    lower, upper = np.percentile(combined, args.percentiles)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise RuntimeError(f"Sampled an invalid window [{lower}, {upper}]")
    return float(lower), float(upper), {
        "sample_file_count": len(sample_paths),
        "sample_voxel_count": int(combined.size),
        "sample_failures": failures,
        "fixed_display_window": False,
    }


def input_list_hash(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def configuration_signature(job: ProfileJob, args) -> str:
    config = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "intensity_space": job.intensity_space,
        "percentiles": list(args.percentiles),
        "sample_files": args.sample_files,
        "sample_voxels": args.sample_voxels,
        "seed": args.seed,
        "valid_hu_min": args.valid_hu_min,
        "batch_size": args.batch_size,
        "variance_ddof": 0,
        "smoke_test": args.smoke_test,
        "smoke_files": args.smoke_files if args.smoke_test else None,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def checkpoint_path(job: ProfileJob, output_dir: Path) -> Path:
    return output_dir / ".checkpoints" / f"{safe_name(job.dataset_name)}__{safe_name(job.subdataset_name)}.json"


def profile_output_path(job: ProfileJob, output_dir: Path) -> Path:
    return output_dir / f"{job.dataset_name}_ct_stats.json"


def initial_checkpoint(job: ProfileJob, args, lower: float, upper: float, sample_metadata: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": job.dataset_name,
        "subdataset_name": job.subdataset_name,
        "configuration_signature": configuration_signature(job, args),
        "input_list_hash": input_list_hash(job.paths),
        "window_lower": lower,
        "window_upper": upper,
        "intensity_space": job.intensity_space,
        "sample_metadata": sample_metadata,
        "completed_batch_ids": [],
        "accumulator": asdict(WelfordAccumulator()),
        "success_file_count": 0,
        "empty_foreground_file_count": 0,
        "failed_file_count": 0,
        "failures": [],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_checkpoint(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(path, state)
    LOGGER.debug(
        "Checkpointed %s (%d completed batches, %d successes, %d empty, %d failures)",
        path, len(state.get("completed_batch_ids", [])),
        state.get("success_file_count", 0), state.get("empty_foreground_file_count", 0),
        state.get("failed_file_count", 0),
    )


def merge_batch_result(state: dict, result: dict) -> None:
    accumulator = WelfordAccumulator.from_dict(state["accumulator"])
    accumulator.merge(WelfordAccumulator.from_dict(result["accumulator"]))
    state["accumulator"] = asdict(accumulator)
    state["success_file_count"] += int(result["success_file_count"])
    state["empty_foreground_file_count"] = (
        int(state.get("empty_foreground_file_count", 0))
        + int(result.get("empty_foreground_file_count", 0))
    )
    state["failures"].extend(result["failures"])
    state["failed_file_count"] = len(state["failures"])
    batch_id = int(result["batch_id"])
    if batch_id >= 0 and batch_id not in state["completed_batch_ids"]:
        state["completed_batch_ids"].append(batch_id)
        state["completed_batch_ids"].sort()


def retry_prior_failures(state: dict, job: ProfileJob, args, checkpoint: Path, bar) -> None:
    prior = list(state.get("failures", []))
    if not prior:
        return
    LOGGER.info("Retrying %d previously failed files for %s/%s", len(prior), job.dataset_name, job.subdataset_name)
    state["failures"] = []
    for failure in prior:
        result = process_batch(
            -1, [failure["path"]], job.intensity_space,
            state["window_lower"], state["window_upper"], args.valid_hu_min,
        )
        merge_batch_result(state, result)
    state["failed_file_count"] = len(state["failures"])
    save_checkpoint(checkpoint, state)


def batches(paths: Sequence[str], batch_size: int) -> list[tuple[int, tuple[str, ...]]]:
    return [
        (index // batch_size, tuple(paths[index:index + batch_size]))
        for index in range(0, len(paths), batch_size)
    ]


def compute_full_statistics(job: ProfileJob, args, state: dict, checkpoint: Path) -> dict:
    all_batches = batches(job.paths, args.batch_size)
    completed = set(int(value) for value in state["completed_batch_ids"])
    pending = [(batch_id, paths) for batch_id, paths in all_batches if batch_id not in completed]
    already_accounted = state["success_file_count"] + state["failed_file_count"]
    bar = tqdm(
        total=len(job.paths), initial=min(already_accounted, len(job.paths)),
        desc=f"  {job.dataset_name}/{job.subdataset_name}", unit="img", leave=False,
        position=1,
    )

    def accept(result: dict) -> None:
        merge_batch_result(state, result)
        for failure in result["failures"]:
            LOGGER.warning("Unreadable CT source %s: %s", failure["path"], failure["error"])
        bar.update(result["file_count"])
        bar.set_postfix(
            phase="statistics", ok=state["success_file_count"],
            empty=state.get("empty_foreground_file_count", 0),
            failed=state["failed_file_count"],
            refresh=False,
        )
        save_checkpoint(checkpoint, state)

    retry_prior_failures(state, job, args, checkpoint, bar)
    if not pending:
        bar.close()
        return state

    if args.workers == 1:
        try:
            for batch_id, batch_paths in pending:
                accept(process_batch(batch_id, batch_paths, job.intensity_space, state["window_lower"], state["window_upper"], args.valid_hu_min))
        except KeyboardInterrupt:
            save_checkpoint(checkpoint, state)
            bar.close()
            raise
        bar.close()
        return state

    remaining = list(pending)
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=configure_worker)
    futures = {}
    pool_failed = False
    try:
        while remaining or futures:
            while remaining and len(futures) < args.workers * 2:
                batch_id, batch_paths = remaining.pop(0)
                try:
                    future = pool.submit(
                        process_batch, batch_id, batch_paths, job.intensity_space,
                        state["window_lower"], state["window_upper"], args.valid_hu_min,
                    )
                    futures[future] = (batch_id, batch_paths)
                except (BrokenProcessPool, OSError) as exc:
                    LOGGER.exception("Could not submit worker batch; switching to sequential mode: %s", exc)
                    remaining.insert(0, (batch_id, batch_paths))
                    remaining.extend(futures.values())
                    futures.clear()
                    pool_failed = True
                    break
            if pool_failed:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                batch_id, batch_paths = futures.pop(future)
                try:
                    accept(future.result())
                except (BrokenProcessPool, OSError) as exc:
                    LOGGER.exception("Worker pool failed; switching remaining work to sequential mode: %s", exc)
                    remaining.insert(0, (batch_id, batch_paths))
                    remaining.extend(futures.values())
                    futures.clear()
                    pool_failed = True
                    break
                except Exception as exc:
                    LOGGER.exception(
                        "Worker batch %d raised unexpectedly; retrying that batch sequentially: %s",
                        batch_id, exc,
                    )
                    accept(process_batch(
                        batch_id, batch_paths, job.intensity_space,
                        state["window_lower"], state["window_upper"], args.valid_hu_min,
                    ))
            if pool_failed:
                break
    except KeyboardInterrupt:
        save_checkpoint(checkpoint, state)
        pool.shutdown(wait=False, cancel_futures=True)
        bar.close()
        raise
    finally:
        if not pool_failed:
            pool.shutdown(wait=True, cancel_futures=False)
        else:
            pool.shutdown(wait=False, cancel_futures=True)

    if pool_failed:
        completed_now = set(state["completed_batch_ids"])
        retry_batches = [(i, p) for i, p in remaining if i not in completed_now]
        for batch_id, batch_paths in retry_batches:
            accept(process_batch(batch_id, batch_paths, job.intensity_space, state["window_lower"], state["window_upper"], args.valid_hu_min))
    bar.close()
    return state


def load_or_create_state(job: ProfileJob, args, checkpoint: Path) -> dict:
    expected_config = configuration_signature(job, args)
    expected_inputs = input_list_hash(job.paths)
    existing = read_json(checkpoint)
    if existing is not None:
        matches = (
            existing.get("configuration_signature") == expected_config
            and existing.get("input_list_hash") == expected_inputs
        )
        if not matches:
            if not args.force:
                raise RuntimeError(
                    f"Checkpoint configuration/input mismatch for {job.dataset_name}/{job.subdataset_name}; "
                    "use --force or a new output directory"
                )
            checkpoint.unlink()
        elif args.resume:
            LOGGER.info("Resuming checkpoint %s", checkpoint)
            return existing
        elif not args.force:
            raise RuntimeError(f"Checkpoint exists at {checkpoint}; use --resume or --force")
        else:
            checkpoint.unlink()

    lower, upper, sample_metadata = estimate_window(job, args)
    state = initial_checkpoint(job, args, lower, upper, sample_metadata)
    save_checkpoint(checkpoint, state)
    return state


def existing_profile_status(job: ProfileJob, args) -> str:
    output = profile_output_path(job, args.output_dir)
    document = read_json(output, {}) or {}
    profile = (document.get("profiles") or {}).get(job.subdataset_name)
    if not profile:
        return "missing"
    matches = (
        profile.get("configuration_signature") == configuration_signature(job, args)
        and profile.get("input_list_hash") == input_list_hash(job.paths)
    )
    if matches:
        return "complete"
    return "mismatch"


def promote_profile(job: ProfileJob, args, state: dict) -> dict:
    accumulator = WelfordAccumulator.from_dict(state["accumulator"])
    if accumulator.count <= 0 or not np.isfinite(accumulator.std) or accumulator.std <= 0:
        raise RuntimeError(f"No usable statistics for {job.dataset_name}/{job.subdataset_name}")
    failures = state.get("failures", [])
    failure_rate = len(failures) / len(job.paths)
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Profile not promoted: {len(failures)}/{len(job.paths)} files failed "
            f"({failure_rate:.2%}, allowed {args.max_failure_rate:.2%})"
        )
    lower = float(state["window_lower"])
    upper = float(state["window_upper"])
    profile = {
        "intensity_space": job.intensity_space,
        "window_lower": lower,
        "window_upper": upper,
        "window_width": upper - lower,
        "window_level": (upper + lower) / 2.0,
        "mean": accumulator.mean,
        "std": accumulator.std,
        "count": accumulator.count,
        "variance_ddof": 0,
        "stats_basis": "smoke_sample_after_window" if args.smoke_test else "full_after_window",
        "window_stats_compatible": not args.smoke_test,
        "window_percentiles": list(args.percentiles),
        "source_image_count": len(job.paths),
        "discovered_source_image_count": job.total_source_image_count or len(job.paths),
        "successful_image_count": state["success_file_count"],
        "foreground_image_count": (
            state["success_file_count"] - state.get("empty_foreground_file_count", 0)
        ),
        "empty_foreground_image_count": state.get("empty_foreground_file_count", 0),
        "failed_image_count": len(failures),
        "failure_details": failures,
        "sample_metadata": state["sample_metadata"],
        "configuration_signature": state["configuration_signature"],
        "input_list_hash": state["input_list_hash"],
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    output = profile_output_path(job, args.output_dir)
    document = read_json(output, {}) or {}
    if document and (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("dataset_name") != job.dataset_name
    ):
        raise RuntimeError(f"Refusing to modify incompatible staged profile file: {output}")
    document.setdefault("schema_version", SCHEMA_VERSION)
    document.setdefault("dataset_name", job.dataset_name)
    document.setdefault("profiles", {})[job.subdataset_name] = profile
    atomic_write_json(output, document)
    return profile


def run_job(job: ProfileJob, args) -> str:
    status = existing_profile_status(job, args)
    if status == "complete" and not args.force:
        LOGGER.info("Skipping completed profile %s/%s", job.dataset_name, job.subdataset_name)
        return "skipped"
    if status == "mismatch" and not args.force:
        raise RuntimeError(
            f"Completed profile configuration/input mismatch for {job.dataset_name}/{job.subdataset_name}; "
            "use --force or a new output directory"
        )
    checkpoint = checkpoint_path(job, args.output_dir)
    state = load_or_create_state(job, args, checkpoint)
    started = time.monotonic()
    state = compute_full_statistics(job, args, state, checkpoint)
    profile = promote_profile(job, args, state)
    checkpoint.unlink(missing_ok=True)
    LOGGER.info(
        "Completed %s/%s in %.1fs: window=[%.3f, %.3f], mean=%.5f, std=%.5f, voxels=%d, failures=%d",
        job.dataset_name, job.subdataset_name, time.monotonic() - started,
        profile["window_lower"], profile["window_upper"], profile["mean"], profile["std"],
        profile["count"], profile["failed_image_count"],
    )
    return "completed"


def configure_logging(output_dir: Path, level: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"compute_ct_stats_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter("%(asctime)s %(processName)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(getattr(logging, level.upper()))
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler], force=True)
    return log_path


def parse_args(argv=None):
    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full-profile",), default="full-profile")
    parser.add_argument("--metadata-csv", type=Path, default=Path("/data/research/datasets.csv"))
    parser.add_argument("--groups-dir", type=Path, default=Path("/data/DatasetIndexes/Groups"))
    parser.add_argument("--data-root", type=Path, default=Path("/data/research"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/DatasetIndexes/CTStats_v2"))
    parser.add_argument("--workers", type=int, default=min(4, max(1, cpu_count // 2)))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-files", type=int, default=32)
    parser.add_argument("--sample-voxels", type=int, default=1_000_000)
    parser.add_argument("--percentiles", type=float, nargs=2, default=DEFAULT_PERCENTILES, metavar=("LOW", "HIGH"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-hu-min", type=float, default=-1024.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--dataset", action="append", default=[], help="Only process this dataset (repeatable)")
    parser.add_argument("--subdataset", action="append", default=[], help="Only process this subdataset (repeatable)")
    parser.add_argument("--display-scaled-dataset", action="append", default=list(DEFAULT_DISPLAY_DATASETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run the full pipeline on a small deterministic sample and write only to _smoke_test",
    )
    parser.add_argument(
        "--smoke-files", type=int, default=2,
        help="Source images per CT subdataset in --smoke-test mode (default: 2)",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.batch_size < 1 or args.sample_files < 1 or args.sample_voxels < 1 or args.smoke_files < 1:
        parser.error("workers, batch-size, sample-files, sample-voxels, and smoke-files must be positive")
    if not (0 <= args.percentiles[0] < args.percentiles[1] <= 100):
        parser.error("percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if not (0 <= args.max_failure_rate <= 1):
        parser.error("max-failure-rate must be between 0 and 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        args.output_dir = args.output_dir / "_smoke_test"
    log_path = configure_logging(args.output_dir, args.log_level)
    LOGGER.info("Log file: %s", log_path)
    try:
        jobs = collect_jobs(args)
    except Exception:
        LOGGER.exception("Could not build the CT profile job list")
        return 2
    if not jobs:
        LOGGER.error("No matching CT subdatasets were found")
        return 2

    datasets = []
    for job in jobs:
        if job.dataset_name not in datasets:
            datasets.append(job.dataset_name)
    LOGGER.info("Found %d CT datasets and %d CT subdataset profiles", len(datasets), len(jobs))
    if args.smoke_test:
        LOGGER.warning(
            "SMOKE TEST: using at most %d images per subdataset; outputs are non-production and isolated at %s",
            args.smoke_files, args.output_dir,
        )
    for job in jobs:
        LOGGER.info(
            "PLAN dataset=%s subdataset=%s files=%d/%d intensity=%s output=%s",
            job.dataset_name, job.subdataset_name, len(job.paths),
            job.total_source_image_count or len(job.paths), job.intensity_space,
            profile_output_path(job, args.output_dir),
        )
    if args.dry_run:
        return 0

    jobs_by_dataset = {name: [job for job in jobs if job.dataset_name == name] for name in datasets}
    failures = []
    overall = tqdm(datasets, desc="CT datasets", unit="dataset")
    try:
        for dataset_name in overall:
            overall.set_postfix(dataset=dataset_name, refresh=False)
            dataset_failed = False
            for job in jobs_by_dataset[dataset_name]:
                try:
                    run_job(job, args)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    dataset_failed = True
                    failures.append((job, str(exc)))
                    LOGGER.exception("Profile failed for %s/%s", job.dataset_name, job.subdataset_name)
            if dataset_failed:
                overall.set_postfix(dataset=dataset_name, status="failed", refresh=False)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Completed batches are checkpointed; rerun with --resume.")
        return 130
    finally:
        overall.close()

    if failures:
        LOGGER.error("%d profile(s) did not complete:", len(failures))
        for job, error in failures:
            LOGGER.error("  %s/%s: %s", job.dataset_name, job.subdataset_name, error)
        return 1
    LOGGER.info("All requested CT profiles completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
