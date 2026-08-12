#!/usr/bin/env python3
"""Validate a rebuilt RWKV-MedSAM2 container before launching training."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable


DEFAULT_REPO_ROOT = Path("/RWKV-MedSAM2")
DEFAULT_DATA_ROOT = Path("/data")
DEFAULT_PREPROCESSED_ROOT = Path("/data/Preprocessed_v2.1")
DEFAULT_CHECKPOINT_ROOT = Path("/data/TrainingCheckpoints")
DEFAULT_LOADERS_PATH = Path("/data/loaders32.pkl")
DEFAULT_EXPECTED_DATASETS = 56
DEFAULT_EXPECTED_GPUS = 4
LOCK_TIMEOUT_SECONDS = 8


class PreflightError(RuntimeError):
    """A required readiness check failed."""


@dataclass(frozen=True)
class Outcome:
    detail: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    status: str
    label: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def check(self, label: str, operation: Callable[[], str | Outcome]) -> None:
        try:
            value = operation()
            outcome = value if isinstance(value, Outcome) else Outcome(str(value))
        except Exception as exc:  # Keep running so one command reports every problem.
            self._add("FAIL", label, str(exc) or type(exc).__name__)
            return

        self._add("PASS", label, outcome.detail)
        for warning in outcome.warnings:
            self._add("WARN", label, warning)

    def skip(self, label: str, detail: str) -> None:
        self._add("SKIP", label, detail)

    def _add(self, status: str, label: str, detail: str) -> None:
        result = Result(status, label, detail)
        self.results.append(result)
        print(f"[{status}] {label}: {detail}", flush=True)

    def finish(self) -> int:
        failures = sum(result.status == "FAIL" for result in self.results)
        warnings = sum(result.status == "WARN" for result in self.results)
        passed = sum(result.status == "PASS" for result in self.results)
        skipped = sum(result.status == "SKIP" for result in self.results)
        print(
            f"\nPreflight summary: {passed} passed, {warnings} warnings, "
            f"{failures} failed, {skipped} skipped.",
            flush=True,
        )
        if failures:
            print("NOT READY: fix every failed check before launching training.", flush=True)
            return 1
        print("READY: required checks passed. Review warnings before launching training.", flush=True)
        return 0


def positive_int_env(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise PreflightError(f"{name} must be an integer; got {raw!r}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PreflightError(f"{name} must be at least {minimum}; got {value}")
    return value


def format_gib(byte_count: int) -> str:
    return f"{byte_count / (1024 ** 3):.1f} GiB"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PreflightError(f"missing {description}: {path}")
    if path.stat().st_size <= 0:
        raise PreflightError(f"empty {description}: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise PreflightError(f"missing {description}: {path}")


def run_command(command: list[str], *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PreflightError(f"command is not installed: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(f"command timed out after {timeout}s: {' '.join(command)}") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        raise PreflightError(
            f"command failed ({exc.returncode}): {' '.join(command)}"
            + (f" — {message}" if message else "")
        ) from exc


def mount_details(path: Path, *, require_nfs: bool) -> str:
    if not os.path.ismount(path):
        raise PreflightError(f"{path} is not a mount point")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise PreflightError(f"{path} is not readable and writable")

    result = run_command(["findmnt", "-n", "-T", str(path), "-o", "FSTYPE,OPTIONS"])
    fields = result.stdout.strip().split(None, 1)
    if len(fields) != 2:
        raise PreflightError(f"could not parse mount details for {path}: {result.stdout.strip()!r}")
    filesystem, raw_options = fields
    options = set(raw_options.split(","))

    if require_nfs and not filesystem.lower().startswith("nfs"):
        raise PreflightError(f"expected an NFS mount, found {filesystem}")
    if "rw" not in options:
        raise PreflightError(f"mount is not read-write: {raw_options}")
    if "soft" in options:
        raise PreflightError(f"unsafe soft NFS mount detected: {raw_options}")
    if "nolock" in options:
        raise PreflightError(f"NFS locking is disabled with nolock: {raw_options}")
    if require_nfs and "hard" not in options:
        raise PreflightError(f"NFS mount does not report the required hard option: {raw_options}")

    return f"{filesystem} mounted read-write with safe options ({raw_options})"


def local_runtime_details(paths: Iterable[Path]) -> str:
    checked = []
    for path in paths:
        require_directory(path, "local runtime directory")
        result = run_command(["findmnt", "-n", "-T", str(path), "-o", "FSTYPE"])
        filesystem = result.stdout.strip().lower()
        if filesystem.startswith("nfs"):
            raise PreflightError(f"lock-sensitive runtime directory is on NFS: {path}")
        checked.append(f"{path}={filesystem}")
    return ", ".join(checked)


def _lock_probe(kind: str, target: Path) -> int:
    if kind in {"flock", "lockf"}:
        with target.open("a+b") as handle:
            if kind == "flock":
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                try:
                    fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise PreflightError("POSIX lock was unexpectedly held") from exc
                    raise
        return 0

    if kind == "sqlite":
        connection = sqlite3.connect(target, timeout=2.0)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE lock_probe (value INTEGER)")
            connection.execute("INSERT INTO lock_probe VALUES (1)")
            connection.commit()
        finally:
            connection.close()
        return 0

    raise PreflightError(f"unknown lock probe: {kind}")


def lock_details(path: Path) -> str:
    require_directory(path, "lock-test directory")
    probe_dir = Path(tempfile.mkdtemp(prefix=".rwkv-preflight-", dir=path))
    try:
        probes = {
            "flock": probe_dir / "flock.bin",
            "lockf": probe_dir / "lockf.bin",
            "sqlite": probe_dir / "sqlite.db",
        }
        for kind, target in probes.items():
            try:
                run_command(
                    [sys.executable, str(Path(__file__).resolve()), "--_lock-probe", kind, str(target)],
                    timeout=LOCK_TIMEOUT_SECONDS,
                )
            except PreflightError as exc:
                raise PreflightError(
                    f"{kind} failed on {path}; the NFS lock manager, rpc.statd/rpcbind, "
                    f"or its firewall path is still broken ({exc})"
                ) from exc
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    return "temporary writes, flock, POSIX lockf, and SQLite locking passed"


def shared_memory_details(minimum_gib: int) -> str:
    usage = shutil.disk_usage("/dev/shm")
    required = minimum_gib * 1024 ** 3
    if usage.total < required:
        raise PreflightError(
            f"/dev/shm is {format_gib(usage.total)}; at least {minimum_gib} GiB is required"
        )
    return f"{format_gib(usage.total)} total, {format_gib(usage.free)} free"


def data_capacity_details(data_root: Path, minimum_free_gib: int) -> str:
    usage = shutil.disk_usage(data_root)
    required = minimum_free_gib * 1024 ** 3
    if usage.free < required:
        raise PreflightError(
            f"only {format_gib(usage.free)} free; require at least {minimum_free_gib} GiB"
        )
    return f"{format_gib(usage.free)} free of {format_gib(usage.total)}"


def gpu_inventory_details(expected: int, max_utilization: int, max_memory_mib: int) -> str:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) < expected:
        raise PreflightError(f"expected at least {expected} GPUs, but nvidia-smi reported {len(rows)}")
    rows = rows[:expected]

    require_a100 = os.environ.get("RWKV_REQUIRE_A100", "1") != "0"
    descriptions = []
    busy = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 6:
            raise PreflightError(f"could not parse nvidia-smi row: {row!r}")
        index, name, uuid, total_raw, used_raw, utilization_raw = fields
        if require_a100 and "A100" not in name.upper():
            raise PreflightError(f"GPU {index} is {name}, not an A100")
        try:
            total_mib = int(total_raw)
            used_mib = int(used_raw)
            utilization = int(utilization_raw)
        except ValueError as exc:
            raise PreflightError(f"non-numeric nvidia-smi values in row: {row!r}") from exc
        descriptions.append(
            f"GPU {index} {name} ({uuid[-12:]}, {used_mib}/{total_mib} MiB, {utilization}% util)"
        )
        if used_mib > max_memory_mib or utilization > max_utilization:
            busy.append(f"GPU {index}: {used_mib} MiB used, {utilization}% utilization")

    if busy:
        raise PreflightError(
            "GPU resources are already busy: " + "; ".join(busy)
            + f" (limits: {max_memory_mib} MiB and {max_utilization}%)"
        )
    return "; ".join(descriptions)


def torch_details(expected: int) -> str:
    torch = importlib.import_module("torch")
    if not str(torch.__version__).startswith("2.7.1"):
        raise PreflightError(f"expected PyTorch 2.7.1, found {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise PreflightError(f"expected PyTorch CUDA 12.8, found {torch.version.cuda}")
    if torch.backends.cudnn.version() != 90701:
        raise PreflightError(
            f"expected PyTorch cuDNN 9.7.1 (90701), found {torch.backends.cudnn.version()}"
        )
    if not torch.cuda.is_available():
        raise PreflightError("torch.cuda.is_available() is false")
    if torch.cuda.device_count() < expected:
        raise PreflightError(
            f"expected at least {expected} CUDA devices, found {torch.cuda.device_count()}"
        )

    devices = []
    for index in range(expected):
        properties = torch.cuda.get_device_properties(index)
        capability = (properties.major, properties.minor)
        if capability != (8, 0):
            raise PreflightError(
                f"GPU {index} compute capability is {capability}, but this image targets 8.0"
            )
        with torch.cuda.device(index):
            value = (torch.ones(4, device=f"cuda:{index}") * 2).sum().item()
            if value != 8.0:
                raise PreflightError(f"CUDA arithmetic smoke test failed on GPU {index}")
            torch.cuda.synchronize(index)
        devices.append(f"cuda:{index}={properties.name} sm_{properties.major}{properties.minor}")
    torch.cuda.empty_cache()
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}; " + ", ".join(devices)


def native_extension_details(expected_gpus: int) -> str:
    cv2 = importlib.import_module("cv2")
    if cv2.__version__ != "4.10.0":
        raise PreflightError(f"expected OpenCV 4.10.0, found {cv2.__version__}")
    build_info = cv2.getBuildInformation()
    build_lines = [line.strip() for line in build_info.splitlines()]
    cuda_lines = [line for line in build_lines if line.startswith("NVIDIA CUDA:")]
    if not cuda_lines or "YES (ver 12.8" not in cuda_lines[0]:
        raise PreflightError(f"OpenCV does not report CUDA 12.8: {cuda_lines}")
    for feature in ("CUFFT", "CUBLAS", "FAST_MATH"):
        if feature not in cuda_lines[0]:
            raise PreflightError(f"OpenCV CUDA build is missing {feature}: {cuda_lines[0]}")
    cudnn_lines = [line for line in build_lines if line.startswith("cuDNN:")]
    if not cudnn_lines or "YES (ver 9.7.1" not in cudnn_lines[0]:
        raise PreflightError(f"OpenCV does not report cuDNN 9.7.1: {cudnn_lines}")
    ffmpeg_lines = [line for line in build_lines if line.startswith("FFMPEG:")]
    if not ffmpeg_lines or ffmpeg_lines[0].split(":", 1)[1].strip() != "YES":
        raise PreflightError(f"OpenCV does not report FFmpeg support: {ffmpeg_lines}")
    cxx_lines = [line for line in build_lines if line.startswith("C++ standard:")]
    if not cxx_lines or not cxx_lines[0].split(":", 1)[1].strip().startswith("17"):
        raise PreflightError(f"OpenCV was not compiled in CUDA 12.8 C++17 mode: {cxx_lines}")
    cv_gpu_count = int(cv2.cuda.getCudaEnabledDeviceCount())
    if cv_gpu_count < expected_gpus:
        raise PreflightError(
            f"OpenCV sees {cv_gpu_count} CUDA devices; expected at least {expected_gpus}"
        )

    cupy = importlib.import_module("cupy")
    cupy_count = int(cupy.cuda.runtime.getDeviceCount())
    if cupy_count < expected_gpus:
        raise PreflightError(f"CuPy sees {cupy_count} CUDA devices; expected at least {expected_gpus}")

    importlib.import_module("sam2._C")
    importlib.import_module("rwkv_medsam2")
    extension_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions"))
    require_directory(extension_root, "local torch extension cache")
    return (
        f"OpenCV {cv2.__version__} CUDA={cv_gpu_count} cuDNN=9.7.1 FFmpeg=YES C++17, CuPy CUDA={cupy_count}, "
        f"SAM2 and RWKV extensions imported; cache={extension_root}"
    )


def python_package_details() -> str:
    modules = (
        "hydra",
        "huggingface_hub",
        "nibabel",
        "numpy",
        "openpyxl",
        "pandas",
        "pycocotools",
        "pydicom",
        "sklearn",
        "SimpleITK",
        "torchvision",
        "tqdm",
        "yaml",
    )
    for name in modules:
        importlib.import_module(name)
    return f"Python {sys.version.split()[0]}; imported {len(modules)} required packages"


def artifact_details(
    repo_root: Path,
    data_root: Path,
    preprocessed_root: Path,
    checkpoint_root: Path,
    loaders_path: Path,
) -> str:
    required_directories = {
        repo_root: "repository",
        preprocessed_root: "v2.1 preprocessing output",
        checkpoint_root: "checkpoint root",
        data_root / "DatasetIndexes": "dataset indexes",
        data_root / "jrbonill" / "oxford_medsam2_env" / "Medical-SAM2": "Oxford teacher source",
    }
    for required, description in required_directories.items():
        require_directory(required, description)

    required_files = {
        loaders_path: "saved DataLoaders",
        checkpoint_root / "pretrained" / "sam2.1_hiera_tiny.pt": "SAM2.1 tiny initialization checkpoint",
        checkpoint_root / "pretrained" / "sam2_hiera_tiny.pt": "Oxford SAM2 tiny initialization checkpoint",
        checkpoint_root / "sam2.1" / "MedSAM2_pretrain.pth": "Oxford MedSAM2 initialization checkpoint",
        data_root / "DatasetIndexes" / "Tasks" / "datasets_tasks.json": "dataset task map",
        data_root / "jrbonill" / "oxford_medsam2_env" / "Medical-SAM2" / "sam2_train" / "build_sam.py": "Oxford teacher model source",
        repo_root / "ext" / "sam2" / "configs" / "sam2.1" / "sam2.1_vcr.yaml": "distillation config",
        repo_root / "ext" / "sam2" / "configs" / "sam2.1" / "sam2.1_vcr_nodistill.yaml": "no-distillation config",
        repo_root / "notebooks" / "Oxford-Refine_v9_launch.ipynb": "teacher launch notebook",
        repo_root / "notebooks" / "training_launch.ipynb": "distillation launch notebook",
        repo_root / "notebooks" / "training_nodistill_launch.ipynb": "no-distillation launch notebook",
        repo_root / "notebooks" / "training_sam2.1_base_launch.ipynb": "SAM2.1 base launch notebook",
    }
    for required, description in required_files.items():
        require_file(required, description)

    if not os.access(checkpoint_root, os.R_OK | os.W_OK | os.X_OK):
        raise PreflightError(f"checkpoint root is not writable: {checkpoint_root}")
    return f"validated {len(required_directories)} directories and {len(required_files)} required files"


def config_details(repo_root: Path, preprocessed_root: Path, checkpoint_root: Path) -> Outcome:
    yaml = importlib.import_module("yaml")
    config_dir = repo_root / "ext" / "sam2" / "configs" / "sam2.1"
    config_paths = {
        "distillation": config_dir / "sam2.1_vcr.yaml",
        "no-distillation": config_dir / "sam2.1_vcr_nodistill.yaml",
    }
    warnings = []
    loaded = {}
    for name, path in config_paths.items():
        config = yaml.safe_load(path.read_text())
        loaded[name] = config
        configured_root = Path(config["dripp"]["output_dir"])
        if configured_root.resolve() != preprocessed_root.resolve():
            raise PreflightError(f"{name} uses {configured_root}, expected {preprocessed_root}")
        initialization = Path(config["init"]["student_ckpt_path"])
        require_file(initialization, f"{name} initialization checkpoint")
        task_map = Path(config["dripp"]["tasks_file"])
        require_file(task_map, f"{name} task map")
        output = Path(config["ckpt_path"])
        try:
            output.resolve().relative_to(checkpoint_root.resolve())
        except ValueError as exc:
            raise PreflightError(
                f"{name} checkpoint output is outside {checkpoint_root}: {output}"
            ) from exc
        if output.exists() and not output.is_dir():
            raise PreflightError(f"{name} checkpoint output exists but is not a directory: {output}")
        if output.is_dir() and any(output.iterdir()):
            warnings.append(
                f"{name} checkpoint directory is non-empty and a new run can replace latest/best files: {output}"
            )

    distill_training = loaded["distillation"]["training"]
    nodistill_training = loaded["no-distillation"]["training"]
    if float(distill_training.get("lambda_distill_2d", 0)) <= 0 and float(
        distill_training.get("lambda_distill_3d", 0)
    ) <= 0:
        raise PreflightError("distillation config has both distillation weights disabled")
    if float(nodistill_training.get("lambda_distill_2d", 0)) != 0 or float(
        nodistill_training.get("lambda_distill_3d", 0)
    ) != 0:
        raise PreflightError("no-distillation config has a nonzero distillation weight")

    teacher_checkpoint = Path(loaded["distillation"]["teacher"]["ckpt_path"])
    require_file(teacher_checkpoint, "refined teacher checkpoint")

    notebook_expectations = {
        "training_launch.ipynb": str(config_paths["distillation"].relative_to(repo_root)),
        "training_nodistill_launch.ipynb": str(config_paths["no-distillation"].relative_to(repo_root)),
        "training_sam2.1_base_launch.ipynb": str(checkpoint_root / "sam2.1" / "sam2.1_hiera_t512_base_v2_1.pth"),
        "Oxford-Refine_v9_launch.ipynb": str(preprocessed_root),
    }
    for notebook_name, expected_text in notebook_expectations.items():
        notebook_path = repo_root / "notebooks" / notebook_name
        json.loads(notebook_path.read_text())
        if expected_text not in notebook_path.read_text():
            raise PreflightError(f"{notebook_name} does not contain expected path {expected_text}")

    teacher_notebook = (repo_root / "notebooks" / "Oxford-Refine_v9_launch.ipynb").read_text()
    teacher_paths = (
        str(checkpoint_root / "pretrained" / "sam2_hiera_tiny.pt"),
        str(checkpoint_root / "sam2.1" / "MedSAM2_pretrain.pth"),
        str(checkpoint_root / "teacher"),
    )
    for expected_path in teacher_paths:
        if expected_path not in teacher_notebook:
            raise PreflightError(
                f"Oxford-Refine_v9_launch.ipynb does not contain persistent path {expected_path}"
            )
    forbidden_teacher_paths = (
        'SITE = f"{BASE}/site"',
        'f"{REPO}/checkpoints"',
        "/data/jrbonill/oxford_refine_runs",
    )
    for forbidden_path in forbidden_teacher_paths:
        if forbidden_path in teacher_notebook:
            raise PreflightError(
                f"Oxford-Refine_v9_launch.ipynb still uses obsolete path {forbidden_path}"
            )

    return Outcome(
        f"active configs and four clean launch notebooks use persistent v2.1 paths; teacher={teacher_checkpoint}",
        tuple(warnings),
    )


def manifest_details(preprocessed_root: Path, loaders_path: Path, expected_datasets: int) -> str:
    nested = sorted(preprocessed_root.glob("*/*_groups.json"))
    reports = sorted(preprocessed_root.glob("*/preprocessing_report.json"))
    legacy = sorted(preprocessed_root.glob("*/groupings.json"))
    counts = (len(nested), len(reports), len(legacy))
    if counts != (expected_datasets, expected_datasets, expected_datasets):
        raise PreflightError(
            "expected "
            f"{expected_datasets} nested manifests, reports, and legacy manifests; found {counts}"
        )
    for manifest in (*nested, *reports, *legacy):
        if manifest.stat().st_size <= 0:
            raise PreflightError(f"empty preprocessing artifact: {manifest}")

    newest_manifest = max(path.stat().st_mtime_ns for path in nested)
    if loaders_path.stat().st_mtime_ns < newest_manifest:
        raise PreflightError(
            f"{loaders_path} is older than a DRIPP manifest and must be rebuilt"
        )
    return (
        f"{expected_datasets} datasets have nested manifests, reports, and compatibility manifests; "
        "saved loaders are newer than every nested manifest"
    )


def loader_details(loaders_path: Path, preprocessed_root: Path) -> str:
    from rwkv_medsam2.train_sam2 import load_data_loaders
    from rwkv_medsam2.utils.teacher_refinement import source_paths

    train_loader, val_loader, test_loader = load_data_loaders(str(loaders_path))
    expected_prefix = os.path.normcase(os.path.abspath(preprocessed_root)) + os.sep
    pieces = []
    for name, loader in (
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ):
        sequences = loader.dataset.sequences
        sources = sorted(source_paths(sequences))
        wrong = [source for source in sources if not source.startswith(expected_prefix)]
        if wrong:
            raise PreflightError(
                f"{name} loader contains {len(wrong)} sources outside {preprocessed_root}: {wrong[:3]}"
            )
        missing_samples = [source for source in sources[:5] if not Path(source).is_file()]
        if missing_samples:
            raise PreflightError(f"{name} loader sample paths are missing: {missing_samples}")
        pieces.append(
            f"{name}={len(sequences)} sequences/{len(loader)} batches/workers={loader.num_workers}"
        )
    return "; ".join(pieces)


def checksum_manifest_details(checkpoint_root: Path, verify_hashes: bool) -> str:
    manifest = checkpoint_root / "SHA256SUMS"
    require_file(manifest, "checkpoint checksum manifest")
    entries: list[tuple[str, Path]] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise PreflightError(f"malformed checksum line {line_number}")
        digest, relative = fields
        try:
            int(digest, 16)
        except ValueError as exc:
            raise PreflightError(f"invalid SHA-256 on line {line_number}") from exc
        target = checkpoint_root / relative.removeprefix("./")
        require_file(target, f"checkpoint listed on line {line_number}")
        entries.append((digest.lower(), target))

    if verify_hashes:
        for expected, target in entries:
            hasher = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != expected:
                raise PreflightError(f"checksum mismatch: {target}")
        return f"verified SHA-256 for all {len(entries)} checkpoint files"
    return f"all {len(entries)} manifest entries exist (content hashing skipped; use --verify-checksums)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the rebuilt four-GPU container is ready for RWKV-MedSAM2 training."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--loaders", type=Path, default=DEFAULT_LOADERS_PATH)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip unpickling and inspecting the 256 MB saved DataLoaders.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Hash every backed-up checkpoint. This reads roughly 26 GiB and can take several minutes.",
    )
    parser.add_argument("--_lock-probe", nargs=2, metavar=("KIND", "PATH"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args._lock_probe:
        return _lock_probe(args._lock_probe[0], Path(args._lock_probe[1]))

    reporter = Reporter()
    try:
        expected_gpus = positive_int_env("RWKV_EXPECTED_GPU_COUNT", DEFAULT_EXPECTED_GPUS)
        expected_datasets = positive_int_env(
            "RWKV_EXPECTED_DATASET_COUNT", DEFAULT_EXPECTED_DATASETS
        )
        minimum_shm = positive_int_env("RWKV_MIN_SHM_GB", 0, allow_zero=True)
        minimum_data_free = positive_int_env("RWKV_MIN_DATA_FREE_GB", 50, allow_zero=True)
        max_gpu_utilization = positive_int_env(
            "RWKV_MAX_GPU_UTILIZATION", 10, allow_zero=True
        )
        max_gpu_memory = positive_int_env(
            "RWKV_MAX_GPU_MEMORY_USED_MB", 2048, allow_zero=True
        )
    except PreflightError as exc:
        reporter._add("FAIL", "preflight settings", str(exc))
        return reporter.finish()

    print("RWKV-MedSAM2 training preflight\n", flush=True)
    skip_nfs_checks = os.environ.get("RWKV_SKIP_NFS_CHECKS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if skip_nfs_checks:
        for label in (
            "/data NFS mount",
            "/root NFS mount",
            "/data NFS locking",
            "/root NFS locking",
        ):
            reporter.skip(label, "RWKV_SKIP_NFS_CHECKS was requested")
    else:
        reporter.check("/data NFS mount", lambda: mount_details(args.data_root, require_nfs=True))
        reporter.check("/root NFS mount", lambda: mount_details(Path("/root"), require_nfs=True))
        reporter.check("/data NFS locking", lambda: lock_details(args.data_root))
        reporter.check("/root NFS locking", lambda: lock_details(Path("/root")))
    reporter.check(
        "local runtime storage",
        lambda: local_runtime_details(
            (
                Path(os.environ.get("JUPYTER_RUNTIME_DIR", "/var/tmp/jupyter_runtime")),
                Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")),
            )
        ),
    )
    reporter.check("shared memory", lambda: shared_memory_details(minimum_shm))
    reporter.check(
        "persistent free space", lambda: data_capacity_details(args.data_root, minimum_data_free)
    )
    reporter.check(
        "idle A100 inventory",
        lambda: gpu_inventory_details(
            expected_gpus, max_gpu_utilization, max_gpu_memory
        ),
    )
    reporter.check("PyTorch CUDA smoke test", lambda: torch_details(expected_gpus))
    reporter.check("native CUDA extensions", lambda: native_extension_details(expected_gpus))
    reporter.check("Python packages", python_package_details)
    reporter.check(
        "required training artifacts",
        lambda: artifact_details(
            args.repo_root,
            args.data_root,
            args.preprocessed_root,
            args.checkpoint_root,
            args.loaders,
        ),
    )
    reporter.check(
        "active configs and notebooks",
        lambda: config_details(args.repo_root, args.preprocessed_root, args.checkpoint_root),
    )
    reporter.check(
        "DRIPP manifests and loader freshness",
        lambda: manifest_details(args.preprocessed_root, args.loaders, expected_datasets),
    )
    if args.quick:
        reporter.skip("saved DataLoader integrity", "--quick was requested")
    else:
        reporter.check(
            "saved DataLoader integrity", lambda: loader_details(args.loaders, args.preprocessed_root)
        )
    reporter.check(
        "checkpoint backup manifest",
        lambda: checksum_manifest_details(args.checkpoint_root, args.verify_checksums),
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
