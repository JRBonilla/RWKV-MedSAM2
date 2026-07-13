#!/usr/bin/env python3
"""Console supervisor for the strict video benchmark.

This script intentionally keeps the benchmark implementation in
notebooks/strict_video_protocol_lib.py. The parent process supervises a worker
subprocess so native CUDA/PyTorch crashes do not erase the progress trail.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = Path("/data/rwkv_medsam2_model_comparisons")
DEFAULT_LOADERS_PKL = Path("/data/loaders32.pkl")
DEFAULT_MODELS = [
    "rwkv_medsam2_distill",
    "rwkv_medsam2_nodistill",
    "sam2_1_base",
    "oxford_medical_sam2",
    "uoft_medsam2",
]
DEFAULT_PROMPT_MODES = ["box", "mask", "mixed"]
DEFAULT_PROTOCOLS = ["validation_forward"]
DEFAULT_REUSE_STATE_MODELS = ["rwkv_medsam2_distill"]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def split_values(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or list(default)


def split_int_values(values: list[str] | None) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    tmp.replace(path)


def tail_text(path: Path, max_lines: int = 120, max_bytes: int = 4_000_000) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
                handle.readline()
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except Exception as exc:
        return f"<failed to read {path}: {type(exc).__name__}: {exc}>"


def copy_if_present(src: Path, dst: Path) -> None:
    try:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    except Exception:
        pass


def find_helper_path(override: str | None = None) -> Path:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            REPO_ROOT / "notebooks" / "strict_video_protocol_lib.py",
            Path.cwd() / "notebooks" / "strict_video_protocol_lib.py",
            Path.cwd() / "strict_video_protocol_lib.py",
            Path("/RWKV-MedSAM2/notebooks/strict_video_protocol_lib.py"),
            Path("/workspace/RWKV-MedSAM2/notebooks/strict_video_protocol_lib.py"),
            Path("/data/RWKV-MedSAM2/notebooks/strict_video_protocol_lib.py"),
            Path("/data/rwkv_medsam2/notebooks/strict_video_protocol_lib.py"),
            Path("/data/jrbonill/RWKV-MedSAM2/notebooks/strict_video_protocol_lib.py"),
            Path("/data/jrbonill/rwkv_medsam2/notebooks/strict_video_protocol_lib.py"),
            Path.home() / "notebooks" / "strict_video_protocol_lib.py",
            Path.home() / "strict_video_protocol_lib.py",
        ]
    )
    for candidate in candidates:
        try:
            candidate = candidate.expanduser().resolve()
        except Exception:
            candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find strict_video_protocol_lib.py")


def load_strict_helper(helper_path: str | None = None):
    helper = find_helper_path(helper_path)
    print(f"Helper path: {helper}", flush=True)
    spec = importlib.util.spec_from_file_location("strict_video_protocol_lib", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper from {helper}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strict_video_protocol_lib"] = module
    spec.loader.exec_module(module)
    module.OrderedDict = OrderedDict
    return module


def format_progress(status: dict[str, Any] | None, heartbeat: dict[str, Any] | None) -> str:
    if status:
        pct = float(status.get("percent_done") or 0.0)
        runs = f"{status.get('runs_done') or 0}/{status.get('runs_total') or 0}"
        model = status.get("model") or ""
        dataset = status.get("current_dataset") or ""
        case = f"{status.get('current_case_number_for_model') or 0}/{status.get('current_case_total_for_model') or 0}"
        elapsed = status.get("elapsed_text") or "unknown"
        eta = status.get("eta_text") or "unknown"
        sec_case = status.get("mean_sec_per_case_visit")
        sec_case_text = f"{float(sec_case):.2f}s/case" if sec_case is not None else "unknown/case"
        remaining = status.get("overall_dataset_visits_remaining_including_current")
        remaining_text = f", datasets remaining {remaining}" if remaining is not None else ""
        return (
            f"{pct:.2f}% runs {runs}, model {model}, case {case}, dataset {dataset}, "
            f"{sec_case_text}, elapsed {elapsed}, eta {eta}{remaining_text}"
        )
    if heartbeat:
        pct = float(heartbeat.get("percent_done") or 0.0)
        runs = f"{heartbeat.get('runs_done') or 0}/{heartbeat.get('runs_total') or 0}"
        phase = heartbeat.get("phase") or "unknown"
        model = heartbeat.get("model") or ""
        dataset = heartbeat.get("current_dataset") or ""
        elapsed = heartbeat.get("elapsed_text") or "unknown"
        return f"{pct:.2f}% runs {runs}, phase {phase}, model {model}, dataset {dataset}, elapsed {elapsed}"
    return "status not written yet"


def latest_resume_dir(previous_resume: Path, attempt_dir: Path) -> Path:
    checkpoint_names = ["per_case_results.jsonl", "per_case_results.csv", "strict_video_protocol_results.csv"]
    for name in checkpoint_names:
        path = attempt_dir / name
        try:
            if path.is_file() and path.stat().st_size > 0:
                return attempt_dir
        except Exception:
            pass
    return previous_resume


def terminate_process(proc: subprocess.Popen[str], grace_sec: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.time() + grace_sec
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.25)
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def write_crash_report(
    *,
    session_dir: Path,
    attempt_dir: Path,
    attempt_number: int,
    reason: str,
    exit_code: int | None,
    previous_resume_dir: Path,
    next_resume_dir: Path,
    attempt_log: Path,
    last_console_lines: deque[str],
) -> Path:
    report_dir = session_dir / "crash_reports" / f"attempt_{attempt_number:03d}_{timestamp()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    status_path = attempt_dir / "benchmark_status.json"
    heartbeat_path = attempt_dir / "benchmark_heartbeat.json"
    heartbeat_jsonl_path = attempt_dir / "benchmark_heartbeat.jsonl"
    fault_log_path = attempt_dir / "python_fault_handler.log"

    status = read_json(status_path)
    heartbeat = read_json(heartbeat_path)
    copy_if_present(status_path, report_dir / "benchmark_status.json")
    copy_if_present(heartbeat_path, report_dir / "benchmark_heartbeat.json")

    (report_dir / "benchmark_heartbeat_tail.jsonl").write_text(
        tail_text(heartbeat_jsonl_path, max_lines=120),
        encoding="utf-8",
    )
    (report_dir / "python_fault_handler_tail.log").write_text(
        tail_text(fault_log_path, max_lines=200),
        encoding="utf-8",
    )
    (report_dir / "console_tail.log").write_text(
        "\n".join(last_console_lines),
        encoding="utf-8",
    )
    copy_if_present(attempt_log, report_dir / "attempt_console.log")

    summary_lines = [
        "# Strict Video Console Crash Report",
        "",
        f"- Written at: `{now_text()}`",
        f"- Reason: `{reason}`",
        f"- Exit code: `{exit_code}`",
        f"- Attempt: `{attempt_number}`",
        f"- Attempt output dir: `{attempt_dir}`",
        f"- Previous resume dir: `{previous_resume_dir}`",
        f"- Next resume dir: `{next_resume_dir}`",
    ]
    if heartbeat:
        summary_lines.extend(
            [
                "",
                "## Latest Heartbeat",
                "",
                f"- Updated at: `{heartbeat.get('updated_at')}`",
                f"- Phase: `{heartbeat.get('phase')}`",
                f"- Model: `{heartbeat.get('model')}`",
                f"- Dataset: `{heartbeat.get('current_dataset')}`",
                f"- Dataset index: `{heartbeat.get('dataset_index')}`",
                f"- Prompt mode: `{heartbeat.get('prompt_mode')}`",
                f"- Runs: `{heartbeat.get('runs_done')}/{heartbeat.get('runs_total')}`",
                f"- Percent: `{float(heartbeat.get('percent_done') or 0.0):.4f}%`",
                f"- Elapsed: `{heartbeat.get('elapsed_text')}`",
                f"- CUDA allocated MB: `{heartbeat.get('cuda_allocated_mb')}`",
                f"- CUDA reserved MB: `{heartbeat.get('cuda_reserved_mb')}`",
            ]
        )
    if status:
        last_row = status.get("last_row") or {}
        summary_lines.extend(
            [
                "",
                "## Latest Status",
                "",
                f"- Progress: `{float(status.get('percent_done') or 0.0):.4f}%`",
                f"- Runs: `{status.get('runs_done')}/{status.get('runs_total')}`",
                f"- Model: `{status.get('model')}`",
                f"- Dataset: `{status.get('current_dataset')}`",
                f"- Elapsed: `{status.get('elapsed_text')}`",
                f"- ETA: `{status.get('eta_text')}`",
                f"- Last row status: `{last_row.get('status')}`",
                f"- Last row model: `{last_row.get('model')}`",
                f"- Last row dataset index: `{last_row.get('dataset_index')}`",
                f"- Last row prompt mode: `{last_row.get('prompt_mode')}`",
            ]
        )
    summary_lines.extend(
        [
            "",
            "## Included Files",
            "",
            "- `benchmark_status.json`",
            "- `benchmark_heartbeat.json`",
            "- `benchmark_heartbeat_tail.jsonl`",
            "- `python_fault_handler_tail.log`",
            "- `console_tail.log`",
            "- `attempt_console.log`",
        ]
    )
    (report_dir / "crash_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report_dir


def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume-dir", help="Directory to resume from.")
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE), help="Base directory for console sessions.")
    parser.add_argument("--loaders-pkl", default=str(DEFAULT_LOADERS_PKL), help="Pickled loaders path.")
    parser.add_argument("--helper-path", default=None, help="Optional strict_video_protocol_lib.py path.")
    parser.add_argument("--models", nargs="+", default=None, help="Models to benchmark; accepts space or comma separated values.")
    parser.add_argument("--prompt-modes", nargs="+", default=None, help="Prompt modes; accepts space or comma separated values.")
    parser.add_argument("--protocols", nargs="+", default=None, help="Protocols; accepts space or comma separated values.")

    parser.add_argument("--smoke-run", action="store_true", help="Run smoke selection instead of the full benchmark.")
    parser.add_argument("--smoke-cases-per-dataset", type=int, default=1)
    parser.add_argument("--smoke-case-selection", default="per_dataset")
    parser.add_argument("--no-smoke-prefer-sequence-cases", action="store_true")
    parser.add_argument("--smoke-datasets", nargs="+", default=None, help="Optional smoke dataset filter.")
    parser.add_argument("--smoke-max-scan-per-dataset", type=int, default=250)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--case-indices", nargs="+", default=None, help="Explicit dataset indices; accepts space or comma separated values.")

    parser.add_argument("--partial-save-every-rows", type=int, default=None)
    parser.add_argument("--status-update-every-rows", type=int, default=25)
    parser.add_argument("--status-update-every-sec", type=float, default=60.0)
    parser.add_argument("--checkpoint-flush-every-rows", type=int, default=1)
    parser.add_argument("--visual-top-n-datasets", type=int, default=None)

    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-order-2d-before-3d", action="store_true")
    parser.add_argument("--no-save-model-separated-outputs", action="store_true")
    parser.add_argument("--no-cache-cases-cpu", action="store_true")
    parser.add_argument("--no-cache-prompt-plans-cpu", action="store_true")
    parser.add_argument("--write-model-outputs-on-partial", action="store_true")
    parser.add_argument("--reuse-prompt-mode-state", action="store_true")
    parser.add_argument("--reuse-prompt-mode-state-models", nargs="+", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Console supervisor for the strict video benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_benchmark_args(parser)
    parser.add_argument("--max-restarts", type=int, default=100, help="Maximum automatic worker restarts after the first attempt.")
    parser.add_argument("--heartbeat-stale-sec", type=float, default=3600.0, help="Kill/restart worker if heartbeat is older than this many seconds.")
    parser.add_argument("--progress-interval-sec", type=float, default=60.0, help="Supervisor progress print interval.")
    parser.add_argument("--no-auto-restart", action="store_true", help="Write crash report and stop instead of auto-resuming.")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-number", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def normalized_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "models": split_values(args.models, DEFAULT_MODELS),
        "prompt_modes": split_values(args.prompt_modes, DEFAULT_PROMPT_MODES),
        "protocols": split_values(args.protocols, DEFAULT_PROTOCOLS),
        "smoke_datasets": split_values(args.smoke_datasets, []) if args.smoke_datasets else None,
        "case_indices": split_int_values(args.case_indices),
        "reuse_prompt_mode_state_models": split_values(args.reuse_prompt_mode_state_models, DEFAULT_REUSE_STATE_MODELS),
        "partial_save_every_rows": (
            int(args.partial_save_every_rows)
            if args.partial_save_every_rows is not None
            else (100 if args.smoke_run else 25000)
        ),
    }


def worker_main(args: argparse.Namespace) -> int:
    if not args.resume_dir:
        raise SystemExit("--resume-dir is required")
    if not args.output_dir:
        raise SystemExit("--output-dir is required in worker mode")

    cfg = normalized_config(args)
    output_dir = as_path(args.output_dir)
    resume_dir = as_path(args.resume_dir)
    session_dir = as_path(args.session_dir)
    loaders_pkl = as_path(args.loaders_pkl)
    assert output_dir is not None
    assert resume_dir is not None
    assert loaders_pkl is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "console_worker_info.json",
        {
            "pid": os.getpid(),
            "started_at": now_text(),
            "attempt_number": int(args.attempt_number or 0),
            "output_dir": str(output_dir),
            "resume_dir": str(resume_dir),
            "session_dir": str(session_dir) if session_dir else None,
            "loaders_pkl": str(loaders_pkl),
            "argv": sys.argv,
        },
    )

    print(f"Worker PID: {os.getpid()}", flush=True)
    print(f"Worker output_dir: {output_dir}", flush=True)
    print(f"Worker resume_dir: {resume_dir}", flush=True)
    print(f"Worker loaders_pkl: {loaders_pkl}", flush=True)

    svp = load_strict_helper(args.helper_path)
    svp.set_determinism()
    train_loader, val_loader, test_loader = svp.load_benchmark_loaders(str(loaders_pkl))
    _ = train_loader, val_loader
    test_dataset = getattr(test_loader, "dataset", test_loader)
    print(f"Test cases after exclusion: {len(test_dataset)}", flush=True)

    outputs = svp.run_strict_video_benchmark(
        test_dataset,
        output_dir=output_dir,
        benchmark_models=cfg["models"],
        prompt_modes=cfg["prompt_modes"],
        protocols=cfg["protocols"],
        smoke_run=bool(args.smoke_run),
        smoke_cases_per_dataset=int(args.smoke_cases_per_dataset),
        smoke_case_selection=str(args.smoke_case_selection),
        smoke_prefer_sequence_cases=not bool(args.no_smoke_prefer_sequence_cases),
        smoke_datasets=cfg["smoke_datasets"],
        smoke_max_scan_per_dataset=int(args.smoke_max_scan_per_dataset),
        max_test_cases=args.max_test_cases,
        benchmark_case_indices=cfg["case_indices"],
        thresholds=svp.THRESHOLDS,
        primary_threshold=svp.PRIMARY_THRESHOLD,
        compute_hd95_2d=True,
        compute_hd95_2d_for_3d_cases=False,
        compute_hd95_3d=False,
        tensorboard=not bool(args.no_tensorboard),
        partial_save_every_rows=int(cfg["partial_save_every_rows"]),
        status_update_every_rows=int(args.status_update_every_rows),
        status_update_every_sec=float(args.status_update_every_sec),
        checkpoint_flush_every_rows=int(args.checkpoint_flush_every_rows),
        order_2d_before_3d=not bool(args.no_order_2d_before_3d),
        save_model_separated_outputs=not bool(args.no_save_model_separated_outputs),
        fail_fast=bool(args.fail_fast),
        visual_top_n_datasets=args.visual_top_n_datasets,
        resume_from_dir=resume_dir,
        resume_completed_rows=True,
        cache_cases_cpu=not bool(args.no_cache_cases_cpu),
        cache_prompt_plans_cpu=not bool(args.no_cache_prompt_plans_cpu),
        write_model_outputs_on_partial=bool(args.write_model_outputs_on_partial),
        reuse_prompt_mode_state=bool(args.reuse_prompt_mode_state),
        reuse_prompt_mode_state_models=cfg["reuse_prompt_mode_state_models"],
    )
    write_json(
        output_dir / "console_worker_info.json",
        {
            "pid": os.getpid(),
            "completed_at": now_text(),
            "attempt_number": int(args.attempt_number or 0),
            "output_dir": str(outputs.get("output_dir", output_dir)),
            "resume_dir": str(resume_dir),
            "session_dir": str(session_dir) if session_dir else None,
            "status": "complete",
        },
    )
    print(f"Benchmark complete: {outputs['output_dir']}", flush=True)
    return 0


def worker_command(args: argparse.Namespace, session_dir: Path, attempt_dir: Path, resume_dir: Path, attempt_number: int) -> list[str]:
    cfg = normalized_config(args)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--resume-dir",
        str(resume_dir),
        "--output-dir",
        str(attempt_dir),
        "--session-dir",
        str(session_dir),
        "--attempt-number",
        str(attempt_number),
        "--output-base",
        str(args.output_base),
        "--loaders-pkl",
        str(args.loaders_pkl),
        "--models",
        *cfg["models"],
        "--prompt-modes",
        *cfg["prompt_modes"],
        "--protocols",
        *cfg["protocols"],
        "--smoke-cases-per-dataset",
        str(args.smoke_cases_per_dataset),
        "--smoke-case-selection",
        str(args.smoke_case_selection),
        "--smoke-max-scan-per-dataset",
        str(args.smoke_max_scan_per_dataset),
        "--status-update-every-rows",
        str(args.status_update_every_rows),
        "--status-update-every-sec",
        str(args.status_update_every_sec),
        "--checkpoint-flush-every-rows",
        str(args.checkpoint_flush_every_rows),
        "--partial-save-every-rows",
        str(cfg["partial_save_every_rows"]),
        "--reuse-prompt-mode-state-models",
        *cfg["reuse_prompt_mode_state_models"],
    ]
    if args.helper_path:
        cmd.extend(["--helper-path", str(args.helper_path)])
    if args.smoke_run:
        cmd.append("--smoke-run")
    if args.no_smoke_prefer_sequence_cases:
        cmd.append("--no-smoke-prefer-sequence-cases")
    if cfg["smoke_datasets"]:
        cmd.extend(["--smoke-datasets", *cfg["smoke_datasets"]])
    if args.max_test_cases is not None:
        cmd.extend(["--max-test-cases", str(args.max_test_cases)])
    if cfg["case_indices"]:
        cmd.extend(["--case-indices", *[str(x) for x in cfg["case_indices"]]])
    if args.visual_top_n_datasets is not None:
        cmd.extend(["--visual-top-n-datasets", str(args.visual_top_n_datasets)])
    if args.no_tensorboard:
        cmd.append("--no-tensorboard")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.no_order_2d_before_3d:
        cmd.append("--no-order-2d-before-3d")
    if args.no_save_model_separated_outputs:
        cmd.append("--no-save-model-separated-outputs")
    if args.no_cache_cases_cpu:
        cmd.append("--no-cache-cases-cpu")
    if args.no_cache_prompt_plans_cpu:
        cmd.append("--no-cache-prompt-plans-cpu")
    if args.write_model_outputs_on_partial:
        cmd.append("--write-model-outputs-on-partial")
    if args.reuse_prompt_mode_state:
        cmd.append("--reuse-prompt-mode-state")
    return cmd


def start_log_reader(
    proc: subprocess.Popen[str],
    log_handle,
    last_console_lines: deque[str],
) -> threading.Thread:
    def _read() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            last_console_lines.append(line.rstrip("\n"))
            print(line, end="", flush=True)

    thread = threading.Thread(target=_read, name="worker-log-reader", daemon=True)
    thread.start()
    return thread


def run_attempt(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    attempt_dir: Path,
    resume_dir: Path,
    attempt_number: int,
) -> tuple[int | None, str, Path, deque[str]]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_log = session_dir / "logs" / f"attempt_{attempt_number:03d}.log"
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = worker_command(args, session_dir, attempt_dir, resume_dir, attempt_number)

    write_json(
        attempt_dir / "console_attempt_info.json",
        {
            "attempt_number": attempt_number,
            "started_at": now_text(),
            "output_dir": str(attempt_dir),
            "resume_dir": str(resume_dir),
            "session_dir": str(session_dir),
            "command": cmd,
        },
    )
    (session_dir / "latest_attempt_dir.txt").write_text(str(attempt_dir) + "\n", encoding="utf-8")
    (session_dir / "latest_resume_dir.txt").write_text(str(resume_dir) + "\n", encoding="utf-8")

    print("", flush=True)
    print(f"[supervisor] Starting attempt {attempt_number}", flush=True)
    print(f"[supervisor] Attempt output dir: {attempt_dir}", flush=True)
    print(f"[supervisor] Resume source dir: {resume_dir}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    popen_kwargs: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "env": env,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    last_console_lines: deque[str] = deque(maxlen=240)
    stale_reason = ""
    with attempt_log.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n=== attempt {attempt_number} start {now_text()} ===\n")
        log_handle.write("Command: " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, **popen_kwargs)
        reader = start_log_reader(proc, log_handle, last_console_lines)
        last_progress = 0.0
        try:
            while proc.poll() is None:
                now = time.time()
                if args.progress_interval_sec > 0 and now - last_progress >= float(args.progress_interval_sec):
                    status = read_json(attempt_dir / "benchmark_status.json")
                    heartbeat = read_json(attempt_dir / "benchmark_heartbeat.json")
                    print(f"[supervisor {now_text()}] {format_progress(status, heartbeat)}", flush=True)
                    last_progress = now

                heartbeat_path = attempt_dir / "benchmark_heartbeat.json"
                if float(args.heartbeat_stale_sec) > 0:
                    if heartbeat_path.is_file():
                        age = now - heartbeat_path.stat().st_mtime
                        if age > float(args.heartbeat_stale_sec):
                            stale_reason = f"heartbeat_stale_{int(age)}s"
                            print(
                                f"[supervisor] Heartbeat stale for {age:.0f}s; terminating worker.",
                                flush=True,
                            )
                            terminate_process(proc)
                            break
                    elif now - attempt_dir.stat().st_mtime > float(args.heartbeat_stale_sec):
                        age = now - attempt_dir.stat().st_mtime
                        stale_reason = f"heartbeat_missing_{int(age)}s"
                        print(
                            f"[supervisor] No heartbeat after {age:.0f}s; terminating worker.",
                            flush=True,
                        )
                        terminate_process(proc)
                        break
                time.sleep(1.0)
        except KeyboardInterrupt:
            stale_reason = "keyboard_interrupt"
            print("\n[supervisor] Keyboard interrupt received; terminating worker.", flush=True)
            terminate_process(proc)
        finally:
            code = proc.wait()
            reader.join(timeout=5.0)
            log_handle.write(f"=== attempt {attempt_number} exit {now_text()} code={code} reason={stale_reason or 'process_exit'} ===\n")

    reason = stale_reason or ("complete" if code == 0 else "worker_exit")
    return code, reason, attempt_log, last_console_lines


def supervisor_main(args: argparse.Namespace) -> int:
    if not args.resume_dir:
        raise SystemExit("--resume-dir is required")
    resume_dir = as_path(args.resume_dir)
    output_base = as_path(args.output_base)
    assert resume_dir is not None
    assert output_base is not None
    if not resume_dir.exists():
        raise SystemExit(f"Resume directory does not exist: {resume_dir}")
    output_base.mkdir(parents=True, exist_ok=True)

    session_dir = output_base / f"strict_video_console_{timestamp()}"
    session_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_command = f"tensorboard --logdir {session_dir} --host 0.0.0.0 --port 6006"
    write_json(
        session_dir / "console_session.json",
        {
            "started_at": now_text(),
            "session_dir": str(session_dir),
            "initial_resume_dir": str(resume_dir),
            "tensorboard_command": tensorboard_command,
            "auto_restart": not bool(args.no_auto_restart),
            "max_restarts": int(args.max_restarts),
            "heartbeat_stale_sec": float(args.heartbeat_stale_sec),
            "argv": sys.argv,
        },
    )

    print(f"Console session dir: {session_dir}", flush=True)
    print(f"Initial resume source dir: {resume_dir}", flush=True)
    print(f"TensorBoard command: {tensorboard_command}", flush=True)
    print("Stop with Ctrl+C. The supervisor will write a report and print the next resume directory.", flush=True)

    restarts_used = 0
    attempt_number = 1
    current_resume_dir = resume_dir
    while True:
        attempt_dir = session_dir / f"strict_video_protocol_{timestamp()}_attempt{attempt_number:03d}"
        code, reason, attempt_log, last_console_lines = run_attempt(
            args,
            session_dir=session_dir,
            attempt_dir=attempt_dir,
            resume_dir=current_resume_dir,
            attempt_number=attempt_number,
        )
        next_resume_dir = latest_resume_dir(current_resume_dir, attempt_dir)
        (session_dir / "latest_resume_dir.txt").write_text(str(next_resume_dir) + "\n", encoding="utf-8")

        if code == 0 and reason == "complete":
            print(f"[supervisor] Benchmark complete. Final output dir: {attempt_dir}", flush=True)
            write_json(
                session_dir / "console_session_complete.json",
                {
                    "completed_at": now_text(),
                    "session_dir": str(session_dir),
                    "final_output_dir": str(attempt_dir),
                    "attempts": attempt_number,
                    "restarts_used": restarts_used,
                },
            )
            return 0

        report_dir = write_crash_report(
            session_dir=session_dir,
            attempt_dir=attempt_dir,
            attempt_number=attempt_number,
            reason=reason,
            exit_code=code,
            previous_resume_dir=current_resume_dir,
            next_resume_dir=next_resume_dir,
            attempt_log=attempt_log,
            last_console_lines=last_console_lines,
        )
        print(f"[supervisor] Worker stopped: reason={reason}, exit_code={code}", flush=True)
        print(f"[supervisor] Crash report: {report_dir}", flush=True)
        print(f"[supervisor] Next resume directory: {next_resume_dir}", flush=True)

        if reason == "keyboard_interrupt":
            return 130
        if args.no_auto_restart:
            return int(code or 1)
        if restarts_used >= int(args.max_restarts):
            print(f"[supervisor] Max restarts reached ({args.max_restarts}); stopping.", flush=True)
            return int(code or 1)

        restarts_used += 1
        attempt_number += 1
        current_resume_dir = next_resume_dir
        print(f"[supervisor] Restarting from {current_resume_dir} (restart {restarts_used}/{args.max_restarts}).", flush=True)
        time.sleep(5.0)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.worker:
            return worker_main(args)
        return supervisor_main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fatal runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
