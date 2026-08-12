#!/usr/bin/env python3
"""Supervise a training process and resume it after crashes or stale heartbeats."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("distill", "nodistill", "base"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--adopt-pid", type=int, default=0)
    parser.add_argument("--stale-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a training command is required after --")
    return args


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def proc_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


def proc_alive(pid: int, identity: str | None) -> bool:
    if identity is None:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0] != "Z" and fields[19] == identity
    except (FileNotFoundError, IndexError, OSError):
        return False


def descendant_pids(root_pid: int) -> list[int]:
    parents: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            parent = int(fields[1])
            parents.setdefault(parent, []).append(int(entry.name))
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    found: list[int] = []
    pending = list(parents.get(root_pid, []))
    while pending:
        pid = pending.pop()
        found.append(pid)
        pending.extend(parents.get(pid, []))
    return found


def terminate_adopted(pid: int, grace_seconds: float) -> None:
    targets = [pid, *descendant_pids(pid)]
    for target in targets:
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        remaining = [target for target in targets if Path(f"/proc/{target}").exists()]
        if not remaining:
            return
        time.sleep(0.5)
    for target in reversed(targets):
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass


def terminate_child(child: subprocess.Popen, grace_seconds: float) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def choose_resume(checkpoint_dir: Path) -> Path | None:
    candidates = [checkpoint_dir / "recovery_latest.pth", checkpoint_dir / "latest.pth"]
    available = [path for path in candidates if path.is_file()]
    return max(available, key=lambda path: path.stat().st_mtime) if available else None


def command_with_resume(command: list[str], resume: Path | None) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token == "--resume":
            skip_next = True
            continue
        if token.startswith("--resume="):
            continue
        cleaned.append(token)
    if resume is not None:
        cleaned.extend(["--resume", str(resume)])
    return cleaned


def main() -> int:
    args = parse_args()
    run_dir = Path(args.experiment_root) / args.mode / args.run_id
    checkpoint_dir = Path(args.checkpoint_root) / args.mode / args.run_id
    status_path = run_dir / "status.json"
    supervisor_path = run_dir / "supervisor_status.json"
    safe_lock_name = f"rwkv-training-{args.mode}-{args.run_id}".replace("/", "_")
    lock_path = Path(tempfile.gettempdir()) / f"{safe_lock_name}.lock"
    run_dir.mkdir(parents=True, exist_ok=True)

    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[training-supervisor] another supervisor holds {lock_path}", flush=True)
        return 2

    stopping = False
    child: subprocess.Popen | None = None
    adopted_pid = int(args.adopt_pid)
    adopted_identity = proc_identity(adopted_pid) if adopted_pid else None

    def request_stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[training-supervisor] received signal {signum}; stopping", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    restart_count = 0
    current_pid = adopted_pid
    current_identity = adopted_identity
    attempt_started = time.time()
    adopted = bool(adopted_pid and adopted_identity)
    if adopted_pid and not adopted:
        print(f"[training-supervisor] adopt PID {adopted_pid} is not alive; launching recovery", flush=True)
    elif adopted:
        print(f"[training-supervisor] adopting PID {adopted_pid} for {args.mode}/{args.run_id}", flush=True)
    else:
        print(f"[training-supervisor] starting {args.mode}/{args.run_id}: {' '.join(args.command)}", flush=True)
        child = subprocess.Popen(list(args.command), start_new_session=True)
        current_pid = child.pid
        current_identity = proc_identity(child.pid)

    while True:
        state = read_json(status_path)
        if state.get("state") == "complete":
            atomic_json(supervisor_path, {
                "state": "complete", "mode": args.mode, "run_id": args.run_id,
                "restarts": restart_count, "updated_utc": now_text(),
            })
            print(f"[training-supervisor] {args.mode}/{args.run_id} is complete", flush=True)
            return 0

        if stopping:
            if adopted and current_pid:
                terminate_adopted(current_pid, args.terminate_grace_seconds)
            elif child is not None:
                terminate_child(child, args.terminate_grace_seconds)
            atomic_json(supervisor_path, {
                "state": "stopped", "mode": args.mode, "run_id": args.run_id,
                "restarts": restart_count, "updated_utc": now_text(),
            })
            return 130

        alive = proc_alive(current_pid, current_identity) if adopted else child is not None and child.poll() is None
        if alive:
            try:
                activity_time = status_path.stat().st_mtime
            except FileNotFoundError:
                activity_time = attempt_started
            if not adopted:
                activity_time = max(activity_time, attempt_started)
            stale_for = max(0.0, time.time() - activity_time)
            atomic_json(supervisor_path, {
                "state": "monitoring", "mode": args.mode, "run_id": args.run_id,
                "pid": current_pid, "adopted": adopted, "restarts": restart_count,
                "training_state": state.get("state"), "stale_for_seconds": stale_for,
                "stale_limit_seconds": args.stale_seconds, "updated_utc": now_text(),
            })
            if args.stale_seconds > 0 and stale_for > args.stale_seconds:
                print(
                    f"[training-supervisor] stale heartbeat for {stale_for:.0f}s "
                    f"(limit {args.stale_seconds:.0f}s); terminating PID {current_pid}", flush=True,
                )
                if adopted:
                    terminate_adopted(current_pid, args.terminate_grace_seconds)
                elif child is not None:
                    terminate_child(child, args.terminate_grace_seconds)
            else:
                time.sleep(max(0.25, args.poll_seconds))
                continue

        state = read_json(status_path)
        if state.get("state") == "complete":
            continue
        if restart_count >= args.max_restarts:
            atomic_json(supervisor_path, {
                "state": "failed", "mode": args.mode, "run_id": args.run_id,
                "error": "maximum restarts reached", "restarts": restart_count,
                "updated_utc": now_text(),
            })
            print(f"[training-supervisor] maximum restarts reached ({args.max_restarts})", flush=True)
            return 1

        resume = choose_resume(checkpoint_dir)
        if resume is None:
            atomic_json(supervisor_path, {
                "state": "failed", "mode": args.mode, "run_id": args.run_id,
                "error": f"no recovery checkpoint under {checkpoint_dir}",
                "restarts": restart_count, "updated_utc": now_text(),
            })
            print(f"[training-supervisor] no recovery checkpoint under {checkpoint_dir}", flush=True)
            return 1

        restart_count += 1
        command = command_with_resume(list(args.command), resume)
        print(
            f"[training-supervisor] restart {restart_count}/{args.max_restarts} "
            f"from {resume}: {' '.join(command)}", flush=True,
        )
        child = subprocess.Popen(command, start_new_session=True)
        adopted = False
        current_pid = child.pid
        current_identity = proc_identity(child.pid)
        attempt_started = time.time()
        atomic_json(supervisor_path, {
            "state": "restarted", "mode": args.mode, "run_id": args.run_id,
            "pid": child.pid, "resume": str(resume), "restarts": restart_count,
            "updated_utc": now_text(),
        })
        time.sleep(min(max(0.25, args.poll_seconds), 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
