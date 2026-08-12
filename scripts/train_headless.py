#!/usr/bin/env python3
"""Headless RWKV-MedSAM2/SAM2.1 training with distributed synchronization."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CUDA_HOME = "/usr/local/cuda"
NVIDIA_ROOT = "/usr/local/lib/python3.10/dist-packages/nvidia"
TORCH_ROOT = "/usr/local/lib/python3.10/dist-packages/torch"
os.environ.setdefault("CUDA_HOME", CUDA_HOME)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
os.environ["PATH"] = (
    f"{CUDA_HOME}/bin:{CUDA_HOME}/nvvm/bin:" + os.environ.get("PATH", "")
)
_runtime_libraries = [
    f"{TORCH_ROOT}/lib", f"{NVIDIA_ROOT}/cudnn/lib",
    f"{CUDA_HOME}/lib64", "/usr/local/lib",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(
    _runtime_libraries + [os.environ.get("LD_LIBRARY_PATH", "")]
).rstrip(":")
if os.environ.get("RWKV_HEADLESS_ENV_READY") != "1":
    os.environ["RWKV_HEADLESS_ENV_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rwkv_medsam2.dataset import BalancedTaskBatchSampler
from rwkv_medsam2.distributed import (
    DistributedBatchShard,
    barrier,
    broadcast_string,
    cleanup_distributed,
    initialize_distributed,
)
from rwkv_medsam2.train_sam2 import (
    CheckpointManager,
    build_student_predictor,
    build_teacher_predictor,
    collate_sam2_batch,
    distillation_enabled,
    load_config,
    load_data_loaders,
    setup_optimizer_and_scheduler,
    train_epoch,
    validate_epoch,
)


class NullWriter:
    """No-op TensorBoard writer used by non-primary ranks."""

    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("distill", "nodistill", "base"), required=True)
    parser.add_argument("--loaders", default="/data/loaders32.pkl")
    parser.add_argument("--experiment-root", default="/data/TrainingLogs/retrain_20260805")
    parser.add_argument("--checkpoint-root", default="/data/TrainingCheckpoints/retrain_20260805")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=150)
    parser.add_argument("--val-frequency", type=int, default=5)
    parser.add_argument("--monitor-every-steps", type=int, default=10)
    parser.add_argument("--recovery-minutes", type=float, default=120.0)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--disable-images", action="store_true", default=True)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def configure_logger(path: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"rwkv_headless_rank_{rank}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def build_config(args: argparse.Namespace):
    name = "sam2.1_vcr.yaml" if args.mode != "nodistill" else "sam2.1_vcr_nodistill.yaml"
    config = load_config(str(Path("ext/sam2/configs/sam2.1") / name))
    if args.mode == "base":
        base = OmegaConf.load("ext/sam2/configs/sam2.1/sam2.1_hiera_t512.yaml")
        config.model = base.model
        config._config_path = "sam2.1/sam2.1_hiera_t512.yaml"
        config.init.ignore_prefixes = []
        config.training.lambda_distill_2d = 0.0
        config.training.lambda_distill_3d = 0.0
    config.training.batch_size = int(args.global_batch_size)
    config.training.num_workers = int(args.num_workers)
    config.training.epochs = int(args.epochs)
    config.training.val_freq = int(args.val_frequency)
    config.logging.disable_images = bool(args.disable_images)
    return config


class LimitedBatchSampler:
    """Expose only the first N batches from a deterministic sampler."""

    def __init__(self, batch_sampler, limit: int):
        self.batch_sampler = batch_sampler
        self.limit = int(limit)
        self.batch_size = int(batch_sampler.batch_size)

    def __iter__(self):
        for index, batch in enumerate(self.batch_sampler):
            if index >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(self.limit, len(self.batch_sampler))

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


class ResumableBatchSampler:
    """Skip completed batches before DataLoader workers fetch their samples.

    ``__len__`` deliberately reports the full epoch length so training progress,
    checkpoint boundaries, and scheduler accounting retain their original batch
    coordinates after a mid-epoch resume.
    """

    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.batch_size = int(batch_sampler.batch_size)
        self.start_batch = 0

    def __iter__(self):
        for index, batch in enumerate(self.batch_sampler):
            if index >= self.start_batch:
                yield batch

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def set_start_batch(self, start_batch: int):
        start_batch = int(start_batch)
        if not 0 <= start_batch <= len(self.batch_sampler):
            raise ValueError(
                f"resume batch {start_batch} is outside epoch length "
                f"{len(self.batch_sampler)}"
            )
        self.start_batch = start_batch


def make_loaders(config, args: argparse.Namespace, context):
    loaded_train, loaded_val, loaded_test = load_data_loaders(args.loaders)
    with open(config.dripp.tasks_file, "r", encoding="utf-8") as handle:
        tasks_map = json.load(handle)

    num_samples = (
        max(600, args.max_steps_per_epoch * args.global_batch_size)
        if args.max_steps_per_epoch > 0
        else None
    )
    target_dim_ratio = {2: 0.5, 3: 0.5} if args.pilot else None
    global_sampler = BalancedTaskBatchSampler(
        loaded_train.dataset.sequences,
        tasks_map=tasks_map,
        batch_size=args.global_batch_size,
        drop_last=True,
        seed=int(config.training.seed),
        num_samples=num_samples,
        min_per_bucket=1,
        target_dim_ratio=target_dim_ratio,
    )
    if args.max_steps_per_epoch > 0:
        global_sampler = LimitedBatchSampler(global_sampler, args.max_steps_per_epoch)
    train_sampler = (
        DistributedBatchShard(global_sampler, context.rank, context.world_size)
        if context.enabled
        else global_sampler
    )
    train_sampler = ResumableBatchSampler(train_sampler)
    kwargs = {
        "dataset": loaded_train.dataset,
        "batch_sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_sam2_batch,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**kwargs)

    val_count = min(max(0, args.val_samples), len(loaded_val.dataset))
    if val_count:
        val_sampler = BalancedTaskBatchSampler(
            loaded_val.dataset.sequences,
            tasks_map=tasks_map,
            batch_size=1,
            drop_last=False,
            seed=int(config.training.seed),
            num_samples=val_count,
            min_per_bucket=1,
            target_dim_ratio=target_dim_ratio,
        )
        val_loader = DataLoader(
            loaded_val.dataset,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_sam2_batch,
        )
    else:
        val_loader = None
    return train_loader, val_loader, loaded_test


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(student, optimizer, scheduler, epoch, batch_idx, global_step):
    return {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": None,
        "epoch": int(epoch),
        "resume_epoch": int(epoch),
        "resume_batch_idx": int(batch_idx) + 1,
        "global_step": int(global_step),
    }


def restore_checkpoint(path: str, student, optimizer, scheduler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return (
        int(state.get("resume_epoch", int(state.get("epoch", -1)) + 1)),
        int(state.get("resume_batch_idx", 0)),
        int(state.get("global_step", 0)),
    )


def primary_metric(metrics: dict) -> tuple[str | None, float]:
    preferred = (
        "iou_best", "iou@0.5", "iou@avg_thr",
        "iou3d@best", "iou3d@0.5", "iou3d@best_vol",
    )
    for key in preferred:
        if key in metrics:
            return key, float(metrics[key])
    candidates = [key for key in metrics if "iou" in key]
    if not candidates:
        return None, float("-inf")
    key = max(candidates, key=lambda item: float(metrics[item]))
    return key, float(metrics[key])


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return float(tensor.item())


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    device = torch.device("cuda", context.local_rank)
    if args.global_batch_size % context.world_size != 0:
        raise ValueError(
            f"global batch {args.global_batch_size} must divide world size {context.world_size}"
        )

    config = build_config(args)
    if args.teacher_checkpoint:
        config.teacher.ckpt_path = args.teacher_checkpoint
    config.training.device = str(device)
    config.runtime = OmegaConf.create({"is_main_process": context.is_main})

    proposed = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = broadcast_string(proposed)
    run_name = f"{args.mode}/{run_id}"
    log_dir = Path(args.experiment_root) / run_name
    checkpoint_dir = Path(args.checkpoint_root) / run_name
    config.ckpt_path = str(checkpoint_dir)
    config.logging.output_dir = str(log_dir)
    config.logging.filename = "training.log"

    logger = configure_logger(log_dir / "training.log", context.rank)
    writer = (
        SummaryWriter(log_dir=str(log_dir), flush_secs=30, max_queue=100)
        if context.is_main
        else NullWriter(Path("/tmp") / f"rwkv_rank_{context.rank}")
    )
    status_path = log_dir / "status.json"
    last_monitor = time.monotonic()
    last_monitor_step = 0
    last_recovery = last_monitor

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        train_loader, val_loader, test_loader = make_loaders(config, args, context)
        train_batches_per_epoch = len(train_loader)
        logger.info(
            "mode=%s run_id=%s rank=%d/%d device=%s global_batch=%d local_batch=%d "
            "train_batches=%d val_batches=%d",
            args.mode, run_id, context.rank, context.world_size, device,
            args.global_batch_size, args.global_batch_size // context.world_size,
            train_batches_per_epoch, len(val_loader) if val_loader is not None else 0,
        )

        student = build_student_predictor(config, logger=logger if context.is_main else None).to(device)
        teacher = build_teacher_predictor(config).to(device) if distillation_enabled(config) else None
        bundle = setup_optimizer_and_scheduler(student, config)
        optimizer, scheduler = bundle["optimizer"], bundle["scheduler"]

        start_epoch = start_batch = global_step = 0
        if args.resume:
            start_epoch, start_batch, global_step = restore_checkpoint(
                args.resume, student, optimizer, scheduler
            )
            logger.info(
                "resumed=%s epoch=%d batch=%d global_step=%d",
                args.resume, start_epoch, start_batch, global_step,
            )

        manager = CheckpointManager(str(checkpoint_dir)) if context.is_main else None
        if context.is_main:
            metadata = {
                "mode": args.mode,
                "run_id": run_id,
                "host": socket.gethostname(),
                "world_size": context.world_size,
                "global_batch_size": args.global_batch_size,
                "num_workers_per_rank": args.num_workers,
                "max_steps_per_epoch": args.max_steps_per_epoch,
                "val_samples": args.val_samples,
                "pilot": args.pilot,
                "config": OmegaConf.to_container(config, resolve=True),
            }
            atomic_json(log_dir / "run.json", metadata)
            writer.add_text("run/metadata", json.dumps(metadata, indent=2), 0)
            writer.add_scalar("run/world_size", context.world_size, 0)
            writer.add_scalar("run/global_batch_size", args.global_batch_size, 0)
            atomic_json(status_path, {
                "state": "starting", "mode": args.mode,
                "run_id": run_id, "global_step": global_step,
            })

        def step_callback(epoch, batch_idx, step, current_student, current_optimizer, current_scheduler, _scaler):
            nonlocal last_monitor, last_monitor_step, last_recovery
            now = time.monotonic()
            if context.is_main and (
                step == 1 or step - last_monitor_step >= max(1, args.monitor_every_steps)
                or now - last_monitor >= 60.0
            ):
                interval = max(now - last_monitor, 1e-6)
                step_delta = max(step - last_monitor_step, 1)
                samples_per_second = step_delta * args.global_batch_size / interval
                total_steps = args.epochs * train_batches_per_epoch
                eta_hours = max(0, total_steps - step) * interval / step_delta / 3600.0
                allocated = torch.cuda.memory_allocated(device) / 1024**2
                reserved = torch.cuda.memory_reserved(device) / 1024**2
                try:
                    utilization = float(torch.cuda.utilization(device))
                except Exception:
                    utilization = float("nan")
                writer.add_scalar("monitor/samples_per_second", samples_per_second, step)
                writer.add_scalar("monitor/eta_hours", eta_hours, step)
                writer.add_scalar("system/gpu_memory_allocated_mib", allocated, step)
                writer.add_scalar("system/gpu_memory_reserved_mib", reserved, step)
                if utilization == utilization:
                    writer.add_scalar("system/gpu_utilization_percent", utilization, step)
                writer.add_scalar("monitor/heartbeat", 1.0, step)
                writer.flush()
                atomic_json(status_path, {
                    "state": "training", "mode": args.mode, "run_id": run_id,
                    "epoch": int(epoch), "batch_idx": int(batch_idx),
                    "global_step": int(step), "samples_per_second": samples_per_second,
                    "eta_hours": eta_hours,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                last_monitor, last_monitor_step = now, step

            recovery_seconds = max(0.0, args.recovery_minutes) * 60.0
            if context.is_main and recovery_seconds > 0 and now - last_recovery >= recovery_seconds:
                atomic_torch_save(
                    checkpoint_dir / "recovery_latest.pth",
                    checkpoint_payload(
                        current_student, current_optimizer, current_scheduler,
                        epoch, batch_idx, step,
                    ),
                )
                writer.add_scalar("monitor/recovery_checkpoint", 1.0, step)
                writer.flush()
                last_recovery = now

        best_metric = float("-inf")
        for epoch in range(start_epoch, args.epochs):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            loader_start_batch = epoch_start_batch
            batch_index_offset = 0
            if hasattr(train_loader.batch_sampler, "set_start_batch"):
                train_loader.batch_sampler.set_start_batch(epoch_start_batch)
                loader_start_batch = 0
                batch_index_offset = epoch_start_batch
            average_loss, global_step, optimizer, scheduler = train_epoch(
                student, teacher, train_loader, optimizer, scheduler, config, None,
                epoch, logger, writer, global_step,
                start_batch_idx=loader_start_batch,
                batch_index_offset=batch_index_offset,
                step_callback=step_callback,
            )
            average_loss = reduce_mean(average_loss, device)
            if context.is_main:
                writer.add_scalar("distributed/train_loss_mean", average_loss, epoch)

            barrier()
            validate_now = (
                val_loader is not None and (
                    epoch == 0
                    or (epoch + 1) % max(1, args.val_frequency) == 0
                    or epoch == args.epochs - 1
                )
            )
            if context.is_main and validate_now:
                # Capture the exact pre-validation boundary. A recovery from this
                # checkpoint re-enters validation without replaying the whole epoch.
                atomic_torch_save(
                    checkpoint_dir / "recovery_latest.pth",
                    checkpoint_payload(
                        student, optimizer, scheduler, epoch,
                        max(0, train_batches_per_epoch - 1), global_step,
                    ),
                )
                atomic_json(status_path, {
                    "state": "validating", "mode": args.mode, "run_id": run_id,
                    "epoch": int(epoch), "validation_batch_idx": -1,
                    "validation_batches": len(val_loader),
                    "global_step": int(global_step),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })

                def validation_callback(batch_idx, total_batches, dataset, subdataset, data_dim):
                    atomic_json(status_path, {
                        "state": "validating", "mode": args.mode, "run_id": run_id,
                        "epoch": int(epoch), "validation_batch_idx": int(batch_idx),
                        "validation_batches": int(total_batches),
                        "dataset": str(dataset), "subdataset": str(subdataset),
                        "data_dim": int(data_dim), "global_step": int(global_step),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    })

                metrics = validate_epoch(
                    student, val_loader, config, epoch, writer,
                    progress_callback=validation_callback,
                )
                key, metric = primary_metric(metrics)
                manager.save_latest(student, optimizer, scheduler, None, epoch, global_step)
                if metric > best_metric:
                    best_metric = metric
                    manager.save_best(
                        student, optimizer, scheduler, None, epoch, metric, global_step
                    )
                logger.info(
                    "epoch=%d validation_key=%s validation_metric=%.6f", epoch, key, metric
                )
            elif context.is_main:
                manager.save_latest(student, optimizer, scheduler, None, epoch, global_step)
            barrier()
            start_batch = 0

        if context.is_main and args.full_test:
            metrics = validate_epoch(student, test_loader, config, args.epochs, writer)
            for key, value in metrics.items():
                writer.add_scalar(f"test/{key}", float(value), args.epochs)
            logger.info("final_test=%s", metrics)

        if context.is_main:
            atomic_json(status_path, {
                "state": "complete", "mode": args.mode, "run_id": run_id,
                "global_step": global_step,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            writer.flush()
            logger.info("training complete")
    except Exception as error:
        if context.is_main:
            atomic_json(status_path, {
                "state": "failed", "mode": args.mode, "run_id": run_id,
                "error": repr(error),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
        logger.exception("training failed")
        raise
    finally:
        writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
