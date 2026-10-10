#!/usr/bin/env python3
"""Headless RWKV-MedSAM2/SAM2.1 training with distributed synchronization."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import socket
import sys
import time
from collections import Counter
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

import numpy as np
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
    parser.add_argument(
        "--variant", choices=("standard", "original", "efficient"),
        default="standard",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--initial-model-state", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=0,
        help="Exact global training-example target before homogeneous batching",
    )
    parser.add_argument("--val-samples", type=int, default=150)
    parser.add_argument("--val-frequency", type=int, default=5)
    parser.add_argument("--monitor-every-steps", type=int, default=10)
    parser.add_argument("--recovery-minutes", type=float, default=120.0)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--disable-images", action="store_true", default=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--safe-training-optimizations", action="store_true")
    parser.add_argument("--efficient-vcr-backbone", action="store_true")
    parser.add_argument(
        "--efficient-vcr-architecture",
        choices=("efficient", "rwkv2", "rwkv2_mlp2"),
        default="efficient",
        help=(
            "Experimental architecture layered on --efficient-vcr-backbone. "
            "The default preserves the previously tested efficient backbone."
        ),
    )
    parser.add_argument("--matched-run", action="store_true")
    parser.add_argument(
        "--match-common-sam2-settings",
        action="store_true",
        help=(
            "Equalize shared non-encoder SAM2 settings across Hiera and VCR "
            "for matched architecture/predictor experiments"
        ),
    )
    parser.add_argument(
        "--ignore-pretrained-image-encoder",
        action="store_true",
        help=(
            "Load pretrained SAM2 components while leaving the complete image "
            "encoder randomly initialized"
        ),
    )
    parser.add_argument(
        "--safe-efficient-backbone", action="store_true",
        help=(
            "Use the tested efficient VCR candidate, per-volume 3D backward, "
            "and external training prefill while keeping SAM2VideoPredictor"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wkv-t-max", type=int, default=256)
    return parser.parse_args()


def safe_training_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "safe_training_optimizations", False)
        or getattr(args, "safe_efficient_backbone", False)
    )


def efficient_backbone_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "efficient_vcr_backbone", False)
        or getattr(args, "safe_efficient_backbone", False)
    )


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
    if bool(getattr(args, "ignore_pretrained_image_encoder", False)):
        config.init.ignore_prefixes = ["image_encoder."]
    if bool(getattr(args, "match_common_sam2_settings", False)):
        config.model.memory_attention.layer.self_attention.feat_sizes = [64, 64]
        config.model.memory_attention.layer.cross_attention.feat_sizes = [64, 64]
        config.model.use_mask_input_as_output_without_sam = False
    config.training.batch_size = int(args.global_batch_size)
    config.training.num_workers = int(args.num_workers)
    config.training.epochs = int(args.epochs)
    config.training.val_freq = int(args.val_frequency)
    config.logging.disable_images = bool(args.disable_images)
    safe_training = safe_training_requested(args)
    efficient_backbone = efficient_backbone_requested(args)
    any_optimization = bool(
        getattr(args, "optimized", False)
        or safe_training
        or efficient_backbone
    )
    config.optimizations = OmegaConf.create({
        "enabled": any_optimization,
        "use_optimized_predictor": bool(getattr(args, "optimized", False)),
        "efficient_vcr_backbone": efficient_backbone,
        "efficient_vcr_architecture": str(
            getattr(args, "efficient_vcr_architecture", "efficient")
        ),
        "external_training_prefill": safe_training,
        "image_feature_chunk_size": 4,
        "training_feature_cache_frames": 8,
        "per_volume_backward": bool(getattr(args, "optimized", False) or safe_training),
        "prediction_gpu_cache_chunks": 2,
        "prediction_cpu_cache_chunks": 64,
        "prediction_pin_cpu_features": True,
        "compile_image_encoder": False,
        "fuse_rwkv_shift_mix": any_optimization,
        "inplace_rwkv_activations": any_optimization,
        "wkv_t_max": int(getattr(args, "wkv_t_max", 256)),
    })
    if any_optimization and args.mode != "base":
        config.model.image_encoder.trunk.fuse_shift_mix = True
        config.model.image_encoder.trunk.inplace_activations = True
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


class FixedBatchSampler:
    """Replay a materialized batch plan without changing cases between passes."""

    def __init__(self, batches, batch_size: int):
        self.batches = [list(batch) for batch in batches]
        self.batch_size = int(batch_size)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class DatasetCoverageBatchSampler:
    """Limit a balanced plan while retaining every eligible dataset."""

    def __init__(self, batch_sampler, sequences, limit: int, seed: int):
        self.batch_sampler = batch_sampler
        self.sequences = sequences
        self.limit = int(limit)
        self.batch_size = int(batch_sampler.batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.eligible_datasets = sorted({
            str(sequences[index].get("dataset", "unknown"))
            for index in batch_sampler.valid_indices
        })

        # A small deterministic candidate bank is enough for one coverage swap
        # per dataset and avoids copying all ~760k eligible indices.
        self.candidates = {}
        for index in batch_sampler.valid_indices:
            dataset = str(sequences[index].get("dataset", "unknown"))
            by_key = self.candidates.setdefault(dataset, {})
            for key in batch_sampler.memberships[index]:
                bank = by_key.setdefault((int(key[0]), str(key[1])), [])
                if len(bank) < 64:
                    bank.append(int(index))

    def __len__(self):
        return min(self.limit, len(self.batch_sampler))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        batches = []
        for index, batch in enumerate(self.batch_sampler):
            if index >= self.limit:
                break
            batches.append(list(batch))
        if len(batches) < self.limit:
            yield from batches
            return

        rng = random.Random(self.seed + self.epoch)

        def analyze_plan():
            used = set()
            counts = Counter()
            positions = {}
            batch_keys = []
            key_counts = Counter()
            for batch_index, batch in enumerate(batches):
                first_index, first_task = batch[0]
                key = (
                    int(self.sequences[int(first_index)].get("dim", -1)),
                    str(first_task),
                )
                batch_keys.append(key)
                key_counts[key] += 1
                for item_index, item in enumerate(batch):
                    index = int(item[0])
                    used.add(index)
                    dataset = str(
                        self.sequences[index].get("dataset", "unknown")
                    )
                    counts[dataset] += 1
                    positions.setdefault(key, []).append(
                        (batch_index, item_index)
                    )
            return used, counts, positions, batch_keys, key_counts

        while True:
            (
                used_indices, dataset_counts, positions_by_key,
                batch_keys, key_counts,
            ) = analyze_plan()
            missing = [
                dataset for dataset in self.eligible_datasets
                if dataset_counts.get(dataset, 0) == 0
            ]
            if not missing:
                break
            rng.shuffle(missing)
            changed = False

            # Prefer a one-item swap that preserves the existing task bucket.
            for missing_dataset in missing:
                choices = []
                for key, indices in self.candidates.get(
                    missing_dataset, {}
                ).items():
                    if key not in positions_by_key:
                        continue
                    for candidate_index in indices:
                        if candidate_index not in used_indices:
                            choices.append((candidate_index, key))
                rng.shuffle(choices)
                for candidate_index, key in choices:
                    positions = list(positions_by_key[key])
                    rng.shuffle(positions)
                    for batch_index, item_index in positions:
                        old_index = int(batches[batch_index][item_index][0])
                        old_dataset = str(
                            self.sequences[old_index].get("dataset", "unknown")
                        )
                        if dataset_counts.get(old_dataset, 0) <= 1:
                            continue
                        batches[batch_index][item_index] = (
                            candidate_index, key[1]
                        )
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue

            # A condensed rare-task draw can omit a task entirely. Insert one
            # valid homogeneous batch for that task, replacing an overrepresented
            # batch of the same dimension.
            for missing_dataset in missing:
                candidate_maps = self.candidates.get(missing_dataset, {})
                keys = list(candidate_maps)
                rng.shuffle(keys)
                for key in keys:
                    new_indices = []
                    for dataset in [missing_dataset, *missing]:
                        for index in self.candidates.get(dataset, {}).get(key, []):
                            if index not in used_indices and index not in new_indices:
                                new_indices.append(index)
                                break
                        if len(new_indices) == self.batch_size:
                            break
                    if len(new_indices) < self.batch_size:
                        for dataset in self.eligible_datasets:
                            for index in self.candidates.get(dataset, {}).get(key, []):
                                if index not in used_indices and index not in new_indices:
                                    new_indices.append(index)
                                if len(new_indices) == self.batch_size:
                                    break
                            if len(new_indices) == self.batch_size:
                                break
                    if len(new_indices) < self.batch_size:
                        continue

                    victim_order = list(range(len(batches)))
                    rng.shuffle(victim_order)
                    for victim_index in victim_order:
                        victim_key = batch_keys[victim_index]
                        if victim_key[0] != key[0] or key_counts[victim_key] <= 1:
                            continue
                        victim_datasets = Counter(
                            str(self.sequences[int(item[0])].get(
                                "dataset", "unknown"
                            ))
                            for item in batches[victim_index]
                        )
                        if any(
                            dataset_counts.get(dataset, 0) <= count
                            for dataset, count in victim_datasets.items()
                        ):
                            continue
                        batches[victim_index] = [
                            (index, key[1]) for index in new_indices
                        ]
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break

            if not changed:
                raise RuntimeError(
                    "Could not represent eligible datasets in condensed plan: "
                    + ", ".join(missing)
                )

        yield from batches


def validate_batch_plan(batches, sampler) -> dict:
    """Assert that condensed swaps retain sampler membership invariants."""
    seen = set()
    for batch_number, batch in enumerate(batches):
        if len(batch) != sampler.batch_size:
            raise RuntimeError(
                f"Batch {batch_number} has {len(batch)} items, expected "
                f"{sampler.batch_size}"
            )
        dimensions = set()
        tasks = set()
        for item in batch:
            index, task_id = int(item[0]), str(item[1])
            if index in seen:
                raise RuntimeError(f"Duplicate dataset index in plan: {index}")
            seen.add(index)
            dimension = int(sampler.sequences[index].get("dim", -1))
            dimensions.add(dimension)
            tasks.add(task_id)
            if (dimension, task_id) not in sampler.memberships[index]:
                raise RuntimeError(
                    f"Invalid task assignment index={index} dim={dimension} "
                    f"task={task_id}"
                )
        if len(dimensions) != 1 or len(tasks) != 1:
            raise RuntimeError(
                f"Batch {batch_number} is not dimension/task homogeneous"
            )
    return {
        "homogeneous_dimension_task_batches": True,
        "all_task_assignments_valid": True,
        "all_indices_unique": True,
    }


def summarize_batch_plan(batches, sequences) -> dict:
    """Describe and hash a sampler plan without loading image pixels."""
    dimensions = Counter()
    datasets = Counter()
    subdatasets = Counter()
    tasks = Counter()
    indices = []
    digest = hashlib.sha256()
    for batch in batches:
        for raw_item in batch:
            if isinstance(raw_item, (tuple, list)):
                index, task_id = int(raw_item[0]), str(raw_item[1])
            else:
                index, task_id = int(raw_item), ""
            sequence = sequences[index]
            dimension = int(sequence.get("dim", -1))
            dataset = str(sequence.get("dataset", "unknown"))
            subdataset = str(sequence.get("subdataset", ""))
            dimensions[str(dimension)] += 1
            datasets[dataset] += 1
            subdatasets[f"{dataset}/{subdataset}"] += 1
            if task_id:
                tasks[task_id] += 1
            indices.append(index)
            digest.update(f"{index}:{task_id}\n".encode("utf-8"))
    return {
        "batches": len(batches),
        "training_examples": len(indices),
        "unique_dataset_indices": len(set(indices)),
        "duplicate_dataset_indices": len(indices) - len(set(indices)),
        "dimension_counts": dict(sorted(dimensions.items())),
        "approximate_input_frames": (
            int(dimensions.get("2", 0)) + 8 * int(dimensions.get("3", 0))
        ),
        "dataset_counts": dict(sorted(datasets.items())),
        "subdataset_counts": dict(sorted(subdatasets.items())),
        "task_counts": dict(sorted(tasks.items())),
        "plan_sha256": digest.hexdigest(),
    }


def make_loaders(config, args: argparse.Namespace, context):
    loaded_train, loaded_val, loaded_test = load_data_loaders(args.loaders)
    with open(config.dripp.tasks_file, "r", encoding="utf-8") as handle:
        tasks_map = json.load(handle)

    exact_batch_limit = None
    if args.samples_per_epoch > 0:
        num_samples = int(args.samples_per_epoch)
        exact_batch_limit = num_samples // args.global_batch_size
        # Each task bucket drops its own incomplete tail. Give the task-aware
        # sampler a modest reserve, then limit the emitted homogeneous batches
        # to the exact requested count.
        reserve_multiple = (2 * args.global_batch_size) if args.pilot else args.global_batch_size
        reserve = max(96, int(math.ceil(num_samples * 0.10)))
        reserve = int(math.ceil(reserve / reserve_multiple) * reserve_multiple)
        sampler_num_samples = num_samples + reserve
    elif args.max_steps_per_epoch > 0:
        num_samples = max(
            600, args.max_steps_per_epoch * args.global_batch_size
        )
        sampler_num_samples = num_samples
    else:
        num_samples = None
        sampler_num_samples = None
    target_dim_ratio = {2: 0.5, 3: 0.5} if args.pilot else None
    raw_sampler = BalancedTaskBatchSampler(
        loaded_train.dataset.sequences,
        tasks_map=tasks_map,
        batch_size=args.global_batch_size,
        drop_last=True,
        seed=int(config.training.seed),
        num_samples=sampler_num_samples,
        min_per_bucket=1,
        target_dim_ratio=target_dim_ratio,
    )

    global_sampler = raw_sampler
    if exact_batch_limit is not None and (args.matched_run or args.safe_efficient_backbone):
        global_sampler = DatasetCoverageBatchSampler(
            raw_sampler, loaded_train.dataset.sequences,
            exact_batch_limit, int(config.training.seed),
        )
    elif exact_batch_limit is not None:
        global_sampler = LimitedBatchSampler(global_sampler, exact_batch_limit)
    elif args.max_steps_per_epoch > 0:
        global_sampler = LimitedBatchSampler(
            global_sampler, args.max_steps_per_epoch
        )

    global_sampler.set_epoch(0)
    first_epoch_batches = list(global_sampler)
    plan_validation = validate_batch_plan(first_epoch_batches, raw_sampler)
    train_plan_summary = summarize_batch_plan(
        first_epoch_batches, loaded_train.dataset.sequences
    )
    train_plan_summary["validation"] = plan_validation
    eligible_datasets = list(
        getattr(global_sampler, "eligible_datasets", [])
    )
    if eligible_datasets:
        selected_datasets = set(train_plan_summary["dataset_counts"])
        train_plan_summary["eligible_dataset_count"] = len(eligible_datasets)
        train_plan_summary["missing_eligible_datasets"] = sorted(
            set(eligible_datasets) - selected_datasets
        )
    if (
        args.samples_per_epoch > 0
        and train_plan_summary["training_examples"] != args.samples_per_epoch
    ):
        raise RuntimeError(
            "Sampler produced "
            f"{train_plan_summary['training_examples']} examples, expected "
            f"{args.samples_per_epoch}. Choose a target compatible with the "
            "batch size and dimension ratio."
        )
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
    if args.matched_run:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(config.training.seed) + 12345)
        kwargs["generator"] = train_generator
    train_loader = DataLoader(**kwargs)

    val_count = min(max(0, args.val_samples), len(loaded_val.dataset))
    val_plan_summary = None
    if val_count:
        val_selector = BalancedTaskBatchSampler(
            loaded_val.dataset.sequences,
            tasks_map=tasks_map,
            batch_size=1,
            drop_last=False,
            seed=int(config.training.seed),
            num_samples=val_count,
            min_per_bucket=1,
            target_dim_ratio=target_dim_ratio,
        )
        if args.matched_run or args.safe_efficient_backbone:
            val_selector.set_epoch(0)
            val_batches = list(val_selector)
            val_plan_summary = summarize_batch_plan(
                val_batches, loaded_val.dataset.sequences
            )
            val_batch_sampler = FixedBatchSampler(val_batches, batch_size=1)
        else:
            val_batch_sampler = val_selector
        val_loader = DataLoader(
            loaded_val.dataset,
            batch_sampler=val_batch_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_sam2_batch,
        )
    else:
        val_loader = None
    sampling_summary = {
        "seed": int(config.training.seed),
        "requested_examples_per_epoch": args.samples_per_epoch or None,
        "sampler_candidate_budget": sampler_num_samples,
        "changes_each_epoch": True,
        "task_aware_balancing": True,
        "matched_run": bool(args.matched_run),
        "data_loader_seed": (
            int(config.training.seed) + 12345 if args.matched_run else None
        ),
        "target_dimension_ratio": target_dim_ratio,
        "epoch_0_training_plan": train_plan_summary,
        "fixed_validation_plan": val_plan_summary,
    }
    return train_loader, val_loader, loaded_test, sampling_summary


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


def load_initial_model_state(path: str, student):
    """Load an exact prepared model initialization before optimizer creation."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = student.load_state_dict(model_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Initial model state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return state.get("metadata", {}) if isinstance(state, dict) else {}


def reset_runtime_seed(seed: int) -> None:
    """Reset main-process stochastic state for matched trajectories."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    if args.wkv_t_max < 256:
        raise ValueError("--wkv-t-max must be at least 256 for 512x512 VCR inputs")
    if args.samples_per_epoch and args.max_steps_per_epoch:
        raise ValueError(
            "Use either --samples-per-epoch or --max-steps-per-epoch, not both"
        )
    if args.samples_per_epoch < 0:
        raise ValueError("--samples-per-epoch cannot be negative")
    if args.samples_per_epoch and args.samples_per_epoch % args.global_batch_size:
        raise ValueError("--samples-per-epoch must be divisible by the global batch size")
    if (
        args.pilot
        and args.samples_per_epoch
        and args.samples_per_epoch % (2 * args.global_batch_size)
    ):
        raise ValueError(
            "With a 50/50 2D/3D plan, --samples-per-epoch must be divisible "
            "by twice the global batch size"
        )
    safe_training = safe_training_requested(args)
    efficient_backbone = efficient_backbone_requested(args)
    if efficient_backbone and args.mode == "base":
        raise ValueError("Efficient VCR backbone flags are not valid for Base")
    if (args.optimized or safe_training or efficient_backbone) and args.mode != "base":
        # VCR is imported lazily when Hydra constructs the model, so this
        # compile-time CUDA specialization is set before the extension loads.
        os.environ.setdefault("VRWKV_T_MAX", str(args.wkv_t_max))

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
    run_name = (
        f"{args.mode}/{run_id}"
        if args.variant == "standard"
        else f"{args.variant}/{args.mode}/{run_id}"
    )
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
        if args.optimized or safe_training or efficient_backbone:
            # The image size and encoder chunk sizes are stable, so cuDNN can
            # select faster convolution kernels during discarded warm-up work.
            torch.backends.cudnn.benchmark = True
        train_loader, val_loader, test_loader, sampling_summary = make_loaders(
            config, args, context
        )
        train_batches_per_epoch = int(sampling_summary["epoch_0_training_plan"]["batches"])
        logger.info(
            "mode=%s run_id=%s rank=%d/%d device=%s global_batch=%d local_batch=%d "
            "train_batches=%d val_batches=%d optimized_predictor=%s "
            "safe_training=%s efficient_backbone=%s variant=%s",
            args.mode, run_id, context.rank, context.world_size, device,
            args.global_batch_size, args.global_batch_size // context.world_size,
            train_batches_per_epoch, len(val_loader) if val_loader is not None else 0,
            args.optimized, safe_training, efficient_backbone, args.variant,
        )

        if context.is_main:
            atomic_json(log_dir / "sampling_plan_summary.json", sampling_summary)
            writer.add_text(
                "run/sampling_plan", json.dumps(sampling_summary, indent=2), 0
            )
            train_plan = sampling_summary["epoch_0_training_plan"]
            logger.info(
                "sampling examples=%d batches=%d dimensions=%s datasets=%d "
                "unique=%d plan_sha256=%s",
                train_plan["training_examples"], train_plan["batches"],
                train_plan["dimension_counts"], len(train_plan["dataset_counts"]),
                train_plan["unique_dataset_indices"], train_plan["plan_sha256"],
            )
        if args.preflight_only:
            if context.is_main:
                atomic_json(status_path, {
                    "state": "preflight_complete", "mode": args.mode,
                    "run_id": run_id, "sampling": sampling_summary,
                })
                logger.info("preflight complete; no model or optimizer was created")
            return

        student = build_student_predictor(config, logger=logger if context.is_main else None).to(device)
        expected_predictor = (
            "OptimizedSAM2VideoPredictor"
            if args.optimized
            else "SAM2VideoPredictor"
        )
        if (
            (args.optimized or safe_training or efficient_backbone)
            and type(student).__name__ != expected_predictor
        ):
            raise TypeError(
                f"Matched path expected {expected_predictor}; got "
                f"{type(student).__name__}"
            )
        initial_state_metadata = {}
        if args.initial_model_state:
            initial_path = Path(args.initial_model_state).resolve()
            if not initial_path.is_file():
                raise FileNotFoundError(initial_path)
            initial_state_metadata = load_initial_model_state(
                str(initial_path), student
            )
            logger.info(
                "loaded prepared initialization=%s shared_tensors=%s",
                initial_path, initial_state_metadata.get("shared_tensor_count"),
            )
        teacher = build_teacher_predictor(config).to(device) if distillation_enabled(config) else None
        bundle = setup_optimizer_and_scheduler(student, config)
        optimizer, scheduler = bundle["optimizer"], bundle["scheduler"]

        start_epoch = start_batch = global_step = 0
        if args.resume:
            start_epoch, start_batch, global_step = restore_checkpoint(
                args.resume, student, optimizer, scheduler
            )
            if context.is_main:
                # Discard only unsaved TensorBoard steps beyond the restored
                # checkpoint, so recovery produces one unambiguous loss curve.
                writer.close()
                writer = SummaryWriter(
                    log_dir=str(log_dir), flush_secs=30, max_queue=100,
                    purge_step=global_step + 1,
                )
            logger.info(
                "resumed=%s epoch=%d batch=%d global_step=%d",
                args.resume, start_epoch, start_batch, global_step,
            )

        manager = CheckpointManager(str(checkpoint_dir)) if context.is_main else None
        if context.is_main:
            metadata = {
                "mode": args.mode,
                "variant": args.variant,
                "run_id": run_id,
                "host": socket.gethostname(),
                "world_size": context.world_size,
                "global_batch_size": args.global_batch_size,
                "num_workers_per_rank": args.num_workers,
                "max_steps_per_epoch": args.max_steps_per_epoch,
                "samples_per_epoch": args.samples_per_epoch,
                "val_samples": args.val_samples,
                "pilot": args.pilot,
                "optimized": args.optimized,
                "cpu_threads_per_rank": torch.get_num_threads(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                "safe_training_optimizations": safe_training,
                "efficient_vcr_backbone": efficient_backbone,
                "efficient_vcr_architecture": getattr(
                    args, "efficient_vcr_architecture", "efficient"
                ),
                "matched_run": args.matched_run,
                "match_common_sam2_settings": bool(
                    args.match_common_sam2_settings
                ),
                "ignore_pretrained_image_encoder": bool(
                    args.ignore_pretrained_image_encoder
                ),
                "initial_model_state": args.initial_model_state or None,
                "resume_checkpoint": args.resume or None,
                "initial_state_metadata": initial_state_metadata,
                "predictor_class": type(student).__name__,
                "runtime_model_settings": {
                    "use_mask_input_as_output_without_sam": bool(
                        student.use_mask_input_as_output_without_sam
                    ),
                    "world_size": context.world_size,
                    "global_batch_size": args.global_batch_size,
                    "local_batch_size": args.global_batch_size // context.world_size,
                },
                "sampling": sampling_summary,
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
                writer.add_scalar("progress/overall_percent", 100.0 * step / total_steps, step)
                writer.add_scalar("progress/epoch", epoch + 1, step)
                writer.add_scalar("progress/completed_steps", step, step)
                writer.flush()
                atomic_json(status_path, {
                    "state": "training", "mode": args.mode, "run_id": run_id,
                    "epoch": int(epoch), "batch_idx": int(batch_idx),
                    "global_step": int(step), "samples_per_second": samples_per_second,
                    "eta_hours": eta_hours,
                    "total_steps": int(total_steps),
                    "total_epochs": int(args.epochs),
                    "overall_percent": 100.0 * step / total_steps,
                    "world_size": context.world_size,
                    "runtime_mask_bypass": bool(student.use_mask_input_as_output_without_sam),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                last_monitor, last_monitor_step = now, step

            recovery_seconds = max(0.0, args.recovery_minutes) * 60.0
            if context.is_main and recovery_seconds > 0 and (step == 1 or now - last_recovery >= recovery_seconds):
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
        best_path = checkpoint_dir / "best.pth"
        if context.is_main and args.resume and best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            saved_metric = best_state.get("metric")
            if saved_metric is not None:
                best_metric = float(saved_metric)
                logger.info("restored best validation metric=%.6f", best_metric)
            del best_state
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
            if args.matched_run:
                reset_runtime_seed(int(config.training.seed) + 10000 * epoch)
            if epoch_start_batch >= train_batches_per_epoch:
                # A pre-validation recovery already includes scheduler.step().
                # Retry validation without advancing the schedule a second time.
                logger.info("epoch=%d training already complete; retrying validation", epoch)
            else:
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

                if args.matched_run:
                    reset_runtime_seed(
                        int(config.training.seed) + 900000 + int(epoch)
                    )
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
            if args.matched_run:
                reset_runtime_seed(int(config.training.seed) + 1900000)
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
