# Training entry point and loop utilities for RWKV-MedSAM2.
#
# Handles config loading, data loader construction, student/teacher predictor
# setup, checkpointing, optimization, training, validation, and final testing.
import os
import json
import math
import random
import logging
import argparse
import pickle
import copy
import time
import html
import re
from collections import defaultdict

from tqdm import tqdm
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

import cupy as cp  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from ext.sam2.build_sam import build_sam2_video_predictor

from .functions.func_2d import train_step_2d, validate_step_2d
from .functions.func_3d import train_step_3d, validate_step_3d
from .functions.func_metrics import (
    TaskAggregator,
    ModalityAggregator,
    dice_iou,
    ece,
    fg_bin,
    sigmoid_np,
    try_auc,
)

from .dataset import BalancedTaskBatchSampler
from .utils.vis import save_vis_gif, frames_to_tb_video
from .utils.preprocessing import load_datasets

def load_config(config_path):
    """
    Loads a configuration file from a given path and resolves any variable interpolation.
    If the config contains a 'seed' field under 'training', sets the Python random seeds for reproducibility.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        OmegaConf.DictConfig: The resolved configuration.
    """
    config = OmegaConf.load(config_path)
    # Resolve any variable interpolation
    OmegaConf.resolve(config)

    # Set Python random seeds for reproducibility
    seed = getattr(config.training, "seed", 42)
    if seed is not None:
        random.seed(seed)
        cp.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return config

def setup_logger(log_cfg):
    """
    Sets up a logger for this module that logs to both the console and a timed rotating log file.

    Args:
        log_cfg (OmegaConf.DictConfig): A configuration containing the following fields:
            - output_dir (str):   The directory where the logs will be saved.
            - log_filename (str): The filename of the log, which will be prepended with the current date and appended with '.log'.
            - level (str):        The logging level to use for console logging. Should be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
            - file_level (str):   The logging level to use for file logging. Should be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
            - interval (int):     The interval at which the log file will be rotated, in minutes.
            - backups (int):      The number of log files to keep before deleting the oldest one.

    Returns:
        logger (logging.Logger): The configured logger.
    """
    # Ensure the output directory exists
    os.makedirs(log_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(log_cfg.output_dir, log_cfg.filename)

    # Set up the logger
    logger = logging.getLogger("RWKV-MedSAM2")
    logger.setLevel(getattr(logging, log_cfg.level.upper()))

    # Stream (console) handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_cfg.level.upper()))

    # File handler
    fh = TimedRotatingFileHandler(
        filename=log_path,
        when="M",
        interval=int(log_cfg.interval),
        backupCount=int(log_cfg.backups),
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, log_cfg.file_level.upper()))

    # Formatter
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)

    # Set formatters
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Print info
    logger.info(f"Logging to console at {log_cfg.level} and to {log_path} at {log_cfg.file_level}")

    return logger

class CheckpointManager:
    """
    Manage latest, best, and epoch-numbered training checkpoints.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self, ckpt_dir):
        """
        Initialize checkpoint storage.

        Args:
            ckpt_dir (str): Directory where checkpoints are written.

        Returns:
            None.
        """
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_metric = float("-inf")

    def _unwrap_model(self, model):
        """
        Return the underlying model when wrapped.

        Args:
            model (nn.Module): Model or wrapper with a ``module`` attribute.

        Returns:
            nn.Module: Unwrapped model when available.
        """
        return model.module if hasattr(model, "module") else model

    def _state_dict(self, model, optimizer, scheduler, scaler, epoch, global_step, metric=None):
        """
        Build a serializable checkpoint state.

        Args:
            model (nn.Module): Model to serialize.
            optimizer (torch.optim.Optimizer | None): Optimizer state source.
            scheduler (object | None): Scheduler state source.
            scaler (object | None): AMP scaler state source.
            epoch (int): Epoch index.
            global_step (int): Global step count.
            metric (float | None): Optional validation metric.

        Returns:
            dict: Checkpoint payload.
        """
        return {
            "model": self._unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metric": None if metric is None else float(metric),
        }

    def _save(self, filename, model, optimizer, scheduler, scaler, epoch, global_step, metric=None):
        """
        Write a checkpoint file to disk.

        Args:
            filename (str): Output filename within the checkpoint directory.
            model (nn.Module): Model to serialize.
            optimizer (torch.optim.Optimizer | None): Optimizer state source.
            scheduler (object | None): Scheduler state source.
            scaler (object | None): AMP scaler state source.
            epoch (int): Epoch index.
            global_step (int): Global step count.
            metric (float | None): Optional validation metric.

        Returns:
            str: Saved checkpoint path.
        """
        path = os.path.join(self.ckpt_dir, filename)
        state = self._state_dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            metric=metric,
        )
        torch.save(state, path)
        return path

    def save_latest(self, model, optimizer, scheduler, scaler, epoch, global_step):
        """
        Save the rolling latest checkpoint.

        Args:
            model (nn.Module): Model to serialize.
            optimizer (torch.optim.Optimizer | None): Optimizer state source.
            scheduler (object | None): Scheduler state source.
            scaler (object | None): AMP scaler state source.
            epoch (int): Epoch index.
            global_step (int): Global step count.

        Returns:
            str: Saved checkpoint path.
        """
        return self._save(
            filename="latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            metric=None,
        )

    def save_best(self, model, optimizer, scheduler, scaler, epoch, metric, global_step):
        """
        Save a best checkpoint if ``metric`` improves.

        Args:
            model (nn.Module): Model to serialize.
            optimizer (torch.optim.Optimizer | None): Optimizer state source.
            scheduler (object | None): Scheduler state source.
            scaler (object | None): AMP scaler state source.
            epoch (int): Epoch index.
            metric (float): Candidate best metric.
            global_step (int): Global step count.

        Returns:
            str | None: Saved path when improved, otherwise None.
        """
        if float(metric) > self.best_metric:
            self.best_metric = float(metric)
            return self._save(
                filename="best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metric=metric,
            )
        return None

    def save_epoch(self, model, optimizer, scheduler, scaler, epoch, global_step, metric=None):
        """
        Save an epoch-numbered checkpoint.

        Args:
            model (nn.Module): Model to serialize.
            optimizer (torch.optim.Optimizer | None): Optimizer state source.
            scheduler (object | None): Scheduler state source.
            scaler (object | None): AMP scaler state source.
            epoch (int): Epoch index.
            global_step (int): Global step count.
            metric (float | None): Optional validation metric.

        Returns:
            str: Saved checkpoint path.
        """
        return self._save(
            filename=f"epoch_{int(epoch):04d}.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            metric=metric,
        )

    def load(self, path, model, map_location="cpu"):
        """
        Load model weights and return optimizer/scheduler resume state.

        Args:
            path (str): Checkpoint path.
            model (nn.Module): Model receiving the saved state dict.
            map_location (str | torch.device): Device mapping for ``torch.load``.

        Returns:
            tuple[int, dict | None, dict | None, int]: Next epoch, optimizer state,
                scheduler state, and global step.
        """
        state = torch.load(path, map_location=map_location)
        self._unwrap_model(model).load_state_dict(state["model"])
        metric = state.get("metric", None)
        if metric is not None:
            self.best_metric = float(metric)
        return (
            int(state.get("epoch", -1)) + 1,
            state.get("optimizer", None),
            state.get("scheduler", None),
            int(state.get("global_step", 0)),
        )

def collate_sam2_batch(samples, allow_trim_2d=False):
    """
    Collate homogeneous SAM2 sequence samples into a batch.

    The collate step stacks image and mask tensors, converts train-time prompts
    from sample-major to time-major layout, and preserves per-sample validation
    prompt caches.

    Args:
        samples (list[dict]): Dataset samples from ``SegmentationSequenceDataset``.
        allow_trim_2d (bool): Whether to trim 2D samples to the shortest sequence.

    Returns:
        dict: Batched tensors, prompts, metadata, and validation prompt caches.
    """
    if len(samples) == 0:
        raise ValueError("Empty batch.")

    dims = []
    t_list = []
    c_list = []
    lead_axis_flags = []

    for s in samples:
        img = s["image"]
        if img.ndim == 4:  # [T,C,H,W]
            lead_axis_flags.append(0)
            t_list.append(int(img.shape[0]))
            c_list.append(int(img.shape[1]))
        elif img.ndim == 5 and img.shape[0] == 1:  # [1,T,C,H,W]
            lead_axis_flags.append(1)
            t_list.append(int(img.shape[1]))
            c_list.append(int(img.shape[2]))
        else:
            raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")

        if 'dim' in s:
            dims.append(int(s['dim']))
        else:
            dims.append(3 if ('axis_lengths' in s or 'spacing' in s) else 2)

    if len(set(dims)) != 1:
        raise ValueError(f"Mixed 2D/3D samples in one batch: {dims}")
    data_dim = dims[0]

    if data_dim == 3:
        if len(set(t_list)) != 1:
            raise ValueError(f"3D batches require equal T per sample; got {t_list}")
        T = t_list[0]
    else:
        if allow_trim_2d:
            T = min(t_list)
        else:
            if len(set(t_list)) != 1:
                raise ValueError(f"2D batches require equal T when trimming is disabled; got {t_list}")
            T = t_list[0]

    out_c = 3 if any(c == 3 for c in c_list) else 1

    def _slice_to_T(x, flag):
        """Trim a tensor to the selected sequence length."""
        if flag == 0:
            y = x[:T]
        else:
            y = x[:, :T]
        if y.ndim == 5 and y.shape[0] == 1:
            y = y.squeeze(0)
        return y

    def _to_channels(y):
        """Match image channels to the batch output channel count."""
        if y.shape[1] == out_c:
            return y
        if y.shape[1] == 1 and out_c == 3:
            return y.repeat(1, 3, 1, 1)
        if y.shape[1] == 3 and out_c == 1:
            return y.mean(dim=1, keepdim=True)
        raise ValueError(f"Cannot convert channels {y.shape[1]} -> {out_c}")

    imgs_buf, masks_buf = [], []
    ds_names, subds_names, spacings = [], [], []
    task_ids, task_labels = [], []
    modalities = []

    for s, flag in zip(samples, lead_axis_flags):
        img = _slice_to_T(s['image'], flag)
        msk = s['mask']
        if msk.ndim == 5 and msk.shape[0] == 1:
            msk = msk[:, :T].squeeze(0)
        else:
            msk = msk[:T]
        if msk.ndim == 3:
            msk = msk.unsqueeze(1)
        if msk.shape[1] != 1:
            msk = msk[:, :1]

        img = _to_channels(img)

        imgs_buf.append(img)
        masks_buf.append(msk)

        if 'dataset' in s:
            ds_names.append(s['dataset'])
        if 'subdataset' in s:
            subds_names.append(s['subdataset'])
        if 'task_id' in s:
            task_ids.append(s['task_id'])
        if 'task_label' in s:
            task_labels.append(s['task_label'])
        if 'modality' in s:
            modalities.append(s['modality'])

        sp = s.get('spacing', None)
        if sp is not None:
            if torch.is_tensor(sp):
                spacings.append(sp.reshape(-1))
            elif isinstance(sp, (list, tuple)):
                spacings.append(torch.as_tensor(sp, dtype=torch.float32).reshape(-1))
            else:
                spacings.append(sp)

    images = torch.stack(imgs_buf, dim=0)
    masks = torch.stack(masks_buf, dim=0)
    B = len(samples)

    def _tm_points():
        """Convert point prompts to time-major layout."""
        out = []
        for t in range(T):
            row = []
            for b in range(B):
                per_seq = samples[b].get('pt_list', None)
                x = None if (per_seq is None or t >= len(per_seq)) else per_seq[t]
                if x is None or (torch.is_tensor(x) and x.numel() == 0):
                    row.append(torch.empty((0, 2), dtype=torch.int64))
                else:
                    row.append(x)
            out.append(row)
        return out

    def _tm_labels():
        """Convert point labels to time-major layout."""
        out = []
        for t in range(T):
            row = []
            for b in range(B):
                per_seq = samples[b].get('p_label', None)
                x = None if (per_seq is None or t >= len(per_seq)) else per_seq[t]
                if x is None or (torch.is_tensor(x) and x.numel() == 0):
                    row.append(torch.empty((0,), dtype=torch.int64))
                else:
                    row.append(x)
            out.append(row)
        return out

    def _tm_boxes():
        """Convert box prompts to time-major layout."""
        out = []
        for t in range(T):
            row = []
            for b in range(B):
                per_seq = samples[b].get('bbox', None)
                x = None if (per_seq is None or t >= len(per_seq)) else per_seq[t]
                row.append(x if x is not None else None)
            out.append(row)
        return out

    def _tm_mask_prompts():
        """Convert mask prompts to time-major layout."""
        out = []
        for t in range(T):
            row = []
            for b in range(B):
                per_seq = samples[b].get('m_prompt', None)
                x = None if (per_seq is None or t >= len(per_seq)) else per_seq[t]
                if x is None:
                    row.append(None)
                else:
                    try:
                        if hasattr(x, "numel") and x.numel() == 0:
                            row.append(None)
                        else:
                            if x.ndim == 2:
                                x = x.unsqueeze(0)
                            row.append(x)
                    except Exception:
                        row.append(None)
            out.append(row)
        return out

    out = {
        'image':            images,
        'mask':             masks,
        'pt_list':          _tm_points(),
        'p_label':          _tm_labels(),
        'bbox':             _tm_boxes(),
        'm_prompt':         _tm_mask_prompts(),
        'val_prompt_cache': [s.get('val_prompt_cache', None) for s in samples],
        'dim':              torch.as_tensor([data_dim] * B, dtype=torch.int64),
    }

    if ds_names:
        out['dataset'] = ds_names
    if subds_names:
        out['subdataset'] = subds_names
    if spacings:
        try:
            out['spacing'] = torch.stack(spacings, dim=0)
        except Exception:
            out['spacing'] = spacings
    if task_ids:
        out['task_id'] = task_ids
    if task_labels:
        out['task_label'] = task_labels
    if modalities:
        out['modality'] = modalities

    return out

def get_data_loaders(config):
    """
    Build train, validation, and test data loaders from cached datasets.

    Args:
        config (OmegaConf.DictConfig): Training configuration with cache, DRIPP,
            sampler, and dataloader settings.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders.
    """
    # Choose where to keep the cached, preformatted datasets
    cache_root = config.cache.path

    # 1) Get (or build) SegmentationSequenceDataset objects directly
    train_ds, val_ds, test_ds = load_datasets(config=config, cache_root=cache_root)

    # 2) Same loaders / sampler as before
    tasks_map = json.load(open(config.dripp.tasks_file, "r"))
    sampler = BalancedTaskBatchSampler(train_ds.sequences, tasks_map=tasks_map, batch_size=config.training.batch_size, drop_last=True, seed=int(getattr(config.training, "seed", 42)))

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,  # (no shuffle)
        num_workers=config.training.num_workers,
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_sam2_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_sam2_batch,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_sam2_batch,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader

def save_data_loaders(save_path, train_loaders, val_loaders, test_loaders):
    """
    Save train, validation, and test DataLoaders to disk via pickle.

    Args:
        save_path (str): Path to the output .pkl file.
        train_loaders (DataLoader): Training DataLoader.
        val_loaders (DataLoader): Validation DataLoader.
        test_loaders (DataLoader): Test DataLoader.

    Returns:
        None.
    """
    # Ensure target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Serialize all three loaders in one file
    with open(save_path, "wb") as f:
        pickle.dump({
            "train_loaders": train_loaders,
            "val_loaders":   val_loaders,
            "test_loaders":  test_loaders,
        }, f)

def load_data_loaders(load_path):
    """
    Load train, validation, and test DataLoaders from a pickle file.

    Args:
        load_path (str): Path to the .pkl file created by save_data_loaders.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders.
    """
    with open(load_path, "rb") as f:
        data = pickle.load(f)

    return (
        data["train_loaders"],
        data["val_loaders"],
        data["test_loaders"],
    )

def _partial_load_sam21(model, ckpt_path, ignore_prefixes=(), logger=None):
    """
    Partially load compatible tensors from a SAM2.1 checkpoint.

    Args:
        model (nn.Module): Model receiving compatible checkpoint tensors.
        ckpt_path (str | None): Checkpoint path.
        ignore_prefixes (tuple[str, ...]): State-dict key prefixes to skip.
        logger (logging.Logger | None): Optional logger for load statistics.

    Returns:
        dict[str, int]: Counts of loaded and skipped checkpoint tensors.
    """
    if ckpt_path is None:
        return {"loaded": 0, "skipped_prefix": 0, "skipped_missing": 0, "skipped_shape": 0}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    msd = model.state_dict()
    load_sd = {}
    skipped_prefix = 0
    skipped_missing = 0
    skipped_shape = 0

    for k, v in sd.items():
        if any(k.startswith(p) for p in ignore_prefixes):
            skipped_prefix += 1
            continue
        if k not in msd:
            skipped_missing += 1
            continue
        if tuple(msd[k].shape) != tuple(v.shape):
            skipped_shape += 1
            continue
        load_sd[k] = v

    missing, unexpected = model.load_state_dict(load_sd, strict=False)

    if logger is not None:
        logger.info(
            f"[partial_load] ckpt={ckpt_path} loaded={len(load_sd)} "
            f"skipped_prefix={skipped_prefix} skipped_missing={skipped_missing} skipped_shape={skipped_shape} "
            f"(strict=False missing={len(missing)} unexpected={len(unexpected)})"
        )

    return {
        "loaded": len(load_sd),
        "skipped_prefix": skipped_prefix,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "missing_after": len(missing),
        "unexpected_after": len(unexpected),
    }

def _set_requires_grad(module: nn.Module, flag: bool):
    """
    Set ``requires_grad`` for all parameters in a module.

    Args:
        module (nn.Module): Module to update.
        flag (bool): Desired ``requires_grad`` value.

    Returns:
        None.
    """
    for p in module.parameters():
        p.requires_grad = flag

def _apply_freeze_schedule(model, config, epoch, logger=None):
    """
    Apply the configured pretrained-module freeze schedule for an epoch.

    Args:
        model (nn.Module): Model containing modules listed in the config.
        config (OmegaConf.DictConfig): Training configuration.
        epoch (int): Current epoch index.
        logger (logging.Logger | None): Optional logger.

    Returns:
        None.
    """
    freeze_epochs = int(getattr(getattr(config, "init", {}), "freeze_pretrained_epochs", 0))
    freeze_modules = list(getattr(getattr(config, "init", {}), "freeze_modules", []))

    if freeze_epochs <= 0 or len(freeze_modules) == 0:
        return

    do_freeze = epoch < freeze_epochs
    for name in freeze_modules:
        if not hasattr(model, name):
            continue
        _set_requires_grad(getattr(model, name), not do_freeze)

    if logger is not None and epoch in (0, freeze_epochs):
        logger.info(
            f"[freeze_schedule] epoch={epoch} "
            f"{'FREEZING' if do_freeze else 'UNFREEZING'} modules={freeze_modules}"
        )

def build_student_predictor(config, logger=None):
    """
    Build the train-mode student SAM2 video predictor.

    Args:
        config (OmegaConf.DictConfig): Model and training configuration.
        logger (logging.Logger | None): Optional logger for partial-load stats.

    Returns:
        SAM2VideoPredictor: Student predictor in train mode.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config._config_path,
        ckpt_path=None,  # IMPORTANT: build without strict checkpoint loading
        device=config.training.device,
        mode="train",
        apply_postprocessing=False,
    )
    video_predictor.train()

    # Enable/disable obj-ptrs from config
    use_obj_ptrs = bool(getattr(getattr(config, "init", {}), "use_obj_ptrs_in_encoder", True))
    video_predictor.use_obj_ptrs_in_encoder = use_obj_ptrs

    # Partial-load SAM2.1 tiny weights (everything compatible)
    ckpt_path = getattr(getattr(config, "init", {}), "student_ckpt_path", None)
    ignore_prefixes = tuple(getattr(getattr(config, "init", {}), "ignore_prefixes", []))
    if ckpt_path:
        _partial_load_sam21(video_predictor, ckpt_path, ignore_prefixes=ignore_prefixes, logger=logger)

    return video_predictor

def build_teacher_predictor(config):
    """
    Builds the teacher video predictor for SAM2 training.
    The teacher model is used for distillation.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        video_predictor (SAM2VideoPredictor): The teacher video predictor.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config.teacher._config_path,
        ckpt_path=config.teacher.ckpt_path,
        device=config.training.device,
        mode='eval',
        apply_postprocessing=False,
    )

    def _unwrap_state_dict(obj):
        """Extract a state dict from common checkpoint wrappers."""
        if isinstance(obj, dict):
            for k in ["model", "state_dict", "model_state_dict", "net", "network"]:
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
        return obj if isinstance(obj, dict) else None

    def _strip_prefix(state, prefix):
        """Strip a shared state-dict key prefix."""
        if not isinstance(state, dict) or not state:
            return state
        keys = list(state.keys())
        if all(isinstance(k, str) and k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state.items()}
        return state

    def _pick_module(p):
        """Pick the load target module from a predictor-like object."""
        # predictors often wrap the underlying nn.Module in `.model`
        for attr in ["model", "sam2", "net", "network"]:
            if hasattr(p, attr) and isinstance(getattr(p, attr), torch.nn.Module):
                return getattr(p, attr)
        if isinstance(p, torch.nn.Module):
            return p
        return None

    try:
        ckpt_path = str(config.teacher.ckpt_path)
        ckpt_obj  = torch.load(ckpt_path, map_location="cpu")
        state     = _unwrap_state_dict(ckpt_obj)
        state     = _strip_prefix(state, "module.")
        state     = _strip_prefix(state, "model.")

        target = _pick_module(video_predictor)
        if target is None or state is None:
            logging.getLogger(__name__).warning(
                f"[Teacher] Could not explicitly load state_dict from {ckpt_path}. "
                "Teacher may be partially loaded by build_sam2_video_predictor only."
            )
        else:
            missing, unexpected = target.load_state_dict(state, strict=False)
            n_total = len(target.state_dict())
            n_missing = len(missing)
            loaded_ratio = 0.0 if n_total == 0 else float(n_total - n_missing) / float(n_total)

            logging.getLogger(__name__).info(
                f"[Teacher] Explicit load from {ckpt_path} strict=False: "
                f"missing={len(missing)} unexpected={len(unexpected)} loaded_ratio={loaded_ratio:.3f}"
            )

            # If this is low, your teacher is not your refined model in practice.
            if loaded_ratio < 0.80:
                logging.getLogger(__name__).warning(
                    "[Teacher] Loaded ratio is low. This usually means the teacher config "
                    "does not match the checkpoint architecture, or the checkpoint keys are incompatible."
                )
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[Teacher] Explicit checkpoint load failed: {repr(e)}"
        )


    video_predictor.eval()
    video_predictor.use_obj_ptrs_in_encoder = True
    return video_predictor

def _trainable(params):
    """
    Filter parameters to those that are trainable.

    Args:
        params (Iterable[nn.Parameter]): Candidate parameters.

    Returns:
        list[nn.Parameter]: Parameters with ``requires_grad=True``.
    """
    return [p for p in params if p.requires_grad]

def _unwrap_model(model):
    """
    Return the underlying nn.Module if wrapped in DataParallel / DDP.

    Args:
        model (nn.Module): Model or wrapper.

    Returns:
        nn.Module: Unwrapped model.
    """
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        return model.module
    return model

def setup_optimizer_and_scheduler(model, config):
    """
    Create AdamW parameter groups and a warmup/cosine scheduler.

    Args:
        model (nn.Module): Student predictor, optionally DataParallel-wrapped.
        config (OmegaConf.DictConfig): Training configuration with LR settings.

    Returns:
        dict: Dictionary containing ``optimizer`` and ``scheduler``.
    """
    model = _unwrap_model(model)

    groups = [
        {
            "params": _trainable(
                list(model.image_encoder.parameters()) +
                list(model.sam_prompt_encoder.parameters())
            ),
            "lr": float(getattr(config.training, "lr_backbone", config.training.lr)),
        },
        {
            "params": _trainable(list(model.sam_mask_decoder.parameters())),
            "lr": float(config.training.mask_lr),
        },
        {
            "params": _trainable(
                list(model.obj_ptr_proj.parameters()) +
                list(model.memory_encoder.parameters()) +
                list(model.memory_attention.parameters()) +
                list(model.mask_downsample.parameters())
            ),
            "lr": float(config.training.mem_lr),
        },
    ]

    groups = [g for g in groups if len(g["params"]) > 0]

    opt = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.999),
        weight_decay=float(config.training.weight_decay),
        fused=True,
    )

    total_epochs = int(config.training.epochs)
    warmup_epochs = int(config.training.warmup_epochs)

    def lr_lambda(epoch):
        """Return the warmup/cosine learning-rate multiplier."""
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return {"optimizer": opt, "scheduler": sched}

def _rebuild_optimizer_and_scheduler_after_unfreeze(model, config, old_optimizer, old_scheduler, logger=None):
    """
    Rebuild optimizer and scheduler after modules are unfrozen.

    Args:
        model (nn.Module): Model with newly trainable parameters.
        config (OmegaConf.DictConfig): Training configuration.
        old_optimizer (torch.optim.Optimizer): Previous optimizer.
        old_scheduler (torch.optim.lr_scheduler.LRScheduler): Previous scheduler.
        logger (logging.Logger | None): Optional logger.

    Returns:
        dict: Rebuilt ``optimizer`` and ``scheduler`` preserving compatible state.
    """
    pack = setup_optimizer_and_scheduler(model, config)
    new_optimizer = pack["optimizer"]
    new_scheduler = pack["scheduler"]

    transferred_state_count = 0

    # Preserve optimizer state for parameters that already existed in the old optimizer
    for group in new_optimizer.param_groups:
        for p in group["params"]:
            if p in old_optimizer.state:
                new_optimizer.state[p] = copy.deepcopy(old_optimizer.state[p])
                transferred_state_count += 1

    # Preserve current LR / initial LR for each param group
    for i, new_group in enumerate(new_optimizer.param_groups):
        if i >= len(old_optimizer.param_groups):
            continue
        old_group = old_optimizer.param_groups[i]
        new_group["lr"] = old_group["lr"]
        if "initial_lr" in old_group:
            new_group["initial_lr"] = old_group["initial_lr"]

    # Preserve scheduler position so warmup/cosine continues from the current epoch
    try:
        new_scheduler.load_state_dict(old_scheduler.state_dict())
    except Exception as e:
        if logger is not None:
            logger.warning(
                f"[freeze_schedule] failed to load old scheduler state into rebuilt scheduler: {repr(e)}"
            )

    # Re-apply current group LRs after loading scheduler state, just to be explicit
    for i, new_group in enumerate(new_optimizer.param_groups):
        if i >= len(old_optimizer.param_groups):
            continue
        new_group["lr"] = old_optimizer.param_groups[i]["lr"]
        if "initial_lr" in old_optimizer.param_groups[i]:
            new_group["initial_lr"] = old_optimizer.param_groups[i]["initial_lr"]

    if logger is not None:
        old_last_epoch = getattr(old_scheduler, "last_epoch", None)
        logger.info(
            "[freeze_schedule] rebuilt optimizer/scheduler after unfreeze "
            f"(preserved scheduler.last_epoch={old_last_epoch}, "
            f"transferred_optimizer_states={transferred_state_count})"
        )

    return {"optimizer": new_optimizer, "scheduler": new_scheduler}

def _dim_from_batch(b):
    """
    Extract data dimensionality from a batch dictionary.

    Args:
        b (dict): Batch dictionary containing ``dim`` or a compatible alias.

    Returns:
        int | None: Data dimensionality, or None when it cannot be inferred.
    """
    for k in ["dim", "Dim", "dimension", "is_3d"]:
        if k in b:
            v = b[k]
            if isinstance(v, bool):
                return 3 if v else 2
            if torch.is_tensor(v):
                try:
                    v = int(v.flatten()[0].item())
                except Exception:
                    pass
            if isinstance(v, (int, float)):
                iv = int(v)
                if iv in (2, 3):
                    return iv
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("2", "2d"):
                    return 2
                if s in ("3", "3d", "true"):
                    return 3
    return None

def as_btc(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure image tensors use ``[B, T, C, H, W]`` layout.

    Args:
        x (torch.Tensor): Image tensor with shape ``[B, T, C, H, W]`` or ``[B, C, H, W]``.

    Returns:
        torch.Tensor: Image tensor with an explicit time dimension.
    """
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        return x.unsqueeze(1)
    raise ValueError(f"Unsupported image tensor shape: {tuple(x.shape)}")

def as_bthw_mask(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """
    Ensure mask tensors use ``[B, T, H, W]`` layout.

    Args:
        x (torch.Tensor): Mask tensor with optional singleton channel dimension.
        B (int): Expected batch size.
        T (int): Expected sequence length.

    Returns:
        torch.Tensor: Mask tensor without a channel dimension.
    """
    if x.ndim == 5 and x.shape[2] == 1:
        return x[:, :, 0]
    if x.ndim == 4 and x.shape[0] == B and x.shape[1] == 1:
        return x[:, 0].unsqueeze(1).expand(B, T, *x.shape[-2:])
    if x.ndim == 4 and x.shape[0] == B and x.shape[1] == T:
        return x
    raise ValueError(f"Unsupported mask tensor shape: {tuple(x.shape)}")

def train_epoch(student, teacher, train_loader, optimizer, scheduler, config, scaler, epoch, logger, writer, global_step):
    """
    Run one mixed 2D/3D training epoch.

    Args:
        student (SAM2VideoPredictor): Student model being optimized.
        teacher (SAM2VideoPredictor | None): Teacher model for distillation.
        train_loader (DataLoader): Training loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning-rate scheduler.
        config (OmegaConf.DictConfig): Training configuration.
        scaler (object | None): AMP scaler, currently unused when bf16 autocast is used.
        epoch (int): Epoch index.
        logger (logging.Logger): Training logger.
        writer (SummaryWriter): TensorBoard writer.
        global_step (int): Current global step.

    Returns:
        tuple[float, int, torch.optim.Optimizer, object]: Average loss, updated global
            step, optimizer, and scheduler.
    """
    student.train()
    _apply_freeze_schedule(student, config, epoch, logger=logger)

    freeze_epochs = int(getattr(getattr(config, "init", {}), "freeze_pretrained_epochs", 0))
    if freeze_epochs > 0 and epoch == freeze_epochs:
        rebuilt = _rebuild_optimizer_and_scheduler_after_unfreeze(
            student, config, optimizer, scheduler, logger=logger
        )
        optimizer = rebuilt["optimizer"]
        scheduler = rebuilt["scheduler"]

    total_loss = 0.0
    first_vis_2d = False
    first_vis_3d = False
    skipped_2d = 0
    skipped_3d = 0

    try:
        config.training._epoch = int(epoch)
    except Exception:
        pass

    image_every = int(getattr(getattr(config, "logging", {}), "image_every", 5))
    gif_fps = int(getattr(getattr(config, "logging", {}), "image_fps", 2))
    ping_pong = bool(getattr(getattr(config, "logging", {}), "image_ping_pong", False))

    E_total = int(getattr(getattr(config, "training", {}), "epochs", 1))
    do_viz_this_ep = (
        epoch == 0
        or ((epoch + 1) % image_every == 0)
        or (epoch == E_total - 1)
    )

    gif_dir = os.path.join(writer.log_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    text_every_steps = int(getattr(getattr(config, "logging", {}), "tb_text_every_train_steps", 100))

    def _tb_pre(txt):
        """Wrap text as escaped preformatted HTML."""
        return "<pre style=\"font-family: monospace; white-space: pre;\">" + html.escape(str(txt)) + "</pre>"

    def _tensor_stats(x):
        """Return tensor shape, dtype, and finite-value stats."""
        if not torch.is_tensor(x):
            return {"kind": type(x).__name__}
        y = x.detach().float()
        out = {
            "shape": tuple(y.shape),
            "dtype": str(y.dtype),
            "finite": bool(torch.isfinite(y).all().item()) if y.numel() else True,
            "numel": int(y.numel()),
        }
        if y.numel():
            finite = y[torch.isfinite(y)]
            if finite.numel():
                out.update({
                    "min": float(finite.min().item()),
                    "max": float(finite.max().item()),
                    "mean": float(finite.mean().item()),
                    "std": float(finite.std().item()) if finite.numel() > 1 else 0.0,
                    "absmax": float(finite.abs().max().item()),
                })
        return out

    def _log_nonfinite_skip(stage, batch_idx, batch, extra=None):
        """Log a non-finite training skip."""
        extra = extra or {}
        ds = batch.get("dataset", "NA")
        sub = batch.get("subdataset", "NA")
        task = batch.get("task_id", batch.get("task_ids", "NA"))
        modality = batch.get("modality", "NA")

        msg_lines = [
            f"[NONFINITE_SKIP] stage={stage}",
            f"epoch={epoch}",
            f"global_step={global_step}",
            f"batch_idx={batch_idx}",
            f"dataset={ds}",
            f"subdataset={sub}",
            f"task_id={task}",
            f"modality={modality}",
            f"dim={data_dim}",
            f"lr_backbone={get_lr_group(optimizer, 0):.8f}",
            f"lr_mask={get_lr_group(optimizer, 1):.8f}",
            f"lr_mem={get_lr_group(optimizer, 2):.8f}",
            f"image_stats={_tensor_stats(batch.get('image', None))}",
            f"mask_stats={_tensor_stats(batch.get('mask', None))}",
        ]

        for k, v in extra.items():
            msg_lines.append(f"{k}={v}")

        msg = "\n".join(msg_lines)

        logger.error(msg)
        print(msg)

        if writer is not None:
            writer.add_text("train/error_nonfinite", _tb_pre(msg), global_step)
            writer.flush()

    def _save_crash_artifacts(batch, batch_idx, stage, extra=None):
        """Save model and batch artifacts for a fatal error."""
        extra = extra or {}

        crash_dir = os.path.join(config.ckpt_path, "crash")
        os.makedirs(crash_dir, exist_ok=True)

        ckpt_path = os.path.join(
            crash_dir,
            f"crash_ep{int(epoch):04d}_step{int(global_step):07d}_batch{int(batch_idx):06d}.pth"
        )
        batch_path = os.path.join(
            crash_dir,
            f"crash_batch_ep{int(epoch):04d}_step{int(global_step):07d}_batch{int(batch_idx):06d}.pt"
        )

        state = {
            "model": _unwrap_model(student).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_idx": int(batch_idx),
            "stage": stage,
            "extra": extra,
        }
        torch.save(state, ckpt_path)

        cpu_batch = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                cpu_batch[k] = v.detach().cpu()
            elif isinstance(v, list):
                cpu_batch[k] = [x.detach().cpu() if torch.is_tensor(x) else x for x in v]
            elif isinstance(v, tuple):
                cpu_batch[k] = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in v)
            else:
                cpu_batch[k] = v

        cpu_batch["__crash_meta__"] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_idx": int(batch_idx),
            "stage": stage,
            "extra": extra,
        }
        torch.save(cpu_batch, batch_path)

        return ckpt_path, batch_path

    def _save_and_log_fatal_crash(stage, batch_idx, batch, extra=None):
        """Save crash artifacts and log fatal-error context."""
        extra = extra or {}
        ckpt_path, batch_path = _save_crash_artifacts(batch, batch_idx, stage, extra=extra)
        msg_lines = [
            f"[FATAL_CRASH] stage={stage}",
            f"epoch={epoch}",
            f"global_step={global_step}",
            f"batch_idx={batch_idx}",
            f"crash_checkpoint={ckpt_path}",
            f"crash_batch={batch_path}",
        ]
        for k, v in extra.items():
            msg_lines.append(f"{k}={v}")
        msg = "\n".join(msg_lines)
        logger.exception(msg)
        print(msg)
        if writer is not None:
            writer.add_text("train/error_fatal", _tb_pre(msg), global_step)
            writer.flush()

    def get_lr_group(opt, i):
        """Return a parameter-group learning rate."""
        return float(opt.param_groups[i]["lr"]) if i < len(opt.param_groups) else float("nan")

    lam2d = float(getattr(getattr(config, "training", {}), "lambda_distill_2d", 0.0))
    lam3d = float(getattr(getattr(config, "training", {}), "lambda_distill_3d", 0.0))
    nonprompt_scale = float(getattr(getattr(config, "training", {}), "nonprompt_scale", 1.0))
    heavy_mult = float(getattr(getattr(config, "training", {}), "distill_heavy_mult", 1.0))
    tau = float(getattr(getattr(config, "training", {}), "distill_temperature", 1.0))

    writer.add_scalar("train/lambda_distill_2d", lam2d, epoch)
    writer.add_scalar("train/lambda_distill_3d", lam3d, epoch)
    writer.add_scalar("train/nonprompt_scale", nonprompt_scale, epoch)
    writer.add_scalar("train/distill_heavy_mult", heavy_mult, epoch)
    writer.add_scalar("train/distill_temperature", tau, epoch)

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch", dynamic_ncols=True)

    train_start_time = time.perf_counter()
    last_out_time = train_start_time
    ema = {"2d_data": None, "2d_step": None, "3d_data": None, "3d_step": None}

    def _upd(k, v, beta=0.9):
        """Update an exponential moving average."""
        ema[k] = v if (ema[k] is None) else (beta * ema[k] + (1 - beta) * v)

    def _train_progress_stats(batch_idx):
        """Return training progress counters."""
        elapsed = time.perf_counter() - train_start_time
        batches_done = batch_idx + 1
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else -1
        pct = (100.0 * batches_done / total_batches) if total_batches > 0 else float("nan")
        return elapsed, pct, batches_done, total_batches

    train_vis_prompt_mode = str(
        getattr(getattr(config, "logging", {}), "train_vis_prompt_mode", "box")
    ).strip().lower()
    if train_vis_prompt_mode in ("bbox", "boxes"):
        train_vis_prompt_mode = "box"
    elif train_vis_prompt_mode in ("click", "clicks", "points"):
        train_vis_prompt_mode = "point"
    elif train_vis_prompt_mode in ("mask", "masks"):
        train_vis_prompt_mode = "mask"
    else:
        train_vis_prompt_mode = "box"

    for batch_idx, batch in enumerate(pbar):
        _now = time.perf_counter()
        data_wait = _now - last_out_time

        ds_name = batch.get("dataset", "NA")
        sub_name = batch.get("subdataset", "NA")
        if isinstance(ds_name, (list, tuple)):
            ds_name = ds_name[0]
        if isinstance(sub_name, (list, tuple)):
            sub_name = sub_name[0]
        pbar.set_description(f"Train Epoch {epoch} - [{ds_name}/{sub_name}]")

        data_dim = _dim_from_batch(batch)
        if data_dim is None:
            raise ValueError("Batch missing 'dim'. Dataset must supply 2 or 3.")

        if data_dim == 2:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            try:
                step_out = train_step_2d(student, teacher, optimizer, batch, config, scaler)
            except Exception as e:
                _save_and_log_fatal_crash(
                    stage="train2d/exception",
                    batch_idx=batch_idx,
                    batch=batch,
                    extra={"exception": repr(e)},
                )
                raise

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.perf_counter() - _t0

            if not step_out["ok"]:
                _log_nonfinite_skip(
                    stage=f"train2d/{step_out['skip_reason']}",
                    batch_idx=batch_idx,
                    batch=batch,
                    extra={k: v for k, v in step_out.items() if k != "ok"},
                )
                last_out_time = time.perf_counter()
                skipped_2d += 1
                writer.add_scalar("train2d/skipped_steps", skipped_2d, global_step)
                continue

            loss = float(step_out["loss"])

            writer.add_scalar("train2d/loss", loss, global_step)
            writer.add_scalar("train/lr_backbone", get_lr_group(optimizer, 0), global_step)
            writer.add_scalar("train/lr_mask", get_lr_group(optimizer, 1), global_step)
            writer.add_scalar("train/lr_mem", get_lr_group(optimizer, 2), global_step)

            if (not first_vis_2d) and do_viz_this_ep:
                with torch.no_grad():
                    device = student.device

                    img_seq = batch["image"][0]
                    mask_seq = batch["mask"][0]

                    video_tchw = img_seq.to(device, non_blocking=True).float()

                    gt = mask_seq.to(device, non_blocking=True)
                    if gt.ndim == 4 and gt.shape[1] == 1:
                        gt_thw = gt[:, 0].float()
                    elif gt.ndim == 3:
                        gt_thw = gt.float()
                    else:
                        raise ValueError(f"Unexpected mask shape for 2D viz: {tuple(gt.shape)}")

                    prompt_cache = None
                    if "val_prompt_cache" in batch and len(batch["val_prompt_cache"]) > 0:
                        prompt_cache = batch["val_prompt_cache"][0]

                    if prompt_cache is not None:
                        out = validate_step_2d(
                            student,
                            video_tchw,
                            gt_thw,
                            prompt_cache=prompt_cache,
                            prompt_mode=train_vis_prompt_mode,
                            normalize_coords=True,
                        )
                    else:
                        out = None

                    if out is not None:
                        logits_seq = out["logits"].detach().cpu()

                        gif_path = os.path.join(gif_dir, f"train2d_ep{epoch:04d}.gif")
                        _, frames = save_vis_gif(
                            img_seq,
                            mask_seq,
                            logits_seq,
                            gif_path,
                            fps=gif_fps,
                            threshold=0.5,
                            ping_pong=ping_pong,
                        )
                        writer.add_video("train2d/vis", frames_to_tb_video(frames), epoch, fps=gif_fps)
                        first_vis_2d = True

            _upd("2d_data", data_wait)
            _upd("2d_step", step_time)
            elapsed, pct, batches_done, total_batches = _train_progress_stats(batch_idx)
            pbar.set_postfix(
                loss=f"{float(loss):.4f}",
                progress=f"{pct:.2f}%" if pct == pct else f"{batches_done}",
                elapsed=f"{elapsed:.1f}s",
                data2d=f"{ema['2d_data']:.3f}s" if ema["2d_data"] is not None else "n/a",
                step2d=f"{ema['2d_step']:.3f}s" if ema["2d_step"] is not None else "n/a",
            )

            if writer is not None and text_every_steps > 0 and (global_step % text_every_steps == 0):
                lines = [
                    f"Epoch {epoch} training progress",
                    f"global_step={global_step}",
                    f"batch={batches_done}/{total_batches}",
                    f"progress={pct:.2f}%" if pct == pct else f"progress=batch {batches_done}",
                    f"elapsed_s={elapsed:.2f}",
                    f"dataset={ds_name}/{sub_name}",
                    f"dim={data_dim}",
                    f"loss={float(loss):.6f}",
                    f"lr_backbone={get_lr_group(optimizer, 0):.8f}",
                    f"lr_mask={get_lr_group(optimizer, 1):.8f}",
                    f"lr_mem={get_lr_group(optimizer, 2):.8f}",
                ]
                if ema.get("2d_data") is not None:
                    lines.append(f"ema_2d_data_wait_s={ema['2d_data']:.4f}")
                if ema.get("2d_step") is not None:
                    lines.append(f"ema_2d_step_s={ema['2d_step']:.4f}")
                if ema.get("3d_data") is not None:
                    lines.append(f"ema_3d_data_wait_s={ema['3d_data']:.4f}")
                if ema.get("3d_step") is not None:
                    lines.append(f"ema_3d_step_s={ema['3d_step']:.4f}")

                writer.add_text("train/progress", _tb_pre("\n".join(lines)), global_step)

        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            try:
                step_out = train_step_3d(student, teacher, optimizer, batch, config, scaler)
            except Exception as e:
                _save_and_log_fatal_crash(
                    stage="train3d/exception",
                    batch_idx=batch_idx,
                    batch=batch,
                    extra={"exception": repr(e)},
                )
                raise

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.perf_counter() - _t0

            if not step_out["ok"]:
                _log_nonfinite_skip(
                    stage=f"train3d/{step_out['skip_reason']}",
                    batch_idx=batch_idx,
                    batch=batch,
                    extra={k: v for k, v in step_out.items() if k != "ok"},
                )
                last_out_time = time.perf_counter()
                skipped_3d += 1
                writer.add_scalar("train3d/skipped_steps", skipped_3d, global_step)
                continue

            loss = float(step_out["loss"])
            p_loss = float(step_out["prompt_loss"])
            np_loss = step_out["non_prompt_loss"]
            np_base = step_out["non_prompt_loss_base"]

            writer.add_scalar("train3d/loss", loss, global_step)
            writer.add_scalar("train3d/prompt_loss", p_loss, global_step)
            if np_loss is not None:
                writer.add_scalar("train3d/non_prompt_loss", float(np_loss), global_step)
            if np_base is not None:
                writer.add_scalar("train3d/non_prompt_loss_base", float(np_base), global_step)

            writer.add_scalar("train/lr_backbone", get_lr_group(optimizer, 0), global_step)
            writer.add_scalar("train/lr_mask", get_lr_group(optimizer, 1), global_step)
            writer.add_scalar("train/lr_mem", get_lr_group(optimizer, 2), global_step)

            if (not first_vis_3d) and do_viz_this_ep:
                with torch.no_grad():
                    device = student.device

                    img_seq = batch["image"][0]
                    mask_seq = batch["mask"][0]

                    video_tchw = img_seq.to(device, non_blocking=True).float()

                    gt = mask_seq.to(device, non_blocking=True)
                    if gt.ndim == 4 and gt.shape[1] == 1:
                        gt_thw = gt[:, 0].float()
                    elif gt.ndim == 3:
                        gt_thw = gt.float()
                    else:
                        raise ValueError(f"Unexpected mask shape for 3D viz: {tuple(gt.shape)}")

                    prompt_cache = None
                    if "val_prompt_cache" in batch and len(batch["val_prompt_cache"]) > 0:
                        prompt_cache = batch["val_prompt_cache"][0]

                    if prompt_cache is not None:
                        out = validate_step_3d(
                            student,
                            video_tchw,
                            gt_thw,
                            prompt_cache=prompt_cache,
                            prompt_mode=train_vis_prompt_mode,
                            normalize_coords=True,
                        )
                    else:
                        out = None

                    if out is not None:
                        logits_seq = out["logits"].detach().cpu()

                        gif_path = os.path.join(gif_dir, f"train3d_ep{epoch:04d}.gif")
                        vis_thr = float(getattr(getattr(config, "training", {}), "vis_threshold", 0.55))
                        _, frames = save_vis_gif(
                            img_seq,
                            mask_seq,
                            logits_seq,
                            gif_path,
                            fps=gif_fps,
                            threshold=vis_thr,
                            ping_pong=ping_pong,
                        )
                        writer.add_video("train3d/vis", frames_to_tb_video(frames), epoch, fps=gif_fps)
                        first_vis_3d = True

            _upd("3d_data", data_wait)
            _upd("3d_step", step_time)
            elapsed, pct, batches_done, total_batches = _train_progress_stats(batch_idx)
            pbar.set_postfix(
                loss=f"{float(loss):.4f}",
                progress=f"{pct:.2f}%" if pct == pct else f"{batches_done}",
                elapsed=f"{elapsed:.1f}s",
                data3d=f"{ema['3d_data']:.3f}s" if ema["3d_data"] is not None else "n/a",
                step3d=f"{ema['3d_step']:.3f}s" if ema["3d_step"] is not None else "n/a",
            )

            if writer is not None and text_every_steps > 0 and (global_step % text_every_steps == 0):
                lines = [
                    f"Epoch {epoch} training progress",
                    f"global_step={global_step}",
                    f"batch={batches_done}/{total_batches}",
                    f"progress={pct:.2f}%" if pct == pct else f"progress=batch {batches_done}",
                    f"elapsed_s={elapsed:.2f}",
                    f"dataset={ds_name}/{sub_name}",
                    f"dim={data_dim}",
                    f"loss={float(loss):.6f}",
                    f"lr_backbone={get_lr_group(optimizer, 0):.8f}",
                    f"lr_mask={get_lr_group(optimizer, 1):.8f}",
                    f"lr_mem={get_lr_group(optimizer, 2):.8f}",
                ]
                if ema.get("2d_data") is not None:
                    lines.append(f"ema_2d_data_wait_s={ema['2d_data']:.4f}")
                if ema.get("2d_step") is not None:
                    lines.append(f"ema_2d_step_s={ema['2d_step']:.4f}")
                if ema.get("3d_data") is not None:
                    lines.append(f"ema_3d_data_wait_s={ema['3d_data']:.4f}")
                if ema.get("3d_step") is not None:
                    lines.append(f"ema_3d_step_s={ema['3d_step']:.4f}")

                writer.add_text("train/progress", _tb_pre("\n".join(lines)), global_step)

        total_loss += float(loss)
        global_step += 1
        last_out_time = time.perf_counter()

    try:
        pbar.close()
    except Exception:
        pass

    avg_loss = total_loss / max(1, len(train_loader))
    logger.info(f"[Epoch {epoch}] Skipped steps: 2D={skipped_2d}, 3D={skipped_3d}")
    writer.add_scalar("train2d/skipped_steps_epoch", skipped_2d, epoch)
    writer.add_scalar("train3d/skipped_steps_epoch", skipped_3d, epoch)
    logger.info(f"[Epoch {epoch}] Train avg loss: {avg_loss:.4f}")
    writer.add_scalar("train/loss", float(avg_loss), epoch)

    if ema.get("2d_data") is not None:
        writer.add_scalar("debug/2d_data_wait_ema_s", float(ema["2d_data"]), epoch)
    if ema.get("2d_step") is not None:
        writer.add_scalar("debug/2d_step_time_ema_s", float(ema["2d_step"]), epoch)
    if ema.get("3d_data") is not None:
        writer.add_scalar("debug/3d_data_wait_ema_s", float(ema["3d_data"]), epoch)
    if ema.get("3d_step") is not None:
        writer.add_scalar("debug/3d_step_time_ema_s", float(ema["3d_step"]), epoch)

    writer.flush()
    scheduler.step()
    return avg_loss, global_step, optimizer, scheduler

def validate_epoch(student, val_loader, config, epoch: int, writer=None):
    """
    Run one validation epoch using dataset-produced prompt caches.

    Args:
        student (SAM2VideoPredictor): Student model to evaluate.
        val_loader (DataLoader): Validation loader.
        config (OmegaConf.DictConfig): Validation and logging configuration.
        epoch (int): Epoch index.
        writer (SummaryWriter | None): Optional TensorBoard writer.

    Returns:
        dict[str, float]: Aggregate validation metrics.
    """
    student.eval()

    try:
        tasks_map = json.load(open(config.dripp.tasks_file, "r"))
        tasks_map = {str(k): v for k, v in tasks_map.items()}
    except Exception:
        tasks_map = {}

    task_agg = TaskAggregator(tasks_map=tasks_map)
    modality_agg = ModalityAggregator()

    text_every_batches = int(getattr(getattr(config, "logging", {}), "tb_text_every_val_batches", 50))

    def _tb_pre(txt):
        """Wrap text as escaped preformatted HTML."""
        return "<pre style=\"font-family: monospace; white-space: pre;\">" + html.escape(str(txt)) + "</pre>"

    image_every = int(getattr(getattr(config, "logging", {}), "image_every", 5))
    gif_fps = int(getattr(getattr(config, "logging", {}), "image_fps", 2))
    ping_pong = bool(getattr(getattr(config, "logging", {}), "image_ping_pong", False))
    E_total = int(getattr(getattr(config, "training", {}), "epochs", max(epoch + 1, 1)))
    is_final_epoch = (epoch == E_total - 1)
    do_viz_this_ep = (
        epoch == 0
        or ((epoch + 1) % image_every == 0)
        or is_final_epoch
    )

    gif_dir = None
    if writer is not None:
        gif_dir = os.path.join(writer.log_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)

    vis_tasks_done = set()
    logged_images_2d = False
    logged_images_3d = False

    normalize_coords = True

    thr_probs = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    thr_logits = np.log(thr_probs / (1.0 - thr_probs)).astype(np.float64)

    totals = defaultdict(float)
    counts = defaultdict(int)
    best_thr_sum, best_thr_w = 0.0, 0
    best_thr3d_sum, best_thr3d_w = 0.0, 0

    sum_pred_frac = 0.0
    sum_gt_frac = 0.0
    sum_prob_mean = 0.0
    empty_slice_count_all = 0.0
    empty_slice_count_gt = 0.0
    total_frames_all = 0
    total_frames_gt = 0

    fg_bin_totals_dice = defaultdict(float)
    fg_bin_totals_iou = defaultdict(float)
    fg_bin_counts = defaultdict(int)

    val_cfg = getattr(config, "validation", {})
    compute_global_prob_metrics = bool(getattr(val_cfg, "compute_global_prob_metrics", False))

    all_probs = [] if compute_global_prob_metrics else None
    all_labels = [] if compute_global_prob_metrics else None

    def _normalize_modes(x):
        """Normalize prompt-mode names."""
        if x is None:
            return []
        if isinstance(x, str):
            x = [x]
        out = []
        for m in x:
            mm = str(m).strip().lower()
            if mm in ("point", "points", "click", "clicks"):
                out.append("point")
            elif mm in ("box", "bbox", "boxes"):
                out.append("box")
            elif mm in ("mask", "masks"):
                out.append("mask")
        dedup = []
        for m in out:
            if m not in dedup:
                dedup.append(m)
        return dedup

    default_modes = _normalize_modes(getattr(val_cfg, "default_prompt_modes", ["point", "box", "mask"]))
    if len(default_modes) == 0:
        default_modes = ["point", "box", "mask"]

    prompt_modes_by_task = getattr(val_cfg, "prompt_modes_by_task", {})
    prompt_modes_by_dataset = getattr(val_cfg, "prompt_modes_by_dataset", {})

    def _resolve_prompt_modes(task_id, task_label, dataset_name, subdataset_name):
        """Resolve validation prompt modes for a sample."""
        for key in (task_id, task_label):
            if key is None:
                continue
            key = str(key)
            if key in prompt_modes_by_task:
                modes = _normalize_modes(prompt_modes_by_task[key])
                if len(modes) > 0:
                    return modes

        ds_keys = [
            f"{dataset_name}/{subdataset_name}" if subdataset_name else None,
            str(dataset_name) if dataset_name is not None else None,
            str(subdataset_name) if subdataset_name is not None else None,
        ]
        for key in ds_keys:
            if key is None:
                continue
            if key in prompt_modes_by_dataset:
                modes = _normalize_modes(prompt_modes_by_dataset[key])
                if len(modes) > 0:
                    return modes

        return list(default_modes)

    def _mode_prefix(mode, data_dim):
        """Return the metric prefix for a prompt mode."""
        tag = {"point": "point", "box": "box", "mask": "mask"}[mode]
        return f"{tag}/" if int(data_dim) == 2 else f"{tag}/"

    val_start_time = time.perf_counter()
    last_batch_end_time = val_start_time
    ema_data_wait = None
    ema_step_time = None

    def _ema(old, new, beta=0.9):
        """Update an exponential moving average."""
        return float(new) if old is None else float(beta * old + (1.0 - beta) * new)

    def _safe_metric(name):
        """Return an averaged metric or NaN."""
        c = counts.get(name, 0)
        return float(totals[name] / c) if c > 0 else float("nan")

    def _emit_val_progress(batch_idx, ds_name, sub_name, data_dim):
        """Emit validation progress to TensorBoard."""
        if writer is None:
            return
        if text_every_batches <= 0:
            return
        if ((batch_idx + 1) % text_every_batches) != 0:
            return

        elapsed = time.perf_counter() - val_start_time
        batches_done = batch_idx + 1
        total_batches = len(val_loader) if hasattr(val_loader, "__len__") else -1
        pct = (100.0 * batches_done / total_batches) if total_batches > 0 else float("nan")

        lines = [
            f"Epoch {epoch} validation progress",
            f"batch={batches_done}/{total_batches}",
            f"progress={pct:.2f}%" if pct == pct else f"progress=batch {batches_done}",
            f"dataset={ds_name}/{sub_name}",
            f"dim={data_dim}",
            f"elapsed_s={elapsed:.2f}",
            f"ema_data_wait_s={ema_data_wait:.4f}" if ema_data_wait is not None else "ema_data_wait_s=NA",
            f"ema_step_s={ema_step_time:.4f}" if ema_step_time is not None else "ema_step_s=NA",
        ]

        for key in ("dice@0.5", "iou@0.5", "dice_best", "iou_best", "dice3d@0.5", "iou3d@0.5", "dice3d@best", "iou3d@best"):
            val = _safe_metric(key)
            if val == val:
                lines.append(f"{key}={val:.6f}")

        writer.add_text("val/progress", _tb_pre("\n".join(lines)), epoch * 100000 + batches_done)

    pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}", unit="batch", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        batch_fetch_time = time.perf_counter()
        data_wait = batch_fetch_time - last_batch_end_time
        step_t0 = time.perf_counter()

        data_dim = _dim_from_batch(batch)
        if data_dim is None:
            raise ValueError("Batch missing 'dim' (2 or 3).")

        ds_name = batch.get("dataset", "NA")
        sub_name = batch.get("subdataset", "NA")
        if isinstance(ds_name, (list, tuple)):
            ds_name = ds_name[0]
        if isinstance(sub_name, (list, tuple)):
            sub_name = sub_name[0]
        pbar.set_description(f"Val Epoch {epoch} - [{ds_name}/{sub_name}]")

        t_id = batch.get("task_id", None)
        if isinstance(t_id, (list, tuple)):
            t_id = t_id[0]
        t_lab = batch.get("task_label", None)
        if isinstance(t_lab, (list, tuple)):
            t_lab = t_lab[0]

        task_label = None
        if task_agg is not None:
            task_label = task_agg.label_for(t_id if t_id is not None else t_lab)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_label) if task_label is not None else "task")

        mod = batch.get("modality", None)
        if isinstance(mod, (list, tuple)):
            uniq = sorted({str(x) for x in mod}) if len(mod) else ["unknown"]
            modality_label = uniq[0] if len(uniq) == 1 else "mixed"
        else:
            modality_label = str(mod) if mod is not None else "unknown"

        visualize_this_batch = False
        if do_viz_this_ep and writer is not None:
            if int(data_dim) == 2:
                visualize_this_batch = (not logged_images_2d) if not is_final_epoch else (task_label not in vis_tasks_done)
            else:
                visualize_this_batch = (not logged_images_3d) if not is_final_epoch else (task_label not in vis_tasks_done)

        device = student.device
        imgs = as_btc(batch["image"]).to(device, non_blocking=True).float()
        B, T, C, H, W = imgs.shape

        gt_thw_all = as_bthw_mask(batch["mask"].to(device, non_blocking=True), B, T).float()
        gt_thw_all = (gt_thw_all > 0.5).to(torch.uint8)

        val_prompt_cache_all = batch.get("val_prompt_cache", [None] * B)

        viz_logits = None
        viz_img = None

        for b in range(B):
            video_tchw = imgs[b]
            gt_thw = gt_thw_all[b]

            nonempty = (gt_thw.sum(dim=(1, 2)) > 0)
            n_valid_frames = int(nonempty.sum().item())
            if n_valid_frames == 0:
                continue

            sample_task_id = None
            sample_task_label = None
            sample_ds = ds_name
            sample_sub = sub_name

            if isinstance(batch.get("task_id", None), (list, tuple)) and b < len(batch["task_id"]):
                sample_task_id = batch["task_id"][b]
            elif t_id is not None:
                sample_task_id = t_id

            if isinstance(batch.get("task_label", None), (list, tuple)) and b < len(batch["task_label"]):
                sample_task_label = batch["task_label"][b]
            elif t_lab is not None:
                sample_task_label = t_lab

            if isinstance(batch.get("dataset", None), (list, tuple)) and b < len(batch["dataset"]):
                sample_ds = batch["dataset"][b]
            if isinstance(batch.get("subdataset", None), (list, tuple)) and b < len(batch["subdataset"]):
                sample_sub = batch["subdataset"][b]

            prompt_modes = _resolve_prompt_modes(
                sample_task_id,
                sample_task_label,
                sample_ds,
                sample_sub,
            )

            sample_prompt_cache = val_prompt_cache_all[b] if b < len(val_prompt_cache_all) else None
            if sample_prompt_cache is None:
                continue

            for prompt_mode in prompt_modes:
                if int(data_dim) == 2:
                    out = validate_step_2d(
                        student,
                        video_tchw,
                        gt_thw.to(torch.float32),
                        prompt_cache=sample_prompt_cache,
                        prompt_mode=prompt_mode,
                        normalize_coords=normalize_coords,
                    )
                else:
                    out = validate_step_3d(
                        student,
                        video_tchw,
                        gt_thw.to(torch.float32),
                        prompt_cache=sample_prompt_cache,
                        prompt_mode=prompt_mode,
                        normalize_coords=normalize_coords,
                    )

                if out is None:
                    continue

                logits_t = out["logits"]

                if visualize_this_batch and viz_logits is None and prompt_mode == prompt_modes[0]:
                    viz_logits = logits_t.detach().cpu()
                    viz_img = batch["image"][b].detach().cpu()

                logits_np = logits_t[nonempty].detach().cpu().numpy().astype(np.float32)
                gt_np = gt_thw[nonempty].detach().cpu().numpy().astype(np.uint8)

                probs_np = sigmoid_np(logits_np)
                pred05 = (logits_np > 0.0).astype(np.uint8)

                dices = []
                ious = []
                fg_fracs = []
                for kslice in range(gt_np.shape[0]):
                    d, i = dice_iou(torch.from_numpy(pred05[kslice]), torch.from_numpy(gt_np[kslice]))
                    dices.append(d)
                    ious.append(i)
                    fg_fracs.append(float(gt_np[kslice].mean()))
                dice05 = float(np.mean(dices))
                iou05 = float(np.mean(ious))

                avg_d_list = []
                avg_i_list = []
                best_d = -1.0
                best_i = -1.0
                best_thr = float(thr_logits[0])

                for thr in thr_logits:
                    pred = (logits_np > float(thr)).astype(np.uint8)
                    d_thr = []
                    i_thr = []
                    for kslice in range(gt_np.shape[0]):
                        d, i = dice_iou(torch.from_numpy(pred[kslice]), torch.from_numpy(gt_np[kslice]))
                        d_thr.append(d)
                        i_thr.append(i)
                    d_m = float(np.mean(d_thr))
                    i_m = float(np.mean(i_thr))
                    avg_d_list.append(d_m)
                    avg_i_list.append(i_m)
                    if d_m > best_d:
                        best_d = d_m
                        best_i = i_m
                        best_thr = float(thr)

                dice_avg = float(np.mean(avg_d_list))
                iou_avg = float(np.mean(avg_i_list))

                mode_prefix = _mode_prefix(prompt_mode, data_dim)

                totals[f"{mode_prefix}dice@0.5"] += dice05
                counts[f"{mode_prefix}dice@0.5"] += 1
                totals[f"{mode_prefix}iou@0.5"] += iou05
                counts[f"{mode_prefix}iou@0.5"] += 1

                totals[f"{mode_prefix}dice_best"] += best_d
                counts[f"{mode_prefix}dice_best"] += 1
                totals[f"{mode_prefix}iou_best"] += best_i
                counts[f"{mode_prefix}iou_best"] += 1

                totals[f"{mode_prefix}dice@avg_thr"] += dice_avg
                counts[f"{mode_prefix}dice@avg_thr"] += 1
                totals[f"{mode_prefix}iou@avg_thr"] += iou_avg
                counts[f"{mode_prefix}iou@avg_thr"] += 1

                if prompt_mode == "box":
                    totals["dice@0.5"] += dice05
                    counts["dice@0.5"] += 1
                    totals["iou@0.5"] += iou05
                    counts["iou@0.5"] += 1
                    totals["dice_best"] += best_d
                    counts["dice_best"] += 1
                    totals["iou_best"] += best_i
                    counts["iou_best"] += 1
                    totals["dice@avg_thr"] += dice_avg
                    counts["dice@avg_thr"] += 1
                    totals["iou@avg_thr"] += iou_avg
                    counts["iou@avg_thr"] += 1

                best_thr_sum += best_thr
                best_thr_w += 1

                if int(data_dim) == 3:
                    pred_best_vol = (logits_np > best_thr).astype(np.uint8)
                    d3, i3 = dice_iou(torch.from_numpy(pred_best_vol), torch.from_numpy(gt_np))
                    d3 = float(d3)
                    i3 = float(i3)

                    totals[f"{mode_prefix}dice3d@best"] += d3
                    counts[f"{mode_prefix}dice3d@best"] += 1
                    totals[f"{mode_prefix}iou3d@best"] += i3
                    counts[f"{mode_prefix}iou3d@best"] += 1

                    pred05_vol = (logits_np > 0.0).astype(np.uint8)
                    d305, i305 = dice_iou(torch.from_numpy(pred05_vol), torch.from_numpy(gt_np))
                    d305 = float(d305)
                    i305 = float(i305)

                    totals[f"{mode_prefix}dice3d@0.5"] += d305
                    counts[f"{mode_prefix}dice3d@0.5"] += 1
                    totals[f"{mode_prefix}iou3d@0.5"] += i305
                    counts[f"{mode_prefix}iou3d@0.5"] += 1

                    best_thr3d_sum += best_thr
                    best_thr3d_w += 1

                    if prompt_mode == "box":
                        totals["dice3d@best"] += d3
                        counts["dice3d@best"] += 1
                        totals["iou3d@best"] += i3
                        counts["iou3d@best"] += 1
                        totals["dice3d@0.5"] += d305
                        counts["dice3d@0.5"] += 1
                        totals["iou3d@0.5"] += i305
                        counts["iou3d@0.5"] += 1

                probs_flat = probs_np.reshape(-1).astype(np.float64)
                gt_flat = gt_np.reshape(-1).astype(np.uint8)
                if compute_global_prob_metrics:
                    all_probs.append(probs_flat)
                    all_labels.append(gt_flat)

                sum_pred_frac += float(pred05.mean()) * int(gt_np.shape[0])
                sum_prob_mean += float(probs_np.mean()) * int(gt_np.shape[0])
                total_frames_all += int(gt_np.shape[0])

                if int(gt_np.shape[0]) > 0:
                    sum_gt_frac += float(gt_np.mean()) * int(gt_np.shape[0])
                    total_frames_gt += int(gt_np.shape[0])

                empty_slice_count_all += float((gt_np.sum(axis=(1, 2)) == 0).sum())

                for frac, d, i in zip(fg_fracs, dices, ious):
                    key = fg_bin(frac)
                    fg_bin_totals_dice[key] += float(d)
                    fg_bin_totals_iou[key] += float(i)
                    fg_bin_counts[key] += 1

                if task_agg is not None:
                    sample_metrics = {
                        "dice@0.5": dice05,
                        "iou@0.5": iou05,
                        "dice_best": best_d,
                        "iou_best": best_i,
                        "dice@avg_thr": dice_avg,
                        "iou@avg_thr": iou_avg,
                    }
                    if int(data_dim) == 3:
                        sample_metrics.update({
                            "dice3d@0.5": d305,
                            "iou3d@0.5": i305,
                            "dice3d@best": d3,
                            "iou3d@best": i3,
                        })
                    task_agg.update(
                        task_label if task_label is not None else safe_task,
                        sample_metrics,
                        slice_weight=int(gt_np.shape[0]),
                        vol_weight=1,
                        dim=int(data_dim),
                        n_items=1,
                    )

                if modality_agg is not None:
                    sample_metrics = {
                        "dice@0.5": dice05,
                        "iou@0.5": iou05,
                        "dice_best": best_d,
                        "iou_best": best_i,
                        "dice@avg_thr": dice_avg,
                        "iou@avg_thr": iou_avg,
                    }
                    if int(data_dim) == 3:
                        sample_metrics.update({
                            "dice3d@0.5": d305,
                            "iou3d@0.5": i305,
                            "dice3d@best": d3,
                            "iou3d@best": i3,
                        })
                    modality_agg.update(
                        modality_label,
                        sample_metrics,
                        slice_weight=int(gt_np.shape[0]),
                        vol_weight=1,
                        dim=int(data_dim),
                        n_items=1,
                    )

        if visualize_this_batch and writer is not None and viz_logits is not None and viz_img is not None and gif_dir is not None:
            if is_final_epoch:
                gif_name = f"val{'2d' if int(data_dim)==2 else '3d'}_task-{safe_task}_ep{epoch:04d}.gif"
                tb_tag = f"val{'2d' if int(data_dim)==2 else '3d'}/vis/{safe_task}"
                vis_tasks_done.add(task_label)
            else:
                gif_name = f"val{'2d' if int(data_dim)==2 else '3d'}_ep{epoch:04d}.gif"
                tb_tag = f"val{'2d' if int(data_dim)==2 else '3d'}/vis"
                if int(data_dim) == 2:
                    logged_images_2d = True
                else:
                    logged_images_3d = True

            gif_path = os.path.join(gif_dir, gif_name) if gif_dir is not None else gif_name
            try:
                img_seq = viz_img
                mask_seq = batch["mask"][0].detach().cpu()
                _, frames_v = save_vis_gif(img_seq, mask_seq, viz_logits, gif_path, fps=gif_fps, threshold=0.5, ping_pong=ping_pong)
                writer.add_video(tb_tag, frames_to_tb_video(frames_v), epoch, fps=gif_fps)
            except Exception as e:
                print("[WARN] Visualization skipped:", repr(e))

        step_time = time.perf_counter() - step_t0
        ema_data_wait = _ema(ema_data_wait, data_wait)
        ema_step_time = _ema(ema_step_time, step_time)
        _emit_val_progress(batch_idx, ds_name, sub_name, data_dim)
        last_batch_end_time = time.perf_counter()

        if int(data_dim) == 2 and counts.get("dice@0.5", 0) > 0:
            pbar.set_postfix(dice2d=f"{(totals['dice@0.5']/counts['dice@0.5']):.3f}")
        if int(data_dim) == 3 and counts.get("dice3d@0.5", 0) > 0:
            pbar.set_postfix(dice3d=f"{(totals['dice3d@0.5']/counts['dice3d@0.5']):.3f}")

    try:
        pbar.close()
    except Exception:
        pass

    metrics = {k: float(totals[k] / max(1, counts.get(k, 0))) for k in totals.keys()}
    if best_thr_w > 0:
        metrics["best_thr"] = float(best_thr_sum / best_thr_w)
    if best_thr3d_w > 0:
        metrics["best_thr3d"] = float(best_thr3d_sum / best_thr3d_w)

    if compute_global_prob_metrics and len(all_probs) > 0:
        probs_all = np.concatenate(all_probs).astype(np.float64)
        labels_all = np.concatenate(all_labels).astype(np.uint8)
        metrics["ece"] = float(ece(probs_all, labels_all, n_bins=15))
        roc, pr = try_auc(labels_all, probs_all)
        metrics["roc_auc"] = float(roc) if not math.isnan(roc) else float("nan")
        metrics["pr_auc"] = float(pr) if not math.isnan(pr) else float("nan")

    for key in ["fg<0.05%", "fg0.05-0.2%", "fg0.2-1%", "fg>1%"]:
        c = int(fg_bin_counts.get(key, 0))
        if c > 0:
            metrics[f"dice@0.5|{key}"] = float(fg_bin_totals_dice[key] / c)
            metrics[f"iou@0.5|{key}"] = float(fg_bin_totals_iou[key] / c)

    if total_frames_all > 0:
        metrics["debug/val_pred_fg_frac"] = float(sum_pred_frac / max(1, total_frames_all))
        metrics["debug/val_prob_mean"] = float(sum_prob_mean / max(1, total_frames_all))
        metrics["debug/val_empty_slice_rate_all"] = float(empty_slice_count_all / max(1, total_frames_all))
        if total_frames_gt > 0:
            metrics["debug/val_gt_fg_frac"] = float(sum_gt_frac / max(1, total_frames_gt))
            metrics["debug/val_empty_slice_rate_gt"] = float(empty_slice_count_gt / max(1, total_frames_all))

    if writer is not None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                writer.add_scalar(f"val/{k}", float(v), epoch)

        if "dice3d@0.5" in metrics:
            writer.add_scalar("val3d/dice@0.5", float(metrics["dice3d@0.5"]), epoch)
        if "dice3d@best" in metrics:
            writer.add_scalar("val3d/dice@best", float(metrics["dice3d@best"]), epoch)
        if "dice3d@best_vol" in metrics:
            writer.add_scalar("val3d/dice@best_vol", float(metrics["dice3d@best_vol"]), epoch)
        if "iou3d@0.5" in metrics:
            writer.add_scalar("val3d/iou@0.5", float(metrics["iou3d@0.5"]), epoch)
        if "iou3d@best" in metrics:
            writer.add_scalar("val3d/iou@best", float(metrics["iou3d@best"]), epoch)
        if "iou3d@best_vol" in metrics:
            writer.add_scalar("val3d/iou@best_vol", float(metrics["iou3d@best_vol"]), epoch)

        writer.add_text("val/summary", f"[Epoch {epoch}] Validation metrics: {metrics}", epoch)

        if task_agg is not None:
            task_agg.log_to_tensorboard(writer, epoch, split="val")
            writer.add_text(
                "val/per_task",
                task_agg.format_text_table(
                    top_keys=[
                        "dice_best",
                        "iou_best",
                        "dice@0.5",
                        "iou@0.5",
                        "dice3d@best",
                        "iou3d@best",
                        "dice3d@best_vol",
                        "iou3d@best_vol",
                    ]
                ),
                epoch,
            )

        if modality_agg is not None:
            modality_agg.log_to_tensorboard(writer, epoch, split="val")
            writer.add_text("val/per_modality", modality_agg.format_text_table(), epoch)

        final_elapsed = time.perf_counter() - val_start_time
        writer.add_text(
            "val/progress_final",
            _tb_pre(
                f"Epoch {epoch} validation complete\n"
                f"batches={len(val_loader) if hasattr(val_loader, '__len__') else 'NA'}\n"
                f"elapsed_s={final_elapsed:.2f}\n"
                + "\n".join(f"{k}={v}" for k, v in metrics.items())
            ),
            epoch,
        )

        writer.flush()

    return metrics

def main(config_path, resume, multi_gpu, amp):
    """
    Run the full train/validation/test workflow from a config file.

    Args:
        config_path (str): Path to the YAML config.
        resume (str | None): Optional checkpoint path to resume.
        multi_gpu (bool): Whether to wrap models in DataParallel when possible.
        amp (bool): Retained CLI flag for compatibility; current steps use bf16 autocast.

    Returns:
        None.
    """
    config = load_config(config_path)
    logger = setup_logger(config.logging)

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, test_loader = get_data_loaders(config)

    student = build_student_predictor(config, logger=logger).to(config.training.device)
    teacher = build_teacher_predictor(config).to(config.training.device)

    opt_all = setup_optimizer_and_scheduler(student, config)
    optimizer, scheduler = opt_all["optimizer"], opt_all["scheduler"]

    run_dir = os.path.join(config.logging.output_dir, "tb", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=run_dir, flush_secs=5, max_queue=10)
    writer.add_hparams(
        {
            "bs": int(getattr(config.training, "batch_size", 1)),
            "epochs": int(getattr(config.training, "epochs", 1)),
            "lr_backbone": float(getattr(config.training, "lr_backbone", getattr(config.training, "lr", 1e-4))),
            "lr_mask": float(getattr(config.training, "mask_lr", 1e-4)),
            "lr_mem": float(getattr(config.training, "mem_lr", 1e-4)),
            "weight_decay": float(getattr(config.training, "weight_decay", 0.0)),
            "seed": int(getattr(config.training, "seed", 42)),
        },
        {"val/iou_best": 0.0},
    )
    global_step = 0

    scaler = None
    if multi_gpu and torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student)
        teacher = torch.nn.DataParallel(teacher)

    ckpt_mgr = CheckpointManager(config.ckpt_path)
    start_ep = 0
    if resume:
        start_ep, opt_sd, sched_sd, global_step = ckpt_mgr.load(resume, student)
        optimizer.load_state_dict(opt_sd)
        scheduler.load_state_dict(sched_sd)
        logger.info(f"Resumed from epoch {start_ep}")

    E = int(config.training.epochs)
    val_freq = int(getattr(config.training, "val_freq", 1))

    for epoch in range(start_ep, E):
        avg_loss, global_step, optimizer, scheduler = train_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            scheduler,
            config,
            scaler,
            epoch,
            logger,
            writer,
            global_step,
        )

        if epoch % val_freq == 0 or epoch == E - 1:
            val_metrics = validate_epoch(student, val_loader, config, epoch, writer)

            def _primary_iou(m):
                """
                Select the primary IoU metric key from validation metrics.

                Args:
                    m (dict): Validation metric dictionary.

                Returns:
                    str | None: Selected metric key, or None when no IoU metric exists.
                """
                for k in ("iou_best", "iou@0.5", "iou@avg_thr", "iou3d@best", "iou3d@0.5", "iou3d@best_vol"):
                    if k in m:
                        return k
                iou_keys = [k for k in m.keys() if "iou" in k]
                return max(iou_keys, key=lambda x: m[x]) if iou_keys else None

            prim_key = _primary_iou(val_metrics)
            prim_val = float(val_metrics[prim_key]) if prim_key else -float("inf")

            ckpt_mgr.save_latest(student, optimizer, scheduler, scaler, epoch, global_step)
            if prim_val > getattr(ckpt_mgr, "best_metric", -1):
                ckpt_mgr.save_best(student, optimizer, scheduler, scaler, epoch, metric=prim_val, global_step=global_step)
            logger.info(f"[Epoch {epoch}] Val ({prim_key}): {prim_val:.4f} | all: {val_metrics}")

    final_epoch_idx = E
    test_metrics = validate_epoch(student, test_loader, config, final_epoch_idx, writer)
    for k, v in test_metrics.items():
        writer.add_scalar(f"test/{k}", float(v), final_epoch_idx)
    logger.info(f"Final test: {test_metrics}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=args.resume,
        multi_gpu=args.multi_gpu,
        amp=args.amp
    )
