"""Helpers for strict cached-prompt video-propagation diagnostics.

This module is intentionally notebook-friendly. It keeps the fair protocol
separate from the older direct per-frame external prompt-decoder benchmark.
"""

from __future__ import annotations

import gc
import faulthandler
import glob
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import warnings
import zipfile
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - notebook optional
    plt = None

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception:  # pragma: no cover - optional metric dependency
    binary_erosion = None
    distance_transform_edt = None

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


SEED = 42
TARGET_HW = 512
LOADERS_PKL = "/data/loaders32.pkl"
EXCLUDED_DATASET = "wanglab__CT_DeepLesion-MedSAM2"
EMPTY_LOGIT = -30.0
PRIMARY_THRESHOLD = 0.5
THRESHOLDS = np.linspace(0.05, 0.95, 19, dtype=np.float32)
MIXED_PROMPT_PROBS = {"box": 0.5, "mask": 0.5}

REPO_ROOT_CANDIDATES = [
    os.getcwd(),
    "/RWKV-MedSAM2",
    "/data/jrbonill/RWKV-MedSAM2",
    "/data/jrbonill/rwkv_medsam2",
    "E:/dev/RWKV-MedSAM2",
]

OXFORD_ENV_BASE = "/data/jrbonill/oxford_medsam2_env"
OXFORD_REPO = f"{OXFORD_ENV_BASE}/Medical-SAM2"
OXFORD_SITE = f"{OXFORD_ENV_BASE}/site"
OXFORD_SAM_CKPT = f"{OXFORD_REPO}/checkpoints/sam2_hiera_tiny.pt"
OXFORD_MED_PRETRAIN = f"{OXFORD_REPO}/checkpoints/MedSAM2_pretrain.pth"

UOFT_MEDSAM2_BASE = "/data/jrbonill/medsam2_env/MedSAM2"
UOFT_MEDSAM2_SITE = "/data/jrbonill/medsam2_env/site"
UOFT_MEDSAM2_CKPT = f"{UOFT_MEDSAM2_BASE}/checkpoints/MedSAM2_latest.pt"
UOFT_MEDSAM2_CFG = "sam2.1_hiera_t512.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_repo_root() -> Path:
    for root in REPO_ROOT_CANDIDATES:
        root = Path(str(root)).expanduser().resolve()
        if (root / "rwkv_medsam2" / "train_sam2.py").is_file():
            return root
    raise FileNotFoundError(f"Could not find RWKV-MedSAM2 repo from {REPO_ROOT_CANDIDATES}")


REPO_ROOT = find_repo_root()
os.chdir(REPO_ROOT)
EXT_ROOT = REPO_ROOT / "ext"
for _p in [str(REPO_ROOT), str(EXT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rwkv_medsam2.train_sam2 import build_student_predictor, get_data_loaders, load_config


def set_determinism(seed: int = SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Flash attention.*", category=UserWarning)


def load_saved_loaders_robust(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        return obj[0], obj[1], obj[2]
    if isinstance(obj, dict):
        key_sets = [
            ("train_loader", "val_loader", "test_loader"),
            ("train_loaders", "val_loaders", "test_loaders"),
            ("train", "val", "test"),
        ]
        for keys in key_sets:
            if all(k in obj for k in keys):
                return obj[keys[0]], obj[keys[1]], obj[keys[2]]
        loader_like = [v for v in obj.values() if hasattr(v, "dataset") and hasattr(v, "__iter__")]
        if len(loader_like) >= 3:
            return loader_like[:3]
    raise RuntimeError(f"Unsupported loader pickle format: {pkl_path}")


def apply_dataset_exclusion(loader, excluded_dataset: str = EXCLUDED_DATASET) -> dict[str, Any]:
    if loader is None or not hasattr(loader, "dataset") or not hasattr(loader.dataset, "sequences"):
        return {"before": None, "after": None, "removed": 0}
    before = len(loader.dataset.sequences)
    kept = [s for s in loader.dataset.sequences if str(s.get("dataset", "")).strip() != excluded_dataset]
    removed = before - len(kept)
    loader.dataset.sequences = kept
    if hasattr(loader.dataset, "entry_dims"):
        loader.dataset.entry_dims = [s.get("dim", 2) for s in kept]
    return {"before": before, "after": len(kept), "removed": removed}


def load_benchmark_loaders(loaders_pkl: str = LOADERS_PKL):
    if os.path.exists(loaders_pkl):
        print("Loading saved loaders:", loaders_pkl)
        train_loader, val_loader, test_loader = load_saved_loaders_robust(loaders_pkl)
    else:
        print("Saved loaders missing; rebuilding from VCR config.")
        cfg = load_config(str(REPO_ROOT / "ext" / "sam2" / "configs" / "sam2.1" / "sam2.1_vcr.yaml"))
        train_loader, val_loader, test_loader = get_data_loaders(cfg)
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(name, apply_dataset_exclusion(loader))
    return train_loader, val_loader, test_loader


def bbox_from_mask_np(mask_hw, pad: int = 5) -> np.ndarray:
    mask = np.asarray(mask_hw) > 0
    ys, xs = np.where(mask)
    h, w = mask.shape[-2:]
    if len(xs) == 0:
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)
    return np.array(
        [
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(w - 1, int(xs.max()) + pad),
            min(h - 1, int(ys.max()) + pad),
        ],
        dtype=np.float32,
    )


def ensure_tchw(tensor) -> torch.Tensor:
    x = tensor.detach().float() if torch.is_tensor(tensor) else torch.as_tensor(tensor).float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1) if x.shape[0] != 3 else x.unsqueeze(0)
    elif x.ndim != 4:
        raise ValueError(f"Expected image/mask with 2-4 dims, got shape {tuple(x.shape)}")
    return x


def ensure_image_t3hw(image) -> torch.Tensor:
    x = ensure_tchw(image)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    if x.shape[1] != 3:
        raise ValueError(f"Expected image channels to resolve to 3, got shape {tuple(x.shape)}")
    return x.contiguous()


def tensor_to_numpy_or_none(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_cpu_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu_nested(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_nested(v) for v in value)
    return value


def clone_cpu_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return a CPU-only copy so cached benchmark cases cannot hold CUDA state."""

    out = dict(case)
    for key in ["image", "mask", "valid"]:
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
    if "gt_thw" in out:
        out["gt_thw"] = np.asarray(out["gt_thw"]).copy()
    if "val_prompt_cache" in out:
        out["val_prompt_cache"] = to_cpu_nested(out["val_prompt_cache"])
    out["meta"] = dict(out.get("meta", {}))
    return out


def normalize_meta_value(value, default="unknown") -> str:
    if value is None:
        return str(default)
    if torch.is_tensor(value):
        value = value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        items = [normalize_meta_value(v, default="") for v in value]
        items = [v for v in items if v]
        return ", ".join(items) if items else str(default)
    text = str(value).strip()
    return text if text else str(default)


def metadata_get(sample: dict[str, Any], source_meta: dict[str, Any] | None, keys, default="unknown"):
    for src in [sample, source_meta or {}]:
        for key in keys:
            if key in src and src.get(key) is not None:
                value = normalize_meta_value(src.get(key), default="")
                if value and value.lower() != "none":
                    return value
    return str(default)


def metadata_tasks(sample: dict[str, Any], source_meta: dict[str, Any] | None) -> list[str]:
    for src in [sample, source_meta or {}]:
        tasks = src.get("tasks") if isinstance(src, dict) else None
        if tasks:
            if not isinstance(tasks, (list, tuple, set)):
                tasks = [tasks]
            out = [normalize_meta_value(task, default="unknown") for task in tasks]
            out = [task for task in out if task and task.lower() != "unknown"]
            if out:
                return out
    task = metadata_get(sample, source_meta, ["task_id", "task_label", "task", "task_name"], default="unknown")
    return [task]


def sample_to_case(sample: dict[str, Any], dataset_index=None, source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    image_t = ensure_image_t3hw(sample["image"])
    mask_t = ensure_tchw(sample["mask"])
    t_len = int(image_t.shape[0])
    if mask_t.shape[0] != t_len and mask_t.shape[0] == 1:
        mask_t = mask_t.repeat(t_len, 1, 1, 1)
    if mask_t.shape[1] != 1:
        mask_t = mask_t[:, :1]
    valid = sample.get("valid", torch.ones(t_len, dtype=torch.bool))
    valid_t = valid.detach().bool().flatten() if torch.is_tensor(valid) else torch.as_tensor(valid).bool().flatten()
    if valid_t.numel() < t_len:
        valid_t = F.pad(valid_t, (0, t_len - valid_t.numel()), value=True)
    valid_t = valid_t[:t_len]
    gt_thw = (mask_t[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    cache = sample.get("val_prompt_cache", None)
    if not isinstance(cache, dict):
        raise RuntimeError(f"Missing val_prompt_cache for dataset_index={dataset_index}; strict protocol requires it.")
    chosen = [int(x) for x in cache.get("chosen_frames", []) if 0 <= int(x) < t_len]
    if not chosen:
        raise RuntimeError(f"Invalid val_prompt_cache for dataset_index={dataset_index}; chosen_frames is empty.")
    source_meta = dict(source_meta or {})
    task_list = metadata_tasks(sample, source_meta)
    task_id = task_list[0] if task_list else "unknown"
    task_label = metadata_get(sample, source_meta, ["task_label", "task_name", "task", "task_id"], default=task_id)
    if task_label.lower() == "unknown" and task_id.lower() != "unknown":
        task_label = task_id
    dim_value = metadata_get(sample, source_meta, ["dim", "dimension", "data_dim"], default=2)
    try:
        dim_value = int(dim_value)
    except Exception:
        dim_value = 2
    return {
        "image": image_t,
        "mask": mask_t,
        "valid": valid_t,
        "gt_thw": gt_thw,
        "boxes": np.stack([bbox_from_mask_np(gt_thw[t]) for t in range(t_len)], axis=0),
        "val_prompt_cache": cache,
        "meta": {
            "dataset_index": int(dataset_index) if dataset_index is not None else None,
            "dataset_name": metadata_get(sample, source_meta, ["dataset", "dataset_name", "name"], default="unknown"),
            "case_id": metadata_get(sample, source_meta, ["case_id", "id", "seq_id", "path"], default=dataset_index),
            "subdataset": metadata_get(sample, source_meta, ["subdataset", "sub_dataset"], default="default"),
            "task_id": task_id,
            "task_label": task_label,
            "task_list": json.dumps(task_list),
            "modality": metadata_get(sample, source_meta, ["modality", "subdataset_modality"], default="unknown"),
            "dim": dim_value,
            "seq_idx": metadata_get(sample, source_meta, ["seq_idx", "sequence_index"], default=dataset_index),
        },
    }


def canonical_prompt_mode(prompt_mode: str) -> str:
    mode = str(prompt_mode).strip().lower()
    if mode in ("bbox", "boxes"):
        return "box"
    if mode in ("masks",):
        return "mask"
    return mode


def deterministic_mixed_prompt_type(case: dict[str, Any], frame_idx: int, seed: int = SEED, probs=None) -> str:
    probs = probs or MIXED_PROMPT_PROBS
    box_prob = float(probs.get("box", 0.5))
    dataset_index = case.get("meta", {}).get("dataset_index", "none")
    key = f"{int(seed)}:{dataset_index}:{int(frame_idx)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha1(key).digest()[:8], "little") / float(2**64 - 1)
    return "box" if value < box_prob else "mask"


def cached_prompt_plan_for_case(case: dict[str, Any], prompt_mode: str, prompt_frames=None) -> dict[str, Any]:
    cache = case["val_prompt_cache"]
    mode = canonical_prompt_mode(prompt_mode)
    if mode not in ("box", "mask", "mixed"):
        raise ValueError(f"Strict protocol supports box/mask/mixed only, got {prompt_mode}")
    t_len = int(case["image"].shape[0])
    cached_frames = [int(x) for x in cache.get("chosen_frames", []) if 0 <= int(x) < t_len]
    frames = cached_frames if prompt_frames is None else [int(x) for x in prompt_frames if 0 <= int(x) < t_len]
    if not frames:
        raise RuntimeError("No prompt frames available for strict protocol")
    boxes = np.full((t_len, 4), np.nan, dtype=np.float32)
    masks = None
    frame_prompt_types: dict[int, str] = {}
    actual = []

    box_payload = cache.get("box", None)
    mask_payload = cache.get("mask", None)
    if mode in ("box", "mixed") and not isinstance(box_payload, dict):
        raise RuntimeError("Prompt cache has no payload for mode=box")
    if mode in ("mask", "mixed") and not isinstance(mask_payload, dict):
        raise RuntimeError("Prompt cache has no payload for mode=mask")

    def _load_box(t: int) -> bool:
        cached_boxes = box_payload.get("bbox", [])
        bb = cached_boxes[t] if t < len(cached_boxes) else None
        bb_np = tensor_to_numpy_or_none(bb)
        if bb_np is None or bb_np.size == 0:
            return False
        boxes[t] = np.asarray(bb_np, dtype=np.float32).reshape(-1, 4)[0]
        return True

    def _load_mask(t: int) -> bool:
        nonlocal masks
        cached_masks = mask_payload.get("m_prompt", [])
        mm = cached_masks[t] if t < len(cached_masks) else None
        mm_np = tensor_to_numpy_or_none(mm)
        if mm_np is None or mm_np.size == 0:
            return False
        if masks is None:
            masks = np.zeros_like(case["gt_thw"], dtype=np.float32)
        masks[t] = np.squeeze(mm_np).astype(np.float32)
        return bool(np.any(masks[t] > 0))

    if mode == "box":
        for t in frames:
            if _load_box(t):
                frame_prompt_types[t] = "box"
                actual.append(t)
    elif mode == "mask":
        for t in frames:
            if _load_mask(t):
                frame_prompt_types[t] = "mask"
                actual.append(t)
    else:
        for t in frames:
            preferred = deterministic_mixed_prompt_type(case, t)
            ok = _load_box(t) if preferred == "box" else _load_mask(t)
            chosen = preferred
            if not ok:
                fallback = "mask" if preferred == "box" else "box"
                ok = _load_mask(t) if fallback == "mask" else _load_box(t)
                chosen = fallback
            if ok:
                frame_prompt_types[t] = chosen
                actual.append(t)
    if not actual:
        raise RuntimeError(f"No usable cached {mode} prompts on requested frames {frames}")
    mixed_box_frames = [int(t) for t, m in sorted(frame_prompt_types.items()) if m == "box"]
    mixed_mask_frames = [int(t) for t, m in sorted(frame_prompt_types.items()) if m == "mask"]
    if masks is None:
        h, w = case["gt_thw"].shape[-2:]
        masks = np.zeros((0, h, w), dtype=np.float32)
    return {
        "requested_prompt_mode": prompt_mode,
        "effective_prompt_mode": mode,
        "prompt_frames": actual,
        "cached_prompt_frames": cached_frames,
        "boxes": boxes,
        "mask_prompts": masks,
        "frame_prompt_types": {int(k): str(v) for k, v in frame_prompt_types.items()},
        "mixed_box_prompt_frames": mixed_box_frames,
        "mixed_mask_prompt_frames": mixed_mask_frames,
        "prompt_source": "val_prompt_cache",
    }


def choose_anchor_frame(case: dict[str, Any], prompt_mode: str, plan: dict[str, Any] | None = None) -> int:
    plan = plan or cached_prompt_plan_for_case(case, prompt_mode)
    gt = case["gt_thw"].astype(bool)
    t_len = gt.shape[0]
    fg = np.where(gt.reshape(t_len, -1).sum(axis=1) > 0)[0].astype(int).tolist()
    candidates = [t for t in plan["prompt_frames"] if t in fg] or list(plan["prompt_frames"])
    candidates = [t for t in candidates if t > 0] or candidates
    center = (t_len - 1) / 2.0
    return int(sorted(candidates, key=lambda t: (abs(t - center), -t))[0])


def case_summary_row(case: dict[str, Any], prompt_mode: str = "box") -> dict[str, Any]:
    gt = case["gt_thw"].astype(bool)
    t_len = gt.shape[0]
    fg_counts = gt.reshape(t_len, -1).sum(axis=1)
    fg_frames = np.where(fg_counts > 0)[0].astype(int).tolist()
    return {
        **case["meta"],
        "sequence_length": int(t_len),
        "cached_prompt_frames": json.dumps([int(x) for x in case["val_prompt_cache"].get("chosen_frames", [])]),
        "anchor_frame": int(choose_anchor_frame(case, prompt_mode)),
        "foreground_frames": json.dumps(fg_frames[:80]),
        "foreground_frame_count": int(len(fg_frames)),
        "foreground_fraction_per_frame_head": json.dumps(
            [float(x / (gt.shape[1] * gt.shape[2])) for x in fg_counts[:20]]
        ),
    }


def case_input_diagnostics(case: dict[str, Any], prompt_mode: str | None = None, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    img = case["image"].detach().cpu().float().numpy()
    gt = np.asarray(case["gt_thw"]).astype(bool)
    valid = case["valid"].detach().cpu().numpy().astype(bool)
    out = {
        "image_min": float(np.nanmin(img)),
        "image_mean": float(np.nanmean(img)),
        "image_std": float(np.nanstd(img)),
        "image_max": float(np.nanmax(img)),
        "valid_frame_count": int(valid.sum()),
        "sequence_length": int(gt.shape[0]),
        "gt_foreground_frame_count": int((gt.reshape(gt.shape[0], -1).sum(axis=1) > 0).sum()),
        "gt_fg_frac_input": float(gt.mean()),
    }
    if prompt_mode is None:
        return out
    try:
        plan = plan or cached_prompt_plan_for_case(case, prompt_mode)
        prompt_frames = [int(t) for t in plan.get("prompt_frames", [])]
        out["prompt_frame_count"] = int(len(prompt_frames))
        out["prompt_frames_json"] = json.dumps(prompt_frames)
        h, w = gt.shape[-2:]
        box_areas = []
        for t in prompt_frames:
            bb = np.asarray(plan["boxes"][t], dtype=np.float32).reshape(-1)
            if bb.size >= 4 and np.isfinite(bb[:4]).all():
                x0, y0, x1, y1 = bb[:4]
                box_areas.append(max(0.0, float(x1 - x0 + 1.0)) * max(0.0, float(y1 - y0 + 1.0)) / float(h * w))
        mask_areas = [float((plan["mask_prompts"][t] > 0).mean()) for t in prompt_frames if t < len(plan["mask_prompts"]) and np.any(plan["mask_prompts"][t] > 0)]
        out["prompt_box_area_frac_mean"] = float(np.mean(box_areas)) if box_areas else float("nan")
        out["prompt_mask_area_frac_mean"] = float(np.mean(mask_areas)) if mask_areas else float("nan")
    except Exception:
        out["prompt_frame_count"] = 0
        out["prompt_frames_json"] = "[]"
        out["prompt_box_area_frac_mean"] = float("nan")
        out["prompt_mask_area_frac_mean"] = float("nan")
    return out


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def prepare_metric_arrays(logits, gt) -> dict[str, np.ndarray]:
    logits_arr = np.nan_to_num(np.asarray(logits, dtype=np.float32), nan=EMPTY_LOGIT)
    gt_arr = np.asarray(gt).astype(bool)
    return {
        "logits": logits_arr,
        "gt": gt_arr,
        "probs": sigmoid_np(logits_arr),
    }


def confusion_counts(pred, gt):
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    return (
        int(np.logical_and(p, g).sum()),
        int(np.logical_and(p, ~g).sum()),
        int(np.logical_and(~p, g).sum()),
        int(np.logical_and(~p, ~g).sum()),
    )


def safe_div(num, den, default=float("nan")):
    den = float(den)
    return float(num) / den if den != 0 else default


def dice_iou_from_counts(tp, fp, fn, pred_sum=None, gt_sum=None):
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    if pred_sum is None:
        pred_sum = tp + fp
    if gt_sum is None:
        gt_sum = tp + fn
    both_empty = int(pred_sum) == 0 and int(gt_sum) == 0
    dice = safe_div(2 * tp, 2 * tp + fp + fn, default=1.0 if both_empty else 0.0)
    iou = safe_div(tp, tp + fp + fn, default=1.0 if both_empty else 0.0)
    return float(dice), float(iou)


def binary_dice_iou_np(pred, gt):
    tp, fp, fn, _tn = confusion_counts(pred, gt)
    return dice_iou_from_counts(tp, fp, fn, pred_sum=np.asarray(pred).sum(), gt_sum=np.asarray(gt).sum())


def per_frame_confusion_from_metric_arrays(metric_arrays, threshold: float = PRIMARY_THRESHOLD):
    threshold = float(threshold)
    cache = metric_arrays.setdefault("_per_frame_confusion_cache", {})
    if threshold in cache:
        return cache[threshold]

    gt = metric_arrays["gt"]
    probs = metric_arrays["probs"]
    t_len = int(gt.shape[0])
    flat_gt = gt.reshape(t_len, -1)
    flat_probs = probs.reshape(t_len, -1)
    frame_pixels = int(flat_gt.shape[1])
    gt_count = flat_gt.sum(axis=1, dtype=np.int64)
    pred = flat_probs >= threshold
    pred_count = pred.sum(axis=1, dtype=np.int64)
    tp = np.logical_and(pred, flat_gt).sum(axis=1, dtype=np.int64)
    fp = pred_count - tp
    fn = gt_count - tp
    tn = np.int64(frame_pixels) - tp - fp - fn
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    both_empty = (pred_count == 0) & (gt_count == 0)
    dice = np.divide(2 * tp, dice_den, out=np.where(both_empty, 1.0, 0.0).astype(np.float64), where=dice_den != 0)
    iou = np.divide(tp, iou_den, out=np.where(both_empty, 1.0, 0.0).astype(np.float64), where=iou_den != 0)
    out = {
        "threshold": threshold,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "has_gt": gt_count > 0,
        "has_pred": pred_count > 0,
    }
    cache[threshold] = out
    return out


def threshold_summary_from_metric_arrays(metric_arrays, thresholds=THRESHOLDS):
    thresholds_key = tuple(float(x) for x in thresholds)
    cache = metric_arrays.setdefault("_threshold_summary_cache", {})
    if thresholds_key in cache:
        return cache[thresholds_key]

    gt = metric_arrays["gt"]
    probs = metric_arrays["probs"]
    t_len = int(gt.shape[0])
    flat_gt = gt.reshape(t_len, -1)
    flat_probs = probs.reshape(t_len, -1)
    frame_pixels = int(flat_gt.shape[1])
    voxel_count = int(flat_gt.size)
    gt_count_frame = flat_gt.sum(axis=1, dtype=np.int64)
    fg_frames = gt_count_frame > 0
    gt_total = int(gt_count_frame.sum())
    fg_gt_total = int(gt_count_frame[fg_frames].sum()) if bool(fg_frames.any()) else 0

    rows = []
    for thr in thresholds_key:
        pred = flat_probs >= float(thr)
        pred_count_frame = pred.sum(axis=1, dtype=np.int64)
        tp_frame = np.logical_and(pred, flat_gt).sum(axis=1, dtype=np.int64)
        fp_frame = pred_count_frame - tp_frame
        fn_frame = gt_count_frame - tp_frame
        tn_frame = np.int64(frame_pixels) - tp_frame - fp_frame - fn_frame

        tp = int(tp_frame.sum())
        fp = int(fp_frame.sum())
        fn = int(fn_frame.sum())
        tn = int(tn_frame.sum())
        pred_total = int(pred_count_frame.sum())
        full_dice, full_iou = dice_iou_from_counts(tp, fp, fn, pred_sum=pred_total, gt_sum=gt_total)

        if bool(fg_frames.any()):
            tp_fg = int(tp_frame[fg_frames].sum())
            fp_fg = int(fp_frame[fg_frames].sum())
            fn_fg = int(fn_frame[fg_frames].sum())
            pred_fg_total = int(pred_count_frame[fg_frames].sum())
            fg_dice, fg_iou = dice_iou_from_counts(tp_fg, fp_fg, fn_fg, pred_sum=pred_fg_total, gt_sum=fg_gt_total)
            dice_den_frame = 2 * tp_frame[fg_frames] + fp_frame[fg_frames] + fn_frame[fg_frames]
            iou_den_frame = tp_frame[fg_frames] + fp_frame[fg_frames] + fn_frame[fg_frames]
            fg_frame_dice = np.divide(
                2 * tp_frame[fg_frames],
                dice_den_frame,
                out=np.zeros_like(dice_den_frame, dtype=np.float64),
                where=dice_den_frame != 0,
            )
            fg_frame_iou = np.divide(
                tp_frame[fg_frames],
                iou_den_frame,
                out=np.zeros_like(iou_den_frame, dtype=np.float64),
                where=iou_den_frame != 0,
            )
            fg_mean_frame_dice = float(np.mean(fg_frame_dice))
            fg_mean_frame_iou = float(np.mean(fg_frame_iou))
        else:
            fg_dice = fg_iou = fg_mean_frame_dice = fg_mean_frame_iou = float("nan")

        rows.append(
            {
                "threshold": float(thr),
                "full_sequence_dice": float(full_dice),
                "full_sequence_iou": float(full_iou),
                "fg_volume_dice": float(fg_dice),
                "fg_volume_iou": float(fg_iou),
                "fg_mean_frame_dice": float(fg_mean_frame_dice),
                "fg_mean_frame_iou": float(fg_mean_frame_iou),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pred_fg_frac": float(pred_total / voxel_count) if voxel_count else float("nan"),
            }
        )

    summary = {
        "thresholds": thresholds_key,
        "rows": rows,
        "fg_frames": fg_frames,
        "gt_count_frame": gt_count_frame,
        "gt_total": gt_total,
        "voxel_count": voxel_count,
    }
    cache[thresholds_key] = summary
    return summary


def threshold_suffix(thr) -> str:
    return f"{float(thr):.2f}".replace(".", "p")


def threshold_metric_key(metric: str, thr) -> str:
    return f"{metric}_thr_{threshold_suffix(thr)}"


def logit_from_prob(prob: float) -> float:
    p = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def hd95_binary(pred, gt, spacing=None) -> float:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"HD95 shape mismatch: {pred.shape} vs {gt.shape}")
    if not pred.any() and not gt.any():
        return 0.0
    if pred.any() != gt.any():
        return float("nan")
    if distance_transform_edt is None or binary_erosion is None:
        return float("nan")
    if spacing is None:
        spacing = (1.0,) * pred.ndim
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    gt_border = np.logical_xor(gt, binary_erosion(gt))
    if not pred_border.any() or not gt_border.any():
        return 0.0 if np.array_equal(pred, gt) else float("nan")
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    distances = np.concatenate([dt_gt[pred_border], dt_pred[gt_border]]).astype(np.float64)
    return float(np.percentile(distances, 95)) if distances.size else float("nan")


def per_frame_metrics_from_logits(logits, gt, threshold: float = PRIMARY_THRESHOLD, metric_arrays=None):
    metric_arrays = metric_arrays or prepare_metric_arrays(logits, gt)
    counts = per_frame_confusion_from_metric_arrays(metric_arrays, threshold)
    rows = []
    for t in range(len(counts["gt_count"])):
        rows.append(
            {
                "frame_idx": int(t),
                "threshold": float(threshold),
                "frame_has_gt": bool(counts["has_gt"][t]),
                "frame_has_pred": bool(counts["has_pred"][t]),
                "frame_dice": float(counts["dice"][t]),
                "frame_iou": float(counts["iou"][t]),
                "frame_pred_fg_count": int(counts["pred_count"][t]),
                "frame_gt_fg_count": int(counts["gt_count"][t]),
                "frame_tp": int(counts["tp"][t]),
                "frame_fp": int(counts["fp"][t]),
                "frame_fn": int(counts["fn"][t]),
                "frame_tn": int(counts["tn"][t]),
            }
        )
    return rows


def threshold_metrics_long_from_logits(logits, gt, thresholds=THRESHOLDS, metric_arrays=None):
    metric_arrays = metric_arrays or prepare_metric_arrays(logits, gt)
    return [
        {
            key: row[key]
            for key in [
                "threshold",
                "full_sequence_dice",
                "full_sequence_iou",
                "fg_volume_dice",
                "fg_volume_iou",
                "fg_mean_frame_dice",
                "tp",
                "fp",
                "fn",
                "tn",
                "pred_fg_frac",
            ]
        }
        for row in threshold_summary_from_metric_arrays(metric_arrays, thresholds)["rows"]
    ]


def empty_training_style_metrics(total_frames: int, nonempty_frames: int = 0):
    return {
        "train_nonempty_frames": int(nonempty_frames),
        "train_total_frames": int(total_frames),
        "train_slice_dice@0.5": float("nan"),
        "train_slice_iou@0.5": float("nan"),
        "train_slice_dice_best_case": float("nan"),
        "train_slice_iou_best_case": float("nan"),
        "train_best_prob_threshold_case": float("nan"),
        "train_volume_dice@0.5": float("nan"),
        "train_volume_iou@0.5": float("nan"),
        "train_volume_dice_best_case": float("nan"),
        "train_volume_iou_best_case": float("nan"),
    }


def training_style_metrics_from_logits(logits, gt, thresholds=THRESHOLDS, metric_arrays=None):
    metric_arrays = metric_arrays or prepare_metric_arrays(logits, gt)
    gt = metric_arrays["gt"]
    primary = per_frame_confusion_from_metric_arrays(metric_arrays, PRIMARY_THRESHOLD)
    nonempty = primary["has_gt"]
    out = {
        "train_nonempty_frames": int(nonempty.sum()),
        "train_total_frames": int(gt.shape[0]),
    }
    if not np.any(nonempty):
        out.update(empty_training_style_metrics(gt.shape[0], 0))
        return out

    summary = threshold_summary_from_metric_arrays(metric_arrays, thresholds)
    best_d = -1.0
    best_i = -1.0
    best_thr = float(PRIMARY_THRESHOLD)
    primary_row = None
    best_row = None
    for row in summary["rows"]:
        thr = float(row["threshold"])
        if np.isclose(thr, PRIMARY_THRESHOLD, rtol=0.0, atol=1e-12):
            primary_row = row
        d_m = float(row["fg_mean_frame_dice"])
        i_m = float(row.get("fg_mean_frame_iou", float("nan")))
        if d_m > best_d or (np.isclose(d_m, best_d) and abs(thr - PRIMARY_THRESHOLD) < abs(best_thr - PRIMARY_THRESHOLD)):
            best_d = d_m
            best_i = i_m
            best_thr = thr
            best_row = row

    if primary_row is None:
        tp = int(primary["tp"][nonempty].sum())
        fp = int(primary["fp"][nonempty].sum())
        fn = int(primary["fn"][nonempty].sum())
        pred_sum = int(primary["pred_count"][nonempty].sum())
        gt_sum = int(primary["gt_count"][nonempty].sum())
        vol_d05, vol_i05 = dice_iou_from_counts(tp, fp, fn, pred_sum=pred_sum, gt_sum=gt_sum)
    else:
        vol_d05 = float(primary_row["fg_volume_dice"])
        vol_i05 = float(primary_row["fg_volume_iou"])

    out.update(
        {
            "train_slice_dice@0.5": float(np.mean(primary["dice"][nonempty])),
            "train_slice_iou@0.5": float(np.mean(primary["iou"][nonempty])),
            "train_slice_dice_best_case": float(best_d),
            "train_slice_iou_best_case": float(best_i),
            "train_best_prob_threshold_case": float(best_thr),
            "train_volume_dice@0.5": float(vol_d05),
            "train_volume_iou@0.5": float(vol_i05),
            "train_volume_dice_best_case": float(best_row["fg_volume_dice"]) if best_row is not None else float("nan"),
            "train_volume_iou_best_case": float(best_row["fg_volume_iou"]) if best_row is not None else float("nan"),
        }
    )
    return out


def metrics_from_logits(
    logits,
    gt,
    thresholds=THRESHOLDS,
    primary_threshold: float = PRIMARY_THRESHOLD,
    compute_hd95_2d: bool = True,
    compute_hd95_2d_for_3d: bool = False,
    compute_hd95_3d: bool = False,
    threshold_rows=None,
    metric_arrays=None,
):
    metric_started = time.perf_counter()
    metric_arrays = metric_arrays or prepare_metric_arrays(logits, gt)
    logits = metric_arrays["logits"]
    gt = metric_arrays["gt"]
    probs = metric_arrays["probs"]
    primary_counts = per_frame_confusion_from_metric_arrays(metric_arrays, primary_threshold)
    fg_frames = primary_counts["has_gt"]
    tp = int(primary_counts["tp"].sum())
    fp = int(primary_counts["fp"].sum())
    fn = int(primary_counts["fn"].sum())
    tn = int(primary_counts["tn"].sum())
    pred_count = int(primary_counts["pred_count"].sum())
    gt_count = int(primary_counts["gt_count"].sum())
    voxel_count = int(gt.size)
    full_dice, full_iou = dice_iou_from_counts(tp, fp, fn, pred_sum=pred_count, gt_sum=gt_count)
    if bool(fg_frames.any()):
        tp_fg = int(primary_counts["tp"][fg_frames].sum())
        fp_fg = int(primary_counts["fp"][fg_frames].sum())
        fn_fg = int(primary_counts["fn"][fg_frames].sum())
        pred_fg_count = int(primary_counts["pred_count"][fg_frames].sum())
        gt_fg_count = int(primary_counts["gt_count"][fg_frames].sum())
        fg_dice, fg_iou = dice_iou_from_counts(tp_fg, fp_fg, fn_fg, pred_sum=pred_fg_count, gt_sum=gt_fg_count)
        fg_slice = primary_counts["dice"][fg_frames]
    else:
        fg_dice = fg_iou = float("nan")
        fg_slice = np.asarray([], dtype=np.float64)
    empty_frames = ~fg_frames
    if bool(empty_frames.any()):
        empty_pred_any = primary_counts["pred_count"][empty_frames] > 0
        empty_frame_fpr = float(empty_pred_any.mean())
        fp_e = int(primary_counts["fp"][empty_frames].sum())
        tn_e = int(primary_counts["tn"][empty_frames].sum())
        empty_specificity = safe_div(tn_e, tn_e + fp_e, default=float("nan"))
    else:
        empty_frame_fpr = float("nan")
        empty_specificity = float("nan")

    threshold_case_metrics = {}
    dice_vals = []
    iou_vals = []
    threshold_rows = threshold_rows if threshold_rows is not None else threshold_metrics_long_from_logits(logits, gt, thresholds=thresholds, metric_arrays=metric_arrays)
    for thr_row in threshold_rows:
        thr = float(thr_row["threshold"])
        threshold_case_metrics[threshold_metric_key("dice", thr)] = float(thr_row["full_sequence_dice"])
        threshold_case_metrics[threshold_metric_key("iou", thr)] = float(thr_row["full_sequence_iou"])
        threshold_case_metrics[threshold_metric_key("fg_dice", thr)] = float(thr_row["fg_volume_dice"])
        threshold_case_metrics[threshold_metric_key("fg_iou", thr)] = float(thr_row["fg_volume_iou"])
        dice_vals.append(float(thr_row["full_sequence_dice"]))
        iou_vals.append(float(thr_row["full_sequence_iou"]))

    hd95_2d_sec = 0.0
    hd95_3d_sec = 0.0
    hd95_value = float("nan")
    hd95_3d_value = float("nan")
    should_compute_2d_hd95 = bool(compute_hd95_2d) and (gt.shape[0] == 1 or bool(compute_hd95_2d_for_3d))
    if should_compute_2d_hd95:
        hd_started = time.perf_counter()
        pred = probs >= float(primary_threshold)
        hd_vals = [hd95_binary(pred[t], gt[t]) for t in range(gt.shape[0])]
        finite = [x for x in hd_vals if not (math.isnan(float(x)) or math.isinf(float(x)))]
        hd95_value = float(np.mean(finite)) if finite else float("nan")
        hd95_2d_sec = float(time.perf_counter() - hd_started)
    if bool(compute_hd95_3d) and gt.shape[0] > 1:
        hd_started = time.perf_counter()
        pred = probs >= float(primary_threshold)
        hd95_3d_value = hd95_binary(pred, gt)
        hd95_3d_sec = float(time.perf_counter() - hd_started)

    out = {
        "dice": float(full_dice),
        "iou": float(full_iou),
        "hd95": float(hd95_value),
        "hd95_3d": float(hd95_3d_value),
        "hd95_2d_computed": float(should_compute_2d_hd95),
        "hd95_3d_computed": float(bool(compute_hd95_3d) and gt.shape[0] > 1),
        "hd95_2d_time_sec": float(hd95_2d_sec),
        "hd95_3d_time_sec": float(hd95_3d_sec),
        "dice_avg_threshold": float(np.nanmean(dice_vals)) if dice_vals else float("nan"),
        "iou_avg_threshold": float(np.nanmean(iou_vals)) if iou_vals else float("nan"),
        "logit_min": float(np.nanmin(logits)),
        "logit_mean": float(np.nanmean(logits)),
        "logit_max": float(np.nanmax(logits)),
        "prob_min": float(np.nanmin(probs)),
        "prob_mean": float(np.nanmean(probs)),
        "prob_max": float(np.nanmax(probs)),
        "foreground_frame_count": int(fg_frames.sum()),
        "empty_frame_count": int(empty_frames.sum()),
        "fg_volume_dice": float(fg_dice),
        "fg_volume_iou": float(fg_iou),
        "fg_mean_frame_dice": float(np.mean(fg_slice)) if len(fg_slice) else float("nan"),
        "full_sequence_dice": float(full_dice),
        "full_sequence_iou": float(full_iou),
        "empty_frame_false_positive_rate": empty_frame_fpr,
        "empty_frame_specificity": float(empty_specificity),
        "pred_fg_count": int(pred_count),
        "gt_fg_count": int(gt_count),
        "pred_fg_frac": float(pred_count / voxel_count) if voxel_count else float("nan"),
        "gt_fg_frac": float(gt_count / voxel_count) if voxel_count else float("nan"),
        "false_negative_rate": safe_div(fn, tp + fn, default=0.0),
        "false_positive_rate": safe_div(fp, fp + tn, default=0.0),
        "precision_ppv": safe_div(tp, tp + fp, default=1.0 if pred_count == 0 and gt_count == 0 else 0.0),
        "recall_sensitivity": safe_div(tp, tp + fn, default=1.0 if gt_count == 0 else 0.0),
        "specificity": safe_div(tn, tn + fp, default=0.0),
        "volumetric_similarity": 1.0 - safe_div(abs(pred_count - gt_count), pred_count + gt_count, default=0.0),
        "empty_pred": float(pred_count == 0),
        "empty_gt": float(gt_count == 0),
        "missed_object": float((gt_count > 0) and (pred_count == 0)),
        "hallucination": float((gt_count == 0) and (pred_count > 0)),
        "n_frames": int(gt.shape[0]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    out.update(threshold_case_metrics)
    out.update(training_style_metrics_from_logits(logits, gt, thresholds=thresholds, metric_arrays=metric_arrays))
    out["metric_time_sec"] = float(time.perf_counter() - metric_started)
    return out


@dataclass
class ModelEntry:
    name: str
    family: str
    config_path: str = ""
    checkpoint_dir: str = ""
    checkpoint_path: str = ""
    checkpoint_policy: str = "best_or_latest"
    builder: str = ""
    enabled: bool = True
    notes: str = ""


def default_model_registry() -> dict[str, ModelEntry]:
    return {
        "rwkv_medsam2_distill": ModelEntry(
            name="rwkv_medsam2_distill",
            family="native",
            config_path=str(REPO_ROOT / "ext/sam2/configs/sam2.1/sam2.1_vcr.yaml"),
            checkpoint_dir=str(REPO_ROOT / "checkpoints" / "sam2.1" / "sam2.1_vcr"),
            builder="rwkv_student",
            notes="RWKV-MedSAM2 with distillation.",
        ),
        "rwkv_medsam2_nodistill": ModelEntry(
            name="rwkv_medsam2_nodistill",
            family="native",
            config_path=str(REPO_ROOT / "ext/sam2/configs/sam2.1/sam2.1_vcr_nodistill.yaml"),
            checkpoint_dir=str(REPO_ROOT / "checkpoints" / "sam2.1" / "sam2.1_vcr_nodistill.pth"),
            builder="rwkv_student",
            notes="RWKV-MedSAM2 without distillation.",
        ),
        "sam2_1_base": ModelEntry(
            name="sam2_1_base",
            family="native",
            config_path=str(REPO_ROOT / "ext/sam2/configs/sam2.1/sam2.1_hiera_t512.yaml"),
            checkpoint_dir=str(REPO_ROOT / "checkpoints" / "sam2.1" / "sam2.1_hiera_t512_base"),
            builder="sam21_base_student",
            notes="SAM 2.1 base trained in this project.",
        ),
        "oxford_medical_sam2": ModelEntry(
            name="oxford_medical_sam2",
            family="oxford",
            config_path="sam2_hiera_t",
            checkpoint_dir=os.path.dirname(OXFORD_MED_PRETRAIN),
            checkpoint_path=OXFORD_MED_PRETRAIN,
            builder="oxford_video_predictor",
            notes="Oxford Medical SAM2 video predictor API.",
        ),
        "uoft_medsam2": ModelEntry(
            name="uoft_medsam2",
            family="uoft",
            config_path=UOFT_MEDSAM2_CFG,
            checkpoint_dir=os.path.dirname(UOFT_MEDSAM2_CKPT),
            checkpoint_path=UOFT_MEDSAM2_CKPT,
            builder="uoft_video_predictor_npz",
            notes="UofT MedSAM2 NPZ video predictor API.",
        ),
    }


MODEL_REGISTRY = default_model_registry()


def checkpoint_candidates_from_dir(checkpoint_dir):
    candidates = []
    for name in ["best.pth", "best_val.pth", "latest.pth"]:
        p = os.path.join(checkpoint_dir, name)
        if os.path.isfile(p):
            candidates.append(p)
    candidates.extend(sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))))
    return candidates


def select_checkpoint_candidate(candidates, policy="best_or_latest"):
    candidates = [p for p in candidates if p and os.path.isfile(p)]
    if not candidates:
        return None
    if policy == "latest":
        return max(candidates, key=os.path.getmtime)
    for p in candidates:
        if os.path.basename(p) in ("best.pth", "best_val.pth"):
            return p
    return max(candidates, key=os.path.getmtime)


def checkpoint_search_names(entry: ModelEntry):
    raw = os.path.basename(str(entry.checkpoint_dir or entry.checkpoint_path)).rstrip(os.sep)
    names = {raw}
    names.add(raw[:-4] if raw.endswith(".pth") else raw + ".pth")
    if entry.name == "rwkv_medsam2_distill":
        names.update(["sam2.1_vcr", "sam2.1_vcr.pth"])
    elif entry.name == "rwkv_medsam2_nodistill":
        names.update(["sam2.1_vcr_nodistill", "sam2.1_vcr_nodistill.pth"])
    elif entry.name == "sam2_1_base":
        names.update(["sam2.1_hiera_t512_base", "sam2.1_hiera_t512_base.pth"])
    return [n for n in names if n]


def pick_checkpoint(entry: ModelEntry):
    if entry.checkpoint_path and os.path.isfile(entry.checkpoint_path):
        return entry.checkpoint_path
    if entry.checkpoint_dir and os.path.isfile(entry.checkpoint_dir):
        return entry.checkpoint_dir
    if entry.checkpoint_dir and os.path.isdir(entry.checkpoint_dir):
        found = select_checkpoint_candidate(checkpoint_candidates_from_dir(entry.checkpoint_dir), entry.checkpoint_policy)
        if found:
            return found
    ckpt_root = REPO_ROOT / "checkpoints"
    if ckpt_root.exists():
        for base in [ckpt_root, ckpt_root / "sam2.1"]:
            for name in checkpoint_search_names(entry):
                candidate = base / name
                if candidate.is_file():
                    return str(candidate)
                if candidate.is_dir():
                    found = select_checkpoint_candidate(
                        checkpoint_candidates_from_dir(str(candidate)), entry.checkpoint_policy
                    )
                    if found:
                        return found
    return None


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "network"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_rwkv_config(entry: ModelEntry):
    if entry.builder == "sam21_base_student":
        cfg = load_config(str(REPO_ROOT / "ext/sam2/configs/sam2.1/sam2.1_vcr.yaml"))
        base_cfg = OmegaConf.load(entry.config_path)
        cfg.model = base_cfg.model
        cfg._config_path = "sam2.1/sam2.1_hiera_t512.yaml"
        cfg.ckpt_path = entry.checkpoint_dir
        cfg.logging.filename = "sam2.1_hiera_t512_base.log"
        cfg.init.ignore_prefixes = []
        return cfg
    return load_config(entry.config_path)


def initialize_rwkv_sam2_hydra():
    config_dir = str(REPO_ROOT / "ext" / "sam2" / "configs")
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.2")


def load_native_video_model(entry: ModelEntry):
    ckpt_path = pick_checkpoint(entry)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for {entry.name}: {entry.checkpoint_dir}")
    cfg = load_rwkv_config(entry)
    cfg.training.device = str(DEVICE)
    initialize_rwkv_sam2_hydra()
    model = build_student_predictor(cfg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(unwrap_state_dict(ckpt), strict=False)
    model.eval()
    return {"entry": entry, "predictor": model, "checkpoint": ckpt_path, "missing": len(missing), "unexpected": len(unexpected)}


def disable_fill_holes(obj) -> None:
    for candidate in [obj, getattr(obj, "model", None)]:
        if candidate is not None and hasattr(candidate, "fill_hole_area"):
            try:
                candidate.fill_hole_area = 0
            except Exception:
                pass


def load_oxford_video_model(entry: ModelEntry):
    for p in [OXFORD_REPO, OXFORD_SITE]:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    if not os.path.isfile(OXFORD_SAM_CKPT):
        raise FileNotFoundError(OXFORD_SAM_CKPT)
    if not os.path.isfile(OXFORD_MED_PRETRAIN):
        raise FileNotFoundError(OXFORD_MED_PRETRAIN)
    GlobalHydra.instance().clear()
    from sam2_train.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=entry.config_path,
        ckpt_path=OXFORD_SAM_CKPT,
        device=str(DEVICE),
        mode="train",
        hydra_overrides_extra=[
            f"model.image_size={TARGET_HW}",
            "model.memory_attention.layer.self_attention.feat_sizes=[16,16]",
            "model.memory_attention.layer.cross_attention.feat_sizes=[16,16]",
        ],
    )
    target = predictor.model if hasattr(predictor, "model") else predictor
    sd = unwrap_state_dict(torch.load(OXFORD_MED_PRETRAIN, map_location="cpu"))
    if any(str(k).startswith("module.") for k in sd.keys()):
        sd = {str(k).replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = target.load_state_dict(sd, strict=False)
    target.to(DEVICE).eval()
    predictor.eval()
    disable_fill_holes(predictor)
    return {
        "entry": entry,
        "predictor": predictor,
        "checkpoint": OXFORD_MED_PRETRAIN,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def purge_top_level_module(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def load_uoft_video_model(entry: ModelEntry):
    if not os.path.isdir(UOFT_MEDSAM2_BASE):
        raise FileNotFoundError(UOFT_MEDSAM2_BASE)
    if not os.path.isfile(UOFT_MEDSAM2_CKPT):
        raise FileNotFoundError(UOFT_MEDSAM2_CKPT)
    config_dir = os.path.join(UOFT_MEDSAM2_BASE, "sam2", "configs")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(config_dir)
    purge_top_level_module("sam2")
    for p in [UOFT_MEDSAM2_SITE, UOFT_MEDSAM2_BASE]:
        if p in sys.path:
            sys.path.remove(p)
    for p in [UOFT_MEDSAM2_SITE, UOFT_MEDSAM2_BASE]:
        if p and os.path.isdir(p):
            sys.path.insert(0, p)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        from sam2.build_sam import build_sam2_video_predictor_npz

        predictor = build_sam2_video_predictor_npz(entry.config_path, UOFT_MEDSAM2_CKPT, device=str(DEVICE), mode="eval")
    predictor.eval()
    disable_fill_holes(predictor)
    return {"entry": entry, "predictor": predictor, "checkpoint": UOFT_MEDSAM2_CKPT, "missing": None, "unexpected": None}


def load_model_bundle(model_name: str):
    entry = MODEL_REGISTRY[model_name]
    if entry.family == "native":
        return load_native_video_model(entry)
    if entry.family == "oxford":
        return load_oxford_video_model(entry)
    if entry.family == "uoft":
        return load_uoft_video_model(entry)
    raise ValueError(entry.family)


def api_probe_row(name: str, bundle: dict[str, Any], status="ok", error="") -> dict[str, Any]:
    entry = bundle["entry"]
    predictor = bundle.get("predictor")
    init_methods = [m for m in ["init_state_from_tensor", "val_init_state", "train_init_state", "init_state"] if hasattr(predictor, m)]
    prompt_methods = [m for m in ["add_new_points_or_box", "add_new_bbox", "train_add_new_bbox", "add_new_mask"] if hasattr(predictor, m)]
    try:
        predictor_file = inspect.getfile(type(predictor))
    except Exception:
        predictor_file = ""
    return {
        **asdict(entry),
        "status": status,
        "error": error,
        "checkpoint": bundle.get("checkpoint", ""),
        "missing_keys": bundle.get("missing"),
        "unexpected_keys": bundle.get("unexpected"),
        "predictor_type": type(predictor).__name__ if predictor is not None else "",
        "predictor_file": predictor_file,
        "init_methods": ";".join(init_methods),
        "prompt_methods": ";".join(prompt_methods),
        "has_propagate": bool(hasattr(predictor, "propagate_in_video") or hasattr(predictor, "train_propagate_in_video")),
    }


def load_requested_models(model_names: list[str]):
    bundles = {}
    rows = []
    for name in model_names:
        entry = MODEL_REGISTRY[name]
        started = time.perf_counter()
        try:
            bundle = load_model_bundle(name)
            bundles[name] = bundle
            row = api_probe_row(name, bundle, status="ok", error="")
            row["load_time_sec"] = float(time.perf_counter() - started)
            print(f"Loaded {name}: {bundle.get('checkpoint')}")
        except Exception as exc:
            row = {
                **asdict(entry),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "load_time_sec": float(time.perf_counter() - started),
                "predictor_type": "",
                "predictor_file": "",
                "init_methods": "",
                "prompt_methods": "",
                "has_propagate": False,
            }
            print(f"Failed to load {name}: {row['error']}")
        rows.append(row)
    return bundles, pd.DataFrame(rows)


def external_input_to_01(imgs_t3hw: torch.Tensor) -> torch.Tensor:
    x = imgs_t3hw.detach().float().clone()
    finite = torch.isfinite(x)
    x = torch.where(finite, x, torch.zeros_like(x))
    mn = float(x.min().item())
    mx = float(x.max().item())
    if mn >= 0.0 and mx <= 1.5:
        return x.clamp(0.0, 1.0)
    if mn >= 0.0 and mx <= 255.5:
        return (x / 255.0).clamp(0.0, 1.0)
    return ((x - mn) / max(mx - mn, 1e-6)).clamp(0.0, 1.0)


def imagenet_normalize_01(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device)[:, None, None]
    return (x - mean) / std


def prepare_images_for_model(bundle: dict[str, Any], case: dict[str, Any]):
    imgs = ensure_image_t3hw(case["image"])
    family = bundle["entry"].family
    if family == "native":
        return imgs.to(DEVICE, non_blocking=True).float(), "native_loader_tensor"
    if family == "oxford":
        return (external_input_to_01(imgs) * 255.0).to(DEVICE, non_blocking=True).float(), "external_0_255"
    if family == "uoft":
        return imagenet_normalize_01(external_input_to_01(imgs).to(DEVICE, non_blocking=True).float()), "external_imagenet_0_1"
    raise ValueError(family)


def init_video_state(bundle: dict[str, Any], imgs: torch.Tensor, h: int, w: int):
    predictor = bundle["predictor"]
    family = bundle["entry"].family
    if family == "native" and hasattr(predictor, "init_state_from_tensor"):
        return predictor.init_state_from_tensor(imgs_tensor=imgs, offload_video_to_cpu=False, offload_state_to_cpu=False, mode="eval"), "init_state_from_tensor"
    if family == "oxford":
        if hasattr(predictor, "val_init_state"):
            return predictor.val_init_state(imgs_tensor=imgs, video_height=int(h), video_width=int(w), offload_video_to_cpu=False, offload_state_to_cpu=False), "val_init_state"
        if hasattr(predictor, "train_init_state"):
            return predictor.train_init_state(imgs_tensor=imgs, video_height=int(h), video_width=int(w), offload_video_to_cpu=False, offload_state_to_cpu=False), "train_init_state"
    if family == "uoft" and hasattr(predictor, "init_state"):
        return predictor.init_state(imgs, int(h), int(w), offload_video_to_cpu=False, offload_state_to_cpu=False), "init_state_npz"
    raise RuntimeError(f"No supported tensor init API for {bundle['entry'].name}")


def reset_video_state(bundle: dict[str, Any], state) -> None:
    predictor = bundle["predictor"]
    if hasattr(predictor, "reset_state"):
        try:
            predictor.reset_state(state)
        except Exception:
            pass


def add_cached_prompt(bundle: dict[str, Any], state, plan: dict[str, Any], frame_idx: int) -> str:
    predictor = bundle["predictor"]
    t = int(frame_idx)
    mode = plan["effective_prompt_mode"]
    if mode == "mixed":
        mode = str(plan.get("frame_prompt_types", {}).get(t, "box"))
    if mode == "box":
        box = torch.as_tensor(plan["boxes"][t], dtype=torch.float32, device=DEVICE).reshape(-1)[:4]
        if not torch.isfinite(box).all():
            raise RuntimeError(f"No finite cached box on frame {t}")
        if hasattr(predictor, "add_new_points_or_box"):
            empty_points = torch.empty((1, 0, 2), dtype=torch.float32, device=DEVICE)
            empty_labels = torch.empty((1, 0), dtype=torch.int32, device=DEVICE)
            predictor.add_new_points_or_box(state, t, 1, points=empty_points, labels=empty_labels, box=box, clear_old_points=True, normalize_coords=True)
            return f"{mode}:add_new_points_or_box"
        if hasattr(predictor, "add_new_bbox"):
            predictor.add_new_bbox(state, t, 1, bbox=box, clear_old_points=True, normalize_coords=True)
            return f"{mode}:add_new_bbox"
        if hasattr(predictor, "train_add_new_bbox"):
            predictor.train_add_new_bbox(state, t, 1, bbox=box, clear_old_points=True, normalize_coords=True)
            return f"{mode}:train_add_new_bbox"
        raise RuntimeError(f"{bundle['entry'].name} has no supported box prompt method")
    if mode == "mask":
        mask = torch.as_tensor(plan["mask_prompts"][t] > 0, dtype=torch.bool, device=DEVICE)
        if not mask.any():
            raise RuntimeError(f"No cached mask foreground on frame {t}")
        if hasattr(predictor, "add_new_mask"):
            predictor.add_new_mask(state, t, 1, mask=mask)
            return f"{mode}:add_new_mask"
        raise RuntimeError(f"{bundle['entry'].name} has no mask prompt method")
    raise ValueError(mode)


def propagate_iterator(bundle: dict[str, Any], state, start_frame_idx: int, max_frame_num_to_track=None, reverse=False):
    predictor = bundle["predictor"]
    if hasattr(predictor, "propagate_in_video"):
        kwargs = {"start_frame_idx": int(start_frame_idx), "reverse": bool(reverse)}
        if max_frame_num_to_track is not None:
            kwargs["max_frame_num_to_track"] = int(max_frame_num_to_track)
        return predictor.propagate_in_video(state, **kwargs), "propagate_in_video"
    if hasattr(predictor, "train_propagate_in_video"):
        kwargs = {"start_frame_idx": int(start_frame_idx), "reverse": bool(reverse)}
        if max_frame_num_to_track is not None:
            kwargs["max_frame_num_to_track"] = int(max_frame_num_to_track)
        try:
            return predictor.train_propagate_in_video(state, **kwargs), "train_propagate_in_video"
        except TypeError:
            return predictor.train_propagate_in_video(state, start_frame_idx=int(start_frame_idx)), "train_propagate_in_video"
    raise RuntimeError(f"{bundle['entry'].name} has no propagation API")


def assign_mask_logits(out: np.ndarray, frame_idx: int, mask_logits) -> None:
    ml = mask_logits[0] if isinstance(mask_logits, (tuple, list)) else mask_logits
    if not torch.is_tensor(ml):
        ml = torch.as_tensor(ml)
    ml = ml.detach().float()
    while ml.ndim > 2:
        ml = ml[0]
    arr = ml.detach().cpu().numpy().astype(np.float32)
    out[int(frame_idx)] = np.nan_to_num(arr, nan=EMPTY_LOGIT)


def collect_propagation_logits(iterator, t_len: int, h: int, w: int):
    logits = np.full((t_len, h, w), EMPTY_LOGIT, dtype=np.float32)
    yielded = []
    for item in iterator:
        if len(item) == 3:
            frame_idx, _obj_ids, mask_logits = item
        elif len(item) == 2:
            frame_idx, mask_logits = item
        else:
            raise RuntimeError(f"Unexpected propagation output len={len(item)}")
        assign_mask_logits(logits, int(frame_idx), mask_logits)
        yielded.append(int(frame_idx))
    return logits, yielded


def run_strict_video_protocol(
    bundle: dict[str, Any],
    case: dict[str, Any],
    prompt_mode: str,
    protocol: str,
    plan: dict[str, Any] | None = None,
    anchor: int | None = None,
):
    imgs, preprocess_note = prepare_images_for_model(bundle, case)
    t_len, _c, h, w = imgs.shape
    if anchor is None:
        anchor = choose_anchor_frame(case, prompt_mode)
    anchor = int(anchor)
    if protocol == "validation_forward":
        plan = plan or cached_prompt_plan_for_case(case, prompt_mode)
        prompt_frames = list(plan["prompt_frames"])
        start = 0
        reverse = False
        max_track = t_len
    elif protocol == "single_anchor_forward":
        plan = plan or cached_prompt_plan_for_case(case, prompt_mode, prompt_frames=[anchor])
        prompt_frames = [anchor]
        start = anchor
        reverse = False
        max_track = t_len - anchor
    elif protocol == "single_anchor_bidirectional":
        plan = plan or cached_prompt_plan_for_case(case, prompt_mode, prompt_frames=[anchor])
        prompt_frames = [anchor]
    else:
        raise ValueError(protocol)

    started = time.perf_counter()
    init_time_sec = 0.0
    prompt_time_sec = 0.0
    propagation_time_sec = 0.0
    reset_time_sec = 0.0
    prompt_methods = []
    yielded = []
    init_method = ""
    propagation_method = ""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad()
    with torch.inference_mode(), amp_ctx:
        if protocol != "single_anchor_bidirectional":
            t0 = time.perf_counter()
            state, init_method = init_video_state(bundle, imgs, h, w)
            init_time_sec += float(time.perf_counter() - t0)
            try:
                t0 = time.perf_counter()
                for t in prompt_frames:
                    prompt_methods.append(add_cached_prompt(bundle, state, plan, t))
                prompt_time_sec += float(time.perf_counter() - t0)
                t0 = time.perf_counter()
                iterator, propagation_method = propagate_iterator(bundle, state, start, max_track, reverse=reverse)
                logits, yielded = collect_propagation_logits(iterator, t_len, h, w)
                propagation_time_sec += float(time.perf_counter() - t0)
            finally:
                t0 = time.perf_counter()
                reset_video_state(bundle, state)
                reset_time_sec += float(time.perf_counter() - t0)
            yielded_set = set(int(t) for t in yielded)
            untracked = [int(t) for t in range(t_len) if t not in yielded_set]
        else:
            t0 = time.perf_counter()
            fwd_state, init_fwd = init_video_state(bundle, imgs, h, w)
            init_time_sec += float(time.perf_counter() - t0)
            try:
                t0 = time.perf_counter()
                prompt_methods.append(add_cached_prompt(bundle, fwd_state, plan, anchor))
                prompt_time_sec += float(time.perf_counter() - t0)
                t0 = time.perf_counter()
                fwd_iter, prop_fwd = propagate_iterator(bundle, fwd_state, anchor, t_len - anchor, reverse=False)
                fwd_logits, yielded_fwd = collect_propagation_logits(fwd_iter, t_len, h, w)
                propagation_time_sec += float(time.perf_counter() - t0)
            finally:
                t0 = time.perf_counter()
                reset_video_state(bundle, fwd_state)
                reset_time_sec += float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            rev_state, init_rev = init_video_state(bundle, imgs, h, w)
            init_time_sec += float(time.perf_counter() - t0)
            try:
                t0 = time.perf_counter()
                prompt_methods.append(add_cached_prompt(bundle, rev_state, plan, anchor))
                prompt_time_sec += float(time.perf_counter() - t0)
                t0 = time.perf_counter()
                rev_iter, prop_rev = propagate_iterator(bundle, rev_state, anchor, anchor + 1, reverse=True)
                rev_logits, yielded_rev = collect_propagation_logits(rev_iter, t_len, h, w)
                propagation_time_sec += float(time.perf_counter() - t0)
            finally:
                t0 = time.perf_counter()
                reset_video_state(bundle, rev_state)
                reset_time_sec += float(time.perf_counter() - t0)
            logits = np.full((t_len, h, w), EMPTY_LOGIT, dtype=np.float32)
            if anchor > 0:
                logits[:anchor] = rev_logits[:anchor]
            logits[anchor:] = fwd_logits[anchor:]
            yielded = sorted(set(yielded_fwd + yielded_rev))
            untracked = [int(t) for t in range(t_len) if t not in set(yielded)]
            init_method = f"{init_fwd}+{init_rev}"
            propagation_method = f"{prop_fwd}+{prop_rev}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = float(time.perf_counter() - started)
    return logits, {
        "wall_time_sec": total_time,
        "init_time_sec": float(init_time_sec),
        "prompt_time_sec": float(prompt_time_sec),
        "propagation_time_sec": float(propagation_time_sec),
        "reset_time_sec": float(reset_time_sec),
        "sec_per_frame": float(total_time / t_len) if t_len > 0 else float("nan"),
        "frames_per_sec": float(t_len / total_time) if total_time > 0 else float("nan"),
        "n_frames": int(t_len),
        "n_prompt_frames": int(len(prompt_frames)),
        "preprocess": preprocess_note,
        "init_method": init_method,
        "prompt_method": ";".join(sorted(set(prompt_methods))),
        "propagation_method": propagation_method,
        "requested_prompt_frames": prompt_frames,
        "actual_prompt_frames": prompt_frames,
        "cached_prompt_frames": plan.get("cached_prompt_frames", []),
        "frame_prompt_types": plan.get("frame_prompt_types", {}),
        "mixed_box_prompt_frames": plan.get("mixed_box_prompt_frames", []),
        "mixed_mask_prompt_frames": plan.get("mixed_mask_prompt_frames", []),
        "anchor_frame": int(anchor),
        "yielded_frames": yielded,
        "yielded_frame_count": int(len(set(yielded))),
        "untracked_frames": untracked,
        "untracked_frame_count": int(len(untracked)),
        "strict_equal_protocol": True,
    }


def source_metadata_for_index(test_dataset, idx: int) -> dict[str, Any]:
    idx = int(idx)
    if hasattr(test_dataset, "sequences") and idx < len(test_dataset.sequences):
        seq = test_dataset.sequences[idx]
        if isinstance(seq, dict):
            return dict(seq)
    return {}


def load_case_by_index(test_dataset, idx: int) -> dict[str, Any]:
    sample = test_dataset[int(idx)]
    if not isinstance(sample, dict):
        raise TypeError(f"Expected dict sample, got {type(sample).__name__}")
    return sample_to_case(sample, dataset_index=int(idx), source_meta=source_metadata_for_index(test_dataset, idx))


def parameter_count(bundle: dict[str, Any]) -> int:
    if bundle is not None and "_param_count" in bundle:
        try:
            return int(bundle["_param_count"])
        except Exception:
            pass
    predictor = bundle.get("predictor")
    if predictor is None or not hasattr(predictor, "parameters"):
        return 0
    count = int(sum(p.numel() for p in predictor.parameters()))
    bundle["_param_count"] = count
    return count


def cuda_memory_snapshot(prefix: str = "") -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            f"{prefix}cuda_allocated_mb": 0.0,
            f"{prefix}cuda_reserved_mb": 0.0,
            f"{prefix}cuda_max_allocated_mb": 0.0,
            f"{prefix}cuda_max_reserved_mb": 0.0,
        }
    return {
        f"{prefix}cuda_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
        f"{prefix}cuda_reserved_mb": float(torch.cuda.memory_reserved() / (1024**2)),
        f"{prefix}cuda_max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        f"{prefix}cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }


def prepare_cuda_memory_measurement(clear_cache: bool = True) -> dict[str, float]:
    gc.collect()
    if torch.cuda.is_available():
        if clear_cache:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return cuda_memory_snapshot("baseline_")


def finalize_cuda_memory_measurement(n_frames: int) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out = cuda_memory_snapshot("final_")
    base_alloc = out.get("baseline_cuda_allocated_mb", 0.0)
    return out


def prediction_memory_fields(
    baseline: dict[str, float],
    n_frames: int,
    peak_alloc_override: float | None = None,
    peak_reserved_override: float | None = None,
) -> dict[str, float]:
    final = cuda_memory_snapshot("final_")
    peak_alloc = float(final.get("final_cuda_max_allocated_mb", 0.0))
    peak_reserved = float(final.get("final_cuda_max_reserved_mb", 0.0))
    if peak_alloc_override is not None and math.isfinite(float(peak_alloc_override)):
        peak_alloc = max(peak_alloc, float(peak_alloc_override))
    if peak_reserved_override is not None and math.isfinite(float(peak_reserved_override)):
        peak_reserved = max(peak_reserved, float(peak_reserved_override))
    base_alloc = float(baseline.get("baseline_cuda_allocated_mb", 0.0))
    base_reserved = float(baseline.get("baseline_cuda_reserved_mb", 0.0))
    delta_alloc = max(0.0, peak_alloc - base_alloc)
    delta_reserved = max(0.0, peak_reserved - base_reserved)
    frames = max(int(n_frames), 1)
    return {
        **baseline,
        **final,
        "peak_cuda_mem_allocated_mb": peak_alloc,
        "peak_cuda_mem_reserved_mb": peak_reserved,
        "delta_peak_cuda_mem_allocated_mb": float(delta_alloc),
        "delta_peak_cuda_mem_reserved_mb": float(delta_reserved),
        "delta_peak_cuda_mem_allocated_mb_per_frame": float(delta_alloc / frames),
        "delta_peak_cuda_mem_reserved_mb_per_frame": float(delta_reserved / frames),
    }


def run_strict_video_protocol_measured(
    bundle: dict[str, Any],
    case: dict[str, Any],
    prompt_mode: str,
    protocol: str,
    clear_cache: bool = True,
    plan: dict[str, Any] | None = None,
    anchor: int | None = None,
):
    baseline = prepare_cuda_memory_measurement(clear_cache=clear_cache)
    logits, info = run_strict_video_protocol(bundle, case, prompt_mode, protocol, plan=plan, anchor=anchor)
    info.update(prediction_memory_fields(baseline, int(logits.shape[0])))
    info["param_count"] = parameter_count(bundle)
    return logits, info


def run_validation_forward_prompt_modes_reused_state_measured(
    bundle: dict[str, Any],
    case: dict[str, Any],
    prompt_plans: dict[str, tuple[dict[str, Any], int]],
    clear_cache: bool = True,
):
    if not prompt_plans:
        return {}
    baseline = prepare_cuda_memory_measurement(clear_cache=clear_cache)
    imgs, preprocess_note = prepare_images_for_model(bundle, case)
    t_len, _c, h, w = imgs.shape
    prompt_modes = list(prompt_plans.keys())
    n_modes = max(1, len(prompt_modes))
    init_time_sec = 0.0
    shared_started = time.perf_counter()
    init_method = ""
    out: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad()
    with torch.inference_mode(), amp_ctx:
        t0 = time.perf_counter()
        state, init_method = init_video_state(bundle, imgs, h, w)
        init_time_sec = float(time.perf_counter() - t0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        init_peak = cuda_memory_snapshot("state_reuse_init_")
        init_peak_alloc = float(init_peak.get("state_reuse_init_cuda_max_allocated_mb", 0.0))
        init_peak_reserved = float(init_peak.get("state_reuse_init_cuda_max_reserved_mb", 0.0))
        try:
            for prompt_mode in prompt_modes:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                plan, anchor = prompt_plans[prompt_mode]
                prompt_frames = list(plan["prompt_frames"])
                prompt_methods = []
                yielded = []
                prompt_time_sec = 0.0
                propagation_time_sec = 0.0
                reset_time_sec = 0.0
                propagation_method = ""
                try:
                    t0 = time.perf_counter()
                    for frame_idx in prompt_frames:
                        prompt_methods.append(add_cached_prompt(bundle, state, plan, int(frame_idx)))
                    prompt_time_sec = float(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    iterator, propagation_method = propagate_iterator(bundle, state, 0, int(t_len), reverse=False)
                    logits, yielded = collect_propagation_logits(iterator, int(t_len), int(h), int(w))
                    propagation_time_sec = float(time.perf_counter() - t0)
                finally:
                    t0 = time.perf_counter()
                    reset_video_state(bundle, state)
                    reset_time_sec = float(time.perf_counter() - t0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                allocated_init_time = float(init_time_sec / n_modes)
                total_time = float(allocated_init_time + prompt_time_sec + propagation_time_sec + reset_time_sec)
                yielded_set = set(int(t) for t in yielded)
                untracked = [int(t) for t in range(t_len) if t not in yielded_set]
                info = {
                    "wall_time_sec": total_time,
                    "init_time_sec": allocated_init_time,
                    "prompt_time_sec": float(prompt_time_sec),
                    "propagation_time_sec": float(propagation_time_sec),
                    "reset_time_sec": float(reset_time_sec),
                    "sec_per_frame": float(total_time / t_len) if t_len > 0 else float("nan"),
                    "frames_per_sec": float(t_len / total_time) if total_time > 0 else float("nan"),
                    "n_frames": int(t_len),
                    "n_prompt_frames": int(len(prompt_frames)),
                    "preprocess": preprocess_note,
                    "init_method": init_method,
                    "prompt_method": ";".join(sorted(set(prompt_methods))),
                    "propagation_method": propagation_method,
                    "requested_prompt_frames": prompt_frames,
                    "actual_prompt_frames": prompt_frames,
                    "cached_prompt_frames": plan.get("cached_prompt_frames", []),
                    "frame_prompt_types": plan.get("frame_prompt_types", {}),
                    "mixed_box_prompt_frames": plan.get("mixed_box_prompt_frames", []),
                    "mixed_mask_prompt_frames": plan.get("mixed_mask_prompt_frames", []),
                    "anchor_frame": int(anchor),
                    "yielded_frames": yielded,
                    "yielded_frame_count": int(len(yielded_set)),
                    "untracked_frames": untracked,
                    "untracked_frame_count": int(len(untracked)),
                    "strict_equal_protocol": True,
                    "state_reuse_prompt_modes": True,
                    "state_reuse_prompt_count": int(n_modes),
                    "state_reuse_shared_init_time_sec": float(init_time_sec),
                    "state_reuse_allocated_init_time_sec": float(allocated_init_time),
                    "state_reuse_shared_wall_time_sec": float(time.perf_counter() - shared_started),
                    "state_reuse_memory_baseline": "pre_shared_state_init",
                    "state_reuse_init_peak_cuda_mem_allocated_mb": float(init_peak_alloc),
                    "state_reuse_init_peak_cuda_mem_reserved_mb": float(init_peak_reserved),
                }
                info.update(
                    prediction_memory_fields(
                        baseline,
                        int(t_len),
                        peak_alloc_override=init_peak_alloc,
                        peak_reserved_override=init_peak_reserved,
                    )
                )
                info["param_count"] = parameter_count(bundle)
                out[prompt_mode] = (logits, info)
        finally:
            reset_video_state(bundle, state)
    return out


def unload_model_bundle(bundle: dict[str, Any] | None) -> None:
    if bundle is None:
        return
    try:
        predictor = bundle.get("predictor")
        if predictor is not None and hasattr(predictor, "cpu"):
            predictor.cpu()
    except Exception:
        pass
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def dataset_groups_from_metadata(test_dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    if hasattr(test_dataset, "sequences"):
        for idx, seq in enumerate(test_dataset.sequences):
            groups.setdefault(str(seq.get("dataset", "unknown")), []).append(int(idx))
    else:
        for idx in range(len(test_dataset)):
            try:
                sample = test_dataset[idx]
                name = str(sample.get("dataset", sample.get("dataset_name", "unknown")))
            except Exception:
                name = "unknown"
            groups.setdefault(name, []).append(int(idx))
    return groups


def metadata_dim_for_index(test_dataset, idx: int):
    dims = metadata_dims(test_dataset)
    if dims is not None and int(idx) < len(dims):
        try:
            return int(dims[int(idx)])
        except Exception:
            return None
    if hasattr(test_dataset, "sequences") and int(idx) < len(test_dataset.sequences):
        try:
            return int(test_dataset.sequences[int(idx)].get("dim", 2))
        except Exception:
            return None
    return None


def select_smoke_cases_per_dataset(
    test_dataset,
    cases_per_dataset: int = 2,
    prefer_sequence_cases: bool = True,
    datasets_filter=None,
    max_scan_per_dataset: int = 250,
    case_cache: dict[int, dict[str, Any]] | None = None,
):
    groups = dataset_groups_from_metadata(test_dataset)
    if datasets_filter:
        allowed = {str(x) for x in datasets_filter}
        groups = {k: v for k, v in groups.items() if k in allowed}
    selected = []
    rows = []
    for dataset_name in tqdm(sorted(groups), desc="Select per-dataset smoke cases", unit="dataset"):
        indices = list(groups[dataset_name])
        dim3 = [i for i in indices if metadata_dim_for_index(test_dataset, i) == 3]
        dim2 = [i for i in indices if metadata_dim_for_index(test_dataset, i) != 3]
        ordered = (dim3 + dim2) if prefer_sequence_cases else indices
        if prefer_sequence_cases and dim3 and dim2:
            ordered = [dim3[0], dim2[0]] + [i for i in ordered if i not in {dim3[0], dim2[0]}]
        seen = set()
        valid_count = 0
        errors = []
        for idx in ordered[: max_scan_per_dataset]:
            if idx in seen:
                continue
            seen.add(idx)
            try:
                if case_cache is not None and int(idx) in case_cache:
                    case = case_cache[int(idx)]
                else:
                    case = load_case_by_index(test_dataset, idx)
                    if case_cache is not None:
                        case_cache[int(idx)] = clone_cpu_case(case)
                selected.append(int(idx))
                valid_count += 1
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_index": int(idx),
                        "status": "selected",
                        "selection_reason": "smoke_per_dataset",
                        "dim": int(case["meta"].get("dim", -1)),
                        "sequence_length": int(case["image"].shape[0]),
                        "foreground_frame_count": int((case["gt_thw"].reshape(case["gt_thw"].shape[0], -1).sum(axis=1) > 0).sum()),
                        "cached_prompt_frames": json.dumps([int(x) for x in case["val_prompt_cache"].get("chosen_frames", [])]),
                    }
                )
            except Exception as exc:
                if len(errors) < 3:
                    errors.append(f"{int(idx)}:{type(exc).__name__}:{exc}")
            if valid_count >= int(cases_per_dataset):
                break
        if valid_count < int(cases_per_dataset):
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_index": None,
                    "status": "shortage",
                    "selection_reason": f"selected {valid_count}/{int(cases_per_dataset)} valid cases",
                    "dim": None,
                    "sequence_length": None,
                    "foreground_frame_count": None,
                    "cached_prompt_frames": "",
                    "errors": " | ".join(errors),
                }
            )
    return selected, pd.DataFrame(rows)


def metadata_dims(ds):
    dims = getattr(ds, "entry_dims", None)
    if dims is None and hasattr(ds, "sequences"):
        dims = [s.get("dim", 2) for s in ds.sequences]
    return dims


def safe_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "unknown"


def safe_tensorboard_tag(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "unknown"


def metadata_dataset_name_for_index(test_dataset, idx: int) -> str:
    idx = int(idx)
    if hasattr(test_dataset, "sequences") and idx < len(test_dataset.sequences):
        seq = test_dataset.sequences[idx]
        for key in ["dataset", "dataset_name", "name"]:
            if key in seq and seq.get(key) is not None:
                return str(seq.get(key))
    for attr in ["dataset_names", "entry_datasets", "datasets"]:
        values = getattr(test_dataset, attr, None)
        if values is not None and idx < len(values):
            try:
                return str(values[idx])
            except Exception:
                pass
    return "unknown"


def case_order_metadata(test_dataset, idx: int) -> dict[str, Any]:
    idx = int(idx)
    dim = metadata_dim_for_index(test_dataset, idx)
    try:
        dim_int = int(dim) if dim is not None else 2
    except Exception:
        dim_int = 2
    phase = "3d_sequence" if dim_int == 3 else "2d_single_frame"
    return {
        "dataset_index": idx,
        "dataset_name": metadata_dataset_name_for_index(test_dataset, idx),
        "order_dim": dim_int,
        "order_phase": phase,
        "order_phase_rank": 1 if dim_int == 3 else 0,
    }


def sort_case_indices_2d_then_3d(test_dataset, case_indices: list[int]) -> list[int]:
    unique_indices = list(dict.fromkeys(int(i) for i in case_indices))
    meta = {
        idx: case_order_metadata(test_dataset, idx)
        for idx in tqdm(unique_indices, desc="Order case metadata", unit="case")
    }
    return sorted(
        unique_indices,
        key=lambda idx: (
            int(meta[idx]["order_phase_rank"]),
            str(meta[idx]["dataset_name"]).lower(),
            int(idx),
        ),
    )


def build_case_progress_table(test_dataset, case_indices: list[int]) -> pd.DataFrame:
    rows = [
        case_order_metadata(test_dataset, int(idx))
        for idx in tqdm(case_indices, desc="Build case progress", unit="case")
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["benchmark_order"] = np.arange(1, len(df) + 1, dtype=int)
    df["benchmark_case_total"] = int(len(df))
    df["dataset_case_number"] = df.groupby("dataset_name", dropna=False).cumcount() + 1
    df["dataset_case_total"] = df.groupby("dataset_name", dropna=False)["dataset_index"].transform("count")
    df["dataset_percent_done"] = 100.0 * df["dataset_case_number"] / df["dataset_case_total"].replace(0, np.nan)
    df["phase_case_number"] = df.groupby("order_phase", dropna=False).cumcount() + 1
    df["phase_case_total"] = df.groupby("order_phase", dropna=False)["dataset_index"].transform("count")
    return df


def annotate_case_selection_order(test_dataset, case_selection: pd.DataFrame, case_indices: list[int]) -> pd.DataFrame:
    progress = build_case_progress_table(test_dataset, case_indices)
    if case_selection is None or case_selection.empty:
        return progress
    out = case_selection.copy()
    if progress.empty or "dataset_index" not in out.columns:
        return out
    progress_map = progress.set_index("dataset_index").to_dict("index")

    def lookup(value, key):
        try:
            if pd.isna(value):
                return np.nan
            return progress_map.get(int(float(value)), {}).get(key, np.nan)
        except Exception:
            return np.nan

    for col in [
        "dataset_name",
        "benchmark_order",
        "order_dim",
        "order_phase",
        "dataset_case_number",
        "dataset_case_total",
        "dataset_percent_done",
        "phase_case_number",
        "phase_case_total",
    ]:
        looked_up = out["dataset_index"].apply(lambda value, key=col: lookup(value, key))
        if col in out.columns:
            out[col] = out[col].where(out[col].notna(), looked_up)
        else:
            out[col] = looked_up
    sort_cols = [c for c in ["benchmark_order", "dataset_name"] if c in out.columns]
    return out.sort_values(sort_cols, na_position="last").reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def diagnostic_case_score(case: dict[str, Any]):
    gt = case["gt_thw"].astype(bool)
    t_len = gt.shape[0]
    fg = np.where(gt.reshape(t_len, -1).sum(axis=1) > 0)[0].astype(int).tolist()
    anchor = choose_anchor_frame(case, "box")
    before = any(t < anchor for t in fg)
    after = any(t > anchor for t in fg)
    empty_count = int(t_len - len(fg))
    mixed_empty_foreground = int(0 < len(fg) < t_len)
    balance = min(len(fg), empty_count)
    return (
        int(case["meta"]["dim"] == 3),
        int(t_len > 1),
        mixed_empty_foreground,
        int(anchor > 0),
        int(before and after),
        balance,
        len(fg),
        t_len,
    )


def select_diagnostic_cases(test_dataset, max_cases=6, scan_limit=12000, include_2d=True):
    dims = metadata_dims(test_dataset)
    if dims is not None:
        ordered = [i for i, d in enumerate(dims) if int(d) == 3]
        if include_2d:
            ordered += [i for i, d in enumerate(dims) if int(d) == 2][: max(10, max_cases)]
        ordered = ordered[:scan_limit]
    else:
        ordered = list(range(min(len(test_dataset), scan_limit)))
    candidates = []
    errors = []
    for idx in tqdm(ordered, desc="Select diagnostic cases", unit="case"):
        try:
            case = load_case_by_index(test_dataset, idx)
            candidates.append((diagnostic_case_score(case), idx, case))
        except Exception as exc:
            if len(errors) < 10:
                errors.append((idx, f"{type(exc).__name__}: {exc}"))
        if len(candidates) >= max_cases * 8:
            break
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [case for _score, _idx, case in candidates[:max_cases]]
    if not selected:
        raise RuntimeError(f"No diagnostic cases selected. First errors: {errors}")
    return selected


def base_case_fields(case: dict[str, Any]) -> dict[str, Any]:
    out = {k: case["meta"].get(k) for k in ["dataset_index", "dataset_name", "case_id", "dim", "task_id", "task_label", "task_list", "modality"]}
    out["dataset"] = out.get("dataset_name")
    return out


def prompt_audit_row(case, model_name, prompt_mode, protocol, info, status, error=""):
    t_len = int(case["image"].shape[0])
    requested = [int(x) for x in info.get("requested_prompt_frames", [])]
    actual = [int(x) for x in info.get("actual_prompt_frames", [])]
    cached = [int(x) for x in info.get("cached_prompt_frames", [])]
    return {
        **base_case_fields(case),
        "model": model_name,
        "prompt_mode": prompt_mode,
        "protocol": protocol,
        "status": status,
        "error": error,
        "cached_prompt_frames": json.dumps(cached),
        "requested_prompt_frames": json.dumps(requested),
        "actual_prompt_frames": json.dumps(actual),
        "number_of_prompted_frames": int(len(actual)),
        "sequence_length": int(t_len),
        "boxes_supplied_for_every_frame": bool(len(actual) == t_len and prompt_mode == "box"),
        "only_cached_frames_prompted": bool(set(actual).issubset(set(cached))),
        "only_one_frame_prompted": bool(len(actual) == 1),
        "mask_prompts_used": bool(prompt_mode == "mask" or len(info.get("mixed_mask_prompt_frames", [])) > 0),
        "mixed_box_prompt_frames": json.dumps([int(x) for x in info.get("mixed_box_prompt_frames", [])]),
        "mixed_mask_prompt_frames": json.dumps([int(x) for x in info.get("mixed_mask_prompt_frames", [])]),
        "frame_prompt_types": json.dumps({int(k): str(v) for k, v in info.get("frame_prompt_types", {}).items()}),
        "strict_equal_protocol": bool(info.get("strict_equal_protocol", True)),
    }


def failed_result_row(case, model_name, prompt_mode, protocol, status, error, wall=0.0):
    return {
        **base_case_fields(case),
        "model": model_name,
        "prompt_mode": prompt_mode,
        "protocol": protocol,
        "status": status,
        "error": error,
        "wall_time_sec": float(wall),
    }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 100:
            text = f"{value:.2f}"
        elif abs(value) >= 10:
            text = f"{value:.3f}"
        else:
            text = f"{value:.4f}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def safe_markdown_table(df: pd.DataFrame, max_rows: int = 30, max_cols: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_No rows._"
    show = df.head(int(max_rows)).copy()
    if max_cols is not None:
        show = show.iloc[:, : int(max_cols)]
    columns = [str(c) for c in show.columns]
    header = "| " + " | ".join(markdown_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in show.iterrows():
        rows.append("| " + " | ".join(markdown_cell(row.get(col)) for col in show.columns) + " |")
    suffix = ""
    if len(df) > len(show):
        suffix = f"\n\n_Showing {len(show):,} of {len(df):,} rows._"
    return "\n".join([header, sep, *rows]) + suffix


TB_TEXT_METRIC_COLS = [
    "n_cases",
    "n_datasets",
    "fg_volume_dice",
    "fg_volume_iou",
    "dice",
    "iou",
    "empty_frame_false_positive_rate",
    "sec_per_frame",
    "frames_per_sec",
    "delta_peak_cuda_mem_allocated_mb",
]


def tensorboard_markdown_table(df: pd.DataFrame, id_cols: list[str], max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_No rows._"
    cols = []
    for col in [*id_cols, *TB_TEXT_METRIC_COLS]:
        if col in df.columns and col not in cols:
            cols.append(col)
    if not cols:
        cols = list(df.columns[:10])
    work = df.loc[:, cols].copy()
    sort_cols = [c for c in id_cols if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, na_position="last")
    return safe_markdown_table(work, max_rows=max_rows)


def tensorboard_progress_markdown(status: dict[str, Any]) -> str:
    rows = [
        {
            "scope": "overall",
            "model": f"{status.get('models_done_or_current') or 0}/{status.get('models_total') or 0}",
            "case": f"{status.get('model_case_visit_number') or 0}/{status.get('model_case_visits_total') or 0}",
            "runs": f"{status.get('runs_done') or 0}/{status.get('runs_total') or 0}",
            "percent": f"{float(status.get('percent_done') or 0.0):.2f}%",
            "dataset": status.get("current_dataset") or "",
            "time_per_case": f"{float(status.get('mean_sec_per_case_visit')):.2f}s" if status.get("mean_sec_per_case_visit") is not None else "",
            "elapsed": status.get("elapsed_text") or "",
            "eta": status.get("eta_text") or "",
            "datasets_remaining": status.get("overall_dataset_visits_remaining_including_current") or "",
        },
        {
            "scope": "model",
            "model": status.get("model") or "",
            "case": f"{status.get('current_case_number_for_model') or 0}/{status.get('current_case_total_for_model') or 0}",
            "runs": f"{status.get('model_runs_done') or 0}/{status.get('model_runs_total') or 0}",
            "percent": f"{float(status.get('model_percent_done') or 0.0):.2f}%",
            "dataset": status.get("current_dataset") or "",
            "time_per_case": "",
            "elapsed": "",
            "eta": "",
            "datasets_remaining": (
                f"{status.get('current_model_datasets_remaining_including_current')}/{status.get('current_model_dataset_total')}"
                if status.get("current_model_datasets_remaining_including_current") is not None
                else ""
            ),
        },
        {
            "scope": "dataset",
            "model": status.get("model") or "",
            "case": f"{status.get('current_dataset_case_number') or 0}/{status.get('current_dataset_case_total') or 0}",
            "runs": "",
            "percent": f"{float(status.get('current_dataset_percent_done') or 0.0):.2f}%",
            "dataset": status.get("current_dataset") or "",
            "time_per_case": "",
            "elapsed": "",
            "eta": "",
            "datasets_remaining": "",
        },
    ]
    lines = ["### Benchmark Progress", ""]
    reason = status.get("progress_report_reason")
    if reason:
        lines.extend([f"Report trigger: `{reason}`", ""])
    lines.extend(
        [
            f"- Time per case: `{float(status.get('mean_sec_per_case_visit')):.2f}s`"
            if status.get("mean_sec_per_case_visit") is not None
            else "- Time per case: `unknown`",
            f"- Current elapsed time: `{status.get('elapsed_text') or 'unknown'}`",
            f"- Estimated time remaining: `{status.get('eta_text') or 'unknown'}`",
            "",
        ]
    )
    lines.append(safe_markdown_table(pd.DataFrame(rows), max_rows=10))
    last = status.get("last_row") or {}
    if last:
        last_cols = ["model", "prompt_mode", "protocol", "dataset_name", "dataset_index", "status", "dice", "iou", "sec_per_frame"]
        last_df = pd.DataFrame([{col: last.get(col, "") for col in last_cols if col in last or col in ["model", "prompt_mode", "protocol", "status"]}])
        lines.extend(["", "### Last Prediction", "", safe_markdown_table(last_df, max_rows=1)])
    return "\n".join(lines)


def format_duration(seconds) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        return "unknown"
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(round(seconds))
    days, rem = divmod(seconds_i, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def numeric_mean_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in metric_cols if c in df.columns]
    out = df.groupby(group_cols, dropna=False)[cols].mean(numeric_only=True).reset_index()
    if "dataset_index" in df.columns:
        n = df.groupby(group_cols, dropna=False)["dataset_index"].nunique().reset_index(name="n_cases")
        out = n.merge(out, on=group_cols, how="left")
    return out


def dataset_macro_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in metric_cols if c in df.columns]
    ds_cols = list(group_cols) + ["dataset_name"]
    per_dataset = df.groupby(ds_cols, dropna=False)[cols].mean(numeric_only=True).reset_index()
    out = per_dataset.groupby(group_cols, dropna=False)[cols].mean(numeric_only=True).reset_index()
    counts = per_dataset.groupby(group_cols, dropna=False)["dataset_name"].nunique().reset_index(name="n_datasets")
    cases = df.groupby(group_cols, dropna=False)["dataset_index"].nunique().reset_index(name="n_cases")
    return counts.merge(cases, on=group_cols, how="left").merge(out, on=group_cols, how="left")


def best_threshold_table(thr_df: pd.DataFrame, group_cols: list[str], score_prefix: str = "fg") -> pd.DataFrame:
    if thr_df is None or thr_df.empty:
        return pd.DataFrame()
    dice_col = "fg_volume_dice" if score_prefix == "fg" else "full_sequence_dice"
    iou_col = "fg_volume_iou" if score_prefix == "fg" else "full_sequence_iou"
    work = thr_df.copy()
    for col in [dice_col, iou_col, "threshold"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "dataset_name" not in group_cols:
        per_dataset = work.groupby(group_cols + ["dataset_name", "threshold"], dropna=False)[[dice_col, iou_col]].mean(numeric_only=True).reset_index()
        agg = per_dataset.groupby(group_cols + ["threshold"], dropna=False)[[dice_col, iou_col]].mean(numeric_only=True).reset_index()
        n_datasets = work.groupby(group_cols, dropna=False)["dataset_name"].nunique().reset_index(name="n_datasets")
    else:
        agg = work.groupby(group_cols + ["threshold"], dropna=False)[[dice_col, iou_col]].mean(numeric_only=True).reset_index()
        n_datasets = pd.DataFrame()
    n_cases = work.groupby(group_cols, dropna=False)["dataset_index"].nunique().reset_index(name="n_cases")

    rows = []
    for key, vals in agg.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = {k: v for k, v in zip(group_cols, key)}
        vals = vals.copy()
        vals["_tie"] = (vals["threshold"] - PRIMARY_THRESHOLD).abs()
        dice_row = vals.sort_values([dice_col, "_tie"], ascending=[False, True], na_position="last").iloc[0]
        iou_row = vals.sort_values([iou_col, "_tie"], ascending=[False, True], na_position="last").iloc[0]
        rows.append(
            {
                **base,
                "best_dice_threshold": float(dice_row["threshold"]),
                "dice_at_best_dice_threshold": float(dice_row[dice_col]),
                "iou_at_best_dice_threshold": float(dice_row[iou_col]),
                "best_iou_threshold": float(iou_row["threshold"]),
                "iou_at_best_iou_threshold": float(iou_row[iou_col]),
                "dice_at_best_iou_threshold": float(iou_row[dice_col]),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = n_cases.merge(out, on=group_cols, how="right")
        if len(n_datasets):
            out = n_datasets.merge(out, on=group_cols, how="right")
    return out


RUNNING_TB_METRIC_COLS = [
    "dice",
    "iou",
    "fg_volume_dice",
    "fg_volume_iou",
    "fg_mean_frame_dice",
    "full_sequence_dice",
    "full_sequence_iou",
    "empty_frame_false_positive_rate",
    "empty_frame_specificity",
    "wall_time_sec",
    "sec_per_frame",
    "frames_per_sec",
    "delta_peak_cuda_mem_allocated_mb",
    "delta_peak_cuda_mem_allocated_mb_per_frame",
]

TB_DATASET_SCALAR_MAP = {
    "dice_at_0p5": "dice",
    "iou_at_0p5": "iou",
    "fg_volume_dice": "fg_volume_dice",
    "fg_volume_iou": "fg_volume_iou",
    "delta_peak_allocated_mb": "delta_peak_cuda_mem_allocated_mb",
    "delta_peak_allocated_mb_per_frame": "delta_peak_cuda_mem_allocated_mb_per_frame",
    "sec_per_frame": "sec_per_frame",
    "frames_per_sec": "frames_per_sec",
}

TB_OVERALL_BY_MODEL_SCALAR_MAP = {
    "metrics/dice_at_0p5": "dice",
    "metrics/iou_at_0p5": "iou",
    "metrics/fg_volume_dice": "fg_volume_dice",
    "metrics/fg_volume_iou": "fg_volume_iou",
    "memory/delta_peak_allocated_mb": "delta_peak_cuda_mem_allocated_mb",
    "memory/delta_peak_allocated_mb_per_frame": "delta_peak_cuda_mem_allocated_mb_per_frame",
}


def _ok_results_for_model(result_rows: list[dict[str, Any]], model_name: str) -> pd.DataFrame:
    if not result_rows:
        return pd.DataFrame()
    df = pd.DataFrame(result_rows)
    if "status" not in df.columns or "model" not in df.columns:
        return pd.DataFrame()
    ok = df[df["status"].astype(str).eq("ok") & df["model"].astype(str).eq(str(model_name))].copy()
    for col in RUNNING_TB_METRIC_COLS:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    return ok


def write_tensorboard_completed_dataset_scalars(tb_writer, by_dataset: pd.DataFrame, model_name: str, completed_dataset: str, step: int):
    if tb_writer is None or by_dataset is None or by_dataset.empty:
        return
    safe_model = safe_tensorboard_tag(model_name)
    dataset_rows = by_dataset[by_dataset["dataset_name"].astype(str).eq(str(completed_dataset))].copy()
    for row in dataset_rows.itertuples(index=False):
        data = row._asdict()
        safe_dataset = safe_tensorboard_tag(data.get("dataset_name", completed_dataset))
        safe_prompt = safe_tensorboard_tag(data.get("prompt_mode", "unknown"))
        safe_protocol = safe_tensorboard_tag(data.get("protocol", "unknown"))
        prefix = f"{safe_model}/dataset_metrics/{safe_dataset}/{safe_prompt}/{safe_protocol}"
        for tag_name, col in TB_DATASET_SCALAR_MAP.items():
            val = data.get(col)
            if val is not None and pd.notna(val):
                tb_writer.add_scalar(f"{prefix}/{tag_name}", float(val), step)


def write_tensorboard_overall_by_model_scalars(tb_writer, model_summary: pd.DataFrame, model_name: str, step: int):
    """Log compact running overall scalar streams, separated by model.

    These are model-level running means over successful rows seen so far. They
    are written only when a dataset completes, so TensorBoard stays readable.
    """

    if tb_writer is None or model_summary is None or model_summary.empty:
        return
    safe_model = safe_tensorboard_tag(model_name)
    for row in model_summary.itertuples(index=False):
        data = row._asdict()
        safe_prompt = safe_tensorboard_tag(data.get("prompt_mode", "unknown"))
        safe_protocol = safe_tensorboard_tag(data.get("protocol", "unknown"))
        prefix = f"overall_by_model/{safe_model}/{safe_prompt}/{safe_protocol}"
        for tag_path, col in TB_OVERALL_BY_MODEL_SCALAR_MAP.items():
            val = data.get(col)
            if val is not None and pd.notna(val):
                tb_writer.add_scalar(f"{prefix}/{tag_path}", float(val), step)


def write_tensorboard_running_group_metrics(tb_writer, result_rows: list[dict[str, Any]], model_name: str, step: int, completed_dataset=None):
    """Log running dataset/modality/task summaries for one model.

    These are intentionally running summaries: the value at each step reflects
    all successful rows seen for this model up to that point.
    """

    if tb_writer is None:
        return
    ok = _ok_results_for_model(result_rows, model_name)
    if ok.empty:
        return
    safe_model = safe_tensorboard_tag(model_name)
    metric_cols = [c for c in RUNNING_TB_METRIC_COLS if c in ok.columns]
    if not metric_cols:
        return

    model_summary = numeric_mean_summary(ok, ["prompt_mode", "protocol"], metric_cols)
    by_dataset = numeric_mean_summary(ok, ["dataset_name", "prompt_mode", "protocol"], metric_cols)
    by_modality = dataset_macro_summary(ok, ["modality", "prompt_mode", "protocol"], metric_cols)
    by_task = dataset_macro_summary(ok, ["task_id", "task_label", "prompt_mode", "protocol"], metric_cols)

    if completed_dataset is not None:
        write_tensorboard_overall_by_model_scalars(tb_writer, model_summary, model_name, step)
        write_tensorboard_completed_dataset_scalars(tb_writer, by_dataset, model_name, str(completed_dataset), step)

    if len(by_task):
        by_task = by_task.copy()
        by_task["task"] = by_task["task_label"].where(by_task["task_label"].notna(), by_task["task_id"])

    tb_writer.add_text(f"{safe_model}/tables/running_overall", tensorboard_markdown_table(model_summary, ["prompt_mode", "protocol"], 80), step)
    tb_writer.add_text(f"{safe_model}/tables/running_by_dataset", tensorboard_markdown_table(by_dataset, ["dataset_name", "prompt_mode", "protocol"], 120), step)
    tb_writer.add_text(f"{safe_model}/tables/running_by_modality", tensorboard_markdown_table(by_modality, ["modality", "prompt_mode", "protocol"], 120), step)
    tb_writer.add_text(f"{safe_model}/tables/running_by_task", tensorboard_markdown_table(by_task, ["task_id", "task_label", "prompt_mode", "protocol"], 120), step)
    tb_writer.flush()


def write_model_separated_outputs(
    output_dir,
    benchmark_models,
    api_rows=None,
    result_rows=None,
    audit_rows=None,
    per_frame_rows=None,
    threshold_rows=None,
    memory_rows=None,
    summary_frames=None,
):
    output_dir = Path(output_dir)
    by_model_dir = output_dir / "by_model"
    by_model_dir.mkdir(parents=True, exist_ok=True)
    api_df = pd.DataFrame(api_rows or [])
    result_df = pd.DataFrame(result_rows or [])
    audit_df = pd.DataFrame(audit_rows or [])
    per_frame_df = pd.DataFrame(per_frame_rows or [])
    threshold_df = pd.DataFrame(threshold_rows or [])
    memory_df = pd.DataFrame(memory_rows or [])
    summary_frames = summary_frames or {}

    def filter_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        key = "model" if "model" in df.columns else "name" if "name" in df.columns else None
        if key is None:
            return df.copy()
        return df[df[key].astype(str).eq(str(model_name))].copy()

    manifest_rows = []
    for model_name in benchmark_models:
        model_dir = by_model_dir / safe_filename(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        files_written = {}
        for filename, df in [
            ("api_probe.csv", filter_model(api_df, model_name)),
            ("master_records.csv", filter_model(result_df, model_name)),
            ("prompt_audit.csv", filter_model(audit_df, model_name)),
            ("per_frame_metrics.csv", filter_model(per_frame_df, model_name)),
            ("threshold_metrics_long.csv", filter_model(threshold_df, model_name)),
            ("memory_measurements.csv", filter_model(memory_df, model_name)),
        ]:
            path = model_dir / filename
            df.to_csv(path, index=False)
            files_written[filename] = int(len(df))
            if filename == "master_records.csv":
                jsonl_path = model_dir / "master_records.jsonl"
                df.to_json(jsonl_path, orient="records", lines=True)
                files_written["master_records.jsonl"] = int(len(df))

        for key, df in summary_frames.items():
            if not isinstance(df, pd.DataFrame) or df.empty or "model" not in df.columns:
                continue
            model_df = filter_model(df, model_name)
            if model_df.empty:
                continue
            path = model_dir / f"{safe_filename(key)}.csv"
            model_df.to_csv(path, index=False)
            files_written[path.name] = int(len(model_df))

        manifest = {
            "model": model_name,
            "model_dir": str(model_dir),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": files_written,
        }
        (model_dir / "model_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
        manifest_rows.append({"model": model_name, "model_dir": str(model_dir), **{f"rows_{k}": v for k, v in files_written.items()}})
    pd.DataFrame(manifest_rows).to_csv(by_model_dir / "model_output_manifest.csv", index=False)


def run_strict_video_benchmark(
    test_dataset,
    output_dir,
    benchmark_models=None,
    prompt_modes=None,
    protocols=None,
    smoke_run: bool = True,
    smoke_cases_per_dataset: int = 1,
    smoke_case_selection: str = "per_dataset",
    smoke_prefer_sequence_cases: bool = True,
    smoke_datasets=None,
    smoke_max_scan_per_dataset: int = 250,
    max_test_cases=None,
    benchmark_case_indices=None,
    thresholds=THRESHOLDS,
    primary_threshold: float = PRIMARY_THRESHOLD,
    compute_hd95_2d: bool = True,
    compute_hd95_2d_for_3d_cases: bool = False,
    compute_hd95_3d: bool = False,
    tensorboard: bool = True,
    partial_save_every_rows: int = 100,
    tensorboard_group_update_every_rows: int = 100,
    status_update_every_rows: int = 25,
    status_update_every_sec: float = 60.0,
    checkpoint_flush_every_rows: int = 1,
    order_2d_before_3d: bool = True,
    save_model_separated_outputs: bool = True,
    fail_fast: bool = False,
    visual_top_n_datasets=None,
    resume_from_dir=None,
    resume_completed_rows: bool = False,
    cache_cases_cpu: bool = True,
    cache_prompt_plans_cpu: bool = True,
    write_model_outputs_on_partial: bool = False,
    reuse_prompt_mode_state: bool = False,
    reuse_prompt_mode_state_models=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_models = benchmark_models or [
        "rwkv_medsam2_distill",
        "rwkv_medsam2_nodistill",
        "sam2_1_base",
        "oxford_medical_sam2",
        "uoft_medsam2",
    ]
    prompt_modes = prompt_modes or ["box", "mask", "mixed"]
    protocols = protocols or ["validation_forward"]
    # Kept for older notebook calls; group summaries now update only when a dataset finishes.
    _ = tensorboard_group_update_every_rows
    status_update_every_rows = max(1, int(status_update_every_rows or 1))
    status_update_every_sec = max(0.0, float(status_update_every_sec or 0.0))
    checkpoint_flush_every_rows = max(1, int(checkpoint_flush_every_rows or 1))
    reuse_prompt_mode_state_model_set = None if reuse_prompt_mode_state_models is None else {str(x) for x in reuse_prompt_mode_state_models}
    run_stamp = output_dir.name
    case_cache: dict[int, dict[str, Any]] | None = {} if cache_cases_cpu else None
    prompt_plan_cache: dict[tuple[Any, ...], tuple[dict[str, Any], int, dict[str, Any]]] | None = {} if cache_prompt_plans_cpu else None
    input_diag_cache: dict[tuple[Any, ...], dict[str, Any]] | None = {} if cache_prompt_plans_cpu else None

    print(f"Strict video benchmark output_dir: {output_dir}", flush=True)
    print(
        "Benchmark config: "
        f"models={len(benchmark_models)}, prompt_modes={prompt_modes}, protocols={protocols}, "
        f"smoke_run={smoke_run}, resume_completed_rows={resume_completed_rows}, "
        f"partial_save_every_rows={partial_save_every_rows}, "
        f"status_update_every_rows={status_update_every_rows}, "
        f"checkpoint_flush_every_rows={checkpoint_flush_every_rows}, "
        f"reuse_prompt_mode_state={reuse_prompt_mode_state}, "
        f"reuse_prompt_mode_state_models={sorted(reuse_prompt_mode_state_model_set) if reuse_prompt_mode_state_model_set is not None else 'all'}",
        flush=True,
    )
    if resume_from_dir:
        print(f"Resume source dir: {Path(resume_from_dir)}", flush=True)

    tb_writer = None
    tb_dir = output_dir / "tensorboard"
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=30, max_queue=20)
        except Exception as exc:
            print(f"TensorBoard disabled: {type(exc).__name__}: {exc}")
            tb_writer = None

    print("Selecting benchmark cases...", flush=True)
    if benchmark_case_indices:
        case_indices = [int(i) for i in benchmark_case_indices]
        case_selection = pd.DataFrame({"dataset_index": case_indices, "status": "explicit"})
    elif smoke_run and smoke_case_selection == "per_dataset":
        case_indices, case_selection = select_smoke_cases_per_dataset(
            test_dataset,
            cases_per_dataset=int(smoke_cases_per_dataset),
            prefer_sequence_cases=bool(smoke_prefer_sequence_cases),
            datasets_filter=smoke_datasets,
            max_scan_per_dataset=int(smoke_max_scan_per_dataset),
            case_cache=case_cache,
        )
        case_selection.to_csv(output_dir / "smoke_case_selection.csv", index=False)
        if len(case_selection) and "dataset_name" in case_selection.columns:
            counts = case_selection[case_selection["status"].eq("selected")].groupby("dataset_name").size()
            too_many = counts[counts > int(smoke_cases_per_dataset)]
            if len(too_many):
                raise AssertionError("Smoke selection exceeded SMOKE_CASES_PER_DATASET")
    else:
        n = len(test_dataset) if max_test_cases is None else min(len(test_dataset), int(max_test_cases))
        case_indices = list(range(n))
        case_selection = pd.DataFrame({"dataset_index": case_indices, "status": "selected"})
    if order_2d_before_3d:
        print("Ordering cases 2D before 3D...", flush=True)
        case_indices = sort_case_indices_2d_then_3d(test_dataset, case_indices)
    print(f"Selected {len(case_indices)} benchmark cases.", flush=True)
    if len(case_indices):
        print(f"First case index: {case_indices[0]} | Last case index: {case_indices[-1]}", flush=True)
    print("Building case progress table...", flush=True)
    case_progress = build_case_progress_table(test_dataset, case_indices)
    case_progress_by_index = case_progress.set_index("dataset_index").to_dict("index") if len(case_progress) else {}
    if len(case_progress) and "dataset_name" in case_progress.columns:
        dataset_names_in_order = case_progress["dataset_name"].astype(str).drop_duplicates().tolist()
    else:
        dataset_names_in_order = []
    dataset_order_by_name = {name: i for i, name in enumerate(dataset_names_in_order)}
    dataset_total_count = int(len(dataset_names_in_order))
    case_selection = annotate_case_selection_order(test_dataset, case_selection, case_indices)
    case_selection.to_csv(output_dir / "benchmark_case_selection.csv", index=False)
    if smoke_run and smoke_case_selection == "per_dataset":
        case_selection.to_csv(output_dir / "smoke_case_selection.csv", index=False)

    paths = {
        "api_probe": output_dir / "strict_video_api_probe.csv",
        "result_jsonl": output_dir / "per_case_results.jsonl",
        "results": output_dir / "per_case_results.csv",
        "legacy_results": output_dir / "strict_video_protocol_results.csv",
        "audit": output_dir / "prompt_audit.csv",
        "legacy_audit": output_dir / "strict_video_prompt_audit.csv",
        "audit_jsonl": output_dir / "prompt_audit.jsonl",
        "per_frame": output_dir / "per_frame_metrics.csv",
        "legacy_per_frame": output_dir / "strict_video_per_frame_metrics.csv",
        "per_frame_jsonl": output_dir / "per_frame_metrics.jsonl",
        "threshold": output_dir / "threshold_metrics_long.csv",
        "threshold_jsonl": output_dir / "threshold_metrics_long.jsonl",
        "memory": output_dir / "memory_measurements.csv",
        "memory_jsonl": output_dir / "memory_measurements.jsonl",
        "status": output_dir / "benchmark_status.json",
        "heartbeat": output_dir / "benchmark_heartbeat.json",
        "heartbeat_jsonl": output_dir / "benchmark_heartbeat.jsonl",
        "fault_log": output_dir / "python_fault_handler.log",
    }

    api_rows, result_rows, audit_rows, per_frame_rows, threshold_rows, memory_rows = [], [], [], [], [], []
    completed_keys = set()
    resume_aux_paths: dict[str, Path] = {}

    def row_key(row):
        try:
            return (str(row.get("model")), str(row.get("prompt_mode")), str(row.get("protocol")), int(float(row.get("dataset_index"))))
        except Exception:
            return None

    def get_case(dataset_index: int) -> dict[str, Any]:
        idx = int(dataset_index)
        if case_cache is not None and idx in case_cache:
            return case_cache[idx]
        case = load_case_by_index(test_dataset, idx)
        if case_cache is not None:
            case_cache[idx] = clone_cpu_case(case)
            return case_cache[idx]
        return case

    def get_prompt_plan_for_run(case: dict[str, Any], dataset_index: int, prompt_mode: str, protocol: str):
        idx = int(dataset_index)
        mode = str(prompt_mode)
        protocol = str(protocol)
        cache_key = (idx, mode, protocol)
        if prompt_plan_cache is not None and cache_key in prompt_plan_cache:
            return prompt_plan_cache[cache_key]

        diagnostic_key = (idx, mode, "validation_forward")
        if prompt_plan_cache is not None and diagnostic_key in prompt_plan_cache:
            diagnostic_plan, anchor, _diagnostic_plan = prompt_plan_cache[diagnostic_key]
        else:
            diagnostic_plan = cached_prompt_plan_for_case(case, mode)
            anchor = choose_anchor_frame(case, mode, plan=diagnostic_plan)
            if prompt_plan_cache is not None:
                prompt_plan_cache[diagnostic_key] = (diagnostic_plan, anchor, diagnostic_plan)

        if protocol == "validation_forward":
            plan = diagnostic_plan
        elif protocol in ("single_anchor_forward", "single_anchor_bidirectional"):
            plan = cached_prompt_plan_for_case(case, mode, prompt_frames=[anchor])
        else:
            plan = cached_prompt_plan_for_case(case, mode)

        out = (plan, int(anchor), diagnostic_plan)
        if prompt_plan_cache is not None:
            prompt_plan_cache[cache_key] = out
        return out

    def get_input_diagnostics(case: dict[str, Any], dataset_index: int, prompt_mode: str, diagnostic_plan: dict[str, Any]):
        if input_diag_cache is None:
            return case_input_diagnostics(case, prompt_mode, plan=diagnostic_plan)
        key = (int(dataset_index), str(prompt_mode))
        if key not in input_diag_cache:
            input_diag_cache[key] = case_input_diagnostics(case, prompt_mode, plan=diagnostic_plan)
        return dict(input_diag_cache[key])

    if resume_completed_rows and resume_from_dir:
        resume_dir = Path(resume_from_dir)
        resume_rows = []
        resume_source_path = None
        resume_jsonl_path = resume_dir / "per_case_results.jsonl"
        resume_results_path = resume_dir / "per_case_results.csv"
        print(f"Resume requested from {resume_dir}", flush=True)
        csv_resume_rows = []
        jsonl_resume_rows = []
        bad_jsonl_rows = 0
        if resume_jsonl_path.exists():
            bad_jsonl_rows = 0
            with open(resume_jsonl_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except Exception:
                        bad_jsonl_rows += 1
                        continue
                    if str(row.get("status")) == "ok":
                        jsonl_resume_rows.append(row)
            if bad_jsonl_rows:
                print(f"Ignored {bad_jsonl_rows} malformed JSONL resume rows from {resume_jsonl_path}", flush=True)
        if resume_results_path.exists():
            resume_results = pd.read_csv(resume_results_path)
            resume_results = resume_results[resume_results["status"].astype(str).eq("ok")].copy()
            csv_resume_rows = resume_results.to_dict("records")
        if jsonl_resume_rows and len(jsonl_resume_rows) >= len(csv_resume_rows):
            print(
                f"Loading resume rows from durable checkpoint {resume_jsonl_path.name} "
                f"({len(jsonl_resume_rows)} ok rows; CSV has {len(csv_resume_rows)} ok rows).",
                flush=True,
            )
            keyed_rows: OrderedDict[tuple[str, str, str, int] | tuple[str, str, str, str], dict[str, Any]] = OrderedDict()
            unkeyed_rows = []
            for row in jsonl_resume_rows:
                key = row_key(row)
                if key is None:
                    unkeyed_rows.append(row)
                else:
                    keyed_rows[key] = row
            resume_rows = list(keyed_rows.values()) + unkeyed_rows
            dropped_duplicates = len(jsonl_resume_rows) - len(resume_rows)
            if dropped_duplicates:
                print(f"Dropped {dropped_duplicates} duplicate JSONL resume rows by benchmark key.", flush=True)
            resume_source_path = resume_jsonl_path
        elif csv_resume_rows:
            print(
                f"Loading resume rows from {resume_results_path.name} "
                f"({len(csv_resume_rows)} ok rows; JSONL has {len(jsonl_resume_rows)} ok rows).",
                flush=True,
            )
            resume_rows = csv_resume_rows
            resume_source_path = resume_results_path
        else:
            print("No per_case_results.csv or per_case_results.jsonl found for resume.", flush=True)
        if resume_rows:
            result_rows.extend(resume_rows)
            completed_keys = {k for k in (row_key(r) for r in result_rows) if k is not None}
            for key, filenames in [
                ("audit", ["prompt_audit.csv", "prompt_audit.jsonl"]),
                ("per_frame", ["per_frame_metrics.csv", "per_frame_metrics.jsonl"]),
                ("threshold", ["threshold_metrics_long.csv", "threshold_metrics_long.jsonl"]),
                ("memory", ["memory_measurements.csv", "memory_measurements.jsonl"]),
            ]:
                for filename in filenames:
                    p = resume_dir / filename
                    if p.exists():
                        resume_aux_paths[key] = p
                        print(f"Deferring resume auxiliary table until snapshot/final write: {p.name}", flush=True)
                        break
            print(f"Resumed {len(completed_keys)} successful prediction rows from {resume_source_path}", flush=True)
    started_at = time.time()
    total_runs = len(case_indices) * len(benchmark_models) * len(prompt_modes) * len(protocols)
    done = 0
    print(f"Total planned run rows: {total_runs}", flush=True)
    fault_log_handle = None
    try:
        fault_log_handle = open(paths["fault_log"], "a", encoding="utf-8", buffering=1)
        fault_log_handle.write(f"\n\n=== strict video benchmark fault log start {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===\n")
        faulthandler.enable(file=fault_log_handle, all_threads=True)
        try:
            faulthandler.dump_traceback_later(1800, repeat=True, file=fault_log_handle)
        except Exception:
            pass
    except Exception as exc:
        print(f"Fault-handler log disabled: {type(exc).__name__}: {exc}", flush=True)
        fault_log_handle = None

    def write_heartbeat(
        phase: str,
        model_name: str = "",
        dataset_index=None,
        prompt_mode: str = "",
        protocol: str = "",
        case_meta: dict[str, Any] | None = None,
        last_row: dict[str, Any] | None = None,
        error: str = "",
        append: bool = False,
    ) -> None:
        try:
            elapsed = time.time() - started_at
            case_meta = case_meta or {}
            rec = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at_epoch": float(time.time()),
                "pid": int(os.getpid()),
                "phase": str(phase),
                "model": str(model_name or ""),
                "dataset_index": int(dataset_index) if dataset_index is not None else None,
                "prompt_mode": str(prompt_mode or ""),
                "protocol": str(protocol or ""),
                "runs_done": int(done),
                "runs_total": int(total_runs),
                "percent_done": float(100.0 * done / total_runs) if total_runs else 0.0,
                "elapsed_sec": float(elapsed),
                "elapsed_text": format_duration(elapsed) if "format_duration" in globals() else f"{elapsed:.1f}s",
                "current_dataset": case_meta.get("dataset_name"),
                "current_case_number_for_model": int(case_meta.get("benchmark_order")) if case_meta.get("benchmark_order") is not None and pd.notna(case_meta.get("benchmark_order")) else None,
                "current_case_total_for_model": int(len(case_indices)),
                "current_dataset_case_number": int(case_meta.get("dataset_case_number")) if case_meta.get("dataset_case_number") is not None and pd.notna(case_meta.get("dataset_case_number")) else None,
                "current_dataset_case_total": int(case_meta.get("dataset_case_total")) if case_meta.get("dataset_case_total") is not None and pd.notna(case_meta.get("dataset_case_total")) else None,
                "error": str(error or ""),
                "last_row_status": (last_row or {}).get("status"),
                "last_row_error": (last_row or {}).get("error"),
            }
            try:
                usage = shutil.disk_usage(output_dir)
                rec["output_disk_free_gb"] = float(usage.free / (1024**3))
                rec["output_disk_total_gb"] = float(usage.total / (1024**3))
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    rec.update(
                        {
                            "cuda_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
                            "cuda_reserved_mb": float(torch.cuda.memory_reserved() / (1024**2)),
                            "cuda_max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
                            "cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
                        }
                    )
                except Exception:
                    pass
            tmp_path = paths["heartbeat"].with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(rec, indent=2, default=json_default), encoding="utf-8")
            tmp_path.replace(paths["heartbeat"])
            if append:
                with open(paths["heartbeat_jsonl"], "a", encoding="utf-8") as hb:
                    hb.write(json.dumps(rec, default=json_default) + "\n")
        except Exception:
            pass

    write_heartbeat("run_initialized", append=True)

    def append_rows_to_csv(path: Path, rows: list[dict[str, Any]], header: bool = False) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(path, mode="a", header=header, index=False)

    def write_table_with_resume_base(base_path: Path | None, rows: list[dict[str, Any]], target_path: Path, legacy_path: Path | None = None) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_base = False
        if base_path is not None and Path(base_path).is_file():
            base_path = Path(base_path)
            if base_path.suffix.lower() == ".jsonl":
                first_chunk = True
                for chunk in pd.read_json(base_path, orient="records", lines=True, chunksize=100_000):
                    chunk.to_csv(target_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
                    first_chunk = False
                if first_chunk:
                    pd.DataFrame(rows).head(0).to_csv(target_path, index=False)
            else:
                shutil.copyfile(base_path, target_path)
            wrote_base = True
        else:
            pd.DataFrame(rows).head(0).to_csv(target_path, index=False)
            wrote_base = True
        if rows:
            if wrote_base and target_path.is_file() and target_path.stat().st_size > 0:
                append_rows_to_csv(target_path, rows, header=False)
            else:
                pd.DataFrame(rows).to_csv(target_path, index=False)
        if legacy_path is not None:
            shutil.copyfile(target_path, legacy_path)

    def save_partial(include_model_outputs: bool = False, full_tables: bool = False) -> float:
        save_started = time.perf_counter()
        pd.DataFrame(api_rows).to_csv(paths["api_probe"], index=False)
        pd.DataFrame(result_rows).to_csv(paths["results"], index=False)
        pd.DataFrame(result_rows).to_csv(paths["legacy_results"], index=False)
        if full_tables:
            write_table_with_resume_base(resume_aux_paths.get("audit"), audit_rows, paths["audit"], paths["legacy_audit"])
            write_table_with_resume_base(resume_aux_paths.get("per_frame"), per_frame_rows, paths["per_frame"], paths["legacy_per_frame"])
            write_table_with_resume_base(resume_aux_paths.get("threshold"), threshold_rows, paths["threshold"])
            write_table_with_resume_base(resume_aux_paths.get("memory"), memory_rows, paths["memory"])
        if save_model_separated_outputs and include_model_outputs:
            write_model_separated_outputs(
                output_dir,
                benchmark_models,
                api_rows=api_rows,
                result_rows=result_rows,
                audit_rows=audit_rows,
                per_frame_rows=per_frame_rows,
                threshold_rows=threshold_rows,
                memory_rows=memory_rows,
            )
        return float(time.perf_counter() - save_started)

    def write_status(
        model_name="",
        dataset_index=None,
        last_row=None,
        model_number=None,
        models_total=None,
        model_runs_done=None,
        model_runs_total=None,
        case_meta=None,
        write_progress_report: bool = False,
        progress_report_reason: str = "",
    ):
        status_started = time.perf_counter()
        elapsed = time.time() - started_at
        frac = done / total_runs if total_runs else 0.0
        eta = elapsed * (1.0 - frac) / frac if frac > 0 else float("nan")
        case_meta = case_meta or {}
        model_frac = float(model_runs_done) / float(model_runs_total) if model_runs_done and model_runs_total else 0.0
        model_case_visits_total = int(len(benchmark_models) * len(case_indices))
        current_model_number = int(model_number) if model_number is not None else None
        current_case_number = case_meta.get("benchmark_order")
        model_case_visit_number = None
        if current_model_number is not None and current_case_number is not None and pd.notna(current_case_number):
            model_case_visit_number = int((current_model_number - 1) * len(case_indices) + int(current_case_number))
        runs_per_case = max(1, int(len(prompt_modes) * len(protocols)))
        model_cases_done = min(int(len(case_indices)), int(math.ceil(float(model_runs_done or 0) / runs_per_case)))
        case_visits_done_est = float(done) / float(runs_per_case) if runs_per_case else 0.0
        total_case_visits = int(len(benchmark_models) * len(case_indices))
        mean_sec_per_case_visit = float(elapsed / case_visits_done_est) if case_visits_done_est > 0 else float("nan")
        model_case_visit_percent_done = 100.0 * float(model_case_visit_number or 0) / float(model_case_visits_total or 1)
        current_dataset = str(case_meta.get("dataset_name")) if case_meta.get("dataset_name") is not None else None
        current_dataset_rank = dataset_order_by_name.get(current_dataset) if current_dataset is not None else None
        current_model_datasets_remaining = None
        overall_dataset_visits_remaining = None
        if current_dataset_rank is not None:
            current_model_datasets_remaining = max(0, int(dataset_total_count) - int(current_dataset_rank))
            if current_model_number is not None:
                future_model_dataset_visits = max(0, int(len(benchmark_models)) - int(current_model_number)) * int(dataset_total_count)
                overall_dataset_visits_remaining = int(current_model_datasets_remaining + future_model_dataset_visits)
        status = {
            "run_name": run_stamp,
            "output_dir": str(output_dir),
            "smoke_run": bool(smoke_run),
            "model": model_name,
            "dataset_index": int(dataset_index) if dataset_index is not None else None,
            "models_done_or_current": current_model_number,
            "models_total": int(models_total) if models_total is not None else int(len(benchmark_models)),
            "runs_done": int(done),
            "runs_total": int(total_runs),
            "percent_done": 100.0 * frac,
            "case_visits_done_est": float(case_visits_done_est),
            "case_visits_total": int(total_case_visits),
            "mean_sec_per_case_visit": float(mean_sec_per_case_visit) if math.isfinite(mean_sec_per_case_visit) else None,
            "elapsed_text": format_duration(elapsed),
            "eta_text": format_duration(eta) if math.isfinite(eta) else "unknown",
            "model_runs_done": int(model_runs_done or 0),
            "model_runs_total": int(model_runs_total or 0),
            "model_percent_done": 100.0 * model_frac,
            "model_cases_done": int(model_cases_done),
            "model_cases_total": int(len(case_indices)),
            "model_case_visit_number": model_case_visit_number,
            "model_case_visits_total": model_case_visits_total,
            "model_case_visit_percent_done": float(model_case_visit_percent_done),
            "current_case_number_for_model": int(current_case_number) if current_case_number is not None and pd.notna(current_case_number) else None,
            "current_case_total_for_model": int(len(case_indices)),
            "current_dataset": case_meta.get("dataset_name"),
            "current_dataset_case_number": int(case_meta.get("dataset_case_number")) if case_meta.get("dataset_case_number") is not None and pd.notna(case_meta.get("dataset_case_number")) else None,
            "current_dataset_case_total": int(case_meta.get("dataset_case_total")) if case_meta.get("dataset_case_total") is not None and pd.notna(case_meta.get("dataset_case_total")) else None,
            "current_dataset_percent_done": float(case_meta.get("dataset_percent_done")) if case_meta.get("dataset_percent_done") is not None and pd.notna(case_meta.get("dataset_percent_done")) else None,
            "current_model_dataset_number": int(current_dataset_rank + 1) if current_dataset_rank is not None else None,
            "current_model_dataset_total": int(dataset_total_count),
            "current_model_datasets_remaining_including_current": int(current_model_datasets_remaining) if current_model_datasets_remaining is not None else None,
            "overall_dataset_visits_remaining_including_current": int(overall_dataset_visits_remaining) if overall_dataset_visits_remaining is not None else None,
            "current_order_phase": case_meta.get("order_phase"),
            "current_phase_case_number": int(case_meta.get("phase_case_number")) if case_meta.get("phase_case_number") is not None and pd.notna(case_meta.get("phase_case_number")) else None,
            "current_phase_case_total": int(case_meta.get("phase_case_total")) if case_meta.get("phase_case_total") is not None and pd.notna(case_meta.get("phase_case_total")) else None,
            "elapsed_sec": float(elapsed),
            "eta_sec": float(eta) if math.isfinite(eta) else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "progress_report_reason": progress_report_reason,
            "last_row": last_row or {},
        }
        paths["status"].write_text(json.dumps(status, indent=2, default=json_default), encoding="utf-8")
        write_heartbeat(
            "status_flush",
            model_name=model_name,
            dataset_index=dataset_index,
            case_meta=case_meta,
            last_row=last_row,
            append=True,
        )
        if tb_writer is not None:
            step = int(done)
            tb_writer.add_scalar("progress/percent_done", float(status["percent_done"]), step)
            tb_writer.add_scalar("progress/runs_done", float(done), step)
            tb_writer.add_text("progress/overall", tensorboard_progress_markdown(status), step)
            tb_writer.flush()
        return float(time.perf_counter() - status_started)

    status_write_state = {"last_done": -1, "last_time": 0.0}

    def maybe_write_status(*args, force: bool = False, **kwargs) -> float:
        now = time.time()
        rows_since = int(done) - int(status_write_state["last_done"])
        seconds_since = now - float(status_write_state["last_time"])
        should_write = (
            bool(force)
            or bool(kwargs.get("write_progress_report", False))
            or rows_since >= status_update_every_rows
            or seconds_since >= status_update_every_sec
            or int(done) >= int(total_runs)
        )
        if not should_write:
            return 0.0
        elapsed = write_status(*args, **kwargs)
        status_write_state["last_done"] = int(done)
        status_write_state["last_time"] = now
        return elapsed

    model_runs_total_per_model = len(case_indices) * len(prompt_modes) * len(protocols)
    model_done_counts = {str(model_name): 0 for model_name in benchmark_models}
    tb_completed_datasets = {str(model_name): set() for model_name in benchmark_models}
    tb_dataset_progress_buckets = {str(model_name): {} for model_name in benchmark_models}
    runs_per_case = max(1, int(len(prompt_modes) * len(protocols)))

    def run_key_for(model_name, prompt_mode, protocol, dataset_index):
        return (str(model_name), str(prompt_mode), str(protocol), int(dataset_index))

    def completed_count_for_case(model_name, dataset_index):
        if not completed_keys:
            return 0
        return sum(
            1
            for prompt_mode in prompt_modes
            for protocol in protocols
            if run_key_for(model_name, prompt_mode, protocol, dataset_index) in completed_keys
        )

    def completed_count_for_model_grid(model_name):
        if not completed_keys:
            return 0
        return sum(completed_count_for_case(model_name, dataset_index) for dataset_index in case_indices)

    def first_incomplete_case_for_model(model_name):
        for position, dataset_index in enumerate(case_indices, start=1):
            if completed_count_for_case(model_name, dataset_index) < runs_per_case:
                return position, int(dataset_index)
        return None, None

    def context_for(model_name, model_number, dataset_index, fallback_case_number=None):
        meta = case_progress_by_index.get(int(dataset_index), {}).copy() if dataset_index is not None else {}
        if not meta and dataset_index is not None:
            meta = case_order_metadata(test_dataset, int(dataset_index))
            if fallback_case_number is not None:
                meta["benchmark_order"] = int(fallback_case_number)
                meta["benchmark_case_total"] = int(len(case_indices))
        return {
            "model_name": model_name,
            "dataset_index": dataset_index,
            "model_number": int(model_number),
            "models_total": int(len(benchmark_models)),
            "model_runs_done": int(model_done_counts.get(str(model_name), 0)),
            "model_runs_total": int(model_runs_total_per_model),
            "case_meta": meta,
        }

    def maybe_write_group_tensorboard(model_name, completed_dataset=None):
        if tb_writer is None:
            return
        if completed_dataset is None:
            return
        completed_dataset = str(completed_dataset)
        completed = tb_completed_datasets.setdefault(str(model_name), set())
        if completed_dataset in completed:
            return
        completed.add(completed_dataset)
        safe_model = safe_tensorboard_tag(model_name)
        completion_df = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "last_completed_dataset": completed_dataset,
                    "datasets_completed": int(len(completed)),
                    "global_runs_done": int(done),
                    "global_runs_total": int(total_runs),
                }
            ]
        )
        tb_writer.add_text(
            f"{safe_model}/progress/last_completed_dataset",
            "### Last Completed Dataset\n\n" + safe_markdown_table(completion_df, max_rows=1),
            int(done),
        )
        write_tensorboard_running_group_metrics(tb_writer, result_rows, model_name, int(done), completed_dataset=completed_dataset)

    def maybe_write_dataset_progress_report(model_name, model_number, dataset_index, case_position, case_meta, last_row=None):
        if tb_writer is None:
            return
        dataset_name = str(case_meta.get("dataset_name", "unknown"))
        try:
            done_cases = int(case_meta.get("dataset_case_number") or 0)
            total_cases = int(case_meta.get("dataset_case_total") or 0)
        except Exception:
            return
        if total_cases <= 0 or done_cases <= 0:
            return
        pct = 100.0 * min(done_cases, total_cases) / float(total_cases)
        bucket = min(10, int(math.floor((pct + 1e-9) / 10.0)))
        if done_cases >= total_cases:
            bucket = 10
        if bucket <= 0:
            return
        dataset_buckets = tb_dataset_progress_buckets.setdefault(str(model_name), {})
        last_bucket = int(dataset_buckets.get(dataset_name, 0))
        if bucket <= last_bucket:
            return
        dataset_buckets[dataset_name] = bucket
        report_meta = dict(case_meta)
        report_meta["dataset_percent_done"] = pct
        ctx = context_for(model_name, model_number, dataset_index, case_position)
        ctx["case_meta"] = report_meta
        maybe_write_status(
            **ctx,
            last_row=last_row or {},
            write_progress_report=True,
            progress_report_reason=f"{dataset_name}: >= {bucket * 10}% dataset milestone",
            force=True,
        )

    def write_jsonl_record(handle, row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, default=json_default) + "\n")

    def write_jsonl_records(handle, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            write_jsonl_record(handle, row)

    with (
        open(paths["result_jsonl"], "w", encoding="utf-8") as jf,
        open(paths["audit_jsonl"], "w", encoding="utf-8") as audit_jf,
        open(paths["per_frame_jsonl"], "w", encoding="utf-8") as per_frame_jf,
        open(paths["threshold_jsonl"], "w", encoding="utf-8") as threshold_jf,
        open(paths["memory_jsonl"], "w", encoding="utf-8") as memory_jf,
    ):
        for row in result_rows:
            write_jsonl_record(jf, row)
        if result_rows:
            jf.flush()
            print(f"Seeded {paths['result_jsonl'].name} with {len(result_rows)} resumed result rows.", flush=True)
        for model_number, model_name in enumerate(tqdm(benchmark_models, desc="Models", unit="model"), start=1):
            model_completed_count = completed_count_for_model_grid(model_name)
            first_incomplete_position, first_incomplete_index = first_incomplete_case_for_model(model_name)
            completed_cases_for_model = int(model_completed_count // runs_per_case)
            print(
                f"[{model_number}/{len(benchmark_models)}] {model_name}: "
                f"{model_completed_count}/{model_runs_total_per_model} runs already complete "
                f"({completed_cases_for_model}/{len(case_indices)} full cases).",
                flush=True,
            )
            if model_completed_count >= model_runs_total_per_model:
                done += int(model_runs_total_per_model)
                model_done_counts[str(model_name)] = int(model_runs_total_per_model)
                print(f"Resume fast-skip: {model_name} already has {model_runs_total_per_model} completed runs; skipping model load.")
                continue
            case_loop_start = 0
            if first_incomplete_index is not None:
                case_loop_start = max(0, int(first_incomplete_position) - 1)
                skipped_runs_before_start = int(case_loop_start * runs_per_case)
                if skipped_runs_before_start:
                    done += skipped_runs_before_start
                    model_done_counts[str(model_name)] += skipped_runs_before_start
                print(
                    f"{model_name}: jumping past {case_loop_start} fully completed cases "
                    f"({skipped_runs_before_start} runs) to first incomplete case position "
                    f"{first_incomplete_position}/{len(case_indices)} (dataset_index={first_incomplete_index}).",
                    flush=True,
                )
            bundle = None
            load_started = time.perf_counter()
            load_baseline = prepare_cuda_memory_measurement(clear_cache=True)
            try:
                write_heartbeat("model_load_start", model_name=model_name, append=True)
                print(f"Loading model {model_name}...", flush=True)
                bundle = load_model_bundle(model_name)
                api_row = api_probe_row(model_name, bundle, status="ok", error="")
                api_row.update(load_baseline)
                api_row.update(cuda_memory_snapshot("model_loaded_"))
                api_row["load_time_sec"] = float(time.perf_counter() - load_started)
                api_row["param_count"] = parameter_count(bundle)
                api_rows.append(api_row)
                pd.DataFrame(api_rows).to_csv(paths["api_probe"], index=False)
                print(f"Loaded model {model_name} in {api_row['load_time_sec']:.1f}s.", flush=True)
                write_heartbeat("model_load_done", model_name=model_name, append=True)
            except Exception as exc:
                api_row = {
                    **asdict(MODEL_REGISTRY[model_name]),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "load_time_sec": float(time.perf_counter() - load_started),
                }
                api_rows.append(api_row)
                pd.DataFrame(api_rows).to_csv(paths["api_probe"], index=False)
                print(f"Failed to load {model_name}: {api_row['error']}")
                write_heartbeat("model_load_failed", model_name=model_name, error=api_row["error"], append=True)

            try:
                skipped_cases_for_model = 0
                reported_first_new_case = False
                case_iter = tqdm(
                    case_indices[case_loop_start:],
                    desc=model_name,
                    unit="case",
                    leave=False,
                    total=len(case_indices),
                    initial=case_loop_start,
                )
                for case_position, dataset_index in enumerate(case_iter, start=case_loop_start + 1):
                    case_meta = case_progress_by_index.get(int(dataset_index), {}).copy()
                    if not case_meta:
                        case_meta = case_order_metadata(test_dataset, int(dataset_index))
                        case_meta["benchmark_order"] = int(case_position)
                        case_meta["benchmark_case_total"] = int(len(case_indices))
                    next_dataset_name = None
                    if case_position < len(case_indices):
                        next_meta = case_progress_by_index.get(int(case_indices[case_position]), {})
                        next_dataset_name = next_meta.get("dataset_name")
                    dataset_boundary = next_dataset_name != case_meta.get("dataset_name")
                    case_completed_count = completed_count_for_case(model_name, dataset_index)
                    if case_completed_count >= runs_per_case:
                        done += int(runs_per_case)
                        model_done_counts[str(model_name)] += int(runs_per_case)
                        skipped_cases_for_model += 1
                        if skipped_cases_for_model == 1 or skipped_cases_for_model % 5000 == 0:
                            print(
                                f"{model_name}: fast-skipped {skipped_cases_for_model} completed cases "
                                f"through case position {case_position}/{len(case_indices)}.",
                                flush=True,
                            )
                        continue
                    if not reported_first_new_case:
                        print(
                            f"{model_name}: starting/resuming inference at case position "
                            f"{case_position}/{len(case_indices)}, dataset_index={int(dataset_index)}, "
                            f"dataset={case_meta.get('dataset_name')}, "
                            f"completed prompts for this case={case_completed_count}/{runs_per_case}.",
                            flush=True,
                        )
                        reported_first_new_case = True
                    case_load_started = time.perf_counter()
                    write_heartbeat("case_load_start", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta, append=True)
                    try:
                        case = get_case(dataset_index)
                        case_load_time_sec = float(time.perf_counter() - case_load_started)
                        write_heartbeat("case_load_done", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta)
                    except Exception as exc:
                        case_load_time_sec = float(time.perf_counter() - case_load_started)
                        write_heartbeat("case_load_failed", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta, error=f"{type(exc).__name__}: {exc}", append=True)
                        for prompt_mode in prompt_modes:
                            for protocol in protocols:
                                row = {
                                    "dataset_index": int(dataset_index),
                                    "model": model_name,
                                    "prompt_mode": prompt_mode,
                                    "protocol": protocol,
                                    "status": "case_load_failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "case_load_time_sec": case_load_time_sec,
                                    "prompt_plan_time_sec": 0.0,
                                    "input_diag_time_sec": 0.0,
                                    "metric_arrays_time_sec": 0.0,
                                    "threshold_metrics_time_sec": 0.0,
                                    "metrics_time_sec": 0.0,
                                    "per_frame_metrics_time_sec": 0.0,
                                    "jsonl_write_time_sec": 0.0,
                                    "status_write_time_sec": 0.0,
                                    "partial_save_time_sec": 0.0,
                                    "total_row_time_sec": case_load_time_sec,
                                }
                                result_rows.append(row)
                                jsonl_started = time.perf_counter()
                                write_jsonl_record(jf, row)
                                row["jsonl_write_time_sec"] = float(time.perf_counter() - jsonl_started)
                                done += 1
                                model_done_counts[str(model_name)] += 1
                                row["status_write_time_sec"] = maybe_write_status(**context_for(model_name, model_number, dataset_index, case_position), last_row=row, force=True)
                        if fail_fast:
                            raise
                        maybe_write_dataset_progress_report(
                            model_name,
                            model_number,
                            dataset_index,
                            case_position,
                            case_meta,
                            last_row=result_rows[-1] if result_rows else {},
                        )
                        if dataset_boundary:
                            maybe_write_group_tensorboard(model_name, completed_dataset=case_meta.get("dataset_name"))
                        continue

                    state_reuse_outputs = {}
                    state_reuse_plan_info = {}
                    can_reuse_prompt_state = (
                        bool(reuse_prompt_mode_state)
                        and bundle is not None
                        and (reuse_prompt_mode_state_model_set is None or str(model_name) in reuse_prompt_mode_state_model_set)
                        and [str(p) for p in protocols] == ["validation_forward"]
                        and len(prompt_modes) > 1
                    )
                    if can_reuse_prompt_state:
                        pending_reuse_modes = [
                            str(prompt_mode)
                            for prompt_mode in prompt_modes
                            if run_key_for(model_name, prompt_mode, "validation_forward", dataset_index) not in completed_keys
                        ]
                        if len(pending_reuse_modes) > 1:
                            try:
                                write_heartbeat("state_reuse_start", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta)
                                prompt_plans_for_reuse = {}
                                for prompt_mode in pending_reuse_modes:
                                    plan_started = time.perf_counter()
                                    plan, anchor, diagnostic_plan = get_prompt_plan_for_run(case, dataset_index, prompt_mode, "validation_forward")
                                    state_reuse_plan_info[prompt_mode] = (
                                        plan,
                                        int(anchor),
                                        diagnostic_plan,
                                        float(time.perf_counter() - plan_started),
                                    )
                                    prompt_plans_for_reuse[prompt_mode] = (plan, int(anchor))
                                state_reuse_outputs = run_validation_forward_prompt_modes_reused_state_measured(
                                    bundle,
                                    case,
                                    prompt_plans_for_reuse,
                                    clear_cache=True,
                                )
                                write_heartbeat("state_reuse_done", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta)
                            except Exception as exc:
                                if fail_fast:
                                    raise
                                write_heartbeat("state_reuse_failed", model_name=model_name, dataset_index=dataset_index, case_meta=case_meta, error=f"{type(exc).__name__}: {exc}", append=True)
                                print(
                                    f"{model_name}: prompt-mode state reuse failed for dataset_index={int(dataset_index)}; "
                                    f"falling back to isolated prompt runs ({type(exc).__name__}: {exc}).",
                                    flush=True,
                                )
                                state_reuse_outputs = {}
                                state_reuse_plan_info = {}

                    for prompt_mode in prompt_modes:
                        for protocol in protocols:
                            key = (str(model_name), str(prompt_mode), str(protocol), int(dataset_index))
                            if key in completed_keys:
                                done += 1
                                model_done_counts[str(model_name)] += 1
                                continue
                            if bundle is None:
                                row_started = time.perf_counter()
                                row = failed_result_row(case, model_name, prompt_mode, protocol, "model_load_failed", api_row.get("error", "model load failed"))
                                row.update(
                                    {
                                        "case_load_time_sec": case_load_time_sec,
                                        "prompt_plan_time_sec": 0.0,
                                        "input_diag_time_sec": 0.0,
                                        "metric_arrays_time_sec": 0.0,
                                        "threshold_metrics_time_sec": 0.0,
                                        "metrics_time_sec": 0.0,
                                        "per_frame_metrics_time_sec": 0.0,
                                        "jsonl_write_time_sec": 0.0,
                                        "status_write_time_sec": 0.0,
                                        "partial_save_time_sec": 0.0,
                                        "total_row_time_sec": 0.0,
                                    }
                                )
                                result_rows.append(row)
                                audit_row = prompt_audit_row(case, model_name, prompt_mode, protocol, {}, "model_load_failed", row["error"])
                                audit_rows.append(audit_row)
                                jsonl_started = time.perf_counter()
                                write_jsonl_record(jf, row)
                                write_jsonl_record(audit_jf, audit_row)
                                row["jsonl_write_time_sec"] = float(time.perf_counter() - jsonl_started)
                                done += 1
                                model_done_counts[str(model_name)] += 1
                                row["status_write_time_sec"] = maybe_write_status(**context_for(model_name, model_number, dataset_index, case_position), last_row=row, force=True)
                                row["total_row_time_sec"] = float(time.perf_counter() - row_started)
                                continue
                            row_started = time.perf_counter()
                            prompt_plan_time_sec = 0.0
                            input_diag_time_sec = 0.0
                            metric_arrays_time_sec = 0.0
                            threshold_metrics_time_sec = 0.0
                            metrics_time_sec = 0.0
                            per_frame_metrics_time_sec = 0.0
                            partial_save_time_sec = 0.0
                            status_write_time_sec = 0.0
                            jsonl_write_time_sec = 0.0
                            try:
                                write_heartbeat(
                                    "row_start",
                                    model_name=model_name,
                                    dataset_index=dataset_index,
                                    prompt_mode=prompt_mode,
                                    protocol=protocol,
                                    case_meta=case_meta,
                                    append=True,
                                )
                                if str(protocol) == "validation_forward" and str(prompt_mode) in state_reuse_outputs:
                                    prompt_plan, anchor, diagnostic_plan, prompt_plan_time_sec = state_reuse_plan_info[str(prompt_mode)]
                                    logits, info = state_reuse_outputs[str(prompt_mode)]
                                else:
                                    prompt_plan_started = time.perf_counter()
                                    prompt_plan, anchor, diagnostic_plan = get_prompt_plan_for_run(case, dataset_index, prompt_mode, protocol)
                                    prompt_plan_time_sec = float(time.perf_counter() - prompt_plan_started)
                                    logits, info = run_strict_video_protocol_measured(
                                        bundle,
                                        case,
                                        prompt_mode,
                                        protocol,
                                        clear_cache=True,
                                        plan=prompt_plan,
                                        anchor=anchor,
                                    )
                                write_heartbeat(
                                    "inference_done",
                                    model_name=model_name,
                                    dataset_index=dataset_index,
                                    prompt_mode=prompt_mode,
                                    protocol=protocol,
                                    case_meta=case_meta,
                                )
                                metric_arrays_started = time.perf_counter()
                                metric_arrays = prepare_metric_arrays(logits, case["gt_thw"])
                                metric_arrays_time_sec = float(time.perf_counter() - metric_arrays_started)
                                write_heartbeat(
                                    "metrics_start",
                                    model_name=model_name,
                                    dataset_index=dataset_index,
                                    prompt_mode=prompt_mode,
                                    protocol=protocol,
                                    case_meta=case_meta,
                                )
                                threshold_started = time.perf_counter()
                                case_threshold_rows = threshold_metrics_long_from_logits(logits, case["gt_thw"], thresholds=thresholds, metric_arrays=metric_arrays)
                                threshold_metrics_time_sec = float(time.perf_counter() - threshold_started)
                                metrics_started = time.perf_counter()
                                metrics = metrics_from_logits(
                                    logits,
                                    case["gt_thw"],
                                    thresholds=thresholds,
                                    primary_threshold=primary_threshold,
                                    compute_hd95_2d=compute_hd95_2d,
                                    compute_hd95_2d_for_3d=compute_hd95_2d_for_3d_cases,
                                    compute_hd95_3d=compute_hd95_3d,
                                    threshold_rows=case_threshold_rows,
                                    metric_arrays=metric_arrays,
                                )
                                metrics_time_sec = float(time.perf_counter() - metrics_started)
                                base = base_case_fields(case)
                                input_diag_started = time.perf_counter()
                                input_diag = get_input_diagnostics(case, dataset_index, prompt_mode, diagnostic_plan)
                                input_diag_time_sec = float(time.perf_counter() - input_diag_started)
                                info_keys = [
                                    "wall_time_sec", "init_time_sec", "prompt_time_sec", "propagation_time_sec", "reset_time_sec",
                                    "sec_per_frame", "frames_per_sec", "n_frames", "n_prompt_frames", "preprocess", "init_method",
                                    "prompt_method", "propagation_method", "anchor_frame", "yielded_frame_count", "untracked_frame_count",
                                    "peak_cuda_mem_allocated_mb", "peak_cuda_mem_reserved_mb", "delta_peak_cuda_mem_allocated_mb",
                                    "delta_peak_cuda_mem_reserved_mb", "delta_peak_cuda_mem_allocated_mb_per_frame",
                                    "delta_peak_cuda_mem_reserved_mb_per_frame", "param_count", "state_reuse_prompt_modes",
                                    "state_reuse_prompt_count", "state_reuse_shared_init_time_sec",
                                    "state_reuse_allocated_init_time_sec", "state_reuse_shared_wall_time_sec",
                                    "state_reuse_memory_baseline", "state_reuse_init_peak_cuda_mem_allocated_mb",
                                    "state_reuse_init_peak_cuda_mem_reserved_mb",
                                ]
                                row = {
                                    **base,
                                    "model": model_name,
                                    "model_family": bundle["entry"].family,
                                    "prompt_mode": prompt_mode,
                                    "effective_prompt_mode": canonical_prompt_mode(prompt_mode),
                                    "protocol": protocol,
                                    "status": "ok",
                                    "error": "",
                                    **input_diag,
                                    **metrics,
                                    **{k: info.get(k) for k in info_keys},
                                    "case_load_time_sec": case_load_time_sec,
                                    "prompt_plan_time_sec": prompt_plan_time_sec,
                                    "input_diag_time_sec": input_diag_time_sec,
                                    "metric_arrays_time_sec": metric_arrays_time_sec,
                                    "threshold_metrics_time_sec": threshold_metrics_time_sec,
                                    "metrics_time_sec": metrics_time_sec,
                                    "per_frame_metrics_time_sec": 0.0,
                                    "jsonl_write_time_sec": 0.0,
                                    "status_write_time_sec": 0.0,
                                    "partial_save_time_sec": 0.0,
                                    "total_row_time_sec": 0.0,
                                    "requested_prompt_frames": json.dumps(info.get("requested_prompt_frames", [])),
                                    "yielded_frames": json.dumps(info.get("yielded_frames", [])),
                                    "untracked_frames": json.dumps(info.get("untracked_frames", [])),
                                }
                                result_rows.append(row)
                                audit_row = prompt_audit_row(case, model_name, prompt_mode, protocol, info, "ok")
                                audit_rows.append(audit_row)
                                mem_row = {**base, "model": model_name, "prompt_mode": prompt_mode, "protocol": protocol, "status": "ok"}
                                mem_row.update({k: v for k, v in info.items() if "cuda" in k or k in ["wall_time_sec", "sec_per_frame", "frames_per_sec", "n_frames", "param_count"]})
                                mem_row.update(
                                    {
                                        "case_load_time_sec": case_load_time_sec,
                                        "prompt_plan_time_sec": prompt_plan_time_sec,
                                        "input_diag_time_sec": input_diag_time_sec,
                                        "metric_arrays_time_sec": metric_arrays_time_sec,
                                        "threshold_metrics_time_sec": threshold_metrics_time_sec,
                                        "metrics_time_sec": metrics_time_sec,
                                        "state_reuse_prompt_modes": bool(info.get("state_reuse_prompt_modes", False)),
                                        "state_reuse_prompt_count": info.get("state_reuse_prompt_count"),
                                        "state_reuse_shared_init_time_sec": info.get("state_reuse_shared_init_time_sec"),
                                        "state_reuse_allocated_init_time_sec": info.get("state_reuse_allocated_init_time_sec"),
                                        "state_reuse_memory_baseline": info.get("state_reuse_memory_baseline", "isolated_prompt_init"),
                                    }
                                )
                                memory_rows.append(mem_row)
                                per_frame_started = time.perf_counter()
                                case_per_frame_rows = [
                                    {**base, "model": model_name, "prompt_mode": prompt_mode, "protocol": protocol, **fr, "anchor_frame": int(info.get("anchor_frame", -1))}
                                    for fr in per_frame_metrics_from_logits(logits, case["gt_thw"], threshold=primary_threshold, metric_arrays=metric_arrays)
                                ]
                                per_frame_metrics_time_sec = float(time.perf_counter() - per_frame_started)
                                row["per_frame_metrics_time_sec"] = per_frame_metrics_time_sec
                                per_frame_rows.extend(case_per_frame_rows)
                                case_threshold_output_rows = [
                                    {**base, "model": model_name, "prompt_mode": prompt_mode, "protocol": protocol, **tr}
                                    for tr in case_threshold_rows
                                ]
                                for tr in case_threshold_rows:
                                    threshold_rows.append({**base, "model": model_name, "prompt_mode": prompt_mode, "protocol": protocol, **tr})
                                write_heartbeat(
                                    "jsonl_write_start",
                                    model_name=model_name,
                                    dataset_index=dataset_index,
                                    prompt_mode=prompt_mode,
                                    protocol=protocol,
                                    case_meta=case_meta,
                                )
                                jsonl_started = time.perf_counter()
                                write_jsonl_record(jf, row)
                                write_jsonl_record(audit_jf, audit_row)
                                write_jsonl_record(memory_jf, mem_row)
                                write_jsonl_records(per_frame_jf, case_per_frame_rows)
                                write_jsonl_records(threshold_jf, case_threshold_output_rows)
                                jsonl_write_time_sec = float(time.perf_counter() - jsonl_started)
                                row["jsonl_write_time_sec"] = jsonl_write_time_sec
                            except Exception as exc:
                                write_heartbeat(
                                    "row_failed",
                                    model_name=model_name,
                                    dataset_index=dataset_index,
                                    prompt_mode=prompt_mode,
                                    protocol=protocol,
                                    case_meta=case_meta,
                                    error=f"{type(exc).__name__}: {exc}",
                                    append=True,
                                )
                                row = failed_result_row(case, model_name, prompt_mode, protocol, "failed", f"{type(exc).__name__}: {exc}")
                                row.update(
                                    {
                                        "case_load_time_sec": case_load_time_sec,
                                        "prompt_plan_time_sec": locals().get("prompt_plan_time_sec", 0.0),
                                        "input_diag_time_sec": 0.0,
                                        "metric_arrays_time_sec": locals().get("metric_arrays_time_sec", 0.0),
                                        "threshold_metrics_time_sec": locals().get("threshold_metrics_time_sec", 0.0),
                                        "metrics_time_sec": locals().get("metrics_time_sec", 0.0),
                                        "per_frame_metrics_time_sec": 0.0,
                                        "jsonl_write_time_sec": 0.0,
                                        "status_write_time_sec": 0.0,
                                        "partial_save_time_sec": 0.0,
                                        "total_row_time_sec": 0.0,
                                    }
                                )
                                result_rows.append(row)
                                audit_row = prompt_audit_row(case, model_name, prompt_mode, protocol, {}, "failed", row["error"])
                                audit_rows.append(audit_row)
                                jsonl_started = time.perf_counter()
                                write_jsonl_record(jf, row)
                                write_jsonl_record(audit_jf, audit_row)
                                row["jsonl_write_time_sec"] = float(time.perf_counter() - jsonl_started)
                                if fail_fast:
                                    raise
                            done += 1
                            model_done_counts[str(model_name)] += 1
                            if done % checkpoint_flush_every_rows == 0:
                                flush_started = time.perf_counter()
                                jf.flush()
                                audit_jf.flush()
                                memory_jf.flush()
                                per_frame_jf.flush()
                                threshold_jf.flush()
                                row["jsonl_write_time_sec"] = float(row.get("jsonl_write_time_sec", 0.0) + (time.perf_counter() - flush_started))
                            if int(partial_save_every_rows or 0) > 0 and done % int(partial_save_every_rows) == 0:
                                partial_save_time_sec = save_partial(include_model_outputs=bool(write_model_outputs_on_partial), full_tables=False)
                                row["partial_save_time_sec"] = partial_save_time_sec
                                print(
                                    f"Checkpoint snapshot at row {done}/{total_runs}: "
                                    f"results CSV refreshed in {partial_save_time_sec:.1f}s; large auxiliary CSVs deferred.",
                                    flush=True,
                                )
                            status_write_time_sec = maybe_write_status(**context_for(model_name, model_number, dataset_index, case_position), last_row=result_rows[-1])
                            row["status_write_time_sec"] = status_write_time_sec
                            if bool(row.get("state_reuse_prompt_modes", False)):
                                row["total_row_time_sec"] = float(
                                    sum(
                                        float(row.get(k, 0.0) or 0.0)
                                        for k in [
                                            "prompt_plan_time_sec",
                                            "wall_time_sec",
                                            "metric_arrays_time_sec",
                                            "threshold_metrics_time_sec",
                                            "metrics_time_sec",
                                            "input_diag_time_sec",
                                            "per_frame_metrics_time_sec",
                                            "jsonl_write_time_sec",
                                            "status_write_time_sec",
                                            "partial_save_time_sec",
                                        ]
                                    )
                                )
                            else:
                                row["total_row_time_sec"] = float(time.perf_counter() - row_started)
                            write_heartbeat(
                                "row_done",
                                model_name=model_name,
                                dataset_index=dataset_index,
                                prompt_mode=prompt_mode,
                                protocol=protocol,
                                case_meta=case_meta,
                                last_row=row,
                                append=True,
                            )
                    maybe_write_dataset_progress_report(
                        model_name,
                        model_number,
                        dataset_index,
                        case_position,
                        case_meta,
                        last_row=result_rows[-1] if result_rows else {},
                    )
                    if dataset_boundary:
                        maybe_write_group_tensorboard(model_name, completed_dataset=case_meta.get("dataset_name"))
            finally:
                unload_model_bundle(bundle)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    final_snapshot_time_sec = save_partial(include_model_outputs=False, full_tables=True)
    print(f"Final CSV materialization complete in {final_snapshot_time_sec:.1f}s.", flush=True)
    write_heartbeat("final_csv_materialized", append=True)
    threshold_frame = pd.read_csv(paths["threshold"]) if paths["threshold"].exists() else pd.DataFrame(threshold_rows)
    memory_frame = pd.read_csv(paths["memory"]) if paths["memory"].exists() else pd.DataFrame(memory_rows)
    per_frame_frame = pd.read_csv(paths["per_frame"]) if paths["per_frame"].exists() else pd.DataFrame(per_frame_rows)
    audit_frame = pd.read_csv(paths["audit"]) if paths["audit"].exists() else pd.DataFrame(audit_rows)
    frames = {
        "api_probe": pd.DataFrame(api_rows),
        "results": pd.DataFrame(result_rows),
        "prompt_audit": audit_frame,
        "per_frame": per_frame_frame,
        "threshold_long": threshold_frame,
        "memory": memory_frame,
        "case_selection": case_selection,
    }
    summary_frames = summarize_strict_video_benchmark(frames, output_dir, tb_writer=tb_writer)
    if save_model_separated_outputs:
        write_model_separated_outputs(
            output_dir,
            benchmark_models,
            api_rows=api_rows,
            result_rows=result_rows,
            audit_rows=audit_rows,
            per_frame_rows=per_frame_rows,
            threshold_rows=threshold_rows,
            memory_rows=memory_rows,
            summary_frames=summary_frames,
        )
    visual_paths = make_strict_video_visuals(summary_frames, output_dir, visual_top_n_datasets=visual_top_n_datasets)
    write_strict_video_reports(
        frames,
        summary_frames,
        output_dir,
        run_stamp=run_stamp,
        smoke_run=smoke_run,
        case_count=len(case_indices),
        benchmark_models=benchmark_models,
        prompt_modes=prompt_modes,
        protocols=protocols,
        tb_dir=tb_dir,
        visual_paths=visual_paths,
        tb_writer=tb_writer,
    )
    if tb_writer is not None:
        tb_writer.close()
    write_heartbeat("run_complete", append=True)
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    if fault_log_handle is not None:
        try:
            fault_log_handle.write(f"=== strict video benchmark clean end {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===\n")
            fault_log_handle.close()
        except Exception:
            pass
    return {**frames, **summary_frames, "output_dir": output_dir, "paths": paths, "visual_paths": visual_paths}


def summarize_strict_video_benchmark(frames: dict[str, pd.DataFrame], output_dir, tb_writer=None) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    results = frames.get("results", pd.DataFrame())
    threshold_long = frames.get("threshold_long", pd.DataFrame())
    memory = frames.get("memory", pd.DataFrame())
    ok = results[results["status"].astype(str).eq("ok")].copy() if len(results) else pd.DataFrame()
    status_summary = results.groupby(["model", "prompt_mode", "protocol", "status"], dropna=False).size().reset_index(name="rows") if len(results) else pd.DataFrame()
    failures = results[results["status"].astype(str).ne("ok")].copy() if len(results) else pd.DataFrame()
    status_summary.to_csv(output_dir / "status_summary.csv", index=False)
    failures.to_csv(output_dir / "failures.csv", index=False)
    status_summary.to_csv(output_dir / "smoke_status_summary.csv", index=False)
    failures.to_csv(output_dir / "smoke_failures.csv", index=False)

    metric_cols = [
        "dice", "iou", "fg_volume_dice", "fg_volume_iou", "fg_mean_frame_dice", "full_sequence_dice", "full_sequence_iou",
        "hd95", "false_positive_rate", "false_negative_rate", "precision_ppv", "recall_sensitivity", "specificity",
        "volumetric_similarity", "empty_frame_false_positive_rate", "empty_frame_specificity", "wall_time_sec",
        "sec_per_frame", "frames_per_sec", "peak_cuda_mem_allocated_mb", "peak_cuda_mem_reserved_mb",
        "delta_peak_cuda_mem_allocated_mb", "delta_peak_cuda_mem_allocated_mb_per_frame",
    ]
    summary_case_micro = numeric_mean_summary(ok, ["model", "prompt_mode", "protocol"], metric_cols)
    summary_dataset_macro = dataset_macro_summary(ok, ["model", "prompt_mode", "protocol"], metric_cols)
    summary_by_dataset = numeric_mean_summary(ok, ["dataset_name", "model", "prompt_mode", "protocol"], metric_cols)
    summary_by_modality = dataset_macro_summary(ok, ["modality", "model", "prompt_mode", "protocol"], metric_cols)
    summary_by_task = dataset_macro_summary(ok, ["task_id", "task_label", "model", "prompt_mode", "protocol"], metric_cols)
    best_overall = best_threshold_table(threshold_long, ["model", "prompt_mode", "protocol"], score_prefix="fg")
    best_dataset = best_threshold_table(threshold_long, ["dataset_name", "model", "prompt_mode", "protocol"], score_prefix="fg")
    best_modality = best_threshold_table(threshold_long, ["modality", "model", "prompt_mode", "protocol"], score_prefix="fg")
    best_task = best_threshold_table(threshold_long, ["task_id", "task_label", "model", "prompt_mode", "protocol"], score_prefix="fg")

    if len(memory):
        mem_metrics = [c for c in memory.columns if "cuda" in c or c in ["wall_time_sec", "sec_per_frame", "frames_per_sec", "n_frames"]]
        memory_summary = memory.groupby(["model", "prompt_mode", "protocol"], dropna=False)[mem_metrics].agg(["mean", "max"]).reset_index()
        memory_summary.columns = ["_".join([str(x) for x in col if x != ""]).strip("_") for col in memory_summary.columns]
    else:
        memory_summary = pd.DataFrame()

    if len(ok):
        eff = ok.copy()
        bins = [0, 1, 8, 32, 96, 192, 10_000]
        labels = ["1", "2-8", "9-32", "33-96", "97-192", "193+"]
        eff["frame_count_bin"] = pd.cut(pd.to_numeric(eff["n_frames"], errors="coerce"), bins=bins, labels=labels, include_lowest=True)
        efficiency = eff.groupby(["model", "prompt_mode", "protocol", "frame_count_bin"], observed=False, dropna=False)[
            ["n_frames", "wall_time_sec", "sec_per_frame", "frames_per_sec", "delta_peak_cuda_mem_allocated_mb_per_frame"]
        ].mean(numeric_only=True).reset_index()
    else:
        efficiency = pd.DataFrame()

    outputs = {
        "status_summary": status_summary,
        "failures": failures,
        "summary_case_micro": summary_case_micro,
        "summary_dataset_macro": summary_dataset_macro,
        "summary_by_dataset": summary_by_dataset,
        "summary_by_modality": summary_by_modality,
        "summary_by_task": summary_by_task,
        "best_overall": best_overall,
        "best_dataset": best_dataset,
        "best_modality": best_modality,
        "best_task": best_task,
        "memory_summary": memory_summary,
        "efficiency": efficiency,
    }
    file_names = {
        "summary_case_micro": "summary_model_prompt_case_micro.csv",
        "summary_dataset_macro": "summary_model_prompt_dataset_macro.csv",
        "summary_by_dataset": "summary_by_dataset.csv",
        "summary_by_modality": "summary_by_modality.csv",
        "summary_by_task": "summary_by_task.csv",
        "best_overall": "best_thresholds_overall_dataset_macro.csv",
        "best_dataset": "best_thresholds_by_dataset.csv",
        "best_modality": "best_thresholds_by_modality.csv",
        "best_task": "best_thresholds_by_task.csv",
        "memory_summary": "memory_summary.csv",
        "efficiency": "efficiency_by_frame_count.csv",
    }
    for key, filename in file_names.items():
        outputs[key].to_csv(output_dir / filename, index=False)
    summary_dataset_macro.to_csv(output_dir / "strict_video_summary_by_model.csv", index=False)
    return outputs


def make_strict_video_visuals(summary_frames: dict[str, pd.DataFrame], output_dir, visual_top_n_datasets=None) -> list[Path]:
    if plt is None:
        return []
    output_dir = Path(output_dir)
    vis_dir = output_dir / "comparison_charts"
    by_dataset_dir = vis_dir / "by_dataset"
    by_dataset_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def safe_name(x):
        import re

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x).strip()).strip("_") or "unknown"

    def save_bar(df, x_col, y_col, hue_col, title, path, higher=True):
        if df is None or df.empty or y_col not in df.columns:
            return
        plot_df = df.dropna(subset=[y_col]).copy()
        if plot_df.empty:
            return
        plot_df["label"] = plot_df[x_col].astype(str) + " | " + plot_df[hue_col].astype(str)
        plot_df = plot_df.sort_values(y_col, ascending=not higher)
        fig_h = max(4.5, 0.33 * len(plot_df) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(plot_df["label"], plot_df[y_col], color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel(y_col.replace("_", " "))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    overall = summary_frames.get("summary_dataset_macro", pd.DataFrame())
    save_bar(overall, "model", "fg_volume_dice", "prompt_mode", "Overall Dataset-Macro Foreground Dice", vis_dir / "overall_dataset_macro_fg_dice.png")
    save_bar(overall, "model", "delta_peak_cuda_mem_allocated_mb", "prompt_mode", "Overall Mean Delta Peak CUDA MB", vis_dir / "overall_memory_delta_peak.png", higher=False)
    save_bar(overall, "model", "sec_per_frame", "prompt_mode", "Overall Mean Seconds Per Frame", vis_dir / "overall_sec_per_frame.png", higher=False)

    by_dataset = summary_frames.get("summary_by_dataset", pd.DataFrame())
    if len(by_dataset):
        dataset_names = list(by_dataset["dataset_name"].dropna().astype(str).unique())
        if visual_top_n_datasets is not None:
            dataset_names = dataset_names[: int(visual_top_n_datasets)]
        for dataset_name in tqdm(dataset_names, desc="Dataset charts", unit="dataset"):
            d = by_dataset[by_dataset["dataset_name"].astype(str).eq(dataset_name)].copy()
            save_bar(d, "model", "fg_volume_dice", "prompt_mode", f"{dataset_name}: Foreground Dice", by_dataset_dir / f"{safe_name(dataset_name)}_fg_dice.png")
    pd.DataFrame({"path": [str(p) for p in saved]}).to_csv(vis_dir / "saved_charts.csv", index=False)
    return saved


def write_strict_video_reports(
    frames,
    summary_frames,
    output_dir,
    run_stamp,
    smoke_run,
    case_count,
    benchmark_models,
    prompt_modes,
    protocols,
    tb_dir,
    visual_paths=None,
    tb_writer=None,
):
    output_dir = Path(output_dir)
    visual_paths = visual_paths or []
    shareable_lines = [
        "# Strict Video Benchmark Summary",
        "",
        f"Run: `{run_stamp}`",
        f"Smoke run: `{smoke_run}`",
        f"Cases: `{case_count}`",
        f"Models: `{benchmark_models}`",
        f"Prompt modes: `{prompt_modes}`",
        f"Protocol: `{protocols}`",
        "",
        "Headline scores use dataset-macro averaging so overrepresented datasets do not dominate overall model performance.",
        "Best thresholds are selected after aggregation at group level, not per case.",
        "",
        "## Status",
        safe_markdown_table(summary_frames.get("status_summary", pd.DataFrame()), 100),
        "",
        "## Headline Dataset-Macro Metrics",
        safe_markdown_table(summary_frames.get("summary_dataset_macro", pd.DataFrame()), 100),
        "",
        "## Best Thresholds Overall",
        safe_markdown_table(summary_frames.get("best_overall", pd.DataFrame()), 100),
        "",
        "## Memory Summary",
        safe_markdown_table(summary_frames.get("memory_summary", pd.DataFrame()), 100),
    ]
    detailed_lines = shareable_lines + [
        "",
        "## Dataset Breakdown",
        safe_markdown_table(summary_frames.get("summary_by_dataset", pd.DataFrame()), 150),
        "",
        "## Modality Breakdown",
        safe_markdown_table(summary_frames.get("summary_by_modality", pd.DataFrame()), 150),
        "",
        "## Task Breakdown",
        safe_markdown_table(summary_frames.get("summary_by_task", pd.DataFrame()), 150),
        "",
        "## Efficiency By Frame Count",
        safe_markdown_table(summary_frames.get("efficiency", pd.DataFrame()), 150),
        "",
        "## Output Files",
        "- Per-case results: `per_case_results.csv`",
        "- Prompt audit: `prompt_audit.csv`",
        "- Per-frame metrics: `per_frame_metrics.csv`",
        "- Threshold metrics: `threshold_metrics_long.csv`",
        "- Memory measurements: `memory_measurements.csv`",
        "- Per-model outputs: `by_model/<model>/master_records.csv` plus per-frame, threshold, audit, and memory tables",
        f"- TensorBoard: `{tb_dir}`",
        f"- Charts saved: `{len(visual_paths)}`",
    ]
    shareable_path = output_dir / "shareable_summary.md"
    detailed_path = output_dir / "detailed_summary.md"
    shareable_path.write_text("\n".join(shareable_lines), encoding="utf-8")
    detailed_path.write_text("\n".join(detailed_lines), encoding="utf-8")
    packet_path = output_dir / "summary_packet.zip"
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [
            shareable_path,
            detailed_path,
            output_dir / "summary_model_prompt_dataset_macro.csv",
            output_dir / "summary_by_dataset.csv",
            output_dir / "summary_by_modality.csv",
            output_dir / "summary_by_task.csv",
            output_dir / "best_thresholds_overall_dataset_macro.csv",
            output_dir / "best_thresholds_by_dataset.csv",
            output_dir / "memory_summary.csv",
            output_dir / "efficiency_by_frame_count.csv",
        ]:
            if p.exists():
                zf.write(p, p.name)


def normalize_for_display(img_chw):
    arr = img_chw.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.isfinite(arr).any() else (0, 1)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def nearest_roles(case, anchor: int):
    gt = case["gt_thw"].astype(bool)
    fg = np.where(gt.reshape(gt.shape[0], -1).sum(axis=1) > 0)[0].astype(int).tolist()
    roles = []
    before = [t for t in fg if t < anchor]
    after = [t for t in fg if t > anchor]
    if before:
        roles.append(("before", before[-1]))
    roles.append(("anchor", int(anchor)))
    if after:
        roles.append(("after", after[0]))
    return roles
