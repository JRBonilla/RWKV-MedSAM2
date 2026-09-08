#!/usr/bin/env python3
"""Clean-process GPU/CPU memory benchmark for SAM 2 prediction profiles.

Each invocation builds exactly one backbone/predictor pair so CUDA peak-memory
measurements are not contaminated by another resident model. Run this script
once per profile (resident, offload, or bounded) and compare the JSON outputs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

CUDA_HOME = "/usr/local/cuda"
NVIDIA_ROOT = "/usr/local/lib/python3.10/dist-packages/nvidia"
TORCH_ROOT = "/usr/local/lib/python3.10/dist-packages/torch"
os.environ.setdefault("CUDA_HOME", CUDA_HOME)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
os.environ.setdefault("VRWKV_T_MAX", "256")
os.environ["PATH"] = (
    f"{CUDA_HOME}/bin:{CUDA_HOME}/nvvm/bin:" + os.environ.get("PATH", "")
)
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [
        f"{TORCH_ROOT}/lib",
        f"{NVIDIA_ROOT}/cudnn/lib",
        f"{CUDA_HOME}/lib64",
        "/usr/local/lib",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
).rstrip(":")
if os.environ.get("RWKV_PREDICTOR_MEMORY_ENV_READY") != "1":
    os.environ["RWKV_PREDICTOR_MEMORY_ENV_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import torch

from sam2.build_sam import build_sam2_video_predictor
from rwkv_medsam2.functions.func_3d import validate_step_3d
from rwkv_medsam2.train_sam2 import load_data_loaders


HIERA_CONFIG = "sam2.1/sam2.1_hiera_t512.yaml"
HIERA_CHECKPOINT = (
    "/data/TrainingCheckpoints/retrain_20260805/base/"
    "20260805-195118/best.pth"
)
RWKV_CONFIG = "sam2.1/sam2.1_vcr_nodistill.yaml"
RWKV_CHECKPOINT = (
    "/data/TrainingCheckpoints/retrain_20260805/nodistill/"
    "20260805-195118/best.pth"
)
RWKV_OVERRIDES = []
RWKV_CANDIDATE_OVERRIDES = [
    "++model.image_encoder.trunk.early_downsample_stem=true",
    "++model.image_encoder.trunk.fused_stage_cfg.expand_ratio=1.0",
    "++model.image_encoder.trunk.fused_stage_cfg.depth=1",
    "++model.image_encoder.trunk.fused_stage_cfg.stride=1",
    "++model.image_encoder.trunk.mbconv_stage_cfg.expand_ratio=1.0",
    "++model.image_encoder.trunk.rwkv_depth=2",
    "++model.image_encoder.trunk.mlp_ratio=2.0",
    "++model.image_encoder.trunk.fuse_shift_mix=true",
    "++model.image_encoder.trunk.inplace_activations=true",
]
MATCHED_COMMON_OVERRIDES = [
    "++model.memory_attention.layer.self_attention.feat_sizes=[64,64]",
    "++model.memory_attention.layer.cross_attention.feat_sizes=[64,64]",
    "++model.use_mask_input_as_output_without_sam=false",
]
OPTIMIZED_TARGET = (
    "rwkv_medsam2.optimized_video_predictor."
    "OptimizedSAM2VideoPredictor"
)
OPTIMIZED_BASE_OVERRIDES = [
    "++model.image_feature_chunk_size=4",
    "++model.training_feature_cache_frames=8",
    "++model.prediction_pin_cpu_features=true",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", choices=("hiera", "rwkv", "rwkv_candidate"), required=True
    )
    parser.add_argument(
        "--profile",
        choices=(
            "resident",
            "output_cpu",
            "video_offload",
            "state_offload",
            "balanced",
            "offload",
            "bounded",
        ),
        required=True,
    )
    parser.add_argument("--loaders", default="/data/loaders32.pkl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--checkpoint-model-key",
        choices=("hiera", "rwkv_nodistill", "rwkv_distill"),
        default="",
        help="Select one student from a combined three-model smoke checkpoint.",
    )
    parser.add_argument("--prompt-mode", default="box")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Repeat the real validation volume to this length; 0 keeps its native length.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument(
        "--matched-common-settings",
        action="store_true",
        help="Rebuild shared SAM2 components exactly as in the matched run",
    )
    parser.add_argument(
        "--prediction-gpu-cache-chunks", type=int, default=1,
        help="Optimized predictor GPU feature-chunk limit",
    )
    parser.add_argument(
        "--prediction-cpu-cache-chunks", type=int, default=0,
        help="Optimized predictor CPU feature-chunk limit",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--logits-output", default="")
    return parser.parse_args()


def checkpoint_state(path: str, model_key: str = "") -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model_key:
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, dict) or not isinstance(
            models.get(model_key), dict
        ):
            raise KeyError(
                f"Checkpoint {path} has no models[{model_key!r}] state dict"
            )
        return models[model_key]
    for key in (
        "candidate_model", "model", "state_dict", "model_state_dict"
    ):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    return payload


def build_model(args: argparse.Namespace):
    if args.backbone == "hiera":
        config = HIERA_CONFIG
        checkpoint = args.checkpoint or HIERA_CHECKPOINT
        overrides = []
    elif args.backbone == "rwkv_candidate":
        config = RWKV_CONFIG
        if not args.checkpoint:
            raise ValueError(
                "--checkpoint is required for --backbone=rwkv_candidate"
            )
        checkpoint = args.checkpoint
        overrides = list(RWKV_CANDIDATE_OVERRIDES)
    else:
        config = RWKV_CONFIG
        checkpoint = args.checkpoint or RWKV_CHECKPOINT
        overrides = list(RWKV_OVERRIDES)

    if args.matched_common_settings:
        overrides.extend(MATCHED_COMMON_OVERRIDES)

    predictor_target = "sam2.sam2_video_predictor.SAM2VideoPredictor"
    if args.profile in {"balanced", "bounded"}:
        predictor_target = OPTIMIZED_TARGET
        overrides.extend(OPTIMIZED_BASE_OVERRIDES)
        overrides.extend([
            f"++model.prediction_gpu_cache_chunks={args.prediction_gpu_cache_chunks}",
            f"++model.prediction_cpu_cache_chunks={args.prediction_cpu_cache_chunks}",
        ])

    model = build_sam2_video_predictor(
        config,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=overrides,
        apply_postprocessing=False,
        predictor_target=predictor_target,
    ).eval()
    state = checkpoint_state(checkpoint, args.checkpoint_model_key)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {missing=} {unexpected=}")
    return model, config, checkpoint


def select_case(loaders_path: str, prompt_mode: str):
    _, validation, _ = load_data_loaders(loaders_path)
    for index, sequence in enumerate(validation.dataset.sequences):
        if int(sequence.get("dim", -1)) != 3:
            continue
        sample = validation.dataset[index]
        cache = sample.get("val_prompt_cache")
        if not cache or not cache.get(prompt_mode) or not cache.get("chosen_frames"):
            continue
        masks = sample["mask"]
        target = masks[:, 0] if masks.ndim == 4 else masks
        return (
            sample["image"].cpu(),
            target.cpu(),
            copy.deepcopy(cache),
            {
                "dataset_index": index,
                "dataset": str(sequence.get("dataset", "unknown")),
                "subdataset": str(sequence.get("subdataset", "")),
                "native_frames": int(sample["image"].shape[0]),
            },
        )
    raise RuntimeError(f"No 3D validation case with {prompt_mode} prompts")


def resize_case(video, target, cache, requested_frames: int):
    if requested_frames <= 0 or requested_frames == int(video.shape[0]):
        return video, target, cache
    cache = copy.deepcopy(cache)
    cache["chosen_frames"] = [
        int(frame) for frame in cache.get("chosen_frames", [])
        if int(frame) < requested_frames
    ]
    if not cache["chosen_frames"]:
        raise ValueError(f"--frames={requested_frames} contains no prompt frame")
    indices = torch.arange(requested_frames) % int(video.shape[0])
    return video[indices].contiguous(), target[indices].contiguous(), cache


def current_rss_mib() -> float:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


class RssMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.start_mib = current_rss_mib()
        self.peak_mib = self.start_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, current_rss_mib())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.peak_mib = max(self.peak_mib, current_rss_mib())
        self._stop.set()
        self._thread.join()
        return False


def profile_options(profile: str) -> dict:
    placements = {
        "resident": (False, False, None),
        "output_cpu": (False, False, "cpu"),
        "video_offload": (True, False, None),
        "state_offload": (False, True, None),
        "balanced": (True, False, "cpu"),
        "offload": (True, True, "cpu"),
        "bounded": (True, True, "cpu"),
    }
    video_cpu, state_cpu, output_device = placements[profile]
    return {
        "offload_video_to_cpu": video_cpu,
        "offload_state_to_cpu": state_cpu,
        "output_device": output_device,
    }


def predict(model, video, target, cache, prompt_mode, profile):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = validate_step_3d(
        model,
        video,
        target,
        prompt_cache=cache,
        prompt_mode=prompt_mode,
        normalize_coords=True,
        **profile_options(profile),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if output is None:
        raise RuntimeError("Validation helper returned no prediction")
    return elapsed, output["logits"]


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def dice_at_zero(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = logits.detach().cpu() > 0
    truth = target.detach().cpu() > 0
    intersection = int((prediction & truth).sum())
    denominator = int(prediction.sum()) + int(truth.sum())
    return 1.0 if denominator == 0 else (2.0 * intersection / denominator)


def summarize(values: list[float]) -> dict:
    return {
        "measurements": len(values),
        "median_s": statistics.median(values),
        "mean_s": statistics.mean(values),
        "minimum_s": min(values),
        "values_s": values,
    }


def main() -> None:
    args = arguments()
    if args.warmup < 0 or args.measurements < 1 or args.frames < 0:
        raise ValueError("warmup and frames must be nonnegative; measurements must be positive")
    if args.prediction_gpu_cache_chunks < 1:
        raise ValueError("prediction GPU cache chunks must be positive")
    if args.prediction_cpu_cache_chunks < -1:
        raise ValueError("prediction CPU cache chunks must be at least -1")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    video, target, cache, case = select_case(args.loaders, args.prompt_mode)
    video, target, cache = resize_case(video, target, cache, args.frames)
    case["frames"] = int(video.shape[0])
    model, config, checkpoint = build_model(args)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    for _ in range(args.warmup):
        _, warm_logits = predict(
            model, video, target, cache, args.prompt_mode, args.profile
        )
        del warm_logits
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    times = []
    final_logits = None
    with RssMonitor() as rss:
        for _ in range(args.measurements):
            elapsed, logits = predict(
                model, video, target, cache, args.prompt_mode, args.profile
            )
            times.append(elapsed)
            if final_logits is not None:
                del final_logits
            final_logits = logits

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_cpu_logits = final_logits.detach().cpu()
    payload = {
        "backbone": args.backbone,
        "profile": args.profile,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_model_key": args.checkpoint_model_key or None,
        "predictor_class": type(model).__name__,
        "parameters": parameters,
        "matched_common_settings": bool(args.matched_common_settings),
        "predictor_cache": {
            "image_feature_chunk_size": getattr(model, "image_feature_chunk_size", None),
            "prediction_gpu_cache_chunks": getattr(model, "prediction_gpu_cache_chunks", None),
            "prediction_cpu_cache_chunks": getattr(model, "prediction_cpu_cache_chunks", None),
            "prediction_pin_cpu_features": getattr(model, "prediction_pin_cpu_features", None),
        },
        "case": case,
        "prompt_mode": args.prompt_mode,
        "placement": profile_options(args.profile),
        "timing": summarize(times),
        "gpu_memory": {
            "baseline_allocated_mib": baseline_allocated / 2**20,
            "baseline_reserved_mib": baseline_reserved / 2**20,
            "peak_allocated_mib": peak_allocated / 2**20,
            "peak_reserved_mib": peak_reserved / 2**20,
            "incremental_peak_allocated_mib": (
                max(0, peak_allocated - baseline_allocated) / 2**20
            ),
        },
        "cpu_memory": {
            "start_rss_mib": rss.start_mib,
            "peak_rss_mib": rss.peak_mib,
            "incremental_peak_rss_mib": max(0.0, rss.peak_mib - rss.start_mib),
        },
        "output": {
            "device_during_return": str(final_logits.device),
            "dtype": str(final_cpu_logits.dtype),
            "sha256": tensor_digest(final_cpu_logits),
            "dice_at_probability_0_5": dice_at_zero(final_cpu_logits, target),
            "minimum": float(final_cpu_logits.min()),
            "maximum": float(final_cpu_logits.max()),
            "mean": float(final_cpu_logits.mean()),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered, flush=True)

    if args.logits_output:
        logits_path = Path(args.logits_output)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_cpu_logits, logits_path)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

