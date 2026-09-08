#!/usr/bin/env python3
"""Paired resident-versus-balanced SAM 2 timing under shared GPU load.

The benchmark alternates resident and balanced predictions in AB/BA order on
the same GPU. Ratios are summarized within case first, then bootstrapped across
cases so permanent shared-GPU activity is treated as paired noise.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
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
if os.environ.get("RWKV_PAIRED_PREDICTOR_ENV_READY") != "1":
    os.environ["RWKV_PAIRED_PREDICTOR_ENV_READY"] = "1"
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
RWKV_DISTILL_CONFIG = "sam2.1/sam2.1_vcr.yaml"
RWKV_CHECKPOINT = (
    "/data/TrainingCheckpoints/retrain_20260805/nodistill/"
    "20260805-195118/best.pth"
)
RWKV_DISTILL_CHECKPOINT = (
    "/data/TrainingCheckpoints/retrain_20260805/distill/"
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
DEFAULT_CASES = "2424,2432,3488,63152,63189,64255"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=(
            "hiera",
            "rwkv",
            "rwkv_distill",
            "rwkv_candidate",
            "system",
            "distill_system",
            "candidate_system",
        ),
        required=True,
    )
    parser.add_argument("--loaders", default="/data/loaders32.pkl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resident-checkpoint", default="")
    parser.add_argument("--balanced-checkpoint", default="")
    parser.add_argument(
        "--resident-predictor", choices=("original", "optimized"), default="original"
    )
    parser.add_argument(
        "--balanced-predictor", choices=("original", "optimized"), default="optimized"
    )
    parser.add_argument(
        "--resident-placement", choices=("resident", "balanced"), default="resident"
    )
    parser.add_argument(
        "--balanced-placement", choices=("resident", "balanced"), default="balanced"
    )
    parser.add_argument("--case-indices", default=DEFAULT_CASES)
    parser.add_argument("--prompt-mode", default="box")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--matched-common-settings", action="store_true",
        help="Rebuild shared SAM2 components exactly as in the matched run",
    )
    parser.add_argument("--prediction-gpu-cache-chunks", type=int, default=1)
    parser.add_argument("--prediction-cpu-cache-chunks", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def checkpoint_state(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "candidate_model", "model", "state_dict", "model_state_dict"
    ):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    return payload


def model_spec(backbone: str, checkpoint_override: str):
    if backbone == "hiera":
        return (
            HIERA_CONFIG,
            checkpoint_override or HIERA_CHECKPOINT,
            [],
        )
    if backbone == "rwkv_distill":
        return (
            RWKV_DISTILL_CONFIG,
            checkpoint_override or RWKV_DISTILL_CHECKPOINT,
            list(RWKV_OVERRIDES),
        )
    if backbone == "rwkv_candidate":
        if not checkpoint_override:
            raise ValueError(
                "RWKV candidate comparisons require a checkpoint"
            )
        return (
            RWKV_CONFIG,
            checkpoint_override,
            list(RWKV_CANDIDATE_OVERRIDES),
        )
    if backbone == "rwkv":
        return (
            RWKV_CONFIG,
            checkpoint_override or RWKV_CHECKPOINT,
            list(RWKV_OVERRIDES),
        )
    raise ValueError(f"Unknown single-model backbone: {backbone}")


def build_pair(args: argparse.Namespace):
    if args.backbone in {"system", "distill_system", "candidate_system"}:
        if args.checkpoint:
            raise ValueError(
                "Use separate checkpoint flags for a system comparison"
            )
        resident_config, resident_checkpoint, resident_overrides = model_spec(
            "hiera", args.resident_checkpoint
        )
        candidate_backbone = (
            "rwkv_candidate"
            if args.backbone == "candidate_system"
            else (
                "rwkv_distill"
                if args.backbone == "distill_system"
                else "rwkv"
            )
        )
        balanced_config, balanced_checkpoint, balanced_overrides = model_spec(
            candidate_backbone, args.balanced_checkpoint
        )
    else:
        resident_config, resident_checkpoint, resident_overrides = model_spec(
            args.backbone, args.resident_checkpoint or args.checkpoint
        )
        balanced_config, balanced_checkpoint, balanced_overrides = model_spec(
            args.backbone, args.balanced_checkpoint or args.checkpoint
        )

    if args.matched_common_settings:
        resident_overrides.extend(MATCHED_COMMON_OVERRIDES)
        balanced_overrides.extend(MATCHED_COMMON_OVERRIDES)

    optimized_overrides = list(OPTIMIZED_BASE_OVERRIDES)
    optimized_overrides.extend([
        f"++model.prediction_gpu_cache_chunks={args.prediction_gpu_cache_chunks}",
        f"++model.prediction_cpu_cache_chunks={args.prediction_cpu_cache_chunks}",
    ])

    resident_state = checkpoint_state(resident_checkpoint)
    balanced_state = checkpoint_state(balanced_checkpoint)

    def construct(config, state, overrides, predictor_kind):
        build_overrides = list(overrides)
        predictor_target = "sam2.sam2_video_predictor.SAM2VideoPredictor"
        if predictor_kind == "optimized":
            build_overrides.extend(optimized_overrides)
            predictor_target = OPTIMIZED_TARGET
        model = build_sam2_video_predictor(
            config,
            ckpt_path=None,
            device="cuda",
            mode="eval",
            hydra_overrides_extra=build_overrides,
            apply_postprocessing=False,
            predictor_target=predictor_target,
        ).eval()
        model.load_state_dict(state, strict=True)
        return model

    resident = construct(
        resident_config, resident_state, resident_overrides,
        args.resident_predictor,
    )
    balanced = construct(
        balanced_config, balanced_state, balanced_overrides,
        args.balanced_predictor,
    )
    return (
        resident,
        balanced,
        {"resident": resident_config, "balanced": balanced_config},
        {
            "resident": resident_checkpoint,
            "balanced": balanced_checkpoint,
        },
    )


def load_case(validation, index: int, prompt_mode: str) -> dict:
    sequence = validation.dataset.sequences[index]
    if int(sequence.get("dim", -1)) != 3:
        raise ValueError(f"Validation index {index} is not a 3D sequence")
    sample = validation.dataset[index]
    cache = sample.get("val_prompt_cache")
    if not cache or not cache.get(prompt_mode) or not cache.get("chosen_frames"):
        raise ValueError(
            f"Validation index {index} has no fixed {prompt_mode} prompt"
        )
    masks = sample["mask"]
    target = masks[:, 0] if masks.ndim == 4 else masks
    return {
        "index": index,
        "dataset": str(sequence.get("dataset", "unknown")),
        "subdataset": str(sequence.get("subdataset", "")),
        "modality": str(sequence.get("modality", "unknown")),
        "frames": int(sample["image"].shape[0]),
        "video": sample["image"].cpu(),
        "target": target.cpu(),
        "cache": cache,
    }


PLACEMENT = {
    "resident": {
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "output_device": "cpu",
    },
    "balanced": {
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": False,
        "output_device": "cpu",
    },
}


def predict(model, case: dict, prompt_mode: str, placement: str):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = validate_step_3d(
        model,
        case["video"],
        case["target"],
        prompt_cache=case["cache"],
        prompt_mode=prompt_mode,
        normalize_coords=True,
        **PLACEMENT[placement],
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if output is None:
        raise RuntimeError(
            f"{placement} returned no output for validation index {case['index']}"
        )
    return elapsed, output["logits"].detach().cpu()


def dice_at_zero(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = logits > 0
    truth = target > 0
    intersection = int((prediction & truth).sum())
    denominator = int(prediction.sum()) + int(truth.sum())
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def compare_outputs(
    resident: torch.Tensor, balanced: torch.Tensor, target: torch.Tensor
) -> dict:
    difference = (resident - balanced).abs()
    return {
        "resident_dice": dice_at_zero(resident, target),
        "balanced_dice": dice_at_zero(balanced, target),
        "dice_delta": (
            dice_at_zero(balanced, target) - dice_at_zero(resident, target)
        ),
        "logit_max_abs": float(difference.max()),
        "logit_mean_abs": float(difference.mean()),
        "binary_disagreement_fraction": float(
            ((resident > 0) != (balanced > 0)).float().mean()
        ),
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_case_median(
    values: list[float], samples: int, seed: int
) -> dict:
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [rng.choice(values) for _ in values]
        draws.append(statistics.median(sample))
    return {
        "estimate": statistics.median(values),
        "lower_95": quantile(draws, 0.025),
        "upper_95": quantile(draws, 0.975),
    }


def summarize(args: argparse.Namespace, cases: list[dict]) -> dict:
    case_ratios = [
        statistics.median(pair["time_ratio"] for pair in case["pairs"])
        for case in cases
    ]
    case_dice_deltas = [
        statistics.median(pair["accuracy"]["dice_delta"] for pair in case["pairs"])
        for case in cases
    ]
    case_disagreements = [
        statistics.median(
            pair["accuracy"]["binary_disagreement_fraction"]
            for pair in case["pairs"]
        )
        for case in cases
    ]
    timing_ratio = bootstrap_case_median(
        case_ratios, args.bootstrap_samples, args.seed
    )
    return {
        "case_count": len(cases),
        "case_median_time_ratios": case_ratios,
        "candidate_faster_case_count": sum(ratio < 1.0 for ratio in case_ratios),
        "timing_ratio": timing_ratio,
        "time_change_percent": {
            key: 100.0 * (value - 1.0)
            for key, value in timing_ratio.items()
        },
        "dice_delta_median": statistics.median(case_dice_deltas),
        "dice_delta_minimum": min(case_dice_deltas),
        "dice_delta_maximum": max(case_dice_deltas),
        "binary_disagreement_median": statistics.median(case_disagreements),
        "binary_disagreement_maximum": max(case_disagreements),
    }


def main() -> None:
    args = arguments()
    if args.checkpoint and (args.resident_checkpoint or args.balanced_checkpoint):
        raise ValueError("Do not combine --checkpoint with separate checkpoint flags")
    if args.prediction_gpu_cache_chunks < 1:
        raise ValueError("prediction GPU cache chunks must be positive")
    if args.prediction_cpu_cache_chunks < -1:
        raise ValueError("prediction CPU cache chunks must be at least -1")
    if args.warmup < 0 or args.blocks < 1 or args.bootstrap_samples < 1:
        raise ValueError("warmup must be nonnegative; blocks and bootstrap samples positive")
    indices = [
        int(value.strip())
        for value in args.case_indices.split(",")
        if value.strip()
    ]
    if not indices:
        raise ValueError("--case-indices must contain at least one index")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    _, validation, _ = load_data_loaders(args.loaders)
    resident, balanced, config, checkpoint = build_pair(args)
    first_case = load_case(validation, indices[0], args.prompt_mode)
    placements = {
        "resident": args.resident_placement,
        "balanced": args.balanced_placement,
    }

    for warmup_index in range(args.warmup):
        order = (
            (("resident", resident), ("balanced", balanced))
            if warmup_index % 2 == 0
            else (("balanced", balanced), ("resident", resident))
        )
        for profile, model in order:
            _, logits = predict(
                model, first_case, args.prompt_mode, placements[profile]
            )
            del logits
    gc.collect()
    torch.cuda.synchronize()

    case_results = []
    for case_index in indices:
        case = load_case(validation, case_index, args.prompt_mode)
        pairs = []
        reference_anchor = None
        resident_repeat_disagreements = []

        for block in range(args.blocks):
            order_names = (
                ("resident", "balanced")
                if block % 2 == 0
                else ("balanced", "resident")
            )
            outputs = {}
            times = {}
            for profile in order_names:
                model = resident if profile == "resident" else balanced
                elapsed, logits = predict(
                    model, case, args.prompt_mode, placements[profile]
                )
                times[profile] = elapsed
                outputs[profile] = logits

            if reference_anchor is None:
                reference_anchor = outputs["resident"].clone()
            else:
                resident_repeat_disagreements.append(
                    float(
                        (
                            (reference_anchor > 0)
                            != (outputs["resident"] > 0)
                        )
                        .float()
                        .mean()
                    )
                )

            pairs.append(
                {
                    "block": block,
                    "order": list(order_names),
                    "resident_s": times["resident"],
                    "balanced_s": times["balanced"],
                    "time_ratio": times["balanced"] / times["resident"],
                    "accuracy": compare_outputs(
                        outputs["resident"],
                        outputs["balanced"],
                        case["target"],
                    ),
                }
            )
            del outputs
            gc.collect()

        case_results.append(
            {
                "index": case["index"],
                "dataset": case["dataset"],
                "subdataset": case["subdataset"],
                "modality": case["modality"],
                "frames": case["frames"],
                "pairs": pairs,
                "case_median_time_ratio": statistics.median(
                    pair["time_ratio"] for pair in pairs
                ),
                "resident_repeat_binary_disagreement_median": (
                    statistics.median(resident_repeat_disagreements)
                    if resident_repeat_disagreements
                    else 0.0
                ),
            }
        )
        del case, reference_anchor
        gc.collect()

    payload = {
        "backbone": args.backbone,
        "config": config,
        "checkpoint": checkpoint,
        "resident_predictor": type(resident).__name__,
        "balanced_predictor": type(balanced).__name__,
        "protocol": {
            "case_indices": indices,
            "blocks": args.blocks,
            "warmup": args.warmup,
            "order": "AB/BA alternating within each case",
            "resident_placement": PLACEMENT[args.resident_placement],
            "balanced_placement": PLACEMENT[args.balanced_placement],
            "resident_predictor_requested": args.resident_predictor,
            "balanced_predictor_requested": args.balanced_predictor,
            "matched_common_settings": bool(args.matched_common_settings),
            "optimized_cache": {
                "prediction_gpu_cache_chunks": args.prediction_gpu_cache_chunks,
                "prediction_cpu_cache_chunks": args.prediction_cpu_cache_chunks,
            },
            "timing_note": (
                "Paired ratios are intended for permanently shared GPUs; "
                "absolute times include external contention."
            ),
        },
        "cases": case_results,
        "summary": summarize(args, case_results),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

