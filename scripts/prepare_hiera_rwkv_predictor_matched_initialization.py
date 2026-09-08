#!/usr/bin/env python3
"""Prepare fair Hiera/RWKV initial states for matched predictor training."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("VRWKV_T_MAX", "256")
sys.path.insert(0, str(REPO_ROOT))

# Importing the headless runner establishes the same CUDA/WKV environment and
# re-execs this script once before torch is imported.
from scripts import train_headless as headless

import torch

from rwkv_medsam2.train_sam2 import build_student_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--efficient-vcr-architecture",
        choices=("efficient", "rwkv2", "rwkv2_mlp2"),
        default="rwkv2_mlp2",
    )
    return parser.parse_args()


def model_args(mode: str, architecture: str) -> Namespace:
    return Namespace(
        mode=mode,
        global_batch_size=3,
        num_workers=0,
        epochs=20,
        val_frequency=5,
        disable_images=True,
        optimized=False,
        safe_training_optimizations=True,
        efficient_vcr_backbone=(mode != "base"),
        efficient_vcr_architecture=str(architecture),
        safe_efficient_backbone=False,
        matched_run=True,
        match_common_sam2_settings=True,
        ignore_pretrained_image_encoder=True,
        wkv_t_max=256,
    )


def build(mode: str, device: str, architecture: str):
    config = headless.build_config(model_args(mode, architecture))
    config.training.device = str(device)
    model = build_student_predictor(config).to(device)
    if type(model).__name__ != "SAM2VideoPredictor":
        raise TypeError(
            f"Expected SAM2VideoPredictor for {mode}, got {type(model).__name__}"
        )
    return model, config


def cpu_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def architecture_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
    return digest.hexdigest()


def atomic_save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def common_settings(config) -> dict:
    return {
        "self_attention_feat_sizes": list(
            config.model.memory_attention.layer.self_attention.feat_sizes
        ),
        "cross_attention_feat_sizes": list(
            config.model.memory_attention.layer.cross_attention.feat_sizes
        ),
        "use_mask_input_as_output_without_sam": bool(
            config.model.use_mask_input_as_output_without_sam
        ),
        "ignore_prefixes": list(config.init.ignore_prefixes),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hiera_path = output_dir / "hiera_initial.pth"
    rwkv_path = output_dir / "rwkv_initial.pth"
    metadata_path = output_dir / "initialization_metadata.json"
    for path in (hiera_path, rwkv_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)

    headless.reset_runtime_seed(args.seed)
    print("[initialization] building random-encoder Hiera model", flush=True)
    hiera, hiera_config = build(
        "base", args.device, args.efficient_vcr_architecture
    )
    hiera_state = cpu_state(hiera)
    del hiera
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    headless.reset_runtime_seed(args.seed)
    print("[initialization] building random-encoder RWKV model", flush=True)
    rwkv, rwkv_config = build(
        "nodistill", args.device, args.efficient_vcr_architecture
    )
    rwkv_state = cpu_state(rwkv)
    del rwkv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_encoder_prefix = "image_encoder."
    shared_names = []
    for name, tensor in hiera_state.items():
        if name.startswith(image_encoder_prefix):
            continue
        if name not in rwkv_state:
            continue
        if tuple(rwkv_state[name].shape) != tuple(tensor.shape):
            continue
        rwkv_state[name] = tensor.clone()
        shared_names.append(name)

    if not shared_names:
        raise RuntimeError("No compatible shared SAM2 tensors were found")
    unequal = [
        name for name in shared_names
        if not torch.equal(hiera_state[name], rwkv_state[name])
    ]
    if unequal:
        raise RuntimeError(f"Shared initialization mismatch: {unequal[:5]}")
    if any(name.startswith(image_encoder_prefix) for name in shared_names):
        raise RuntimeError("An image-encoder tensor entered the shared copy set")

    hiera_settings = common_settings(hiera_config)
    rwkv_settings = common_settings(rwkv_config)
    if hiera_settings != rwkv_settings:
        raise RuntimeError(
            "Common SAM2 configuration mismatch: "
            f"Hiera={hiera_settings}, RWKV={rwkv_settings}"
        )

    metadata = {
        "seed": int(args.seed),
        "rwkv_architecture": args.efficient_vcr_architecture,
        "prepared_predictor_class": "SAM2VideoPredictor",
        "shared_tensor_count": len(shared_names),
        "shared_tensors_bit_identical": True,
        "shared_scope": (
            "all shape-compatible tensors outside the complete image_encoder"
        ),
        "pretrained_sam2_shared_components_loaded": True,
        "hiera_image_encoder_pretrained": False,
        "rwkv_image_encoder_pretrained": False,
        "pretrained_checkpoint_ignore_prefixes": ["image_encoder."],
        "hiera_tensor_count": len(hiera_state),
        "rwkv_tensor_count": len(rwkv_state),
        "hiera_image_encoder_tensor_count": sum(
            name.startswith(image_encoder_prefix) for name in hiera_state
        ),
        "rwkv_image_encoder_tensor_count": sum(
            name.startswith(image_encoder_prefix) for name in rwkv_state
        ),
        "hiera_architecture_fingerprint": architecture_fingerprint(hiera_state),
        "rwkv_architecture_fingerprint": architecture_fingerprint(rwkv_state),
        "common_sam2_settings": hiera_settings,
        "hiera_optimizations": headless.OmegaConf.to_container(
            hiera_config.optimizations, resolve=True
        ),
        "rwkv_optimizations": headless.OmegaConf.to_container(
            rwkv_config.optimizations, resolve=True
        ),
    }
    atomic_save(hiera_path, {"model": hiera_state, "metadata": metadata})
    atomic_save(rwkv_path, {"model": rwkv_state, "metadata": metadata})
    headless.atomic_json(metadata_path, metadata)
    print(f"[initialization] Hiera={hiera_path}", flush=True)
    print(f"[initialization] RWKV={rwkv_path}", flush=True)
    print(
        f"[initialization] shared tensors={len(shared_names)} exact=True; "
        "both complete image encoders remain random",
        flush=True,
    )


if __name__ == "__main__":
    main()
