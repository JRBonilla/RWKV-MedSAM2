#!/usr/bin/env python3
"""Count dominant image-encoder MACs for Hiera-T512 and rwkv2_mlp2.

The count includes convolution, linear-projection, and attention matrix
multiplications. It excludes normalization, activation, positional interpolation,
tensor reshaping, elementwise mixing, and the scalar exp/div arithmetic inside the
custom WKV recurrence. This is the convention normally used by model MAC counters.

One multiply-accumulate (MAC) is also reported as two FLOPs. Hardware profiler
"FLOP" conventions can differ, so thesis tables should retain the GMAC column and
state this convention explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EncoderCount:
    trunk_macs: int
    neck_macs: int

    @property
    def image_encoder_macs(self) -> int:
        return self.trunk_macs + self.neck_macs

    def report(self) -> dict[str, float | int]:
        values = asdict(self)
        values.update(
            {
                "image_encoder_macs": self.image_encoder_macs,
                "trunk_gmac": self.trunk_macs / 1e9,
                "neck_gmac": self.neck_macs / 1e9,
                "image_encoder_gmac": self.image_encoder_macs / 1e9,
                "image_encoder_gflops_2_per_mac": 2 * self.image_encoder_macs / 1e9,
            }
        )
        return values


def conv2d_macs(
    out_h: int,
    out_w: int,
    in_channels: int,
    out_channels: int,
    kernel: int,
    groups: int = 1,
) -> int:
    return (
        out_h
        * out_w
        * out_channels
        * (in_channels // groups)
        * kernel
        * kernel
    )


def hiera_t512_count(image_size: int) -> EncoderCount:
    """Count the configured sam2.1_hiera_t512 image encoder."""
    if image_size % 32:
        raise ValueError("Hiera-T512 counting requires image size divisible by 32")

    # PatchEmbed: 7x7, stride 4, padding 3, RGB -> 96 channels.
    height = width = image_size // 4
    dim = 96
    trunk = conv2d_macs(height, width, 3, dim, kernel=7)

    stages = (1, 2, 7, 2)
    stage_ends = tuple(sum(stages[:i]) - 1 for i in range(1, len(stages) + 1))
    q_pool_blocks = {end + 1 for end in stage_ends[:-1]}
    window_spec = (8, 4, 14, 7)
    global_attention_blocks = {5, 7, 9}
    stage_index = 0

    for block_index in range(sum(stages)):
        dim_out = dim
        window = window_spec[stage_index]
        if block_index in global_attention_blocks:
            window = 0

        if block_index - 1 in stage_ends:
            dim_out = 2 * dim
            stage_index += 1

        q_pool = block_index in q_pool_blocks

        # Hiera applies QKV to padded window tokens, but applies the shortcut
        # projection before window padding and its MLP after unpartitioning.
        if window:
            padded_h = math.ceil(height / window) * window
            padded_w = math.ceil(width / window) * window
            window_count = (padded_h // window) * (padded_w // window)
            key_tokens_per_window = window * window
            query_tokens_per_window = (
                (window // 2) * (window // 2) if q_pool else key_tokens_per_window
            )
            qkv_tokens = window_count * key_tokens_per_window
            query_tokens = window_count * query_tokens_per_window
            attention_macs = (
                2
                * window_count
                * query_tokens_per_window
                * key_tokens_per_window
                * dim_out
            )
        else:
            qkv_tokens = height * width
            query_tokens = (
                (height // 2) * (width // 2) if q_pool else height * width
            )
            attention_macs = 2 * query_tokens * qkv_tokens * dim_out

        qkv_macs = qkv_tokens * dim * (3 * dim_out)
        output_projection_macs = query_tokens * dim_out * dim_out
        shortcut_projection_macs = (
            height * width * dim * dim_out if dim != dim_out else 0
        )

        if q_pool:
            height //= 2
            width //= 2

        # Two MLP linears with hidden width 4 * dim_out.
        mlp_macs = height * width * 8 * dim_out * dim_out
        trunk += (
            qkv_macs
            + attention_macs
            + output_projection_macs
            + shortcut_projection_macs
            + mlp_macs
        )
        dim = dim_out

    # FPN lateral 1x1 convolutions at strides 32, 16, 8, and 4.
    neck = sum(
        conv2d_macs(image_size // stride, image_size // stride, channels, 256, 1)
        for stride, channels in ((32, 768), (16, 384), (8, 192), (4, 96))
    )
    return EncoderCount(trunk_macs=trunk, neck_macs=neck)


def rwkv2_mlp2_count(image_size: int) -> EncoderCount:
    """Count the final early-downsampling, depth-2, MLP-ratio-2 VCR encoder."""
    if image_size % 32:
        raise ValueError("rwkv2_mlp2 counting requires image size divisible by 32")

    # Early stem: RGB -> 56 at stride 2, then 56 -> 112 at stride 2.
    trunk = conv2d_macs(image_size // 2, image_size // 2, 3, 56, 3)
    trunk += conv2d_macs(image_size // 4, image_size // 4, 56, 112, 3)

    # One expansion-ratio-1 Fused-MBConv: 112 -> 112, 3x3, stride 1.
    trunk += conv2d_macs(image_size // 4, image_size // 4, 112, 112, 3)

    # Three depth-2, expansion-ratio-1 MBConv stages. Each block consists of a
    # depthwise 3x3 convolution followed by a 1x1 projection.
    height = width = image_size // 4
    for in_channels, out_channels in ((112, 64), (64, 128), (128, 256)):
        height //= 2
        width //= 2
        trunk += conv2d_macs(
            height, width, in_channels, in_channels, kernel=3, groups=in_channels
        )
        trunk += conv2d_macs(height, width, in_channels, out_channels, kernel=1)
        trunk += conv2d_macs(
            height, width, out_channels, out_channels, kernel=3, groups=out_channels
        )
        trunk += conv2d_macs(height, width, out_channels, out_channels, kernel=1)

    # Two RWKV blocks on a 16x16 grid at 512 input. SpatialMix has four CxC
    # projections. ChannelMix with ratio R has (2R + 1) CxC-equivalent MACs.
    tokens = height * width
    channels = 256
    rwkv_depth = 2
    mlp_ratio = 2
    trunk += rwkv_depth * tokens * channels * channels * (4 + 2 * mlp_ratio + 1)

    # The VCR FPN has the same 256-channel outputs but narrower lateral inputs.
    neck = sum(
        conv2d_macs(image_size // stride, image_size // stride, channels, 256, 1)
        for stride, channels in ((32, 256), (16, 128), (8, 64), (4, 112))
    )
    return EncoderCount(trunk_macs=trunk, neck_macs=neck)


def percent_reduction(candidate: int, reference: int) -> float:
    return 100.0 * (1.0 - candidate / reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    hiera = hiera_t512_count(args.image_size)
    rwkv = rwkv2_mlp2_count(args.image_size)
    report = {
        "image_size": args.image_size,
        "counting_convention": (
            "conv, linear, and attention matmul MACs; excludes norm, activation, "
            "interpolation, reshaping, elementwise mixing, and scalar WKV recurrence"
        ),
        "hiera_t512": hiera.report(),
        "rwkv2_mlp2": rwkv.report(),
        "rwkv_reduction_percent": {
            "trunk_macs": percent_reduction(rwkv.trunk_macs, hiera.trunk_macs),
            "neck_macs": percent_reduction(rwkv.neck_macs, hiera.neck_macs),
            "image_encoder_macs": percent_reduction(
                rwkv.image_encoder_macs, hiera.image_encoder_macs
            ),
        },
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Input: {args.image_size}x{args.image_size}")
    print("Model             Trunk GMAC   Neck GMAC   Encoder GMAC   Encoder GFLOPs")
    for name, count in (("Hiera-T512", hiera), ("rwkv2_mlp2", rwkv)):
        print(
            f"{name:<17} {count.trunk_macs / 1e9:>10.3f}"
            f" {count.neck_macs / 1e9:>11.3f}"
            f" {count.image_encoder_macs / 1e9:>14.3f}"
            f" {2 * count.image_encoder_macs / 1e9:>16.3f}"
        )
    print(
        "RWKV reduction: "
        f"trunk={report['rwkv_reduction_percent']['trunk_macs']:.2f}%, "
        f"neck={report['rwkv_reduction_percent']['neck_macs']:.2f}%, "
        f"image_encoder={report['rwkv_reduction_percent']['image_encoder_macs']:.2f}%"
    )
    print("Convention: " + report["counting_convention"])


if __name__ == "__main__":
    main()
