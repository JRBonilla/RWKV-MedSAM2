import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.utils.checkpoint as cp

import math
from typing import Tuple, Optional

# Import custom modules
from refinement import RefinementModule, UncertaintyHead
from ext.vrwkv.vrwkv import VRWKV_SpatialMix, VRWKV_ChannelMix
from ext.vrwkv.utils import resize_pos_embed, DropPath

# CUDA settings for WKV
T_MAX = 8192
wkv_cuda = load(
    name="wkv",
    sources=[
        "../ext/vrwkv/cuda/wkv_op.cpp",
        "../ext/vrwkv/cuda/wkv_cuda.cu"
    ],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-res-usage",
        "--maxrregcount=60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={T_MAX}"
    ],
)

class FusedMbConv(nn.Module):
    """
    Implementation of a fused Mobile Inverted Bottleneck (FusedMbConv):
    Combines expand and depthwise convolutions into a single 3x3 convolution.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - stride (int, optional): Convolution stride. Default is 1.
        - expand_ratio (float, optional): Expansion factor for hidden layer. Default is 4.0.
        - act_layer (nn.Module, optional): Activation layer. Default is nn.ReLU.
        - norm_layer (nn.Module, optional): Normalization layer. Default is nn.BatchNorm2d.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=4.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.expand = (expand_ratio != 1)

        # Fused 3x3 convolution + BN + activation
        self.fused_conv = nn.Conv2d(
            in_channels=in_channels if not self.expand else hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False
        )
        self.fused_bn = norm_layer(hidden_dim)
        self.fused_act = act_layer(inplace=True)

        # Projection layer (1x1 convolution + BN)
        if self.expand:
            self.project_conv = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.project_bn = norm_layer(out_channels)

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.fused_conv(x)
            x = self.fused_bn(x)
            x = self.fused_act(x)

            x = self.project_conv(x)
            x = self.project_bn(x)
        else:
            x = self.fused_conv(x)
            x = self.fused_bn(x)
            x = self.fused_act(x)

        if self.use_res_connect:
            x += identity

        return x

class FusedMbConvStage(nn.Module):
    """
    Stage stacking multiple FusedMbConv blocks.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - depth (int, optional): Number of blocks. Default is 2.
        - stride (int, optional): Convolution stride. Default is 1.
        - expand_ratio (float, optional): Expansion factor for hidden layer. Default is 4.0.
        - act_layer (nn.Module, optional): Activation layer. Default is nn.ReLU.
        - norm_layer (nn.Module, optional): Normalization layer. Default is nn.BatchNorm2d.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        stride=1,
        expand_ratio=4.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                FusedMbConv(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)

class MbConv(nn.Module):
    """
    Classic Mobile Inverted Bottleneck (MBConv).

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - stride (int, optional): Convolution stride. Default is 1.
        - expand_ratio (float, optional): Expansion factor for hidden layer. Default is 4.0.
        - act_layer (nn.Module, optional): Activation layer. Default is nn.ReLU.
        - norm_layer (nn.Module, optional): Normalization layer. Default is nn.BatchNorm2d.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=4.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.expand = (expand_ratio != 1)

        if self.expand:
            self.expand_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.expand_bn = norm_layer(hidden_dim)
            self.expand_act = act_layer(inplace=True)

        self.dw_conv = nn.Conv2d(
            in_channels=hidden_dim if self.expand else in_channels,
            out_channels=hidden_dim if self.expand else in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim if self.expand else in_channels,
            bias=False
        )
        self.dw_bn = norm_layer(hidden_dim)
        self.dw_act = act_layer(inplace=True)

        self.project_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.project_bn = norm_layer(out_channels)

    def forward(self, x):
        identity = x

        if self.expand:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_res_connect:
            x += identity

        return x

class MbConvStage(nn.Module):
    """
    Stage stacking multiple MbConv blocks.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        stride=1,
        expand_ratio=4.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                MbConv(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)

class RWKVBlock(nn.Module):
    """
    Implementation of an RWKV block.
    This block follows a two-step sequence.
        1. LayerNorm + SpatialMix + residual
        2. LayerNorm + ChannelMix + residual

    Optionally applies:
        - DropPath (for Stochastic Depth)
        - Learnable per-branch scaling (layer scale),
        - Checkpointing (to save memory at the cost of extra compute)

    Args:
        - embed_dim (int): Number of channels (C) in input features (B, N, C).
        - total_layers (int): Total number of RWKV layers in the model (used by VRWKV modules).
        - layer_id (int): Index of this layer (starting at 0).
        - mlp_ratio (float, optional): Expansion ratio for ChannelMix. Default: 4.0.
        - drop_path (float, optional): Dropout path rate for stochastic depth. Default: 0.0.
        - shift_mode (str, optional): Shift mode for SpatialMix. Default: "q_shift".
        - channel_gamma (float, optional): Fraction of channels to shift in SpatialMix. Default: 0.25.
        - shift_pixel (int, optional): Number of pixels to shift in SpatialMix. Default: 1.
        - init_mode (str, optional): Initialization mode for VRWKV modules. Default: "fancy".
        - key_norm (bool, optional): Whether to apply LayerNorm inside VRWKV modules. Default: False.
        - use_layer_scale (bool, optional): If True, adds a learnable scaling parameter to each branch. Default: True.
        - layer_scale_init_value (float, optional): Init value for Layer Scale. Default: 1e-2.
        - with_cp (bool, optional): If True, uses torch.utils.checkpoint for checkpointing. Default: False.
    """
    def __init__(
        self,
        embed_dim: int,
        total_layer: int,
        layer_id: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        shift_mode: str = "q_shift",
        channel_gamma: float = 0.25,
        shift_pixel: int = 1,
        init_mode: str = "fancy",
        key_norm: bool = False,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-2,
        with_cp: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.with_cp = with_cp
        self.use_layer_scale = use_layer_scale

        # --- Sub-block A: Spatial (Attention-like) ---
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.spatial_mix = VRWKV_SpatialMix(
            n_embd=embed_dim,
            n_layer=total_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            init_mode=init_mode,
            key_norm=key_norm
        )

        # --- Sub-block B: Channel (Feedforward-like) ---
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.channel_mix = VRWKV_ChannelMix(
            n_embd=embed_dim,
            n_layer=total_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            hidden_rate=mlp_ratio,
            init_mode=init_mode,
            key_norm=key_norm
        )

        # --- DropPath (Stochastic Depth) ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- Layer Scale Params (optional) ---
        if self.use_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, patch_resolution: tuple) -> torch.Tensor:
        def _inner_forward(t):
            # --- Sub-block A: Spatial Mix ---
            t_norm = self.norm1(t)
            out_spatial = self.spatial_mix(t_norm, patch_resolution)
            if self.use_layer_scale:
                out_spatial *= self.gamma1
            t += self.drop_path(out_spatial)

            # --- Sub-block B: Channel Mix ---
            t_norm = self.norm2(t)
            out_channel = self.channel_mix(t_norm, patch_resolution)
            if self.use_layer_scale:
                out_channel *= self.gamma2
            t += self.drop_path(out_channel)

            return t
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

class VCRBackbone(nn.Module):
    """
    V ision
    C NN
    R WKV

    A backbone that:
      1. Uses FusedMbConv for an initial stage,
      2. Uses MbConv for a secondary stage,
      3. Flattens the resulting feature map,
      4. Processes tokens with RWKV blocks,
      5. (Optionally) adds uncertainty & refinement modules.

    Args:
        - in_channels (int): Number of input channels (usually 3 for RGB).
        - fused_stage_cfg (dict): Config dict for the FusedMbConv stage (e.g. out_ch, depth, stride).
        - mbconv_stage_cfg (dict): Config dict for the MbConv stage (similarly).
        - embed_dim (int): Channel dimension after CNN stages for the flattened tokens.
        - rwkv_depth (int): Number of RWKV blocks to apply.
        - mlp_ratio (float): Expansion ratio in each RWKV block’s channel mix.
        - drop_path (float): Drop path rate for RWKV blocks.
        - with_pos_embed (bool): Whether to add a learnable positional embedding after flattening.
        - init_patch_size (Tuple[int,int]): The stride or scale factor from the entire CNN. 
            We use this to guess how large the final feature map might be.
        - use_uncertainty (bool): Whether to attach an uncertainty head.
        - use_refinement (bool): Whether to attach a refinement module.
    """
    def __init__(
        self,
        in_channels: int = 3,
        fused_stage_cfg: Optional[dict] = None,
        mbconv_stage_cfg: Optional[dict] = None,
        embed_dim: int = 768,
        rwkv_depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        with_pos_embed: bool = True,
        init_patch_size: Tuple[int,int] = (16,16),
        use_uncertainty: bool = False,
        use_refinement: bool = False
    ):
        super().__init__()

        # --- Stage 1: FusedMbConv Stage Config ---
        fused_cfg = fused_stage_cfg if fused_stage_cfg is not None else {}
        fused_out_ch = fused_cfg.get("out_ch", 32)
        fused_depth = fused_cfg.get("depth", 2)
        fused_stride = fused_cfg.get("stride", 1)
        self.fused_stage = FusedMbConvStage(
            in_channels=in_channels,
            out_channels=fused_out_ch,
            depth=fused_depth,
            stride=fused_stride,
            expand_ratio=fused_cfg.get("expand_ratio", 4.0),
            act_layer=fused_cfg.get("act_layer", nn.ReLU),
            norm_layer=fused_cfg.get("norm_layer", nn.BatchNorm2d)
        )
        
        # --- Stage 2: MBConv Stage Config ---
        mbconv_cfg = mbconv_stage_cfg if mbconv_stage_cfg is not None else {}
        mbconv_out_ch = mbconv_cfg.get("out_ch", 32)
        mbconv_depth = mbconv_cfg.get("depth", 2)
        mbconv_stride = mbconv_cfg.get("stride", 1)
        self.mbconv_stage = MbConvStage(
            in_channels=fused_out_ch,
            out_channels=mbconv_out_ch,
            depth=mbconv_depth,
            stride=mbconv_stride,
            expand_ratio=mbconv_cfg.get("expand_ratio", 4.0),
            act_layer=mbconv_cfg.get("act_layer", nn.ReLU),
            norm_layer=mbconv_cfg.get("norm_layer", nn.BatchNorm2d)
        )

        # --- Stage 3: Project final CNN output to embed_dim if needed ---
        self.final_cnn_proj = nn.Conv2d(
            in_channels=mbconv_out_ch,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False
        )

        # --- Optional Position Embedding ---
        self.with_pos_embed = with_pos_embed
        if self.with_pos_embed:
            max_hw = init_patch_size[0] * init_patch_size[1]
            self.pos_embed = nn.Parameter(torch.zeros(1, max_hw, embed_dim))
        else:
            self.pos_embed = None

        # --- Stage 4: RWKV Blocks ---
        self.rwkv_blocks = nn.ModuleList([
            RWKVBlock(
                embed_dim=embed_dim,
                total_layer=rwkv_depth,
                layer_id=i,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                shift_mode="q_shift",
                channel_gamma=0.25,
                shift_pixel=1,
                init_mode="fancy",
                key_norm=False,
                use_layer_scale=True,
                layer_scale_init_value=1e-2,
                with_cp=False
            )
            for i in range(rwkv_depth)
        ])

        self.final_norm = nn.LayerNrom(embed_dim, eps=1e-6)


    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the VCR model.
            1) FusedMbConv Stage
            2) MbConv Stage
            3) Final conv projection
            4) Flatten and positional embedding
            5) RWKV Blocks
            6) Final norm
        Returns features
        """
        B, C, H, W = x.shape

        # --- Stage 1: FusedMbConv Stage ---
        x = self.fused_stage(x) # out: (B, fused_out_ch, H, W)

        # --- Stage 2: MBConv Stage ---
        x = self.mbconv_stage(x) # out: (B, mbconv_out_ch, H, W)

        # --- Stage 3: Final Conv Projection ---
        x = self.final_cnn_proj(x) # out: (B, embed_dim, H, W)

        # --- Stage 4: Flatten and Positional Embedding ---
        B, E, Hf, Wf = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous() # (B, N, E), N = Hf * Wf

        if self.with_pos_embed and self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                src_shape=(int(math.sqrt(self.pos_embed.shape[1])),) * 2,
                dst_shape=(Hf, Wf),
                mode="bicubic",
                num_extra_tokens=0
            )

        # --- Stage 5: RWKV Blocks ---
        for rwkv_block in self.rwkv_blocks:
            x = rwkv_block(x, patch_resolution=(Hf, Wf)) # shape remains (B, N, E)

        # --- Stage 6: Final Norm in Flattened Form ---
        x = self.final_norm(x)

        # --- Unflatten to (B, E, Hf, Wf) ---
        x_2d = x.transpose(1, 2).reshape(B, E, Hf, Wf).contiguous()

        return x_2d
