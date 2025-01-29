import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.utils.checkpoint as cp

import math

from typing import Tuple, Optional, List

# Import custom modules for refinement and uncertainty prediction
from .refinement import RefinementModule, UncertaintyHead
from ext.vrwkv.vrwkv import VRWKV_SpatialMix, VRWKV_ChannelMix
from ext.vrwkv.utils import resize_pos_embed, DropPath

class VCRStem(nn.Module):
    """
    Initial stem block that processes raw input images.
    
    Reduces spatial dimensions by 2x while increasing channels through:
    - First 3x3 strided convolution (stride=2)
    - Batch normalization + activation
    - Second 3x3 convolution (stride=1)

    Args:
        in_channels (int): Number of input channels. Default: 3
        out_channels (int): Number of output channels. Default: 32
        act_layer (nn.Module): Activation layer. Default: nn.ReLU
        norm_layer (nn.Module): Normalization layer. Default: nn.BatchNorm2d
        norm_eps (float): Epsilon for normalization. Default: 1e-6
        bias (bool): Whether to use bias in convolutions. Default: False
    """
    def __init__(
            self,
            in_channels=3,
            out_channels=32,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            norm_eps=1e-6,
            bias=False
        ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.norm = norm_layer(out_channels, eps=norm_eps)
        self.act = act_layer(inplace=True)
    
    def forward(self, x):
        # Process input through stem block with 2x spatial reduction
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class FusedMbConv(nn.Module):
    """
    Implementation of a fused Mobile Inverted Bottleneck (FusedMbConv):
    Combines expand and depthwise convolutions into a single 3x3 convolution
    for improved efficiency. Includes optional residual connection when stride=1
    and channels match.

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
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False
        )
        self.fused_bn = norm_layer(hidden_dim)
        self.fused_act = act_layer(inplace=True)

        # Projection layer (1x1 convolution + BN) if needed
        if self.expand or (out_channels != in_channels):
            self.project_conv = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.project_bn = norm_layer(out_channels)
        else:
            self.project_conv = None

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
    Stage stacking multiple FusedMbConv blocks for hierarchical feature extraction.
    First block handles stride and channel changes, subsequent blocks maintain dimensions.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - depth (int, optional): Number of blocks. Default is 2.
        - stride (int, optional): Stride for first block. Default is 1.
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
    Classic Mobile Inverted Bottleneck (MBConv) with separate expansion,
    depthwise, and projection layers. Includes skip connection when possible.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - stride (int, optional): Stride for depthwise conv. Default is 1.
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
    Stage stacking multiple MbConv blocks for hierarchical feature extraction.
    First block handles stride and channel changes, subsequent blocks maintain dimensions.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - depth (int, optional): Number of blocks. Default is 2.
        - stride (int, optional): Stride for first block. Default is 1.
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
    Implementation of an RWKV (Receptance Weighted Key Value) block.
    This block follows a two-step sequence:
        1. LayerNorm -> SpatialMix (attention-like) -> residual
        2. LayerNorm -> ChannelMix (feedforward-like) -> residual

    Optionally applies:
        - DropPath (for stochastic depth regularization)
        - Learnable per-branch scaling (layer scale)
        - Gradient checkpointing (to save memory at the cost of extra compute)

    Args:
        - embed_dim (int): Number of channels (C) in input features (B, N, C).
        - total_layer (int): Total number of RWKV layers in the model (used by VRWKV modules).
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
            hidden_rate=int(mlp_ratio),
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
            # --- Sub-block A: Spatial Mix with residual ---
            t_norm = self.norm1(t)
            out_spatial = self.spatial_mix(t_norm, patch_resolution)
            if self.use_layer_scale:
                out_spatial *= self.gamma1
            t += self.drop_path(out_spatial)

            # --- Sub-block B: Channel Mix with residual ---
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
    Vision CNN-RWKV hybrid backbone that combines:
    1. Efficient mobile convolutions (FusedMbConv and MbConv) for early processing
    2. RWKV blocks for global context modeling
    3. Multi-scale feature outputs compatible with FPN

    The architecture progressively reduces spatial resolution while increasing channels,
    then applies RWKV blocks for long-range dependencies. Outputs features at multiple
    scales for downstream tasks like segmentation.

    Args:
        - in_channels (int): Number of input channels (usually 3 for RGB).
        - embed_dim (int): Channel dimension after CNN stages for the flattened tokens.
        - rwkv_depth (int): Number of RWKV blocks to apply.
        - mlp_ratio (float): Expansion ratio in each RWKV block's channel mix.
        - drop_path (float): Drop path rate for RWKV blocks.
        - with_pos_embed (bool): Whether to add a learnable positional embedding after flattening.
        - init_patch_size (Tuple[int,int]): The stride or scale factor from the entire CNN.
            Used to initialize positional embeddings.
        - fused_stage_cfg (dict, optional): Configuration for the FusedMbConv stage.
        - mbconv_stage_cfg (dict, optional): Configuration for the MbConv stages.
        - use_uncertainty (bool, optional): Whether to use uncertainty heads. Default: False.
        - use_refinement (bool, optional): Whether to use refinement heads. Default: False.
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        rwkv_depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        with_pos_embed: bool = True,
        init_patch_size: Tuple[int,int] = (16,16),
        fused_stage_cfg: dict = None,
        mbconv_stage_cfg: dict = None,
        use_uncertainty: bool = False,
        use_refinement: bool = False,
    ):
        super().__init__()

        stem_out_channels = (fused_stage_cfg or {}).get('out_ch', 112)
        self.stem = VCRStem(in_channels=in_channels, out_channels=stem_out_channels)

        # Use config parameters or defaults
        fused_cfg = fused_stage_cfg or {
            'out_ch': 112,
            'depth': 2,
            'stride': 2,
            'expand_ratio': 4.0
        }
        
        mbconv_cfg = mbconv_stage_cfg or {
            'out_ch': 224,
            'depth': 2,
            'stride': 2,
            'expand_ratio': 4.0
        }

        # Calculate stage channels based on configs
        stage_channels = [
            fused_cfg['out_ch'],              # Stage 0
            mbconv_cfg['out_ch'],             # Stage 1
            mbconv_cfg['out_ch'] * 2,         # Stage 2
            mbconv_cfg['out_ch'] * 4          # Stage 3
        ]

        # Stage 0: Initial FusedMbConv stage
        self.stage0 = FusedMbConvStage(
            in_channels=stem_out_channels,
            out_channels=stage_channels[0],
            depth=fused_cfg['depth'],
            stride=fused_cfg['stride'],
            expand_ratio=fused_cfg['expand_ratio'],
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
        )

        # Stages 1-3: MbConv stages with increasing channels
        self.stage1 = MbConvStage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            depth=mbconv_cfg['depth'],
            stride=mbconv_cfg['stride'],
            expand_ratio=mbconv_cfg['expand_ratio'],
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
        )

        self.stage2 = MbConvStage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            depth=mbconv_cfg['depth'],
            stride=mbconv_cfg['stride'],
            expand_ratio=mbconv_cfg['expand_ratio'],
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
        )

        self.stage3 = MbConvStage(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            depth=mbconv_cfg['depth'],
            stride=mbconv_cfg['stride'],
            expand_ratio=mbconv_cfg['expand_ratio'],
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
        )

        # Positional embedding for final scale
        self.with_pos_embed = with_pos_embed
        if self.with_pos_embed:
            max_hw = init_patch_size[0] * init_patch_size[1]
            self.pos_embed = nn.Parameter(torch.zeros(1, max_hw, stage_channels[-1]))
            # Initialize with learned positional embeddings
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # RWKV blocks on final scale
        self.rwkv_blocks = nn.ModuleList([
            RWKVBlock(
                embed_dim=stage_channels[-1],
                total_layer=rwkv_depth,
                layer_id=i,
                mlp_ratio=int(mlp_ratio),
                drop_path=drop_path,
                shift_mode="q_shift",
                channel_gamma=0.25,
                shift_pixel=1,
                init_mode="fancy",
                key_norm=False,
                use_layer_scale=True,
                layer_scale_init_value=1e-2,
                with_cp=False,
            )
            for i in range(rwkv_depth)
        ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(stage_channels[-1], eps=1e-6)

        # Channel list for FPN (high to low resolution)
        self.channel_list = stage_channels[::-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature maps [feat0, feat1, feat2, feat3_rnn] with channels
            [112, 224, 448, 896] and progressively halved spatial resolution:
            - feat0: 1/2 resolution features from FusedMbConv
            - feat1: 1/4 resolution features from first MbConv
            - feat2: 1/8 resolution features from second MbConv
            - feat3_rnn: 1/16 resolution features processed by RWKV blocks
        """
        # Initial stem block reduces spatial dims by 2x while increasing channels
        x = self.stem(x)
        
        # Process through CNN stages with progressively increasing channels and decreasing resolution
        feat0 = self.stage0(x)      # (B, 112, H/2, W/2)
        feat1 = self.stage1(feat0)  # (B, 224, H/4, W/4) 
        feat2 = self.stage2(feat1)  # (B, 448, H/8, W/8)
        feat3 = self.stage3(feat2)  # (B, 896, H/16, W/16)

        # Process final scale with RWKV
        B, C, Hf, Wf = feat3.shape
        x3 = feat3.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)

        # Add positional embedding
        if self.with_pos_embed and self.pos_embed is not None:
            x3 = x3 + resize_pos_embed(
                self.pos_embed,
                src_shape=(int(math.sqrt(self.pos_embed.shape[1])),) * 2,
                dst_shape=(Hf, Wf),
                mode="bicubic",
                num_extra_tokens=0,
            )

        # Apply RWKV blocks
        for block in self.rwkv_blocks:
            x3 = block(x3, (Hf, Wf))

        # Final norm and reshape
        x3 = self.final_norm(x3)
        feat3_rnn = x3.transpose(1, 2).reshape(B, C, Hf, Wf)

        # Return multi-scale features for FPN
        return [feat0, feat1, feat2, feat3_rnn]