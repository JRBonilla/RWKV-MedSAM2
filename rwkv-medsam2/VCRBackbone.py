import torch
import torch.nn as nn

from ext.vrwkv.vrwkv import VRWKV_SpatialMix, VRWKV_ChannelMix
from ext.vrwkv.vrwkv import Block as VRWKV_Block
from ext.vrwkv.utils import resize_pos_embed, DropPath

T_MAX = 8192

from torch.utils.cpp_extension import load
wkv_cuda = load(
    name="wkv",
    sources=["../ext/vrwkv/cuda/wkv_op.cpp", "../ext/vrwkv/cuda/wkv_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-res-usage", "--maxrregcount=60", "--use_fast_math", '-03', '-Xptxas -O3', f'-DTmax={T_MAX}'],
)

class FusedMbConv(nn.Module):
    """
    Implementation of a fused depthwise convolution + pointwise convolution block.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride of the convolution. Default: 1
        expand_ratio (float, optional): Expansion factor for the hidden layer. Default: 4.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.ReLU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.BatchNorm2d
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 4.0,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # If expand_ratio == 1, skip the expansion
        self.expand = (expand_ratio != 1)

        # 1) Fused 3x3 conv + BN + act
        # This replaces the separate 1x1 expansion + 3x3 depthwise
        self.fused_conv = nn.Conv2d(
            in_channels if not self.expand else hidden_dim,
            hidden_dim if not self.expand else hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
        )
        self.fused_bn = norm_layer(hidden_dim)
        self.fused_act = act_layer(inplace=True)

        # 2) 1x1 conv + BN
        if self.expand:
            self.project_conv = nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.project_bn = norm_layer(out_channels)
        
        # 3) Residual connection
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.expand:
            out = self.fused_conv(x)
            out = self.fused_bn(out)
            out = self.fused_act(out)

            out = self.project_conv(out)
            out = self.project_bn(out)
        else:
            out = self.fused_conv(x)
            out = self.fused_bn(out)
            out = self.fused_act(out)

        if self.use_res_connect:
            out = out + identity

        return out

class FusedMbConvStage(nn.Module):
    """
    Stage that stacks multiple FusedMbConv blocks.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int=2,
            stride: int = 1,
            expand_ratio: float = 4.0,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            # Only apply stride to the first block of the stage
            s = stride if i == 0 else 1
            blocks.append(
                FusedMbConv(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=s,
                    expand_ratio=expand_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class VCRBackbone(nn.Module):
    """
    V ision
    C NN
    R WKV
    """
    def __init__(
            self
    ):
        super().__init__()