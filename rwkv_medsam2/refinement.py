"""Uncertainty and feature refinement heads for the VCR backbone."""

import torch
import torch.nn as nn

class UncertaintyHead(nn.Module):
    """
    Predict a per-pixel uncertainty map from feature tensors.

    Args:
        nn.Module (type): PyTorch module base class.

    Returns:
        None.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the uncertainty head layers.

        Args:
            in_channels (int): Numbers of channels in the input feature map.
            out_channels (int): Number of output channels.

        Returns:
            None.
        """
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Generate uncertainty probabilities for input features.

        Args:
            x (torch.Tensor): Feature map of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: Uncertainty map of shape ``[B, out_channels, H, W]``.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        uncertainty = self.sigmoid(x)
        return uncertainty

class RefinementModule(nn.Module):
    """
    Refine features by amplifying uncertain regions before convolution.

    Args:
        nn.Module (type): PyTorch module base class.

    Returns:
        None.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        """
        Initialize refinement convolutions.

        Args:
            in_channels (int): Number of channels of the input feature map.
            mid_channels (int): Number of channels in the intermediate representation.
            out_channels (int): Number of channels for the refined output.

        Returns:
            None.
        """
        super().__init__()
        # Two convolutional layers for refinement
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, uncertainty_map=None):
        """
        Refine feature maps with an optional uncertainty gate.

        Args:
            x (Tensor): Input feature map (e.g., segmentation features).
            uncertainty_map (Tensor, optional): Uncertainty map indicating regions to refine.
                                                Expected to have values in [0, 1].

        Returns:
            Tensor: Refined output features.
        """
        if uncertainty_map is not None:
            # Use the uncertainty map to modulate the features.
            # Here, areas with higher uncertainty (closer to 1) are amplified for further refinement.
            x_refine = x * (1 + uncertainty_map)
        else:
            x_refine = x

        refined = self.conv1(x_refine)
        refined = self.relu(refined)
        refined = self.conv2(refined)
        refined = self.sigmoid(refined)

        # Fuse the refined output with the original features (residual connection)
        output = x + refined
        return output
