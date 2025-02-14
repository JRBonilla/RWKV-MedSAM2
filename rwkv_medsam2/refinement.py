import torch
import torch.nn as nn

class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Generates an uncertainty map from input features.
        
        Args:
            in_channels (int): Numbers of channels in the input feature map.
            out_channels (int): Number of output channels. Default: 1
        """
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        uncertainty = self.sigmoid(x)
        return uncertainty

class RefinementModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        """
        Refines segmentation features by focusing on regions of high uncertainty.

        Args:
            in_channels (int): Number of channels of the input feature map.
            mid_channels (int): Number of channels in the intermediate representation.
            out_channels (int): Number of channels for the refined output.
        """
        super().__init__()
        # Two convolutional layers for refinement
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, uncertainty_map=None):
        """
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