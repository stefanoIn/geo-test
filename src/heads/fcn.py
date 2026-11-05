import torch.nn as nn
from torchvision.models.segmentation.fcn import FCNHead as TorchvisionFCNHead

class FCNHead(nn.Module):
    """
    FCN segmentation head with GroupNorm.

    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W]
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.head = TorchvisionFCNHead(
            in_channels=in_channels,
            channels=num_classes,
        )

        # Replace BatchNorm with GroupNorm,
        # matching Sen1Floods11 experiments.
        self.head[1] = nn.GroupNorm(
            num_groups=16,
            num_channels=in_channels // 4,
        )

    def forward(self, features):
        return self.head(features)
