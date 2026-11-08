import torch
from torch import nn

class LinearSegmentationHead(nn.Module):
    """Pixel-wise linear probe on frozen spatial features"""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)