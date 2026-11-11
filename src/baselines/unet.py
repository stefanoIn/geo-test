
"""
Conventional U-Net for supervised dense prediction.

Architecture:
    - DoubleConv blocks: Conv3x3 -> BatchNorm -> ReLU (x2)
    - Max-pooling downsampling
    - Transposed-convolution upsampling
    - Encoder-decoder skip connections (concatenation)
    - Channel topology: [64, 128, 256, 512]

All parameters are randomly initialized and trained end-to-end.

References:
    Ronneberger et al. (2015):
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    https://arxiv.org/abs/1505.04597

    Architecture configuration follows the supervised U-Net
    baseline adopted in PANGAEA:
    https://github.com/VMarsocci/pangaea-bench
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.common import concatenate_modalities


class DoubleConv(nn.Module):
    """Two convolutional blocks with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    """Downsampling followed by two convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsampling, skip concatenation and feature refinement."""

    def __init__(
        self,
        deep_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=deep_channels,
            out_channels=deep_channels,
            kernel_size=2,
            stride=2,
        )

        self.refine = DoubleConv(
            in_channels=deep_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, deep, skip):

        x = self.upsample(deep)

        # Match the spatial resolution of the skip feature.
        # Required when downsampling produces odd dimensions.
        diff_h = skip.shape[-2] - x.shape[-2]
        diff_w = skip.shape[-1] - x.shape[-1]

        x = F.pad(
            x,
            (
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
            ),
        )

        # Concatenate along the channel dimension.
        x = torch.cat([skip, x], dim=1)

        return self.refine(x)


class UNet(nn.Module):
    """
    Fully supervised U-Net initialized from scratch.

    Accepts GeoTest modality dictionaries:
        {"sar": [B, 2, H, W]}
        {"optical": [B, 13, H, W]}
        {"sar": ..., "optical": ...}

    Returns:
        Dense predictions [B, num_classes, H, W].
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        modalities: tuple[str, ...],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.modalities = tuple(modalities)

        # -------------------------
        # Encoder
        # -------------------------

        self.stem = DoubleConv(in_channels, 64)

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)

        # -------------------------
        # Decoder
        # -------------------------

        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up4 = UpBlock(64, 64, 64)

        # -------------------------
        # Prediction layer
        # -------------------------

        self.classifier = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, images):

        # Concatenate available input modalities.
        x = concatenate_modalities(
            images,
            self.modalities,
            self.in_channels,
        )

        # Encoder
        x0 = self.stem(x)       # [B, 64, H, W]
        x1 = self.down1(x0)     # [B, 128, H/2, W/2]
        x2 = self.down2(x1)     # [B, 256, H/4, W/4]
        x3 = self.down3(x2)     # [B, 512, H/8, W/8]
        x4 = self.down4(x3)     # [B, 512, H/16, W/16]

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Raw logits (no softmax).
        return self.classifier(x)
