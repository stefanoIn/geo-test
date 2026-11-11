
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.baselines.common import concatenate_modalities


class ResNetUNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        modalities: tuple[str, ...],
        encoder_name: str = "resnet50",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.modalities = tuple(modalities)

        self.network = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    def forward(self, images):
        x = concatenate_modalities(
            images,
            self.modalities,
            self.in_channels,
        )
        return self.network(x)
