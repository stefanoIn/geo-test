import torch.nn as nn

from .use_croma import PretrainedCROMA


class CROMAEncoder(nn.Module):

    def __init__(
        self,
        checkpoint_path,
        size="large",
        modality="both",
        image_resolution=120,
        freeze=True,
    ):
        super().__init__()

        self.modality = modality
        self.freeze = freeze

        self.backbone = PretrainedCROMA(
            pretrained_path=str(checkpoint_path),
            size=size,
            modality=modality,
            image_resolution=image_resolution,
        )

        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward(self, sample):

        outputs = self.backbone(
            SAR_images=sample.get("sar"),
            optical_images=sample.get("optical"),
        )

        prefix = {
            "SAR": "SAR",
            "optical": "optical",
            "both": "joint",
        }[self.modality]

        return {
            "tokens": outputs[f"{prefix}_encodings"],
            "pooled": outputs[f"{prefix}_GAP"],
        }