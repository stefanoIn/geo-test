"""TerraMind v1 encoder exposing spatial patch tokens to the common adapter."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TerraMindEncoder(nn.Module):
    """TerraMind v1 Large, Sentinel-1 GRD, using a local pretrained checkpoint.

    Input:  {"sar": Tensor[B, 2, H, W]} with model-appropriate normalisation.
    Output: {"tokens": Tensor[B, N, 1024]} (last encoder layer).
    """

    def __init__(self, checkpoint_path, freeze=True):
        super().__init__()

        from terratorch.models.backbones.terramind.model.terramind_register import (
            terramind_v1_large,
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        # Keep the checkpoint's original positional embedding grid.
        self.backbone = terramind_v1_large(
            pretrained=False,
            modalities=["S1GRD"],
            img_size=224,
            merge_method="mean",
        )

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        expected = self.backbone.state_dict()
        missing = [key for key in expected if key not in state]
        mismatched = [
            key for key in expected
            if key in state and expected[key].shape != state[key].shape
        ]
        if missing or mismatched:
            raise RuntimeError(
                f"Incompatible TerraMind checkpoint: missing={missing[:5]}, "
                f"shape mismatches={mismatched[:5]}"
            )
        # Strictly load every tensor of the selected S1GRD backbone.
        self.backbone.load_state_dict(
            {key: state[key] for key in expected}, strict=True
        )

        self.freeze = freeze
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, images):
        sar = images["sar"]
        if sar.ndim != 4 or sar.shape[1] != 2:
            raise ValueError("TerraMind S1GRD expects SAR [B, 2, H, W]")

        # TerraMind uses 16x16 patches. 120x120 becomes 128x128.
        height, width = sar.shape[-2:]
        pad_h = (-height) % 16
        pad_w = (-width) % 16
        sar = F.pad(sar, (0, pad_w, 0, pad_h), mode="replicate")

        features = self.backbone({"S1GRD": sar})
        tokens = features[-1]

        expected_tokens = (sar.shape[-2] // 16) * (sar.shape[-1] // 16)
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Unexpected TerraMind tokens: {tuple(tokens.shape)}, "
                f"expected {expected_tokens} spatial tokens"
            )
        return {"tokens": tokens}
