"""Compose an encoder, feature adapter, and dense prediction head."""

import torch
import torch.nn.functional as F
from torch import nn


class DensePredictionModel(nn.Module):
    """Return full-input-resolution logits (or other dense predictions).

    The encoder and adapter define their own feature representation.
    The head must return a spatial tensor [B, C_out, h, w].
    """

    def __init__(self, encoder: nn.Module, adapter: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.head = head

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        input_hw = next(iter(images.values())).shape[-2:]

        features = self.encoder(images)
        spatial_features = self.adapter(features)
        predictions = self.head(spatial_features)

        if predictions.shape[-2:] != input_hw:
            predictions = F.interpolate(
                predictions,
                size=input_hw,
                mode="bilinear",
                align_corners=False,
            )
        return predictions
