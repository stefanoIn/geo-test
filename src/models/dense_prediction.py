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

        # TerraMind may pad input images to whole patches (e.g. 120 -> 128).
        # Decode on the *padded* spatial extent, then remove the padding.
        # Encoders without this metadata (e.g. CROMA) behave as before.
        padded_hw = (
            features.get("padded_size", input_hw)
            if isinstance(features, dict) else input_hw
        )
        if (not isinstance(padded_hw, (tuple, list)) or len(padded_hw) != 2
                or any(type(n) is not int or n <= 0 for n in padded_hw)
                or any(p < original for p, original in zip(padded_hw, input_hw))):
            raise ValueError(f"Invalid encoder padded_size: {padded_hw!r}")

        if predictions.shape[-2:] != tuple(padded_hw):
            predictions = F.interpolate(
                predictions,
                size=tuple(padded_hw),
                mode="bilinear",
                align_corners=False,
            )
        if tuple(padded_hw) != tuple(input_hw):
            predictions = predictions[..., :input_hw[0], :input_hw[1]]
        return predictions
