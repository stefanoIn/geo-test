"""Convert spatial patch tokens into a 2-D feature map."""

from math import isqrt

import torch
from torch import nn


class TokenToSpatial(nn.Module):
    """Convert [B, N, D] tokens to [B, D, grid_h, grid_w].

    If grid_size is omitted, infer a square grid from N. The input must
    contain only spatial patch tokens (no CLS or other special tokens).
    The channel dimension D is not changed.
    """

    def __init__(self, feature_key="tokens", grid_size=None):
        super().__init__()
        self.feature_key = feature_key

        if grid_size is not None:
            if (
                not isinstance(grid_size, (tuple, list))
                or len(grid_size) != 2
                or any(type(n) is not int or n <= 0 for n in grid_size)
            ):
                raise ValueError("grid_size must be [height, width] with positive integers")
            grid_size = tuple(grid_size)
        self.grid_size = grid_size

    def forward(self, features):
        if isinstance(features, dict):
            if self.feature_key not in features:
                raise KeyError(f"Missing encoder output: {self.feature_key!r}")
            tokens = features[self.feature_key]
        else:
            tokens = features

        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("Expected spatial tokens with shape [B, N, D]")

        batch, n_tokens, channels = tokens.shape
        if self.grid_size is None:
            side = isqrt(n_tokens)
            if side * side != n_tokens:
                raise ValueError(
                    f"Cannot infer a square grid from {n_tokens} tokens; "
                    "provide grid_size=[height, width]"
                )
            grid_h, grid_w = side, side
        else:
            grid_h, grid_w = self.grid_size
            if grid_h * grid_w != n_tokens:
                raise ValueError(
                    f"grid_size {self.grid_size} requires {grid_h * grid_w} tokens, "
                    f"received {n_tokens}"
                )

        return tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
