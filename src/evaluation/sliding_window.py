"""Sliding-window inference for spatial predictions, averaging overlap logits."""

import math

import torch


def _pair(value, label):
    if type(value) is int:
        value = (value, value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be an integer or (height, width)")
    if any(type(v) is not int or v <= 0 for v in value):
        raise ValueError(f"{label} must contain positive integers")
    return tuple(value)


def _starts(length, tile, stride):
    last = length - tile
    if stride is None:
        # Backwards-compatible behaviour of the original implementation.
        return torch.linspace(0, last, math.ceil(length / tile)).round().long().tolist()
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)  # Force coverage of the far edge.
    return starts


@torch.inference_mode()
def sliding_window_predict(model, images, tile_size, tile_batch_size=8, stride=None):
    """Return averaged logits [B,C,H,W] from aligned image modalities.

    `stride` sets the step between windows; e.g. tile_size=120, stride=60
    matches the overlapping windows used in the original experiments.
    If omitted, preserves the older evenly-spaced implementation.
    """
    tile_h, tile_w = _pair(tile_size, "tile_size")
    if stride is not None:
        stride_h, stride_w = _pair(stride, "stride")
        if stride_h > tile_h or stride_w > tile_w:
            raise ValueError("stride must be <= tile_size to cover the whole image")
    else:
        stride_h = stride_w = None
    if type(tile_batch_size) is not int or tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be a positive integer")
    if not isinstance(images, dict) or not images:
        raise ValueError("images must be a non-empty dict")

    first = next(iter(images.values()))
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("Image modalities must be [B,C,H,W] tensors")
    batch_size, _, height, width = first.shape
    if batch_size == 0:
        raise ValueError("Empty batch")
    for name, tensor in images.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                or tensor.shape[0] != batch_size or tensor.shape[-2:] != (height, width)
                or tensor.device != first.device):
            raise ValueError(f"{name}: inconsistent shape or device")
    if tile_h > height or tile_w > width:
        raise ValueError("tile_size exceeds image spatial size")

    tops = _starts(height, tile_h, stride_h)
    lefts = _starts(width, tile_w, stride_w)
    locations = [(b, top, left) for b in range(batch_size) for top in tops for left in lefts]

    logits_sum = None
    counts = torch.zeros((batch_size, 1, height, width), device=first.device, dtype=torch.float32)
    for start in range(0, len(locations), tile_batch_size):
        locations_chunk = locations[start:start + tile_batch_size]
        tiles = {
            name: torch.stack([
                tensor[b, :, top:top + tile_h, left:left + tile_w]
                for b, top, left in locations_chunk
            ]) for name, tensor in images.items()
        }
        logits = model(tiles)
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or logits.shape[0] != len(locations_chunk)
                or logits.shape[-2:] != (tile_h, tile_w)
                or logits.shape[1] <= 0):
            raise ValueError("Model must return logits [tile_batch,C,tile_h,tile_w]")
        if logits_sum is None:
            logits_sum = torch.zeros(
                (batch_size, logits.shape[1], height, width),
                device=logits.device, dtype=torch.float32,
            )
        elif logits.shape[1] != logits_sum.shape[1]:
            raise ValueError("Output channel count changed between windows")

        for idx, (b, top, left) in enumerate(locations_chunk):
            logits_sum[b, :, top:top + tile_h, left:left + tile_w] += logits[idx].float()
            counts[b, :, top:top + tile_h, left:left + tile_w] += 1

    if not bool((counts > 0).all()):
        raise RuntimeError("Sliding windows did not cover every pixel")
    return logits_sum / counts
