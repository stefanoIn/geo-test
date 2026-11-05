"""Sliding-window inference for dense, spatial predictions."""

import math
import torch


@torch.inference_mode()
def sliding_window_predict(model, images, tile_size, tile_batch_size=8):
    """Predict a full image by averaging predictions from overlapping tiles.

    Args:
        model: Model accepting a dict of batched modalities and returning
            spatial predictions [B, C_out, tile_h, tile_w]. Set model.eval()
            before calling this function.
        images: Dictionary of aligned image tensors [B, C, H, W].
        tile_size: Integer or (height, width) of the model's input window.
        tile_batch_size: Maximum number of tiles passed to the model at once.

    Returns:
        Float32 predictions [B, C_out, H, W]. For segmentation, these are
        logits, not class labels. The caller computes task-specific metrics.

    Raises:
        ValueError: If input shapes, tile sizes, or predictions are invalid.
    """
    if isinstance(tile_size, int):
        tile_h = tile_w = tile_size
    elif isinstance(tile_size, (tuple, list)) and len(tile_size) == 2:
        tile_h, tile_w = tile_size
    else:
        raise ValueError("tile_size must be an int or (height, width)")

    if (not isinstance(tile_h, int) or not isinstance(tile_w, int)
            or tile_h <= 0 or tile_w <= 0):
        raise ValueError("Tile dimensions must be positive integers")
    if not isinstance(tile_batch_size, int) or tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be a positive integer")
    if not isinstance(images, dict) or not images:
        raise ValueError("images must be a non-empty dictionary")

    first = next(iter(images.values()))
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("Each modality must be a tensor [B, C, H, W]")
    batch_size, _, height, width = first.shape
    if batch_size == 0:
        raise ValueError("Batch size must be greater than zero")

    for name, tensor in images.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                or tensor.shape[0] != batch_size
                or tensor.shape[-2:] != (height, width)
                or tensor.device != first.device):
            raise ValueError(f"{name}: incompatible shape or device")

    if tile_h > height or tile_w > width:
        raise ValueError("Tile size exceeds image size")

    # Evenly distribute windows from the first to the last valid position.
    # This covers the full image and permits overlap at image boundaries.
    top_positions = torch.linspace(
        0, height - tile_h, math.ceil(height / tile_h)
    ).round().long().tolist()
    left_positions = torch.linspace(
        0, width - tile_w, math.ceil(width / tile_w)
    ).round().long().tolist()

    locations = [
        (b, top, left)
        for b in range(batch_size)
        for top in top_positions
        for left in left_positions
    ]

    predictions = None
    counts = torch.zeros((batch_size, 1, height, width),
                         device=first.device, dtype=torch.float32)

    for start in range(0, len(locations), tile_batch_size):
        selected = locations[start:start + tile_batch_size]
        tiles = {
            name: torch.stack([
                tensor[b, :, top:top + tile_h, left:left + tile_w]
                for b, top, left in selected
            ])
            for name, tensor in images.items()
        }
        output = model(tiles)
        if (not isinstance(output, torch.Tensor) or output.ndim != 4
                or output.shape[0] != len(selected)
                or output.shape[-2:] != (tile_h, tile_w)
                or output.shape[1] == 0):
            raise ValueError("Model must return [tile_batch, C_out, tile_h, tile_w]")

        if predictions is None:
            predictions = torch.zeros(
                (batch_size, output.shape[1], height, width),
                device=output.device, dtype=torch.float32,
            )
        elif output.shape[1] != predictions.shape[1]:
            raise ValueError("Model output channel count changed between tiles")

        for i, (b, top, left) in enumerate(selected):
            predictions[b, :, top:top + tile_h, left:left + tile_w] += output[i].float()
            counts[b, :, top:top + tile_h, left:left + tile_w] += 1

    return predictions / counts
