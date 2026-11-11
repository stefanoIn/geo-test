import torch

def concatenate_modalities(images, modalities, in_channels):
    """Convert a modality dictionary into a CNN input tensor."""

    if set(images) != set(modalities):
        raise ValueError(
            f"Expected modalities {modalities}, got {list(images)}"
        )

    tensors = [images[name] for name in modalities]

    if any(x.ndim != 4 for x in tensors):
        raise ValueError("Expected tensors [B, C, H, W]")

    if len({(x.shape[0], *x.shape[-2:]) for x in tensors}) != 1:
        raise ValueError("Modalities must have matching batch and spatial sizes")

    x = torch.cat(tensors, dim=1)

    if x.shape[1] != in_channels:
        raise ValueError(
            f"Expected {in_channels} input channels, got {x.shape[1]}"
        )

    return x
