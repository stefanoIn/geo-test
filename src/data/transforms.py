"""Reusable spatial transformations for image datasets."""

import torch

def random_crop(sample, size, generator=None):
    """
    Apply the same random crop to all spatial tensors.

    Args:
        sample: Dictionary containing image tensors [C, H, W]
                and optionally a target mask [H, W].
        size: Crop size (int or tuple of height, width).
        generator: Optional PyTorch random generator.

    Returns:
        Dictionary containing the cropped tensors.
    """

    if isinstance(size, int):
        crop_h, crop_w = size, size
    else:
        if len(size) != 2:
            raise ValueError(
                "size must be an int or (height, width)"
            )
        crop_h, crop_w = size

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("Crop dimensions must be positive")

    # Identify spatial tensors.
    spatial = {
        key: value
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
        and value.ndim in (2, 3)
    }

    if not spatial:
        raise ValueError("No image or mask tensors found")

    # All modalities and masks must be aligned.
    height, width = next(iter(spatial.values())).shape[-2:]

    if any(
        tensor.shape[-2:] != (height, width)
        for tensor in spatial.values()
    ):
        raise ValueError(
            "Images and target must have the same spatial size"
        )

    if crop_h > height or crop_w > width:
        raise ValueError("Crop size exceeds the input size")

    # Sample coordinates only once.
    top = torch.randint(
        height - crop_h + 1,
        (1,),
        generator=generator,
    ).item()

    left = torch.randint(
        width - crop_w + 1,
        (1,),
        generator=generator,
    ).item()

    # Apply identical coordinates to every spatial tensor.
    return {
        key: (
            value[
                ..., top:top + crop_h, left:left + crop_w
            ]
            if key in spatial else value
        )
        for key, value in sample.items()
    }
