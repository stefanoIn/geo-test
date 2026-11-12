"""Reusable spatial transformations for image datasets."""

import torch

def random_crop(sample, size, generator=None, *, include_metadata=False,
                require_valid_target=False, ignore_index=255):
    """
    Apply the same random crop to all spatial tensors.

    Args:
        sample: Dictionary containing image tensors [C, H, W]
                and optionally a target mask [H, W].
        size: Crop size (int or tuple of height, width).
        generator: Optional PyTorch random generator.
        include_metadata: Attach zero-based crop coordinates without extra RNG draws.
        require_valid_target: Resample windows with no supervised pixels.
        ignore_index: Target value excluded from supervision.

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

    target = sample.get("target")
    if require_valid_target and (not isinstance(target, torch.Tensor) or target.ndim != 2):
        raise ValueError("Valid crop sampling requires a target mask [H, W]")

    # Rejection sampling preserves uniform sampling over valid crop positions.
    for _ in range(32 if require_valid_target else 1):
        top = torch.randint(height - crop_h + 1, (1,), generator=generator).item()
        left = torch.randint(width - crop_w + 1, (1,), generator=generator).item()
        if not require_valid_target or (target[top:top + crop_h, left:left + crop_w] != ignore_index).any():
            break
    else:
        # Sparse labels: enumerate valid windows instead of retrying indefinitely.
        # Integral-image sums keep the fallback uniform over eligible positions.
        valid = (target != ignore_index).to(torch.int64)
        integral = torch.nn.functional.pad(valid.cumsum(0).cumsum(1), (1, 0, 1, 0))
        counts = (integral[crop_h:, crop_w:] - integral[:-crop_h, crop_w:]
                  - integral[crop_h:, :-crop_w] + integral[:-crop_h, :-crop_w])
        positions = (counts > 0).nonzero()
        if len(positions) == 0:
            raise ValueError("Cannot sample a valid crop from an entirely ignored target")
        index = torch.randint(len(positions), (1,), generator=generator).item()
        top, left = positions[index].tolist()

    # Apply identical coordinates to every spatial tensor.
    result = {
        key: (
            value[
                ..., top:top + crop_h, left:left + crop_w
            ]
            if key in spatial else value
        )
        for key, value in sample.items()
    }

    if include_metadata:
        result["crop_metadata"] = {
            "top": top, "left": left, "height": crop_h, "width": crop_w,
            "source_height": height, "source_width": width,
        }
    return result
