"""Reusable image normalisation; no statistic computation or file I/O."""

import torch

class NormalizeFromStats:
    """Normalise each image modality with precomputed per-band statistics.

    Methods:
      - 'zscore': (x - mean) / std
      - 'two_sigma_stretch': clip((x - (mean - 2*std)) / (4*std), 0, 1)

    The optional band_indices are zero-based positions in the raw dataset.
    They select/reorder image bands AND the corresponding statistics.
    """

    def __init__(
        self,
        statistics,
        method="two_sigma_stretch",
        band_indices=None,
        use_8_bit=False,
    ):
        if method not in {"zscore", "two_sigma_stretch"}:
            raise ValueError(f"Unknown normalization method: {method}")
        if use_8_bit and method != "two_sigma_stretch":
            raise ValueError("use_8_bit is supported only for two_sigma_stretch")
        self.statistics = statistics
        self.method = method
        self.band_indices = band_indices or {}
        self.use_8_bit = use_8_bit

    def __call__(self, sample):
        result = sample.copy()

        for modality, stats in self.statistics.items():
            if modality not in sample:
                continue

            image = sample[modality].float()
            if image.ndim != 3:
                raise ValueError(f"{modality}: expected [C,H,W]")
            raw_channels = image.shape[0]
            mean = image.new_tensor(stats["mean"])
            std = image.new_tensor(stats["std"])

            indices = self.band_indices.get(modality)
            if indices is not None:
                image = image[indices]
                # Dataset-derived statistics cover all raw channels. External
                # model statistics may already cover only the selected channels.
                if mean.numel() == raw_channels:
                    mean = mean[indices]
                    std = std[indices]

            if image.shape[0] != mean.numel() or mean.numel() != std.numel():
                raise ValueError(
                    f"{modality}: {image.shape[0]} image bands but "
                    f"{mean.numel()} statistics entries"
                )
            if not bool(torch.all(torch.isfinite(std) & (std > 0))):
                raise ValueError(f"{modality}: invalid standard deviations")

            valid = torch.isfinite(image)
            safe_image = torch.where(valid, image, 0.0)
            mean = mean[:, None, None]
            std = std[:, None, None]

            if self.method == "zscore":
                normalized = (safe_image - mean) / std
            else:
                normalized = ((safe_image - (mean - 2 * std)) / (4 * std)).clamp(0, 1)
                if self.use_8_bit:
                    normalized = (normalized * 255).to(torch.uint8).float() / 255

            result[modality] = torch.where(valid, normalized, 0.0)

        return result
