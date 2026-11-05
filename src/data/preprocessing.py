
from pathlib import Path

import torch
import yaml


def compute_channel_stats(dataset, band_indices=None):
    """
    Compute per-channel mean and standard deviation
    from a dataset yielding raw image tensors.

    Ignores NaN and infinite input values.
    Does not use target labels.

    band_indices optionally selects/reorders channels
    for each modality, using zero-based indexing.
    """

    band_indices = band_indices or {}
    running = {}

    for sample in dataset:
        for modality, image in sample.items():

            if modality == "target":
                continue

            if not isinstance(image, torch.Tensor):
                continue

            indices = band_indices.get(modality)

            if indices is not None:
                image = image[indices]

            x = image.to(torch.float64)

            if x.ndim != 3:
                raise ValueError(
                    f"Expected [C, H, W] for {modality}"
                )

            valid = torch.isfinite(x)
            count = valid.sum(dim=(1, 2)).double()

            values = torch.where(valid, x, 0.0)

            local_mean = (
                values.sum(dim=(1, 2))
                / count.clamp_min(1)
            )

            local_m2 = torch.where(
                valid,
                (x - local_mean[:, None, None]) ** 2,
                0.0,
            ).sum(dim=(1, 2))

            if modality not in running:
                running[modality] = {
                    "count": torch.zeros_like(count),
                    "mean": torch.zeros_like(local_mean),
                    "m2": torch.zeros_like(local_m2),
                }

            state = running[modality]

            if state["count"].shape != count.shape:
                raise ValueError(
                    f"Inconsistent channels for {modality}"
                )

            old_count = state["count"]
            old_mean = state["mean"]

            total = old_count + count
            delta = local_mean - old_mean

            state["mean"] = (
                old_mean
                + delta * count / total.clamp_min(1)
            )

            state["m2"] += (
                local_m2
                + delta.square()
                * old_count * count
                / total.clamp_min(1)
            )

            state["count"] = total

    if not running:
        raise ValueError("No image data found")

    statistics = {}

    for modality, state in running.items():
        count = state["count"]

        if (count == 0).any():
            raise ValueError(
                f"No valid pixels in some {modality} bands"
            )

        std = torch.sqrt(state["m2"] / count)

        if (std <= 0).any():
            raise ValueError(
                f"Zero variance in some {modality} bands"
            )

        statistics[modality] = {
            "band_indices": band_indices.get(modality),
            "mean": state["mean"].tolist(),
            "std": std.tolist(),
            "valid_pixels": count.long().tolist(),
        }

    return statistics


def save_stats(statistics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.safe_dump(statistics, f, sort_keys=False)


def load_stats(path):
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


class NormalizeFromStats:
    """
    Apply previously computed channel statistics.

    method:
        "zscore" -> (x - mean) / std
        "satmae" -> mean +/- 2std, clipped to [0, 1]
    """

    def __init__(
        self,
        statistics,
        method="satmae",
        use_8_bit=False,
    ):
        if method not in {"zscore", "satmae"}:
            raise ValueError(f"Unknown method: {method}")

        self.statistics = statistics
        self.method = method
        self.use_8_bit = use_8_bit

    def __call__(self, sample):
        result = sample.copy()

        for modality, stats in self.statistics.items():

            if modality not in sample:
                continue

            image = sample[modality]

            indices = stats.get("band_indices")
            if indices is not None:
                image = image[indices]

            image = image.float()
            valid = torch.isfinite(image)

            mean = image.new_tensor(
                stats["mean"]
            )[:, None, None]

            std = image.new_tensor(
                stats["std"]
            )[:, None, None]

            if self.method == "zscore":
                normalized = (image - mean) / std

            else:
                normalized = (
                    (image - (mean - 2 * std))
                    / (4 * std)
                ).clamp(0, 1)

                if self.use_8_bit:
                    normalized = (
                        (normalized * 255)
                        .to(torch.uint8)
                        .float() / 255
                    )

            # Fill invalid pixels with zero after
            # normalization, not before estimating stats.
            result[modality] = torch.where(
                valid,
                normalized,
                0.0,
            )

        return result
