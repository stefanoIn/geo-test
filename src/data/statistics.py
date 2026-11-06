"""Model-independent dataset statistics and caching."""

import hashlib
import json
from pathlib import Path

import torch


STATS_VERSION = 2


def compute_channel_stats(dataset):
    """
    Compute per-channel mean and standard deviation
    over all valid pixels in a raw dataset.

    Expected sample:
        {
            "sar": Tensor[C, H, W],
            "optical": Tensor[C, H, W],
            "target": Tensor[H, W]
        }

    Modality names and channel counts are not fixed.

    Requirements:
        - Raw images, without normalization or augmentation.
        - Invalid pixels represented by NaN or infinity.
        - Target labels are ignored.

    Returns:
        {
            modality: {
                "mean": [...],
                "std": [...],
                "valid_pixels": [...]
            }
        }
    """

    running = {}

    for sample in dataset:
        for modality, image in sample.items():

            if modality == "target":
                continue

            if not isinstance(image, torch.Tensor):
                continue

            if image.ndim != 3:
                raise ValueError(
                    f"{modality}: expected [C, H, W], "
                    f"got {tuple(image.shape)}"
                )

            x = image.to(torch.float64)

            valid = torch.isfinite(x)
            count = valid.sum(dim=(1, 2)).to(torch.float64)

            safe = torch.where(valid, x, 0.0)

            local_mean = (
                safe.sum(dim=(1, 2))
                / count.clamp_min(1)
            )

            differences = torch.where(
                valid,
                x - local_mean[:, None, None],
                0.0,
            )

            local_m2 = (
                differences.square().sum(dim=(1, 2))
            )

            if modality not in running:
                running[modality] = {
                    "count": torch.zeros_like(count),
                    "mean": torch.zeros_like(local_mean),
                    "m2": torch.zeros_like(local_m2),
                }

            state = running[modality]

            if state["count"].shape != count.shape:
                raise ValueError(
                    f"{modality}: channel count changed"
                )

            old_count = state["count"]
            total = old_count + count

            delta = local_mean - state["mean"]

            state["mean"] = (
                state["mean"]
                + delta * count / total.clamp_min(1)
            )

            state["m2"] = (
                state["m2"]
                + local_m2
                + delta.square()
                * old_count * count
                / total.clamp_min(1)
            )

            state["count"] = total

    if not running:
        raise ValueError("No image tensors found")

    statistics = {}

    for modality, state in running.items():

        count = state["count"]

        if (count == 0).any():
            raise ValueError(
                f"{modality}: a band has no valid pixels"
            )

        std = (state["m2"] / count).sqrt()

        if (std <= 0).any():
            raise ValueError(
                f"{modality}: a band has zero variance"
            )

        statistics[modality] = {
            "mean": state["mean"].tolist(),
            "std": std.tolist(),
            "valid_pixels": count.long().tolist(),
        }

    return statistics


def save_stats(statistics, path, metadata=None):
    """Save statistics and provenance as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "statistics": statistics,
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_stats(path):
    """Load cached or externally provided statistics."""

    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    return data.get("statistics", data)


def get_or_compute_stats(
    *,
    dataset_factory,
    modalities,
    split_csv,
    dataset_name,
    cache_dir="data/statistics",
):
    """
    Load or compute statistics independently per modality.

    Statistics are identified by:
        - Dataset name
        - Training split CSV contents
        - Modality
        - Source image file metadata
        - Statistics algorithm version

    Each modality is computed only when its cache
    does not already exist.

    dataset_factory(modalities) must return a raw
    Dataset for the requested modalities.

    The Dataset must expose:
        data_list: list of dictionaries mapping
                   modality names to image paths.

    Example:
        get_or_compute_stats(
            dataset_factory=factory,
            modalities=("sar", "optical"),
            split_csv="train.csv",
            dataset_name="sen1floods11",
        )
    """

    split_path = Path(split_csv)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    split_hash = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()

    # Remove duplicates while preserving order.
    modalities = tuple(dict.fromkeys(modalities))

    if not modalities:
        raise ValueError("At least one modality is required")

    result = {}

    for modality in modalities:

        # Build a raw Dataset for this modality only.
        dataset = dataset_factory((modality,))

        if getattr(dataset, "preprocess_func", None) is not None:
            raise ValueError(
                "Statistics require a raw Dataset "
                "(preprocess_func=None)"
            )

        if not hasattr(dataset, "data_list"):
            raise ValueError(
                "Dataset must expose data_list "
                "for cache identification"
            )

        # Create a fingerprint for this specific
        # modality and training subset.
        fingerprint = hashlib.sha256()

        fingerprint.update(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "modality": modality,
                    "split_sha256": split_hash,
                    "stats_version": STATS_VERSION,
                },
                sort_keys=True,
            ).encode("utf-8")
        )

        # Include source image metadata so the cache
        # changes when the underlying files change.
        for paths in dataset.data_list:

            if modality not in paths:
                raise ValueError(
                    f"Missing modality '{modality}' "
                    "in dataset.data_list"
                )

            image_path = Path(paths[modality])

            file_stat = image_path.stat()

            fingerprint.update(
                str(image_path).encode("utf-8")
            )

            fingerprint.update(
                (
                    f"{file_stat.st_size}:"
                    f"{file_stat.st_mtime_ns}"
                ).encode("ascii")
            )

        cache_key = fingerprint.hexdigest()[:16]

        cache_file = (
            Path(cache_dir)
            / dataset_name
            / split_hash[:12]
            / f"{modality}_{cache_key}.json"
        )

        # Reuse previously computed statistics.
        if cache_file.is_file():

            print(
                f"Loading cached statistics "
                f"for {modality}: {cache_file}"
            )

            cached = load_stats(cache_file)

        else:

            print(
                f"Computing statistics "
                f"for {modality}..."
            )

            cached = compute_channel_stats(dataset)

            if set(cached) != {modality}:
                raise ValueError(
                    f"Expected statistics only "
                    f"for '{modality}', "
                    f"got {list(cached)}"
                )

            save_stats(
                cached,
                cache_file,
                metadata={
                    "dataset": dataset_name,
                    "modality": modality,
                    "split_sha256": split_hash,
                    "source_split": str(split_path),
                    "stats_version": STATS_VERSION,
                },
            )

            print(
                f"Saved statistics "
                f"for {modality}: {cache_file}"
            )

        if modality not in cached:
            raise ValueError(
                f"Cache missing modality '{modality}'"
            )

        result[modality] = cached[modality]

    return result
