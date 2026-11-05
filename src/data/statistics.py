"""Model-independent statistics for raw training datasets."""

import hashlib
import json
from pathlib import Path

import torch

STATS_VERSION = 1

def compute_channel_stats(dataset):
    """Compute per-band population mean/std over valid pixels of raw images.

    The dataset must yield dictionaries containing modality tensors [C, H, W].
    The 'target' entry is ignored. Data must not be augmented/normalised.
    All original bands are included; models select bands when preprocessing.
    """
    running = {}

    for sample in dataset:
        for modality, image in sample.items():
            if modality == "target" or not isinstance(image, torch.Tensor):
                continue
            if image.ndim != 3:
                raise ValueError(f"{modality}: expected [C,H,W], got {tuple(image.shape)}")

            x = image.to(torch.float64)
            valid = torch.isfinite(x)
            n = valid.sum(dim=(1, 2)).to(torch.float64)
            safe = torch.where(valid, x, 0.0)
            local_mean = safe.sum(dim=(1, 2)) / n.clamp_min(1)
            local_m2 = torch.where(
                valid, (x - local_mean[:, None, None]).square(), 0.0
            ).sum(dim=(1, 2))

            if modality not in running:
                running[modality] = {
                    "count": torch.zeros_like(n),
                    "mean": torch.zeros_like(local_mean),
                    "m2": torch.zeros_like(local_m2),
                }

            state = running[modality]
            if state["count"].shape != n.shape:
                raise ValueError(f"{modality}: channel count changed across samples")
            old_n = state["count"]
            new_n = old_n + n
            delta = local_mean - state["mean"]
            state["mean"] += delta * n / new_n.clamp_min(1)
            state["m2"] += (
                local_m2 + delta.square() * old_n * n / new_n.clamp_min(1)
            )
            state["count"] = new_n

    if not running:
        raise ValueError("No image tensors found")

    statistics = {}
    for modality, state in running.items():
        count = state["count"]
        if (count == 0).any():
            raise ValueError(f"{modality}: a band contains no valid pixels")
        std = (state["m2"] / count).sqrt()
        if (std <= 0).any():
            raise ValueError(f"{modality}: a band has zero variance")
        statistics[modality] = {
            "mean": state["mean"].tolist(),
            "std": std.tolist(),
            "valid_pixels": count.long().tolist(),
        }

    return statistics


def save_stats(statistics, path, metadata=None):
    """Write statistics and optional provenance to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "statistics": statistics}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stats(path):
    """Load either a cached stats file or a plain external stats dictionary."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("statistics", data)


def get_or_compute_stats(
    dataset,
    *,
    split_csv,
    dataset_name,
    cache_dir="data/statistics",
):
    """Load cached raw-band statistics, or compute them once for a training split.

    The cache identity uses CSV content, source image file metadata, dataset
    name, modalities, and algorithm version. No random seed is required.
    """
    if getattr(dataset, "preprocess_func", None) is not None:
        raise ValueError("Statistics require a raw dataset (preprocess_func=None)")

    split_path = Path(split_csv).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    split_contents = split_path.read_bytes()
    split_digest = hashlib.sha256(split_contents).hexdigest()
    modalities = sorted(getattr(dataset, "modalities", []))

    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps({
        "dataset": dataset_name,
        "modalities": modalities,
        "algorithm_version": STATS_VERSION,
        "split_sha256": split_digest,
    }, sort_keys=True).encode("utf-8"))

    # Detect changes to source imagery, if the dataset exposes its file paths.
    # Labels are intentionally excluded: they are not used for normalisation.
    for paths in getattr(dataset, "data_list", []):
        for modality, image_path in sorted(paths.items()):
            if modality == "target":
                continue
            image_path = Path(image_path)
            file_stat = image_path.stat()
            fingerprint.update(str(image_path.resolve()).encode("utf-8"))
            fingerprint.update(f"{file_stat.st_size}:{file_stat.st_mtime_ns}".encode("ascii"))

    cache_key = fingerprint.hexdigest()[:16]
    cache_file = Path(cache_dir) / dataset_name / f"{cache_key}.json"

    if cache_file.is_file():
        print(f"Loading cached training statistics: {cache_file}")
        return load_stats(cache_file)

    print(f"Computing statistics for training split: {split_path}")
    statistics = compute_channel_stats(dataset)
    save_stats(
        statistics,
        cache_file,
        metadata={
            "dataset": dataset_name,
            "modalities": modalities,
            "algorithm_version": STATS_VERSION,
            "split_sha256": split_digest,
            "source_split": str(split_path),
        },
    )
    print(f"Saved training statistics: {cache_file}")
    return statistics
