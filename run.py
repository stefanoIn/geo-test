"""Inspect the configured dataset, cached statistics, and normalised batches.

This is a data-pipeline smoke test. It does not train or evaluate a model.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.preprocessing import NormalizeFromStats
from src.data.statistics import get_or_compute_stats


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    """Interpret relative paths from the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path):
    with resolve_path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def build_dataset(config, split, preprocess_func=None, modalities=None):
    """Create a Dataset for the requested split and modalities."""
    dataset_cfg = config["dataset"]

    if dataset_cfg["name"] != "sen1floods11":
        raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")

    csv_path = dataset_cfg.get(f"{split}_csv")
    if not csv_path:
        raise ValueError(f"No CSV configured for split '{split}'")

    return Sen1Floods11Dataset(
        split_csv=resolve_path(csv_path),
        modalities=tuple(modalities or dataset_cfg["modalities"]),
        preprocess_func=preprocess_func,
    )


def build_preprocessing(config):
    """Compute/load training statistics and build a reusable normaliser."""
    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocessing"]
    norm_cfg = preprocess_cfg["normalization"]

    if norm_cfg["statistics"] != "training_dataset":
        raise NotImplementedError(
            "This data-pipeline test currently supports only "
            "statistics: training_dataset"
        )

    train_csv = resolve_path(dataset_cfg["train_csv"])
    requested_modalities = tuple(dataset_cfg["modalities"])

    statistics = get_or_compute_stats(
        dataset_factory=lambda modalities: build_dataset(
            config,
            split="train",
            modalities=modalities,
            preprocess_func=None,
        ),
        modalities=requested_modalities,
        split_csv=train_csv,
        dataset_name=dataset_cfg["name"],
        cache_dir=PROJECT_ROOT / "data" / "statistics",
    )

    normalizer = NormalizeFromStats(
        statistics=statistics,
        method=norm_cfg["method"],
        band_indices=preprocess_cfg.get("band_indices"),
        use_8_bit=norm_cfg.get("use_8_bit", False),
    )
    return normalizer, statistics


def build_loaders(config, normalizer):
    """Construct DataLoaders for all splits defined in the YAML."""
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
    num_workers = training_cfg.get("num_workers", 0)
    seed = config.get("experiment", {}).get("seed", 0)
    generator = torch.Generator().manual_seed(seed)

    loaders = {}
    for split in ("train", "val", "test", "bolivia"):
        if not config["dataset"].get(f"{split}_csv"):
            continue

        dataset = build_dataset(config, split, preprocess_func=normalizer)
        batch_size = (
            training_cfg["batch_size"] if split == "train"
            else evaluation_cfg.get("batch_size", 1)
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            generator=generator if split == "train" else None,
        )
    return loaders


def inspect_batch(split, loader, config, statistics):
    """Read one batch and fail if shapes, labels or values are invalid."""
    batch = next(iter(loader))
    target = batch["target"]
    if target.ndim != 3 or target.dtype != torch.long:
        raise ValueError(f"{split}: target should be LongTensor[B,H,W]")

    num_classes = config["head"]["num_classes"]
    allowed = set(range(num_classes)) | {config["preprocessing"].get("ignore_index", 255)}
    labels = set(torch.unique(target).tolist())
    if not labels <= allowed:
        raise ValueError(f"{split}: unexpected target labels: {labels - allowed}")

    print(f"\n[{split}] {len(loader.dataset)} samples; batch target: {tuple(target.shape)}")
    print(f"  target classes in batch: {sorted(labels)}")

    band_indices = config["preprocessing"].get("band_indices", {})
    method = config["preprocessing"]["normalization"]["method"]
    for modality in config["dataset"]["modalities"]:
        image = batch[modality]
        expected_channels = len(
            band_indices.get(modality, statistics[modality]["mean"])
        )
        if image.ndim != 4 or image.shape[1] != expected_channels:
            raise ValueError(
                f"{split}/{modality}: expected [B,{expected_channels},H,W], "
                f"got {tuple(image.shape)}"
            )
        if image.shape[0] != target.shape[0] or image.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"{split}/{modality}: image/target shapes do not align")
        if not torch.isfinite(image).all():
            raise ValueError(f"{split}/{modality}: non-finite values after preprocessing")

        minimum = image.min().item()
        maximum = image.max().item()
        if method == "two_sigma_stretch" and (minimum < -1e-6 or maximum > 1 + 1e-6):
            raise ValueError(f"{split}/{modality}: expected values in [0,1]")

        print(
            f"  {modality}: {tuple(image.shape)}, "
            f"min={minimum:.4f}, max={maximum:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="GeoFM data-pipeline smoke test")
    parser.add_argument(
        "--config",
        default="configs/croma/croma_s1s2_fcn_all_hand_labels.yaml",
        help="YAML experiment configuration (relative to repository root)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("experiment", {}).get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
    print(f"Training split: {resolve_path(config['dataset']['train_csv'])}")

    normalizer, statistics = build_preprocessing(config)
    for modality, stats in statistics.items():
        print(
            f"Statistics ready [{modality}]: "
            f"{len(stats['mean'])} bands; "
            f"valid pixel counts={stats['valid_pixels']}"
        )

    loaders = build_loaders(config, normalizer)
    for split, loader in loaders.items():
        inspect_batch(split, loader, config, statistics)

    print("\nData-pipeline checks passed.")
    if config["preprocessing"].get("crop_size"):
        print(
            "Note: crop_size is configured but not applied yet. "
            "No cropping, model forward, training or evaluation is performed."
        )


if __name__ == "__main__":
    main()
