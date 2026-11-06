import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "data" / "sen1floods11"
HAND_DIR = DATASET_ROOT / "HandLabeled"
SPLITS_DIR = DATASET_ROOT / "splits"


def load_split_paths(split_csv, modalities=("sar", "optical")):
    """Load georeferenced chip paths for the selected modalities and labels."""
    modalities = tuple(modalities)
    if not modalities or set(modalities) - {"sar", "optical"} or len(set(modalities)) != len(modalities):
        raise ValueError(f"Invalid modalities: {modalities}")
    split_csv = Path(split_csv)
    if not split_csv.is_absolute():
        split_csv = PROJECT_ROOT / split_csv
    if not split_csv.is_file():
        raise FileNotFoundError(split_csv)

    samples = []
    seen_chips = set()
    with split_csv.open(newline="") as f:
        for image_name, label_name in csv.reader(f):
            image_name = Path(image_name.strip()).name
            label_name = Path(label_name.strip()).name
            suffix = "_LabelHand.tif"
            if not label_name.endswith(suffix):
                raise ValueError(f"Unexpected label filename: {label_name}")
            chip_id = label_name.removesuffix(suffix)
            if image_name not in (f"{chip_id}_S1Hand.tif", f"{chip_id}_S2Hand.tif"):
                raise ValueError(f"Image/label mismatch: {image_name}, {label_name}")
            if chip_id in seen_chips:
                raise ValueError(f"Duplicate chip: {chip_id}")
            seen_chips.add(chip_id)

            sample = {"target": HAND_DIR / "LabelHand" / f"{chip_id}_LabelHand.tif"}
            if "sar" in modalities:
                sample["sar"] = HAND_DIR / "S1Hand" / f"{chip_id}_S1Hand.tif"
            if "optical" in modalities:
                sample["optical"] = HAND_DIR / "S2Hand" / f"{chip_id}_S2Hand.tif"
            for path in sample.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples found in {split_csv}")
    return samples


class Sen1Floods11Dataset(Dataset):
    """Lazy GeoTIFF loading; optional preprocessing; stable `chip_id` metadata."""

    def __init__(self, split_csv, modalities=("sar", "optical"), preprocess_func=None):
        super().__init__()
        self.modalities = tuple(modalities)
        self.preprocess_func = preprocess_func
        self.data_list = load_split_paths(split_csv, modalities=self.modalities)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        paths = self.data_list[idx]
        sample = {}
        for modality in self.modalities:
            with rasterio.open(paths[modality]) as src:
                image = src.read(masked=True).astype(np.float32).filled(np.nan)
            sample[modality] = torch.from_numpy(image)

        with rasterio.open(paths["target"]) as src:
            mask = src.read(1).astype(np.int64)
        mask[mask == -1] = 255
        sample["target"] = torch.from_numpy(mask)

        # Metadata, not an encoder input. Generic evaluator reads this key only
        # when instructed with sample_id_key='chip_id'.
        sample["chip_id"] = paths["target"].name.removesuffix("_LabelHand.tif")
        if self.preprocess_func is not None:
            sample = self.preprocess_func(sample)
        return sample
