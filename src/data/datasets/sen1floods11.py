import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Run commands from the project folder; paths are relative to that folder.
DATASET_ROOT = Path("data/sen1floods11")
HAND_DIR = DATASET_ROOT / "HandLabeled"
SPLITS_DIR = DATASET_ROOT / "splits"


def load_split_paths(split_csv, modalities=("sar", "optical")):
    """Load georeferenced chip paths for the selected modalities and labels."""
    modalities = tuple(modalities)
    if not modalities or set(modalities) - {"sar", "optical"} or len(set(modalities)) != len(modalities):
        raise ValueError(f"Invalid modalities: {modalities}")
    split_csv = Path(split_csv)
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

    def __init__(self, split_csv, modalities=("sar", "optical"), preprocess_func=None,
                 include_permanent_water=False):
        super().__init__()
        self.modalities = tuple(modalities)
        self.preprocess_func = preprocess_func
        self.data_list = load_split_paths(split_csv, modalities=self.modalities)
        self.excluded_all_ignore_chips = []
        if include_permanent_water:
            for paths in self.data_list:
                chip_id = paths["target"].name.removesuffix("_LabelHand.tif")
                reference = HAND_DIR / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif"
                if not reference.is_file():
                    raise FileNotFoundError(f"AW/PW/FW requires {reference}")
                paths["permanent_water"] = reference

    def exclude_all_ignore_chips(self):
        """Filter training chips only; retain every chip with any valid label."""
        retained = []
        excluded = []
        for paths in self.data_list:
            with rasterio.open(paths["target"]) as src:
                mask = src.read(1)
            if ((mask != -1) & (mask != 255)).any():
                retained.append(paths)
            else:
                excluded.append(paths["target"].name.removesuffix("_LabelHand.tif"))
        if not retained:
            raise ValueError("No training chips contain valid target pixels")
        self.data_list = retained
        self.excluded_all_ignore_chips.extend(excluded)

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
            if "permanent_water" in paths:
                with rasterio.open(paths["permanent_water"]) as reference:
                    if (reference.shape != src.shape or reference.crs != src.crs
                            or not reference.transform.almost_equals(src.transform, precision=1e-10)):
                        raise ValueError(f"JRC/label grid mismatch: {paths['permanent_water']}")
                    permanent = reference.read(1).astype(np.int64)
                if not np.isin(permanent, [0, 1]).all():
                    raise ValueError(f"Expected binary JRC mask: {paths['permanent_water']}")
                sample["permanent_water"] = torch.from_numpy(permanent)
        mask[mask == -1] = 255
        sample["target"] = torch.from_numpy(mask)

        # Metadata, not an encoder input. Generic evaluator reads this key only
        # when instructed with sample_id_key='chip_id'.
        sample["chip_id"] = paths["target"].name.removesuffix("_LabelHand.tif")
        if self.preprocess_func is not None:
            sample = self.preprocess_func(sample)
        return sample
