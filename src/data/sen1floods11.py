import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# ============================================================
# DATASET PATHS
# ============================================================

# Assumes this file is located at:
# GeoTest/src/data/sen1floods11.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "sen1floods11"

HAND_DIR = DATASET_ROOT / "HandLabeled"

SPLITS_DIR = DATASET_ROOT / "splits" 

# ===========================================================
# SPLIT LOADING
# ============================================================

def load_split_paths(split_csv, modalities=("sar", "optical")):
    """
    Read a Sen1Floods11 split CSV and build the paths
    to the requested modalities and their ground-truth mask.

    Args:
        split_csv: Path to the split CSV.
        modalities: Tuple containing "sar", "optical", or both.

    Returns:
        List of dictionaries containing GeoTIFF paths.
    """

    modalities = tuple(modalities)

    if (
        not modalities
        or set(modalities) - {"sar", "optical"}
        or len(modalities) != len(set(modalities))
    ):
        raise ValueError(f"Invalid modalities: {modalities}")

    split_csv = Path(split_csv)

    # Resolve relative paths from the project root.
    if not split_csv.is_absolute():
        split_csv = PROJECT_ROOT / split_csv

    if not split_csv.is_file():
        raise FileNotFoundError(split_csv)

    samples = []
    seen_chips = set()

    with open(split_csv, newline="") as f:

        for image_name, label_name in csv.reader(f):

            image_name = Path(image_name.strip()).name
            label_name = Path(label_name.strip()).name

            # Identify the chip using its label filename.
            suffix = "_LabelHand.tif"

            if not label_name.endswith(suffix):
                raise ValueError(
                    f"Unexpected label filename: {label_name}"
                )

            chip_id = label_name.removesuffix(suffix)

            # Verify the image and mask refer to the same chip.
            valid_image_names = (
                f"{chip_id}_S1Hand.tif",
                f"{chip_id}_S2Hand.tif",
            )

            if image_name not in valid_image_names:
                raise ValueError(
                    f"Image/label mismatch: {image_name}, {label_name}"
                )

            if chip_id in seen_chips:
                raise ValueError(f"Duplicate chip: {chip_id}")

            seen_chips.add(chip_id)

            # Construct paths from the common chip identifier.
            sample = {
                "target": (
                    HAND_DIR / "LabelHand" /
                    f"{chip_id}_LabelHand.tif"
                )
            }

            if "sar" in modalities:
                sample["sar"] = (
                    HAND_DIR / "S1Hand" /
                    f"{chip_id}_S1Hand.tif"
                )

            if "optical" in modalities:
                sample["optical"] = (
                    HAND_DIR / "S2Hand" /
                    f"{chip_id}_S2Hand.tif"
                )

            # Verify that every requested GeoTIFF exists.
            for path in sample.values():
                if not path.is_file():
                    raise FileNotFoundError(path)

            samples.append(sample)

    if not samples:
        raise ValueError(f"No samples found in {split_csv}")

    return samples


# ============================================================
# PYTORCH DATASET
# ============================================================

class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 hand-labelled imagery.

    Supports:
        - Sentinel-1 (SAR)
        - Sentinel-2 (Optical)
        - Sentinel-1 + Sentinel-2

    Loads GeoTIFFs lazily and applies an optional
    preprocessing function.

    Expected preprocessing input/output:
        {
            "sar": Tensor,       # Optional [2, H, W]
            "optical": Tensor,   # Optional [13, H, W]
            "target": Tensor     # [H, W]
        }
    """

    def __init__(
        self,
        split_csv,
        modalities=("sar", "optical"),
        preprocess_func=None,
    ):
        super().__init__()

        self.modalities = tuple(modalities)
        self.preprocess_func = preprocess_func

        # Read the split and construct GeoTIFF paths.
        self.data_list = load_split_paths(
            split_csv=split_csv,
            modalities=self.modalities,
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        paths = self.data_list[idx]
        sample = {}

        # Load the requested satellite modalities.
        for modality in self.modalities:

            with rasterio.open(paths[modality]) as src:
                image = src.read().astype(np.float32)

            sample[modality] = torch.from_numpy(image)

        # Load the ground-truth segmentation mask.
        with rasterio.open(paths["target"]) as src:
            mask = src.read(1).astype(np.int64)

        # Map ignored labels to the configured convention.
        mask[mask == -1] = 255

        sample["target"] = torch.from_numpy(mask)

        # Apply model/experiment-specific preprocessing.
        if self.preprocess_func is not None:
            sample = self.preprocess_func(sample)

        return sample
