"""Visual inspection of Sen1Floods11 flood-segmentation predictions.

For predictions written by the CROMA test notebook, e.g.:
    results/croma_test_non_bol/Bolivia_103757_S1Hand_prediction.tif

The corresponding files are expected at:
    HandLabeled/LabelHand/Bolivia_103757_LabelHand.tif
    HandLabeled/S2Hand/Bolivia_103757_S2Hand.tif

Only numpy, rasterio, and matplotlib are required. This module does not
modify predictions or model preprocessing; Sentinel-2 scaling is for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject

PREDICTION_SUFFIX = "_S1Hand_prediction.tif"
IGNORE_VALUES = (-1, 255)


def _chip_id(prediction_path: Path) -> str:
    name = prediction_path.name
    if not name.endswith(PREDICTION_SUFFIX):
        raise ValueError(
            f"Unexpected prediction filename: {name!r}. "
            f"Expected *{PREDICTION_SUFFIX} (as saved by the test notebook)."
        )
    return name[: -len(PREDICTION_SUFFIX)]


def _same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.shape == b.shape
        and a.crs == b.crs
        and a.transform.almost_equals(b.transform, precision=1e-10)
    )


def _load_case(prediction_path: Path, hand_dir: Path, *, s2_scale: float):
    chip_id = _chip_id(prediction_path)
    label_path = hand_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"
    s2_path = hand_dir / "S2Hand" / f"{chip_id}_S2Hand.tif"

    for path in (prediction_path, label_path, s2_path):
        if not path.is_file():
            extra = (
                " If S2Hand is missing, download the Sen1Floods11 v1.1 "
                "HandLabeled/S2Hand folder."
                if path == s2_path
                else ""
            )
            raise FileNotFoundError(f"Missing raster: {path}.{extra}")

    if s2_scale <= 0:
        raise ValueError("s2_scale must be positive.")

    with rasterio.open(label_path) as label_src, rasterio.open(prediction_path) as pred_src:
        if not _same_grid(label_src, pred_src):
            raise ValueError(
                f"Prediction and label grids differ for {chip_id}. "
                "Check the saved prediction's CRS/transform and dimensions."
            )
        label = label_src.read(1).astype(np.int16)
        prediction = pred_src.read(1).astype(np.int16)
        label_grid = (label_src.crs, label_src.transform, label_src.height, label_src.width)

    label[np.isin(label, IGNORE_VALUES)] = 255
    if not np.isin(label, [0, 1, 255]).all():
        raise ValueError(f"Unexpected values in label for {chip_id}.")
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError(f"Expected binary 0/1 prediction for {chip_id}.")

    # Sen1Floods11 S2Hand contains B1, B2, B3, B4, ... B12 (13 bands).
    # rasterio band indexing starts at 1, so true-color RGB = bands (4, 3, 2).
    with rasterio.open(s2_path) as s2_src:
        if s2_src.count < 4:
            raise ValueError(f"Expected multi-band Sentinel-2 GeoTIFF: {s2_path}")
        rgb = s2_src.read((4, 3, 2), out_dtype="float32")
        label_crs, label_transform, height, width = label_grid
        is_aligned = (
            s2_src.shape == (height, width)
            and s2_src.crs == label_crs
            and s2_src.transform.almost_equals(label_transform, precision=1e-10)
        )
        if not is_aligned:
            if s2_src.crs is None or label_crs is None:
                raise ValueError(f"Cannot align S2 for {chip_id}: missing CRS.")
            aligned = np.zeros((3, height, width), dtype=np.float32)
            for b in range(3):
                reproject(
                    source=rgb[b],
                    destination=aligned[b],
                    src_transform=s2_src.transform,
                    src_crs=s2_src.crs,
                    dst_transform=label_transform,
                    dst_crs=label_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=s2_src.nodata,
                )
            rgb = aligned

    # S2 L1C TOA reflectance is commonly represented with factor 10,000.
    # Scale ~3000 for a brighter, easily inspected display, then clip to [0,1].
    # This affects visualization ONLY, not training/evaluation.
    rgb = np.moveaxis(rgb, 0, -1)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0) / s2_scale, 0, 1)

    valid = label != 255
    tp = np.count_nonzero(valid & (label == 1) & (prediction == 1))
    fp = np.count_nonzero(valid & (label == 0) & (prediction == 1))
    fn = np.count_nonzero(valid & (label == 1) & (prediction == 0))
    union = tp + fp + fn
    water_iou = tp / union if union else float("nan")

    return chip_id, rgb, label, prediction, water_iou


def plot_predictions(
    predictions_dir: str | Path,
    hand_dir: str | Path,
    *,
    n: int = 4,
    seed: int = 42,
    chip_ids: Sequence[str] | None = None,
    show_errors: bool = False,
    s2_scale: float = 3000.0,
    save_path: str | Path | None = None,
):
    """Show S2 RGB | ground truth | prediction for selected test chips.

    Args:
        predictions_dir: Directory containing *_S1Hand_prediction.tif files.
        hand_dir: Sen1Floods11 HandLabeled folder (contains LabelHand/S2Hand).
        n: Number of randomly selected chips if chip_ids is not provided.
        seed: Reproducible random selection seed.
        chip_ids: Explicit IDs, e.g. ['Bolivia_103757'], ignoring n.
        show_errors: Add a fourth panel (correct / false positive / false negative).
        s2_scale: S2 RGB display divisor, NOT a model normalization setting.
        save_path: Optional output path for a PNG/PDF/SVG figure.

    Returns:
        (fig, axes, selected_ids): Matplotlib figure, 2D axes, and IDs.
    """
    predictions_dir = Path(predictions_dir)
    hand_dir = Path(hand_dir)
    files = sorted(predictions_dir.glob(f"*{PREDICTION_SUFFIX}"))
    if not files:
        raise FileNotFoundError(
            f"No *{PREDICTION_SUFFIX} files in {predictions_dir}. "
            "Run test(..., save_predictions=True) first."
        )

    by_id = {_chip_id(p): p for p in files}
    if chip_ids is not None:
        selected_ids = list(chip_ids)
        if not selected_ids:
            raise ValueError("chip_ids cannot be empty.")
        missing = [chip for chip in selected_ids if chip not in by_id]
        if missing:
            raise FileNotFoundError(f"No predictions for chip IDs: {missing}")
    else:
        if not 1 <= n <= len(files):
            raise ValueError(f"n must be between 1 and {len(files)}.")
        rng = np.random.default_rng(seed)
        selected_ids = sorted(rng.choice(list(by_id), size=n, replace=False).tolist())

    count = len(selected_ids)
    cols = 4 if show_errors else 3
    fig, axes = plt.subplots(count, cols, figsize=(4.4 * cols, 4.0 * count), squeeze=False)
    mask_cmap = ListedColormap(["#29323f", "#20bde0", "#c4c7cd"])
    error_cmap = ListedColormap(["#2a3543", "#35b99d", "#f0ab50", "#e85d65", "#c4c7cd"])

    for row, chip_id in enumerate(selected_ids):
        _, rgb, label, pred, iou = _load_case(by_id[chip_id], hand_dir, s2_scale=s2_scale)
        # 0=nonwater, 1=water, 2=unlabeled, identical palette across masks.
        gt_show = np.where(label == 255, 2, label)
        pred_show = np.where(label == 255, 2, pred)  # do not show predictions as evidence on invalid pixels

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Sentinel-2 RGB (B4/B3/B2)", fontsize=11)
        axes[row, 1].imshow(gt_show, cmap=mask_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axes[row, 1].set_title("Ground truth", fontsize=11)
        axes[row, 2].imshow(pred_show, cmap=mask_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
        iou_text = f"{iou:.3f}" if np.isfinite(iou) else "undefined (no water union)"
        axes[row, 2].set_title(f"CROMA prediction · water IoU {iou_text}", fontsize=11)

        if show_errors:
            # Valid: TN=0, TP=1, FP=2, FN=3; invalid=4.
            errors = np.full(label.shape, 4, dtype=np.uint8)
            valid = label != 255
            errors[valid & (label == 0) & (pred == 0)] = 0
            errors[valid & (label == 1) & (pred == 1)] = 1
            errors[valid & (label == 0) & (pred == 1)] = 2
            errors[valid & (label == 1) & (pred == 0)] = 3
            axes[row, 3].imshow(errors, cmap=error_cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
            axes[row, 3].set_title("Error map", fontsize=11)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(chip_id, rotation=90, labelpad=10, fontsize=10)

    legend = [
        Patch(facecolor="#29323f", label="Non-water (0)"),
        Patch(facecolor="#20bde0", label="Water (1)"),
        Patch(facecolor="#c4c7cd", label="Ignore / invalid (255)"),
    ]
    if show_errors:
        legend += [
            Patch(facecolor="#f0ab50", label="False positive"),
            Patch(facecolor="#e85d65", label="False negative"),
        ]
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1.8, w_pad=1.0)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, axes, selected_ids
