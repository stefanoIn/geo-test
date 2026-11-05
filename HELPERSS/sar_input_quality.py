from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from segmentation_metrics import binary_metrics, confusion_counts, metric_table

def audit_sar_pairs(pairs):
    rows = []
    for x_path, y_path in pairs:
        with rasterio.open(x_path) as src:
            raw = src.read()
            usable = np.isfinite(raw).all(axis=0) & (src.read_masks() > 0).all(axis=0)
        with rasterio.open(y_path) as src:
            label = src.read(1)
        if label.shape != usable.shape:
            raise ValueError(f"Input/label shape mismatch: {x_path}")
        valid = np.isin(label, [0, 1])
        rows.append(dict(
            chip_id=Path(x_path).stem.replace("_S1Hand", ""),
            image_path=str(x_path), label_path=str(y_path),
            sar_usable_pixels=int(usable.sum()),
            sar_usable_fraction=float(usable.mean()),
            valid_label_pixels=int(valid.sum()),
            labeled_usable_sar_pixels=int((usable & valid).sum()),
            exclude_missing_sar=not bool(usable.any())))
    return pd.DataFrame(rows)

def report_input_quality(pairs, prediction_dir):
    # Whole-image exclusion only; partially missing images retain all labelled pixels.
    prediction_dir = Path(prediction_dir)
    audit = audit_sar_pairs(pairs)
    audit.to_csv(prediction_dir / "input_quality_audit.csv", index=False)
    audit[audit.exclude_missing_sar].to_csv(prediction_dir / "excluded_missing_sar.csv", index=False)
    counts = np.zeros(4, dtype=np.int64)
    kept = 0
    for (x_path, y_path), exclude in zip(pairs, audit.exclude_missing_sar):
        if exclude:
            continue
        with rasterio.open(prediction_dir / (Path(x_path).stem + "_prediction.tif")) as src:
            pred = src.read(1)
        with rasterio.open(y_path) as src:
            target = src.read(1)
        counts += confusion_counts(pred, target)
        kept += 1
    metrics = dict(binary_metrics(*counts), included_images=kept,
                   excluded_missing_sar_images=int(audit.exclude_missing_sar.sum()))
    metric_table({"usable_sar_images":metrics}, prediction_dir / "usable_sar_images")
    return metrics
