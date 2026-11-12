"""Inspect logged training crops without training or model inference.

From GeoTest:
python experiments/debug/analyze_crop_inputs.py --run-dir runs/.../RUN --epochs 7 8 9
Outputs: RUN/debug/input_analysis/ (statistics and visual crop comparisons).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.data.datasets.sen1floods11 import HAND_DIR
from src.data.preprocessing import NormalizeFromStats


def read_crop(record, folder, masked=True):
    c = record["crop"]
    path = HAND_DIR / folder / f"{record['chip_id']}_{folder}.tif"
    with rasterio.open(path) as src:
        array = src.read(window=rasterio.windows.Window(
            c["left"], c["top"], c["width"], c["height"]), masked=masked)
    return array.astype(np.float32).filled(np.nan) if masked else array


def analyze(run_dir, epochs):
    normalizer = NormalizeFromStats(**json.loads((run_dir / "normalization.json").read_text()))
    config = __import__("yaml").safe_load((run_dir / "config.yaml").read_text())
    rows = [json.loads(line) for line in (run_dir / "debug/training_crops.jsonl").read_text().splitlines()]
    available = {r["epoch"] for r in rows}
    if set(epochs) - available:
        raise ValueError(f"Epochs missing from crop log: {set(epochs) - available}")
    records = []
    for r in rows:
        if r["epoch"] not in epochs:
            continue
        mask = read_crop(r, "LabelHand", masked=False)[0].astype(np.int64)
        mask[mask == -1] = 255
        valid = mask != 255
        assert int(valid.sum()) == r["valid_pixels"], f"Label count changed: {r['chip_id']}"
        labels, counts = np.unique(mask[valid], return_counts=True)
        assert {str(int(k)): int(v) for k, v in zip(labels, counts)} == r["class_pixel_counts"]
        raw = {m: read_crop(r, "S1Hand" if m == "sar" else "S2Hand")
               for m in config["dataset"]["modalities"]}
        normalized = normalizer({m: torch.from_numpy(a) for m, a in raw.items()})
        row = dict(r)
        for m, a in raw.items():
            z = normalized[m].numpy()
            finite = np.isfinite(a)
            row[m] = {
                "missing_fraction": float((~finite).mean()),
                "zero_raw_fraction": float((a == 0).mean()),
                "abs_z_max": float(np.abs(z).max()),
                "abs_z_gt5_fraction": float((np.abs(z) > 5).mean()),
                "band_means": z.mean(axis=(1, 2)).tolist(),
                "band_stds": z.std(axis=(1, 2)).tolist(),
                "band_absmax": np.abs(z).max(axis=(1, 2)).tolist(),
            }
        records.append(row)
        if len(records) % 126 == 0:
            print(f"Inspected {len(records)} crops", flush=True)
    return records


def save_outputs(run_dir, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output = run_dir / "debug/input_analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "crop_statistics.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    epochs = sorted({r["epoch"] for r in records})
    summary = []
    modalities = [m for m in ("sar", "optical") if m in records[0]]
    for e in epochs:
        rr = [r for r in records if r["epoch"] == e]
        for m in modalities:
            summary.append({"epoch": e, "modality": m, "crops": len(rr),
                "water_fraction_valid": sum(r["class_pixel_counts"].get("1", 0) for r in rr) / max(1, sum(r["valid_pixels"] for r in rr)),
                "max_abs_normalized": max(r[m]["abs_z_max"] for r in rr),
                "mean_fraction_abs_normalized_gt5": float(np.mean([r[m]["abs_z_gt5_fraction"] for r in rr])),
                "mean_missing_fraction": float(np.mean([r[m]["missing_fraction"] for r in rr]))})
    with (output / "epoch_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    fig, axes = plt.subplots(len(modalities), 1, figsize=(12, 3 * len(modalities)), squeeze=False)
    for ax, m in zip(axes[:, 0], modalities):
        for e in epochs:
            rr = [r for r in records if r["epoch"] == e]
            batches = sorted({r["batch"] for r in rr})
            ax.plot(batches, [max(r[m]["abs_z_max"] for r in rr if r["batch"] == b) for b in batches], label=f"Epoch {e}")
        ax.set(xlabel="Training batch", ylabel="Maximum absolute normalized value", title=m)
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "batch_extremes.png", dpi=150); plt.close(fig)
    cmap = ListedColormap(["#263440", "#20b9d2"]); cmap.set_bad("#c5c7cb")
    for e in epochs:
        tail = [r for r in records if r["epoch"] == e][-6:]
        fig, axes = plt.subplots(len(tail), 3, figsize=(10, 3 * len(tail)), squeeze=False)
        for axs, r in zip(axes, tail):
            rgb = read_crop(r, "S2Hand")[[3, 2, 1]].transpose(1, 2, 0)
            axs[0].imshow(np.nan_to_num(np.clip(rgb / 3000, 0, 1)))
            axs[0].set_title(f"{r['chip_id']} | batch {r['batch']}\nRGB /3000 (display only)", fontsize=9)
            vh = read_crop(r, "S1Hand")[1]
            axs[1].imshow(vh, cmap="gray", vmin=-30, vmax=0)
            axs[1].set_title("VH: -30 to 0 dB")
            mask = read_crop(r, "LabelHand", masked=False)[0]
            axs[2].imshow(np.ma.masked_where((mask == -1) | (mask == 255), mask), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            c = r["crop"]
            axs[2].set_title(f"Label | top={c['top']}, left={c['left']}\nWater=cyan; land=dark; ignored=grey", fontsize=9)
            for ax in axs: ax.axis("off")
        fig.suptitle(f"Epoch {e}: final three training batches (exact logged crops)")
        fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(output / f"epoch_{e:02d}_last_batches.png", dpi=140); plt.close(fig)
    print(f"Saved analysis to {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    os.chdir(PROJECT)
    torch.set_num_threads(2)
    save_outputs(run_dir, analyze(run_dir, args.epochs))
