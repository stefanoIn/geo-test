"""Save per-chip label, prediction and error figures without changing metrics."""

from pathlib import Path
import re

import numpy as np


def save_prediction_plot(directory, sample_id, target, prediction, num_classes,
                         ignore_index=255, class_names=None, index=0):
    """Write one PNG; ignore pixels are excluded from the error map."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = target.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    valid = target != ignore_index
    names = class_names or [f"Class {i}" for i in range(num_classes)]
    colors = (["#263440", "#20b9d2"] if num_classes == 2 else
              [matplotlib.colormaps["tab20"](i % 20) for i in range(num_classes)])
    cmap = ListedColormap(colors)
    cmap.set_bad("#c5c7cb")
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)
    # Binary errors distinguish false positives and false negatives.
    errors = np.zeros(target.shape, dtype=np.uint8)
    if num_classes == 2:
        errors[valid & (prediction == 1) & (target == 0)] = 1
        errors[valid & (prediction == 0) & (target == 1)] = 2
        error_colors = ["#e5e9ed", "#f2a444", "#e55764", "#c5c7cb"]
        error_names = ["Correct", "False positive", "False negative", "Ignored"]
    else:
        errors[valid & (prediction != target)] = 1
        error_colors = ["#e5e9ed", "#e55764", "#c5c7cb"]
        error_names = ["Correct", "Incorrect", "Ignored"]
    errors[~valid] = len(error_colors) - 1
    figure = Figure(figsize=(12, 4.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for ax, array, title in zip(axes[:2], [target, prediction], ["Ground truth", "Prediction"]):
        ax.imshow(np.ma.array(array, mask=~valid), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
    axes[2].imshow(errors, cmap=ListedColormap(error_colors), vmin=0,
                   vmax=len(error_colors)-1, interpolation="nearest")
    axes[2].set_title("Errors (valid labels only)")
    for ax in axes:
        ax.set_axis_off()
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, names)] +
                   [Patch(color="#c5c7cb", label="Ignored")], loc="upper center",
                   bbox_to_anchor=(0.5, 0), fontsize=8, ncol=2)
    axes[2].legend(handles=[Patch(color=c, label=n) for c, n in zip(error_colors, error_names)],
                   loc="upper center", bbox_to_anchor=(0.5, 0), fontsize=8, ncol=2)
    figure.suptitle(str(sample_id) + (" | No valid labels" if not valid.any() else ""))
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    # Index prevents filename collisions even after sanitizing external chip IDs.
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(sample_id))[:100]
    path = directory / f"{index:06d}_{safe_id}.png"
    figure.savefig(path, dpi=150)
    figure.clear()
    return path
