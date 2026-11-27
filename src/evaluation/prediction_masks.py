"""Lossless export of the integer masks used by segmentation metrics."""

import hashlib
from pathlib import Path
import re

import numpy as np


def save_prediction_masks(directory, sample_id, target, prediction):
    """Save one sample without remapping labels, resizing, or overwriting files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sample_id = str(sample_id)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", sample_id)[:80] or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    path = directory / f"{safe_id}_{digest}.npz"
    # Exclusive creation detects repeated IDs and filename collisions.
    stream = path.open("xb")
    try:
        with stream:
            np.savez_compressed(
                stream,
                sample_id=np.asarray(sample_id),
                prediction=prediction.detach().cpu().numpy(),
                target=target.detach().cpu().numpy(),
            )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path
