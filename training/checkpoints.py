"""Save and restore model checkpoints without depending on the experiment runner."""

import os
import tempfile
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path,
    model: nn.Module,
    epoch: int,
    *,
    optimizer=None,
    scheduler=None,
    metrics=None,
    config=None,
    components=None,
):
    """Save a resumable checkpoint, or only selected model components.

    Args:
        components: None to save the whole model, or a list of module paths,
            e.g. ['head'] or ['adapter', 'head'], to avoid storing a frozen
            pretrained backbone. For partial checkpoints, instantiate the
            same pretrained model *first* before calling load_checkpoint.

    Saving optimizer/scheduler state is optional. The caller decides when a
    checkpoint is 'best' and which file to write; no ranking logic lives here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if components is None:
        model_state = {"model_state_dict": model.state_dict()}
        saved_components = None
    else:
        if isinstance(components, str) or not components:
            raise ValueError("components must be a non-empty list of module names")
        saved_components = list(components)
        if len(set(saved_components)) != len(saved_components):
            raise ValueError("Duplicate component names")
        model_state = {
            "component_state_dicts": {
                name: model.get_submodule(name).state_dict() for name in saved_components
            }
        }

    payload = {
        "epoch": int(epoch),
        "metrics": metrics,
        "config": config,
        "saved_components": saved_components,
        **model_state,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    # Atomic replacement prevents leaving a partially written checkpoint.
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def load_checkpoint(
    path,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    strict=True,
) -> dict:
    """Restore saved weights and optionally optimizer/scheduler state.

    Returns a small metadata dictionary (epoch, metrics, config, components).
    The model must already be instantiated with the same architecture.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "component_state_dicts" in checkpoint:
        for name, state in checkpoint["component_state_dicts"].items():
            model.get_submodule(name).load_state_dict(state, strict=strict)
    else:
        raise ValueError("Checkpoint contains no recognised model state")

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("This checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        if "scheduler_state_dict" not in checkpoint:
            raise ValueError("This checkpoint does not contain scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "config": checkpoint.get("config"),
        "saved_components": checkpoint.get("saved_components"),
    }
