"""One training epoch for dense prediction models.

The caller owns the epoch loop, model, loss, optimizer, and scheduler.
Expected batches: image tensors [B,C,H,W], a target [B,H,W], and
optional metadata such as chip_id. ``criterion`` should return a scalar mean loss;
per-epoch loss aggregation is designed for segmentation CrossEntropyLoss.
"""

import torch
from torch import nn


def train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=None, input_keys=None, batch_observer=None):
    """Train for one epoch; return the mean loss over valid target pixels.

    ``scheduler``, when provided, is stepped after each optimizer update,
    matching the previous CROMA/TerraMind notebook training loops.
    """
    if isinstance(input_keys, str):
        raise TypeError("input_keys must be a sequence of names, not a string")

    model.train()
    ignore_index = getattr(criterion, "ignore_index", 255)
    total_loss = 0.0
    total_weight = 0.0

    for batch in train_loader:
        target = batch["target"].to(device)
        valid = target != ignore_index
        if not valid.any().item():
            continue

        # Only image tensors belong to the model input. Metadata such as
        # chip_id (a list of strings) stays in the batch, on the CPU.
        keys = (
            list(input_keys)
            if input_keys is not None
            else [
                name for name, tensor in batch.items()
                if name != "target"
                and isinstance(tensor, torch.Tensor)
                and tensor.ndim == 4
            ]
        )
        if not keys or "target" in keys:
            raise ValueError("input_keys must specify at least one image tensor")
        images = {}
        for name in keys:
            tensor = batch.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name}: expected batched image tensor [B,C,H,W]")
            images[name] = tensor.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if batch_observer is not None:
            batch_observer(images, target, logits)
            batch_observer = None
        loss = criterion(logits, target)

        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError("Training loss must be a finite scalar")

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # For weighted CrossEntropyLoss(reduction="mean"), the denominator
        # is the sum of the weights of valid target pixels, not their count.
        if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
            batch_weight = criterion.weight[target[valid]].sum().item()
        else:
            batch_weight = valid.sum().item()

        total_loss += loss.item() * batch_weight
        total_weight += batch_weight

    if total_weight == 0:
        raise RuntimeError("No training batches contain valid target pixels")

    return total_loss / total_weight
