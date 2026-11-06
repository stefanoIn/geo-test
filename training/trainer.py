"""One training epoch for dense prediction models.

The caller owns the epoch loop, model, loss, optimizer, and scheduler.
Expected batches: {"sar": ..., "optical": ..., "target": ...}, with either
or both modalities present. ``criterion`` should return a scalar mean loss;
per-epoch loss aggregation is designed for segmentation CrossEntropyLoss.
"""

import torch
from torch import nn


def train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch
       Returns the mean loss over valid target pixels.
    """
    model.train()
    ignore_index = getattr(criterion, "ignore_index", 255)
    total_loss = 0.0
    total_weight = 0.0

    for batch in train_loader:
        target = batch["target"].to(device)
        valid = target != ignore_index
        if not valid.any().item():
            continue

        images = {
            name: tensor.to(device)
            for name, tensor in batch.items()
            if name != "target"
        }

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
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
