"""Evaluate any compatible semantic-segmentation DataLoader; optionally save reports.

Input batches are dictionaries containing image tensors and a target [B,H,W].
This module has no knowledge of model architecture, dataset or split names.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn

from src.evaluation.metrics import SegmentationMetrics
from src.evaluation.sliding_window import sliding_window_predict


def _sample_ids(batch, sample_id_key, batch_size, start_index):
    if sample_id_key is None:
        return [f"sample_{start_index + idx:06d}" for idx in range(batch_size)]
    if sample_id_key not in batch:
        raise ValueError(f"Missing sample ID field {sample_id_key!r}")
    values = batch[sample_id_key]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) != batch_size:
        raise ValueError(f"{sample_id_key!r} must contain one value per sample")
    return [str(value) for value in values]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    device,
    num_classes: int,
    criterion=None,
    ignore_index: int = 255,
    tile_size=None,
    tile_batch_size: int = 8,
    stride=None,
    input_keys=None,
    target_key: str = "target",
    return_per_sample: bool = False,
    sample_id_key: str | None = None,
    class_names=None,
) -> dict:
    """Return global counts/metrics, loss, and optional per-sample counts/metrics.

    An image can be evaluated with direct inference (tile_size=None) or with
    sliding-window inference. Samples with all pixels ignored still appear in
    per-sample reports, but do not affect confusion-matrix counts or loss.

    The caller owns preprocessing, choosing validation vs test data, and saving.
    For weighted CrossEntropyLoss(mean), global loss is aggregated with target
    weights. For other criteria, it is a valid-pixel-weighted mean of batch
    losses, which need not equal the criterion applied to the whole dataset.

    `sample_id_key` may identify an optional metadata field in each batch, e.g.
    'chip_id'. When omitted, samples receive sequential IDs. Set input_keys
    explicitly if your batches have additional tensor-valued metadata.
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        if criterion.ignore_index != ignore_index:
            raise ValueError("criterion.ignore_index differs from evaluator.ignore_index")
        if criterion.reduction != "mean":
            raise ValueError("CrossEntropyLoss must use reduction='mean'")
    if isinstance(input_keys, str):
        raise TypeError("input_keys must be a sequence of input names, not a string")

    metrics = SegmentationMetrics(
        num_classes=num_classes, ignore_index=ignore_index, class_names=class_names
    )
    per_sample = [] if return_per_sample else None
    total_loss = 0.0
    total_loss_weight = 0.0
    next_sample_index = 0
    previous_mode = model.training
    model.eval()

    try:
        for batch in dataloader:
            if not isinstance(batch, dict) or target_key not in batch:
                raise ValueError(f"Each batch must be a dict containing {target_key!r}")

            targets = batch[target_key].to(device=device, dtype=torch.long)
            if targets.ndim != 3:
                raise ValueError("Segmentation target must have shape [B,H,W]")

            selected = (
                list(input_keys)
                if input_keys is not None
                else [
                    key for key, value in batch.items()
                    if key not in {target_key, sample_id_key} and isinstance(value, torch.Tensor)
                ]
            )
            if not selected:
                raise ValueError("No model input tensors in the batch")
            if target_key in selected or (sample_id_key is not None and sample_id_key in selected):
                raise ValueError("input_keys must contain only model inputs")

            images = {}
            for key in selected:
                if key not in batch or not isinstance(batch[key], torch.Tensor):
                    raise ValueError(f"Missing tensor input {key!r}")
                images[key] = batch[key].to(device)

            logits = (
                model(images)
                if tile_size is None
                else sliding_window_predict(
                    model, images, tile_size=tile_size, tile_batch_size=tile_batch_size, stride=stride
                )
            )
            expected = (targets.shape[0], num_classes, *targets.shape[-2:])
            if logits.shape != expected:
                raise ValueError(f"Model logits must be {expected}; got {tuple(logits.shape)}")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model produced non-finite logits")

            predictions = logits.argmax(dim=1)
            metrics.update(predictions, targets)

            if return_per_sample:
                ids = _sample_ids(batch, sample_id_key, targets.shape[0], next_sample_index)
                for idx, sample_id in enumerate(ids):
                    sample_metrics = SegmentationMetrics(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        class_names=class_names,
                    )
                    sample_metrics.update(predictions[idx], targets[idx])
                    per_sample.append({"sample_id": sample_id, **sample_metrics.compute()})

            next_sample_index += targets.shape[0]
            valid = targets != ignore_index
            if criterion is not None and bool(valid.any()):
                loss = criterion(logits, targets)
                if loss.ndim != 0 or not math.isfinite(float(loss.item())):
                    raise ValueError("Criterion must return a finite scalar loss")
                if isinstance(criterion, nn.CrossEntropyLoss):
                    if criterion.weight is not None:
                        loss_weight = float(
                            criterion.weight.to(targets.device)[targets[valid]].sum().item()
                        )
                    else:
                        loss_weight = float(valid.sum().item())
                else:
                    loss_weight = float(valid.sum().item())
                if loss_weight > 0:
                    total_loss += float(loss.item()) * loss_weight
                    total_loss_weight += loss_weight
    finally:
        model.train(previous_mode)

    result = metrics.compute()
    if result["valid_pixels"] == 0:
        raise ValueError("No valid labelled pixels were found in the supplied data")
    result["loss"] = (
        total_loss / total_loss_weight if criterion is not None and total_loss_weight > 0
        else None
    )
    result["num_samples"] = next_sample_index
    if return_per_sample:
        result["per_sample"] = per_sample
    return result


def save_evaluation_report(path, results: dict, metadata: dict | None = None) -> Path:
    """Write JSON report (e.g. split, seed and checkpoint name in metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(results)
    if metadata is not None:
        report["metadata"] = metadata
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    return path
