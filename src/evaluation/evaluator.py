"""Evaluate a semantic segmentation model on any compatible DataLoader.

Batch contract: a mapping containing a target [B,H,W] and image tensors.
The evaluator does not know the dataset name, split name or encoder type.
"""

import math

import torch
from torch import nn

from src.evaluation.metrics import SegmentationMetrics
from src.evaluation.sliding_window import sliding_window_predict


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
    input_keys=None,
    target_key: str = "target",
) -> dict:
    """Evaluate over the batches received, with direct or tiled inference.

    Args:
        model: Receives a dictionary of image tensors and outputs [B,C,H,W] logits.
        dataloader: Any iterable yielding dictionaries of input tensors and a target.
        device: torch.device or string, e.g. 'cuda'.
        num_classes: Number of semantic classes (at least 2).
        criterion: Optional scalar-valued criterion, e.g. CrossEntropyLoss.
        ignore_index: Pixels excluded from metrics and loss aggregation.
        tile_size: None for a direct forward; integer or (height,width) for tiled
            inference using the existing sliding_window_predict function.
        tile_batch_size: Max tiles per model forward during tiled inference.
        input_keys: Optional names of inputs. By default, all tensor-valued
            batch entries except the target are passed to the model.
        target_key: Key containing the integer segmentation mask.

    Returns:
        Dictionary of aggregated segmentation metrics and 'loss' (None when
        no criterion is supplied). For weighted CrossEntropyLoss(mean), the
        reported loss is correctly aggregated by the sum of target weights.
        Other scalar per-batch losses are aggregated by valid-pixel count;
        non-additive losses such as Dice are not a global dataset-level Dice.

    Notes:
        Inputs must have been preprocessed by the caller. This function does
        not save results, choose checkpoints or distinguish validation/test.
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        if criterion.ignore_index != ignore_index:
            raise ValueError("criterion.ignore_index differs from evaluator.ignore_index")
        if criterion.reduction != "mean":
            raise ValueError("CrossEntropyLoss must use reduction='mean'")
    if isinstance(input_keys, str):
        raise TypeError("input_keys must be a sequence of input names, not a string")

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    total_loss_weight = 0.0
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
                else [k for k, v in batch.items() if k != target_key and isinstance(v, torch.Tensor)]
            )
            if not selected:
                raise ValueError("No model input tensors in the batch")
            if target_key in selected:
                raise ValueError("input_keys must not include the target")

            images = {}
            for key in selected:
                if key not in batch or not isinstance(batch[key], torch.Tensor):
                    raise ValueError(f"Missing tensor input {key!r}")
                images[key] = batch[key].to(device)

            logits = (
                model(images)
                if tile_size is None
                else sliding_window_predict(
                    model, images, tile_size=tile_size, tile_batch_size=tile_batch_size
                )
            )
            expected = (targets.shape[0], num_classes, *targets.shape[-2:])
            if logits.shape != expected:
                raise ValueError(f"Model logits must be {expected}; got {tuple(logits.shape)}")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model produced non-finite logits")

            predictions = logits.argmax(dim=1)
            metrics.update(predictions, targets)

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
    return result
