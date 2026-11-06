"""Dataset-independent metrics for semantic segmentation.

Metrics accumulate a single confusion matrix across all samples and batches.
Rows are ground-truth classes; columns are predicted classes.
Ignored ground-truth pixels do not contribute.
"""

import torch


class SegmentationMetrics:
    """Accumulate pixel-level segmentation metrics for an arbitrary number of classes.

    An undefined class metric (zero denominator) is returned as None, and is
    excluded from macro averages. This makes the results JSON-compatible.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update counts using integer predicted/target class labels of equal shape."""
        if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("predictions and targets must be PyTorch tensors")
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if predictions.dtype.is_floating_point or targets.dtype.is_floating_point:
            raise TypeError("predictions and targets must contain integer class labels")

        valid = targets != self.ignore_index
        if not bool(valid.any()):
            return

        gt = targets[valid].long()
        pred = predictions[valid].long()
        if bool(((gt < 0) | (gt >= self.num_classes)).any()):
            raise ValueError("Target class outside [0, num_classes) or ignore_index")
        if bool(((pred < 0) | (pred >= self.num_classes)).any()):
            raise ValueError("Predicted class outside [0, num_classes)")

        counts = torch.bincount(
            gt * self.num_classes + pred,
            minlength=self.num_classes * self.num_classes,
        )
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict:
        """Return global/pixel-weighted metrics and a per-class breakdown."""
        cm = self.confusion_matrix
        tp = cm.diag()
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        fp = predicted - tp
        fn = support - tp
        union = support + predicted - tp

        def ratios(numerators, denominators):
            return [
                float(n) / float(d) if int(d) else None
                for n, d in zip(numerators.tolist(), denominators.tolist())
            ]

        def mean_defined(values):
            usable = [x for x in values if x is not None]
            return sum(usable) / len(usable) if usable else None

        iou = ratios(tp, union)
        precision = ratios(tp, predicted)
        recall = ratios(tp, support)
        f1 = ratios(2 * tp, 2 * tp + fp + fn)
        total = int(cm.sum().item())

        return {
            "miou": mean_defined(iou),
            "macro_f1": mean_defined(f1),
            "pixel_accuracy": float(tp.sum().item()) / total if total else None,
            "iou_per_class": iou,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support.tolist(),
            "valid_pixels": total,
            "confusion_matrix": cm.tolist(),
        }
