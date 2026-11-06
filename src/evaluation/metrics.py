"""Dataset-agnostic pixel counts and metrics for semantic segmentation.

Rows of the confusion matrix are ground-truth classes; columns are predictions.
Ignored ground-truth pixels do not contribute to the confusion matrix.
"""

import torch


class SegmentationMetrics:
    """Accumulate integer counts across batches, then derive segmentation metrics.

    Undefined ratios are returned as None and excluded from macro averages.
    For binary segmentation, class 1 is treated as the positive class.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, class_names=None):
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if class_names is not None:
            if len(class_names) != num_classes or any(not isinstance(n, str) for n in class_names):
                raise ValueError("class_names must contain one string per class")
            class_names = list(class_names)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        self.total_pixels = 0

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Add class-label predictions and targets with identical spatial shapes."""
        if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("predictions and targets must be PyTorch tensors")
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if predictions.device != targets.device:
            raise ValueError("predictions and targets must be on the same device")
        if predictions.dtype.is_floating_point or targets.dtype.is_floating_point:
            raise TypeError("predictions and targets must contain integer class labels")

        valid = targets != self.ignore_index
        self.total_pixels += targets.numel()
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
        """Return raw counts and metrics, all in JSON-compatible Python types."""
        cm = self.confusion_matrix
        tp = cm.diag()
        support = cm.sum(dim=1)  # Actual pixels per class.
        predicted = cm.sum(dim=0)  # Predicted pixels per class.
        fp = predicted - tp
        fn = support - tp
        valid_pixels = int(cm.sum().item())
        tn = valid_pixels - tp - fp - fn  # One-vs-rest count for EACH class.
        union = tp + fp + fn

        def ratios(numerators, denominators):
            return [
                float(n) / float(d) if int(d) else None
                for n, d in zip(numerators.tolist(), denominators.tolist())
            ]

        def mean_defined(values):
            usable = [value for value in values if value is not None]
            return sum(usable) / len(usable) if usable else None

        iou = ratios(tp, union)
        precision = ratios(tp, predicted)
        recall = ratios(tp, support)
        f1 = ratios(2 * tp, 2 * tp + fp + fn)
        correct = int(tp.sum().item())
        incorrect = valid_pixels - correct
        per_class = [
            {
                "class_id": idx,
                "class_name": self.class_names[idx] if self.class_names is not None else str(idx),
                "tp": int(tp[idx]),
                "fp": int(fp[idx]),
                "fn": int(fn[idx]),
                "tn": int(tn[idx]),
                "support": int(support[idx]),
                "predicted": int(predicted[idx]),
                "iou": iou[idx],
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
            }
            for idx in range(self.num_classes)
        ]
        result = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "total_pixels": self.total_pixels,
            "valid_pixels": valid_pixels,
            "ignored_pixels": self.total_pixels - valid_pixels,
            "correct_pixels": correct,
            "incorrect_pixels": incorrect,
            "pixel_accuracy": correct / valid_pixels if valid_pixels else None,
            "miou": mean_defined(iou),
            "macro_f1": mean_defined(f1),
            "iou_per_class": iou,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support.tolist(),
            "predicted_per_class": predicted.tolist(),
            "tp_per_class": tp.tolist(),
            "fp_per_class": fp.tolist(),
            "fn_per_class": fn.tolist(),
            "tn_per_class": tn.tolist(),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
        if self.num_classes == 2:
            # In a binary task, class 1 is positive, class 0 negative.
            # These 4 values partition valid pixels exactly once.
            result["binary_counts"] = {
                "positive_class": 1,
                "negative_class": 0,
                "tp": int(cm[1, 1]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tn": int(cm[0, 0]),
            }
        return result
