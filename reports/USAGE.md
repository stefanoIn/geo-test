# Detailed segmentation evaluation

This update changes only:

- `src/evaluation/metrics.py`
- `src/evaluation/evaluator.py`

Leave `training/checkpoints.py`, `run.py`, and the encoders unchanged.

## Evaluate any compatible DataLoader

```python
from src.evaluation.evaluator import evaluate, save_evaluation_report

results = evaluate(
    model=model,
    dataloader=val_loader,
    device=device,
    num_classes=2,
    class_names=["non_water", "water"],
    criterion=criterion,
    ignore_index=255,
    tile_size=120,
    tile_batch_size=1,
    input_keys=["sar", "optical"],  # adjust to available input keys
    return_per_sample=True,
    sample_id_key="chip_id",  # optional; omit if dataset does not provide it
)

save_evaluation_report(
    "runs/my_experiment/validation_metrics.json",
    results,
    metadata={
        "split": "validation",
        "seed": 234,
        "experiment": "my_experiment",
        "checkpoint": "best.pt",
    },
)
```

**Compatibility:** The current Sen1Floods11 dataset returns `sar`, `optical`, and `target`, but not `chip_id`. Omit `sample_id_key` to use sequential `sample_000000` identifiers, or have the dataset return a stable identifier for reproducible per-chip reports. Do not pass metadata tensors to the model: set `input_keys` when necessary.

## Count conventions

- `confusion_matrix[true_class][predicted_class]`.
- `valid_pixels` = pixels whose target differs from `ignore_index`.
- `ignored_pixels` = pixels excluded by that index.
- `correct_pixels` + `incorrect_pixels` = `valid_pixels`.
- For binary classification, class **1** is the positive class. `binary_counts` contains TP, FP, FN, TN, and their sum equals `valid_pixels`.
- `tp_per_class`, `fp_per_class`, `fn_per_class`, `tn_per_class` are *one-vs-rest* counts for each class. Do not sum `tn_per_class` to obtain total correct pixels.
- `per_class` contains the same counts plus IoU, precision, recall, F1 and support.
- When a ratio has a zero denominator it is `null` in JSON; such values are excluded from macro averages.
- `per_sample` is optional; when enabled it includes the exact same count fields and metrics for each item in the loader, including all-ignored samples.

## Experiment-level logging

The evaluator never decides whether it is evaluating validation or test data, nor does it choose checkpoints. The caller controls the DataLoader and saves the report with metadata. When the experiment runner is implemented, it can call `evaluate()` each epoch, choose the best checkpoint, and save the metric history and split-level reports.
