# Assemble one CROMA experiment using the GeoTest modules

A Python-first tutorial, 6 November 2026.

This tutorial builds one concrete experiment: **CROMA Large + FCN on Sen1Floods11 hand-labelled data**, using Sentinel-1 and Sentinel-2. You load settings from a YAML file in the first cell, then assemble each module explicitly in Python. No command-line arguments or automatic experiment runner are required.

The numbered sections are intended to be executed in order in a notebook in the project folder, or copied in order into a Python script in that folder. Optional alternatives at the end are not part of the main sequence. Use your existing environment containing PyTorch, torchvision, NumPy, rasterio, einops and the project's other dependencies. Data and weights must already be local. This document was written without launching training.

“All hand-labelled data” means use the **full hand-labelled training split**, not merge every labelled chip into training. Validation selects the checkpoint; test and Bolivia measure held-out performance. Training on all four splits would invalidate those evaluations.

## What you are assembling

```text
Sen1Floods11Dataset
  -> raw SAR, optical, target, chip_id
  -> normalization and synchronized crop
  -> DataLoader
  -> CROMAEncoder
  -> TokenToSpatial
  -> FCNHead
  -> DensePredictionModel: full-size class logits
  -> train_one_epoch / evaluate
  -> checkpoint and metric files
```

The encoder supplies representations, the adapter arranges them as a spatial grid, and the head learns water/non-water predictions. The trainer optimizes this composed model; it does not need CROMA-specific code.

## 1. Read the experiment YAML and import the modules

Run with the project folder as your current working directory. All paths below are relative; no machine-specific root path is needed.

```python
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json
import random

import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.statistics import get_or_compute_stats
from src.data.preprocessing import NormalizeFromStats
from src.data.transforms import random_crop
from src.encoders.croma.croma import CROMAEncoder
from src.adapters.token_to_spatial import TokenToSpatial
from src.heads.fcn import FCNHead
from src.models.dense_prediction import DensePredictionModel
from src.evaluation.evaluator import evaluate, save_evaluation_report
from training.trainer import train_one_epoch
from training.checkpoints import save_checkpoint, load_checkpoint

config_path = Path("configs/croma/croma_s1s2_fcn_all_hand_labels.yaml")
with config_path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

# These aliases make the module calls below easier to read.
data_cfg = config["dataset"]
model_cfg = config["model"]
train_cfg = config["training"]
eval_cfg = config["evaluation"]
preprocess_cfg = config["preprocessing"]
loss_cfg = train_cfg["loss"]
optimizer_cfg = train_cfg["optimizer"]

if data_cfg["name"] != "sen1floods11" or model_cfg["name"] != "croma":
    raise ValueError("This walkthrough assembles CROMA on Sen1Floods11")
if config["head"]["name"] != "fcn" or config["adapter"]["name"] != "token_to_spatial":
    raise ValueError("This walkthrough uses TokenToSpatial and FCN")
if config["head"]["num_classes"] != 2 or loss_cfg["ignore_index"] != 255:
    raise ValueError("This walkthrough uses binary labels with ignore_index=255")

seed = config["experiment"]["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modalities = tuple(data_cfg["modalities"])
croma_modality = model_cfg["modality"]
expected = {"SAR": {"sar"}, "optical": {"optical"}, "both": {"sar", "optical"}}
if set(modalities) != expected[croma_modality]:
    raise ValueError("Dataset and encoder modalities disagree")
pretrained_path = Path(model_cfg["checkpoint"])
crop_size = preprocess_cfg["crop_size"]
batch_size = train_cfg["batch_size"]
epochs = train_cfg["epochs"]
validate_every = train_cfg.get("validate_every", 5)
learning_rate = optimizer_cfg["learning_rate"]
weight_decay = optimizer_cfg["weight_decay"]
freeze_encoder = model_cfg["freeze_encoder"]
class_weights = loss_cfg.get("class_weights")
ignore_index = loss_cfg["ignore_index"]
if model_cfg["image_resolution"] != crop_size or eval_cfg["tile_size"] != crop_size:
    raise ValueError("Keep CROMA image_resolution, crop_size and tile_size equal here")

split_paths = {
    split: Path(data_cfg[f"{split}_csv"])
    for split in ("train", "val", "test", "bolivia")
}
assert pretrained_path.is_file(), pretrained_path
assert all(path.is_file() for path in split_paths.values())
print("Device:", device)
```

The YAML is now the source of experiment settings. To change the learning rate, batch size or loss weighting, edit the YAML and rerun the construction cells; changing a file does not update already-created Python objects. Add `training.validate_every: 5` explicitly if desired (this tutorial defaults to 5).

Check `model.checkpoint` before starting: the inspected YAML uses `pretrained_models/croma/CROMA_large.pt`, while the observed local file is `pretrained_models/CROMA_large.pt`. Correct that YAML path to your actual relative file location. This document does not modify the YAML.

The existing settings are not a claim of optimal hyperparameters or exact reproduction of an older run. batch_size and class_weights are scientific choices. Seeds help repeatability but do not guarantee deterministic CUDA execution.

## 2. Load the raw datasets and check the splits

`Sen1Floods11Dataset` needs a split CSV, selected modalities and optionally a preprocessing callable. At this stage leave preprocessing off.

```python
raw_datasets = {
    split: Sen1Floods11Dataset(
        split_csv=path,
        modalities=modalities,
        preprocess_func=None,
    )
    for split, path in split_paths.items()
}

seen = {}
for split, dataset in raw_datasets.items():
    print(split, len(dataset))
    for paths in dataset.data_list:
        chip = paths["target"].name
        if chip in seen:
            raise ValueError(f"Chip appears in {seen[chip]} and {split}: {chip}")
        seen[chip] = split

sample = raw_datasets["train"][0]
print(sample["chip_id"])
print(sample["sar"].shape, sample["optical"].shape, sample["target"].shape)
```

The loader returns image tensors [channels,height,width], target [height,width] and a string chip_id. Labels -1 become 255. Raster nodata becomes NaN for preprocessing to handle. S1 and S2 filenames are paired by chip identifier. Matching dimensions alone do not prove acquisition dates or geospatial alignment.

Changing `split_csv` selects a custom label budget or geographic split. It does not move imagery: the current loader reads from data/sen1floods11/HandLabeled.

## 3. Compute normalization statistics from training images only

For this CROMA example, use the existing training-derived two-sigma stretch. The statistics module computes each modality independently and reuses its cache when possible.

```python
def raw_training_dataset(selected_modalities):
    return Sen1Floods11Dataset(
        split_csv=split_paths["train"],
        modalities=selected_modalities,
        preprocess_func=None,
    )

statistics = None
if preprocess_cfg["normalization"]["statistics"] == "training_dataset":
    statistics = get_or_compute_stats(
        dataset_factory=raw_training_dataset,
        modalities=modalities,
        split_csv=split_paths["train"],
        dataset_name=data_cfg["name"],
        cache_dir="data/statistics",
    )
normalizer = NormalizeFromStats.from_config(config, training_statistics=statistics)

```

CROMA optical input uses 12 bands, so this removes B10 from the 13-band Sen1Floods11 L1C tensor. Removing a band does not convert L1C to L2A. The YAML supplies band_indices, normalization method and use_8_bit. from_config applies those settings; pretraining_fixed is currently supported for TerraMind, not CROMA. The stretch computes clip((x-(mean-2*std))/(4*std),0,1). Statistics are calculated from finite training-image values, not validation/test data. Invalid image pixels are filled with zero after normalization; the target mask is not automatically changed.

`NormalizeFromStats` parameters:

| Parameter | Meaning |
|---|---|
| statistics | Dictionary keyed by sar/optical, each containing mean and std arrays |
| method | two_sigma_stretch or zscore |
| band_indices | Zero-based band positions; selects/reorders image bands and corresponding full-band statistics |
| use_8_bit | Quantize the stretched result to 256 levels; only valid with two_sigma_stretch |

## 4. Define training preprocessing and create DataLoaders

Training uses a random crop; validation/test retain complete chips. The same crop is applied to SAR, optical and labels.

```python
class TrainingPreprocess:
    def __init__(self, normalizer, crop_size):
        self.normalizer = normalizer
        self.crop_size = crop_size

    def __call__(self, sample):
        sample = random_crop(sample, size=self.crop_size)
        return self.normalizer(sample)

train_dataset = Sen1Floods11Dataset(
    split_csv=split_paths["train"],
    modalities=modalities,
    preprocess_func=TrainingPreprocess(normalizer, crop_size),
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_cfg.get("num_workers", 0),
)

evaluation_loaders = {
    split: DataLoader(
        Sen1Floods11Dataset(
            split_csv=split_paths[split],
            modalities=modalities,
            preprocess_func=normalizer,
        ),
        batch_size=eval_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )
    for split in ("val", "test", "bolivia")
}
```

Keep training.num_workers: 0 for this sequential Windows notebook tutorial; multiple workers may need script-level setup and a main guard. An epoch visits all training chips once, sampling one crop from each. This is not every pixel of every chip per epoch. No flips or rotations are added by this example.

## 5. Create the encoder

```python
encoder = CROMAEncoder(
    checkpoint_path=pretrained_path,
    size=model_cfg["size"],
    modality=croma_modality,
    image_resolution=model_cfg["image_resolution"],
    freeze=freeze_encoder,
)
```

| Parameter | Options / consequence |
|---|---|
| checkpoint_path | Matching local pretrained checkpoint |
| size | large (1024-dimensional features) or base (768), requiring matching weights |
| modality | SAR, optical or both; case-sensitive |
| image_resolution | Spatial input size; keep consistent with training crop and inference tile |
| freeze | True disables backbone gradients and keeps the backbone in eval mode |

The wrapper returns a dictionary with tokens [B,N,D] and pooled [B,D]. For both, tokens are the joint CROMA representation. Segmentation uses tokens, not pooled. CROMA has 8-pixel patches: a 120x120 input gives a 15x15 spatial token grid.

## 6. Create the adapter and head, then assemble the model

```python
adapter = TokenToSpatial(
    feature_key=config["adapter"]["feature_key"],
    grid_size=config["adapter"].get("grid_size"),
)
head = FCNHead(
    in_channels=config["head"]["in_channels"],
    num_classes=config["head"]["num_classes"],
)
model = DensePredictionModel(encoder=encoder, adapter=adapter, head=head)
model = model.to(device)

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
```

`TokenToSpatial` reshapes tokens into [B,1024,15,15]. grid_size=None infers a square grid; for rectangular inputs specify the actual grid as (height,width). It does not change feature width.

`FCNHead` converts spatial features into two class logits per spatial location, using GroupNorm. in_channels=1024 is feature width, not number of input spectral bands. `DensePredictionModel` interpolates the head output to input resolution, producing [B,2,120,120]. With encoders that pad inputs, it handles their padded_size metadata before cropping.

UPerNet is not implemented in the current head factory. It would require a suitable head and intermediate-feature interface, not just changing a name.

## 7. Create the loss, optimizer and optional scheduler

```python
if loss_cfg["name"] != "cross_entropy":
    raise ValueError("Add an explicit criterion implementation for this loss")

weights = None if class_weights is None else torch.tensor(
    class_weights, dtype=torch.float32, device=device
)
criterion = nn.CrossEntropyLoss(
    weight=weights, ignore_index=ignore_index, reduction="mean",
)
optimizer_types = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}
optimizer_name = optimizer_cfg["name"]
optimizer_options = {"lr": learning_rate, "weight_decay": weight_decay}
if optimizer_name == "sgd":
    optimizer_options["momentum"] = optimizer_cfg.get("momentum", 0.9)
optimizer = optimizer_types[optimizer_name](
    [p for p in model.parameters() if p.requires_grad], **optimizer_options
)

scheduler_cfg = train_cfg.get("scheduler") or {"name": "none"}
if scheduler_cfg["name"] == "none":
    scheduler = None
elif scheduler_cfg["name"] == "cosine_annealing_warm_restarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_cfg["t0_epochs"] * len(train_loader)),
        T_mult=scheduler_cfg.get("t_mult", 1),
        eta_min=scheduler_cfg.get("eta_min", 0.0),
    )
else:
    raise ValueError("Add the scheduler's construction and stepping policy explicitly")

```

Pass raw logits to CE, not softmax probabilities. Targets are integer class labels, not one-hot arrays. Keep ignore_index=255 with this dataset. Class weights [1,8] increase the water contribution; None removes weighting. Use reduction="mean" with the current trainer's aggregation.

To change LR, edit training.optimizer.learning_rate in YAML, reload it and rebuild the optimizer. To fine-tune the backbone, set freeze_encoder=False before constructing it, then rebuild the optimizer to include its trainable parameters. That changes both memory requirements and the experiment protocol.

## 8. Prepare the run folder and evaluation call

You decide what a run contains; checkpointing does not create an experiment protocol for you. This example keeps configuration choices, actual normalization values, checkpoints and metrics together.

```python
run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
run_dir = Path(config["output"]["directory"]) / run_id
run_dir.mkdir(parents=True, exist_ok=False)

settings = {
    "seed": seed,
    "modalities": list(modalities),
    "croma_modality": croma_modality,
    "pretrained_path": str(pretrained_path),
    "encoder_size": model_cfg["size"],
    "freeze_encoder": freeze_encoder,
    "crop_size": crop_size,
    "batch_size": batch_size,
    "epochs": epochs,
    "validate_every": validate_every,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "class_weights": class_weights,
    "ignore_index": ignore_index,
    "scheduler": scheduler_cfg,
    "evaluation_tile_size": eval_cfg["tile_size"],
    "evaluation_stride": eval_cfg.get("stride"),
    "split_paths": {key: str(value) for key, value in split_paths.items()},
}
(run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
(run_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
normalization_settings = {
    "statistics": normalizer.statistics,
    "method": normalizer.method,
    "band_indices": normalizer.band_indices,
    "use_8_bit": normalizer.use_8_bit,
}
(run_dir / "normalization.json").write_text(
    json.dumps(normalization_settings, indent=2), encoding="utf-8"
)

# Record the identity of the pretrained weights and split files.
import hashlib

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

(run_dir / "provenance.json").write_text(json.dumps({
    "pretrained_sha256": file_sha256(pretrained_path),
    "split_sha256": {key: file_sha256(path) for key, path in split_paths.items()},
}, indent=2), encoding="utf-8")

def evaluate_split(loader):
    return evaluate(
        model=model,
        dataloader=loader,
        device=device,
        num_classes=2,
        criterion=criterion,
        ignore_index=ignore_index,
        tile_size=eval_cfg["tile_size"],
        stride=eval_cfg.get("stride"),
        tile_batch_size=eval_cfg["tile_batch_size"],
        input_keys=modalities,
        return_per_sample=eval_cfg.get("return_per_sample", True),
        sample_id_key=eval_cfg.get("sample_id_key", "chip_id"),
        class_names=eval_cfg.get("class_names", ["non_water", "water"]),
    )
```

`evaluate` sets evaluation mode, disables gradients, and restores the previous mode afterward. Tiling covers full chips and averages overlapping logits before assigning classes. This is different from evaluating one random crop. tile_batch_size controls inference memory independently of training.batch_size.

`input_keys` prevents chip_id from becoming a model input. return_per_sample=True adds chip-level metrics to the returned dictionary. The evaluator has no knowledge of which loader is validation/test: that is your choice in the calling code.

## 9. Train, validate and save the best checkpoint

This is the step that actually starts training when you execute it. Run the earlier construction sections first. Set epochs=1 before this step for an initial functionality check; use a fresh run for the subsequent full experiment.

```python
if epochs < 1 or validate_every < 1:
    raise ValueError("epochs and validate_every must be positive")

history = []
best_score = float("-inf")
best_path = run_dir / "checkpoints/best.pt"
last_path = run_dir / "checkpoints/last.pt"

# Frozen backbone: save only trainable components to avoid duplicating weights.
# Unfrozen backbone: save the full model.
components = ["adapter", "head"] if freeze_encoder else None

for epoch in range(1, epochs + 1):
    lr_start = optimizer.param_groups[0]["lr"]
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        input_keys=modalities,
    )
    row = {"epoch": epoch, "train_loss": train_loss, "lr_start": lr_start}
    validation_summary = None

    if epoch % validate_every == 0 or epoch == epochs:
        validation = evaluate_split(evaluation_loaders["val"])
        save_evaluation_report(
            run_dir / "metrics" / f"validation_{epoch:03d}.json",
            validation,
            metadata={"split": "val", "epoch": epoch},
        )
        validation_summary = {key: value for key, value in validation.items()
                              if key != "per_sample"}
        row["validation"] = validation_summary
        metric_name = eval_cfg.get("checkpoint_metric", "miou")
        if metric_name == "iou":
            score = validation["iou_per_class"][eval_cfg.get("checkpoint_class", 1)]
        elif metric_name == "miou":
            score = validation["miou"]
        else:
            raise ValueError("This example selects by iou or miou (maximize)")

        if score is not None and score > best_score:
            best_score = score
            save_checkpoint(
                best_path, model, epoch,
                optimizer=optimizer, scheduler=scheduler,
                metrics=validation_summary, config=settings, components=components,
            )

    save_checkpoint(
        last_path, model, epoch,
        optimizer=optimizer, scheduler=scheduler,
        metrics=validation_summary, config=settings, components=components,
    )
    history.append(row)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "metrics/history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )
    message = f"Epoch {epoch:03d}/{epochs} | LR {lr_start:.2e} | Train loss {train_loss:.4f}"
    if validation_summary is not None:
        message += (f" | Water IoU {validation_summary['iou_per_class'][1]}"
                    f" | mIoU {validation_summary['miou']}")
    print(message, flush=True)
```

`train_one_epoch` switches the model to training mode, skips all-ignore batches, computes the loss, runs backward and updates the optimizer. It returns the epoch's aggregated loss. Frozen encoder wrappers keep their backbone in evaluation mode despite model.train().

The outer loop follows evaluation.checkpoint_metric: iou with checkpoint_class: 1 selects water IoU; miou selects mean class IoU. Both are maximized on validation only. Do not average validation metrics across epochs to choose a checkpoint.

`save_checkpoint` accepts a file path, model and epoch, with optional optimizer, scheduler, metrics, config and components. components=None saves the whole model; ['adapter','head'] saves those modules only. TokenToSpatial currently has no learned weights, but including it is harmless. Partial checkpoints require the same frozen pretrained backbone when loaded.

## 10. Reload the best weights and evaluate held-out chips

The model after the loop holds the last epoch's weights. Load best.pt before final evaluation.

```python
if not best_path.is_file():
    raise RuntimeError("No valid best checkpoint was selected")

checkpoint_info = load_checkpoint(best_path, model, map_location=device)
checkpoint_hash = file_sha256(best_path)
print("Selected epoch:", checkpoint_info["epoch"])

# Unique folder avoids overwriting earlier evaluations if you rerun this cell.
evaluation_dir = run_dir / "metrics" / ("evaluation_" + uuid4().hex[:8])
for split in ("test", "bolivia"):
    results = evaluate_split(evaluation_loaders[split])
    save_evaluation_report(
        evaluation_dir / f"{split}.json",
        results,
        metadata={
            "split": split,
            "checkpoint": str(best_path),
            "checkpoint_sha256": checkpoint_hash,
            "selected_epoch": checkpoint_info["epoch"],
            "seed": seed,
        },
    )
    print(split, "mIoU:", results["miou"], "water IoU:", results["iou_per_class"][1])
```

`load_checkpoint` restores weights in an already constructed model. It does not construct the architecture, normalizer or loaders. Do not pass optimizer/scheduler for test-only evaluation. Full-scene outputs must follow the same tiling and preprocessing protocol as validation.

Test and Bolivia remain separate. When comparing literature, match the split and metric definition rather than averaging these scores.

## 11. Read the results

```python
report = json.loads((evaluation_dir / "test.json").read_text())
print("mIoU (%):", 100 * report["miou"])
print("Water IoU (%):", 100 * report["iou_per_class"][1])
print("Water precision:", report["precision_per_class"][1])
print("Water recall:", report["recall_per_class"][1])
print("Water F1:", report["f1_per_class"][1])
print("Pixel accuracy:", report["pixel_accuracy"])
print("Confusion matrix:", report["confusion_matrix"])

recall = report["recall_per_class"][1]
precision = report["precision_per_class"][1]
omission = None if recall is None else 1 - recall
commission = None if precision is None else 1 - precision
chip_values = [sample["iou_per_class"][1] for sample in report.get("per_sample", [])
               if sample["iou_per_class"][1] is not None]
mean_per_chip_water_iou = sum(chip_values) / len(chip_values) if chip_values else None
print("Omission:", omission, "Commission:", commission)
print("Mean per-chip water IoU:", mean_per_chip_water_iou)
print("Chips with defined water IoU:", len(chip_values))
```

The binary confusion matrix is [[TN,FP],[FN,TP]], with target classes as rows. Water IoU is TP/(TP+FP+FN). mIoU averages the defined water/non-water IoUs. Mean per-chip water IoU averages individual chip ratios, so it is different from pooled water IoU. Undefined ratios are None/null, not automatically zero.

Omission/commission are derived here rather than emitted directly by the module.

For Sen1Floods11 AW (all water), PW (permanent water), and FW (flood water), add `aw_pw_fw: true` under `evaluation` in YAML. `run.py` automatically loads the matching `HandLabeled/JRCWaterHand` masks for validation, test and Bolivia. Omit the setting or use `false` to disable it. The reference masks are never model inputs.

When assembling the experiment manually, pass `include_permanent_water=True` to the evaluation datasets and `aw_pw_fw=True` to `evaluate(...)`. Training datasets do not need the masks. Set both arguments from `config["evaluation"].get("aw_pw_fw", False)`.

Reports contain `aw_pw_fw.summary.AW`, `.PW`, and `.FW`, each with `mean_chip_iou`, `total_iou` (pooled IoU), TP/FP/FN/TN, and included/excluded chip counts. With `return_per_sample=True`, each sample also contains its category counts, inclusion flag and chip IoU. Existing saved reports are not updated automatically; evaluate their checkpoint again to obtain these fields.

The calculation matches the previous notebooks: PW ignores labelled water outside JRC permanent water; FW ignores labelled water inside JRC permanent water; both retain labelled background. A wet chip with no water of the requested category is excluded from that category's average. Originally dry chips remain included. Legacy per-chip IoU is `(TP + 1e-7)/(TP + FP + FN + 1e-7)`, so empty unions (including all-ignore chips) score 1. Pooled IoU is unsmoothed and returns null for an empty union. These legacy category averages therefore differ from averaging only defined per-chip IoUs in the generic report. AW is not the arithmetic average of PW and FW.

## 12. How to change parts of the experiment

The following are alternatives, not additional cells to execute blindly after training. Change a component before assembling a new experiment, then create a fresh run and record the changed settings.

### S1-only or S2-only CROMA

| Inputs | modalities variable | croma_modality variable |
|---|---|---|
| S1 | ('sar',) | 'SAR' |
| S2 | ('optical',) | 'optical' |
| S1+S2 | ('sar','optical') | 'both' |

Keep only the relevant band_indices for clarity and change the output parent folder name accordingly. Dataset, normalization and encoder must agree. Set dataset.modalities and model.modality in YAML, then reload it before constructing datasets, statistics and the encoder. Base CROMA also requires its own checkpoint, size='base' and FCN in_channels=768.

### Another optimizer or learning rate

The optimizer is a PyTorch object, not a custom GeoTest module. For example:

```python
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3, momentum=0.9, weight_decay=0.01,
)
```

Adam and AdamW are also available through PyTorch. Equal weight_decay values do not imply identical regularization across optimizer types. If freeze=False, all trainable parameters here share one LR; separate backbone/head rates require parameter groups.

### Cosine warm restarts

The main sequence already constructs this scheduler when YAML selects cosine_annealing_warm_restarts. Its direct module call is:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10 * len(train_loader),
    T_mult=2,
    eta_min=0.0,
)
```

The trainer steps after every successful optimizer update, so T_0 is in updates here. All-ignore skipped batches do not advance it. Set training.scheduler fields in YAML so the saved configuration records this choice.

An epoch-based scheduler should instead be stepped outside train_one_epoch, with scheduler=None passed into the trainer. ReduceLROnPlateau needs a validation metric and should be called only when validation runs. Do not step twice.

### Another loss

Unweighted/weighted cross-entropy is already handled by changing class_weights. Other losses must be explicitly implemented with the current two-class logits and ignored labels in mind. For example, this optional batch water-Dice objective illustrates the interface:

```python
class WaterDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        valid = target != self.ignore_index
        probabilities = logits.softmax(dim=1)[:, 1][valid]
        water = (target[valid] == 1).to(probabilities.dtype)
        intersection = (probabilities * water).sum()
        return 1 - (2 * intersection + self.smooth) / (
            probabilities.sum() + water.sum() + self.smooth
        )

# Optional replacement before a new training run:
# criterion = WaterDiceLoss(ignore_index=255)
```

This is not an installed module or a recommendation that Dice is better. Dice is batch-dependent; the trainer/evaluator's weighted mean of batch losses is not dataset-global Dice loss. Pure Dice can behave poorly on no-water batches. Document the objective and loss aggregation. BCEWithLogitsLoss is not a direct replacement for the existing two-logit integer-target contract.

### TerraMind instead of CROMA

Choose this alternative before preprocessing/model construction; do not apply it to a trained CROMA head. For S1+S2, replace the CROMA training-statistics normalizer and encoder with:

```python
from src.data.preprocessing import TERRAMIND_STATS
from src.encoders.terramind.terramind import TerraMindEncoder

normalizer = NormalizeFromStats(
    statistics={
        "sar": TERRAMIND_STATS["S1GRD"],
        "optical": TERRAMIND_STATS["S2L1C"],
    },
    method="zscore",
    band_indices={"sar": [0, 1], "optical": list(range(13))},
    use_8_bit=False,
)
encoder = TerraMindEncoder(
    checkpoint_path="pretrained_models/TerraMind_v1_large.pt",
    size="large",
    modalities=["S1GRD", "S2L1C"],
    input_mapping={"sar": "S1GRD", "optical": "S2L1C"},
    merge_method="mean",
    freeze=True,
)
```

Then rebuild datasets/loaders with this normalizer and create a fresh adapter/head/model as in section 6. Set pretrained_path, output folder and saved settings to describe TerraMind rather than CROMA. Do not compute training statistics when using the fixed values above.

TerraMind Large with mean fusion has feature width 1024, so the same FCN dimensions work. Base needs 768 and matching weights. concat fusion changes width (2048 for two Large image modalities). S1GRD/S1RTC require VV,VH in dB; S2L1C requires 13 native-DN bands including B10. S2L2A uses 12 and different statistics. Selecting a name does not convert products. The existing hand-labelled loader supplies the dataset's acquisitions, not arbitrary alternative products.

TerraMind pads to multiples of 16. The composed model already uses its padding metadata to resize then crop logits. The same train/evaluate/checkpoint modules can be reused because the final output remains segmentation logits.

## 13. Evaluate later, without retraining

In a new session, import the modules and reconstruct the same architecture from settings.json, load the same pretrained weights, and restore the head using load_checkpoint. Instead of recomputing normalization, reconstruct it from the saved values:

```python
# Set existing_run to the relative directory printed by your earlier run.
existing_run = Path("runs/sen1floods11/croma/fcn_s1s2/REPLACE_WITH_RUN_ID")
saved_settings = json.loads((existing_run / "settings.json").read_text())
saved_normalization = json.loads((existing_run / "normalization.json").read_text())
normalizer = NormalizeFromStats(**saved_normalization)

encoder = CROMAEncoder(
    checkpoint_path=saved_settings["pretrained_path"],
    size=saved_settings["encoder_size"],
    modality=saved_settings["croma_modality"],
    image_resolution=saved_settings["crop_size"],
    freeze=saved_settings["freeze_encoder"],
)
feature_width = 1024 if saved_settings["encoder_size"] == "large" else 768
model = DensePredictionModel(
    encoder, TokenToSpatial(), FCNHead(feature_width, 2)
).to(device)
metadata = load_checkpoint(existing_run / "checkpoints/best.pt", model, map_location=device)

test_dataset = Sen1Floods11Dataset(
    split_csv=saved_settings["split_paths"]["test"],
    modalities=tuple(saved_settings["modalities"]),
    preprocess_func=normalizer,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
results = evaluate(
    model, test_loader, device, num_classes=2,
    criterion=None, ignore_index=255,
    tile_size=saved_settings["evaluation_tile_size"],
    stride=saved_settings["evaluation_stride"], tile_batch_size=1,
    input_keys=saved_settings["modalities"], sample_id_key="chip_id",
    return_per_sample=True, class_names=["non_water", "water"],
)
print(results["miou"], results["iou_per_class"][1])
```

This reload example is specifically for the main CROMA experiment. criterion=None omits loss calculation, not segmentation metrics. No optimizer is required for testing. For strict provenance, compare current pretrained/split hashes with provenance.json before evaluating; retain unchanged imagery too, since the example does not hash every raster.

To resume optimization, construct the matching optimizer/scheduler and pass them to load_checkpoint using last.pt. Continue from returned epoch + 1 and recover history and best-validation score. Current checkpoints do not save random-generator or DataLoader state, so this is not guaranteed to reproduce an uninterrupted trajectory exactly.

## What this tutorial does and does not establish

The main sequence assembles a complete experiment using existing public module interfaces. It keeps held-out data separate and shows exactly where the loss, optimizer, preprocessing and checkpoint choices live. It does not claim equivalence to PANGAEA, authors' baselines or past notebook scores without checking the entire protocol. Repeated seeds and a matched data/evaluation setup are needed for meaningful model comparisons.

This is a new tutorial document. The separate EXPERIMENT_MANUAL.md and project implementation were not modified when writing it. Snippets were checked for syntax and local API compatibility; full training and GPU inference were not run.
