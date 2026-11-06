# GeoTest offline experiment manual

Reviewed: 6 November 2026. This manual explains how the current modules fit together and includes a complete reference runner. No training, dependency installation or source-code changes were performed to create this document.

## 1. Prepare the offline environment

Open a terminal in the project folder and select your existing Python environment. All commands below are run from that folder; no specific drive, username or installation location is required. If your environment is stored locally as `.venv`, you can activate it with `.\.venv\Scripts\Activate.ps1` on Windows. Otherwise activate your environment through your IDE or environment manager.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

Use the populated environment, not the global Python that previously lacked NumPy. Once packages, satellite data and pretrained checkpoints are local, the workflow needs no internet. The wrappers load local weights rather than requesting pretrained downloads.

Observed installed versions: Python 3.12.10; torch 2.14.0+cu130; torchvision 0.29.0+cu130; terratorch 1.2.5; torchgeo 0.8.1; numpy 2.5.3; rasterio 1.5.1; PyYAML 6.0.3; einops 0.8.2; timm 1.0.29. This is an environment inventory, not a fresh compatibility certification. Preserve the working environment for offline use; a requirements list alone cannot reinstall dependencies without their wheel files. You can record it using `python -m pip freeze > environment.txt`.

The loader expects:

```text
data/sen1floods11/
  HandLabeled/{S1Hand,S2Hand,LabelHand}/
  splits/all_hand_labels/
    flood_train_data.csv
    flood_valid_data.csv
    flood_test_data.csv
    flood_bolivia_data.csv
pretrained_models/
  CROMA_large.pt
  TerraMind_v1_large.pt
```

Split CSVs have no header and contain image filename,label filename. Either S1Hand or S2Hand image names identify chips. Custom CSVs select different existing chips; the imagery directory itself is currently fixed inside the dataset module.

**Path correction:** the current CROMA YAML points to pretrained_models/croma/CROMA_large.pt, but the observed checkpoint is pretrained_models/CROMA_large.pt. Correct model.checkpoint before using it, unless you intentionally relocate the checkpoint. The manual does not change the YAML.

## 2. Module responsibilities

```text
YAML -> Dataset -> normalization + shared training crop -> DataLoader
     -> encoder -> token-to-spatial adapter -> FCN -> logits
     -> train_one_epoch -> validation -> best checkpoint
     -> load best checkpoint -> test/Bolivia -> JSON reports
```

| Module | Interface / responsibility |
|---|---|
| src/data/datasets/sen1floods11.py | Sen1Floods11Dataset(split_csv, modalities, preprocess_func); raw acquisitions, targets and chip IDs |
| src/data/statistics.py | get_or_compute_stats(...); cache statistics from raw training images |
| src/data/preprocessing.py | NormalizeFromStats.from_config(config, training_statistics=None) |
| src/data/transforms.py | random_crop(sample, size); identical crop for all images and target |
| src/models/factory.py | build_model(config); builds encoder, adapter and FCN |
| src/encoders/croma/croma.py | SAR, optical or joint CROMA tokens |
| src/encoders/terramind/terramind.py | Product mapping, TerraMind tokens and padded_size metadata |
| src/adapters/token_to_spatial.py | [B,N,D] -> [B,D,grid_h,grid_w] |
| src/heads/fcn.py | GroupNorm FCN -> class logits |
| src/models/dense_prediction.py | Compose components, interpolate logits and remove padding |
| training/trainer.py | train_one_epoch(...); backward and optimizer updates |
| src/evaluation/sliding_window.py | Tiled inference, averaging overlapping logits |
| src/evaluation/metrics.py | Confusion counts and derived metrics |
| src/evaluation/evaluator.py | evaluate(...), save_evaluation_report(...) |
| training/checkpoints.py | save_checkpoint(...), load_checkpoint(...) |

A sample contains sar [2,H,W] and/or optical [13,H,W] before band selection, target [H,W] with integer labels 0/1/255, and string chip_id metadata. DataLoader adds a batch dimension. Logits are [B,2,H,W]. Do not apply softmax before CrossEntropyLoss. The evaluator uses argmax. The common output interface lets both encoders use the same trainer.

## 3. Read YAML and choose normalization

```python
from pathlib import Path
import yaml
path = Path("configs/terramind/terramind_v1_large_s1_fcn_all_hand_labels.yaml")
with path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)
print(config["model"]["name"])
print(config["training"]["optimizer"]["learning_rate"])
```

For fixed normalization, the relevant YAML fragment is:

```yaml
model:
  name: terramind_v1_large
  modalities: [S1GRD, S2L1C]
  input_mapping: {sar: S1GRD, optical: S2L1C}
preprocessing:
  normalization:
    method: zscore
    statistics: pretraining-fixed
    use_8_bit: false
```

This fragment belongs inside a complete experiment configuration. It is not standalone.

```python
from src.data.preprocessing import NormalizeFromStats
normalizer = NormalizeFromStats.from_config(config)
```

Relative paths are interpreted from the terminal working directory. Run the examples from the project folder, rather than from reports/ or src/.

The mapping selects fixed values from TERRAMIND_STATS in preprocessing.py. Use the same normalizer for all splits. For statistics: training_dataset, first call get_or_compute_stats on raw training data, then pass the result as training_statistics. Never estimate these statistics from test/validation imagery. The full example includes both branches.

| Product | Input convention for fixed statistics |
|---|---|
| S1GRD / S1RTC | VV then VH, in dB; different statistics per product |
| S2L1C | 13 native-DN bands: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 |
| S2L2A | 12 native-DN bands in the same order without B10 |

Do not divide DN by 10000 or apply display stretching before fixed normalization. Selecting a product name does not convert L1C to L2A or GRD to RTC. Current Sen1Floods11 optical inputs are handled as L1C. Dropping B10 for CROMA does not perform atmospheric correction.

## 4. Complete reference runner

There is currently no root run.py. Copy the complete next Python block into GeoTest/run_experiment.py yourself; this manual does not create that file. It is a runnable reference implementation using the existing modules, not a claim that a full experiment has already been validated.

It supports mean cross-entropy, AdamW/Adam/SGD, no scheduler or update-based cosine warm restarts, validation-selected checkpoints and separate test/Bolivia reports. The example introduces optional training.validate_every (default 1). It uses num_workers: 0 for straightforward Windows execution. It adds no augmentation beyond random crops, no mixed precision and no UPerNet. It therefore does not automatically reproduce legacy notebook protocols.

```python
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.preprocessing import NormalizeFromStats
from src.data.statistics import get_or_compute_stats
from src.data.transforms import random_crop
from src.models.factory import build_model
from src.evaluation.evaluator import evaluate, save_evaluation_report
from training.trainer import train_one_epoch
from training.checkpoints import save_checkpoint, load_checkpoint

def read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainPreprocess:
    def __init__(self, normalizer, crop_size):
        self.normalizer = normalizer
        self.crop_size = crop_size

    def __call__(self, sample):
        return self.normalizer(random_crop(sample, self.crop_size))


def make_dataset(config, split, preprocess=None, modalities=None):
    return Sen1Floods11Dataset(
        split_csv=Path(config["dataset"][f"{split}_csv"]),
        modalities=modalities or tuple(config["dataset"]["modalities"]),
        preprocess_func=preprocess,
    )


def prepare_normalizer(config):
    statistics = None
    if config["preprocessing"]["normalization"]["statistics"] == "training_dataset":
        statistics = get_or_compute_stats(
            dataset_factory=lambda modalities: make_dataset(config, "train", modalities=modalities),
            modalities=config["dataset"]["modalities"],
            split_csv=Path(config["dataset"]["train_csv"]),
            dataset_name=config["dataset"]["name"],
            cache_dir=Path("data/statistics"),
        )
    return NormalizeFromStats.from_config(config, training_statistics=statistics)


def make_loaders(config, normalizer, training):
    datasets, loaders = {}, {}
    splits = ("train", "val", "test", "bolivia") if training else ("test", "bolivia")
    for split in splits:
        if not config["dataset"].get(f"{split}_csv"):
            continue
        transform = normalizer
        if split == "train":
            transform = TrainPreprocess(normalizer, config["preprocessing"]["crop_size"])
        datasets[split] = make_dataset(config, split, transform)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=(config["training"]["batch_size"] if split == "train"
                        else config["evaluation"].get("batch_size", 1)),
            shuffle=(split == "train"), num_workers=0, drop_last=False,
        )
    # Check overlap between all splits loaded by this invocation.
    seen = {}
    for split, dataset in datasets.items():
        for paths in dataset.data_list:
            chip = paths["target"].name
            if chip in seen:
                raise ValueError(f"Split overlap: {chip} in {seen[chip]} and {split}")
            seen[chip] = split
    if training and not {"train", "val"} <= set(loaders):
        raise ValueError("Training requires train_csv and val_csv")
    return loaders


def make_loss(config, device):
    options = config["training"].get("loss", {})
    if options.get("name", "cross_entropy") != "cross_entropy":
        raise ValueError("This reference runner implements cross_entropy only")
    if options.get("ignore_index", 255) != 255:
        raise ValueError("The current dataset uses ignore_index=255")
    weights = options.get("class_weights")
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if weights.numel() != config["head"]["num_classes"] or not bool(
            (torch.isfinite(weights) & (weights > 0)).all()
        ):
            raise ValueError("Use one positive finite weight per class")
    return nn.CrossEntropyLoss(weight=weights, ignore_index=255, reduction="mean")


def make_optimizer(config, model):
    options = config["training"]["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    factories = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                 "sgd": torch.optim.SGD}
    name = options["name"].lower()
    if name not in factories:
        raise ValueError(f"Unsupported optimizer: {name}")
    kwargs = dict(lr=options["learning_rate"], weight_decay=options.get("weight_decay", 0))
    if name == "sgd":
        kwargs["momentum"] = options.get("momentum", 0.9)
    return factories[name](params, **kwargs)


def make_scheduler(config, optimizer, batches_per_epoch):
    options = config["training"].get("scheduler") or {"name": "none"}
    if options["name"] == "none":
        return None
    if options["name"] != "cosine_annealing_warm_restarts":
        raise ValueError("Supported schedulers: none, cosine_annealing_warm_restarts")
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(options["t0_epochs"] * batches_per_epoch),
        T_mult=options.get("t_mult", 1), eta_min=options.get("eta_min", 0.0),
    )


def run_evaluation(config, model, loader, criterion, device):
    options = config["evaluation"]
    return evaluate(
        model, loader, device, num_classes=config["head"]["num_classes"],
        criterion=criterion, ignore_index=255,
        tile_size=options.get("tile_size"),
        tile_batch_size=options.get("tile_batch_size", 1), stride=options.get("stride"),
        input_keys=config["dataset"]["modalities"],
        return_per_sample=options.get("return_per_sample", True),
        sample_id_key=options.get("sample_id_key", "chip_id"),
        class_names=options.get("class_names"),
    )


def selection_score(config, results):
    options = config["evaluation"]
    key = options.get("checkpoint_metric", "miou")
    if key == "iou":
        return results["iou_per_class"][options.get("checkpoint_class", 1)]
    if key != "miou":
        raise ValueError("Reference selection supports iou or miou, maximized")
    return results["miou"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--evaluate-run", help="Evaluate best.pt in a saved run; no training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if bool(args.config) == bool(args.evaluate_run):
        parser.error("Provide exactly one of --config or --evaluate-run")
    training = args.evaluate_run is None
    if training:
        config = read_yaml(Path(args.config))
    else:
        run_dir = Path(args.evaluate_run)
        config = read_yaml(run_dir / "config.yaml")
    if config["dataset"]["name"] != "sen1floods11":
        raise ValueError("This reference data pipeline implements Sen1Floods11 only")
    if config["training"].get("num_workers", 0) != 0:
        raise ValueError("This simple reference runner uses num_workers: 0")
    seed = config["experiment"].get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if training:
        normalizer = prepare_normalizer(config)
    else:
        saved = json.loads((run_dir / "normalization.json").read_text())
        normalizer = NormalizeFromStats(**saved)
    loaders = make_loaders(config, normalizer, training)
    model = build_model(config).to(device)
    criterion = make_loss(config, device)

    if training:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        run_dir = Path(config["output"]["directory"]) / tag
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json(run_dir / "normalization.json", dict(
            statistics=normalizer.statistics, method=normalizer.method,
            band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
        ))
        write_json(run_dir / "provenance.json", {
            "pretrained_path": str(Path(config["model"]["checkpoint"])),
            "pretrained_sha256": sha256(Path(config["model"]["checkpoint"])),
            "splits": {split: sha256(Path(config["dataset"][f"{split}_csv"]))
                       for split in loaders},
            "seed": seed,
        })
        optimizer = make_optimizer(config, model)
        scheduler = make_scheduler(config, optimizer, len(loaders["train"]))
        epochs = config["training"]["epochs"]
        every = config["training"].get("validate_every", 1)
        if epochs < 1 or every < 1:
            raise ValueError("epochs and validate_every must be positive")
        components = ["adapter", "head"] if config["model"].get("freeze_encoder", True) else None
        best, history = float("-inf"), []
        print("Run directory:", run_dir, flush=True)
        for epoch in range(1, epochs + 1):
            lr_start = optimizer.param_groups[0]["lr"]
            loss = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device,
                scheduler=scheduler, input_keys=config["dataset"]["modalities"],
            )
            row = {"epoch": epoch, "train_loss": loss, "lr_start": lr_start,
                   "lr_end": optimizer.param_groups[0]["lr"]}
            validation = None
            if epoch % every == 0 or epoch == epochs:
                validation = run_evaluation(config, model, loaders["val"], criterion, device)
                save_evaluation_report(run_dir / "metrics" / f"val_{epoch:03d}.json", validation,
                                       metadata={"split": "val", "epoch": epoch})
                row["validation"] = {k: v for k, v in validation.items() if k != "per_sample"}
                score = selection_score(config, validation)
                if score is not None and score > best:
                    best = score
                    save_checkpoint(run_dir / "checkpoints/best.pt", model, epoch,
                                    optimizer=optimizer, scheduler=scheduler,
                                    metrics=row["validation"], config=config, components=components)
            history.append(row)
            write_json(run_dir / "metrics/history.json", history)
            save_checkpoint(run_dir / "checkpoints/last.pt", model, epoch,
                            optimizer=optimizer, scheduler=scheduler, metrics=row.get("validation"),
                            config=config, components=components)
            message = f"Epoch {epoch:03d}/{epochs} | LR start {lr_start:.2e} | Train loss {loss:.4f}"
            if validation is not None:
                message += f" | Water IoU {validation['iou_per_class'][1]} | mIoU {validation['miou']}"
            print(message, flush=True)

    provenance = json.loads((run_dir / "provenance.json").read_text())
    if not training:
        if sha256(Path(config["model"]["checkpoint"])) != provenance["pretrained_sha256"]:
            raise ValueError("Pretrained weights changed since training")
        for split in loaders:
            if sha256(Path(config["dataset"][f"{split}_csv"])) != provenance["splits"].get(split):
                raise ValueError(f"The {split} split changed since training")
    checkpoint = run_dir / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("No best checkpoint: inspect validation metrics")
    metadata = load_checkpoint(checkpoint, model, map_location=device)
    checkpoint_hash = sha256(checkpoint)
    evaluation_dir = run_dir / "metrics" / ("evaluation_" + uuid4().hex[:8])
    for split in ("test", "bolivia"):
        if split not in loaders:
            continue
        results = run_evaluation(config, model, loaders[split], criterion, device)
        save_evaluation_report(evaluation_dir / f"{split}.json", results, metadata={
            "split": split, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash, "epoch": metadata["epoch"], "seed": seed,
        })
        print(split, "mIoU:", results["miou"], "water IoU:", results["iou_per_class"][1])
    print("Saved run:", run_dir)


if __name__ == "__main__":
    main()
```

Training draws one random crop per chip per epoch. Validation/testing cover full chips through sliding windows. The saved normalization.json prevents evaluation from recomputing or changing normalization. Seeds improve repeatability but do not enforce deterministic CUDA algorithms.

## 5. Commands

After copying the reference block into run_experiment.py:

```powershell
python run_experiment.py --config configs/terramind/terramind_v1_large_s1_fcn_all_hand_labels.yaml
python run_experiment.py --config configs/terramind/terramind_v1_large_s1s2_fcn_all_hand_labels.yaml
python run_experiment.py --config configs/croma/croma_s1s2_fcn_all_hand_labels.yaml
```

Execute these separately. Each trains and then evaluates best.pt. First try a copied YAML with epochs: 1, then start a fresh full run. This checks functionality, not scientific performance. To validate every five epochs add training.validate_every: 5; the last epoch is always validated too.

To test an existing run without retraining:

```powershell
python run_experiment.py --evaluate-run runs/sen1floods11/terramind/fcn_s1/REPLACE_WITH_RUN_ID
```

Use the actual printed run ID. The command loads that run's configuration, normalization and best.pt. It writes a separate evaluation directory. Both the original pretrained file and saved run must remain available. Use --device cpu only if needed; large models can be slow on CPU.

## 6. Configuration parameter reference

These fields are interpreted by the modules and the reference runner. YAML does not implement behavior itself: an unhandled option has no effect.

| Field | Meaning / supported values |
|---|---|
| experiment.seed | Python/NumPy/PyTorch seed |
| experiment.name | Identifier saved with the configuration |
| dataset.name | sen1floods11 in this runner |
| dataset.modalities | [sar], [optical], or [sar, optical] |
| dataset.train_csv / val_csv / test_csv / bolivia_csv | Local split paths |
| model.checkpoint | Project-relative pretrained file, e.g. pretrained_models/CROMA_large.pt |
| model.freeze_encoder | true: train head; false: train encoder and head |
| adapter.name | token_to_spatial only |
| adapter.feature_key | tokens |
| adapter.grid_size | null to infer square grid; [height,width] for explicit grid |
| head.name | fcn only; UPerNet is not implemented in this factory |
| head.in_channels | Feature width, not satellite-band count |
| head.num_classes | 2 for non-water/water |
| preprocessing.crop_size | Training window; square recommended with grid_size: null |
| preprocessing.band_indices | Zero-based raw-band selection per dataset key |
| normalization.statistics | training_dataset or pretraining-fixed, inside preprocessing |
| normalization.method | zscore or two_sigma_stretch; fixed policy requires zscore |
| normalization.use_8_bit | Optional quantization for stretch, false for zscore |
| training.epochs | Outer-loop epoch count |
| training.batch_size | Number of crops per optimizer update |
| training.num_workers | 0 in reference runner; more workers require changing the runner |
| training.validate_every | Example-runner field, default 1; final epoch also validated |
| training.loss.class_weights | null for unweighted CE or [non_water_weight,water_weight] |
| training.optimizer.learning_rate | Initial learning rate |
| training.optimizer.weight_decay | Optimizer-specific regularization parameter |
| evaluation.tile_size | Window size; null for direct inference only if model/memory support it |
| evaluation.stride | Window step; 60 for overlapping 120-pixel windows |
| evaluation.tile_batch_size | Windows per forward pass, separate from image batch size |
| evaluation.batch_size | Whole chips per loader batch |
| evaluation.return_per_sample | Include per-chip reports |
| evaluation.sample_id_key | chip_id |
| evaluation.checkpoint_metric | iou or miou in the example runner |
| evaluation.checkpoint_class | 1 selects water when checkpoint_metric is iou |
| output.directory | Parent folder for unique runs |

### CROMA modality selection

```yaml
model:
  name: croma
  size: large
  checkpoint: pretrained_models/CROMA_large.pt
  modality: SAR
  image_resolution: 120
  freeze_encoder: true
```

This fragment is for S1. CROMA's modality field is case-sensitive: SAR, optical or both. Match dataset.modalities accordingly. Large feature width is 1024; Base is 768 and requires its own checkpoint. CROMA uses 8-pixel patches: keep crop_size, tile_size and image_resolution compatible and normally equal. For optical data use band_indices.optical: [0,1,2,3,4,5,6,7,8,9,11,12]. Removing B10 is not conversion to L2A.

### TerraMind modality selection

```yaml
model:
  name: terramind_v1_large
  size: large
  checkpoint: pretrained_models/TerraMind_v1_large.pt
  modalities: [S1GRD, S2L1C]
  input_mapping: {sar: S1GRD, optical: S2L1C}
  merge_method: mean
  freeze_encoder: true
```

S1-only: modalities [S1GRD], mapping {sar: S1GRD}, dataset.modalities [sar]. S2-only: modalities [S2L1C], mapping {optical: S2L1C}, dataset.modalities [optical]. Keep the corresponding band selections. Current TerraMind L1C input uses all 13 bands; the wrapper does not expose subset-band patch-embedding adaptation.

Base requires name terramind_v1_base, size base, its matching checkpoint and head.in_channels 768. Large uses 1024. mean/max fusion preserve feature width; concat multiplies it by the number of image modalities (two Large inputs -> 2048). The factory validates TerraMind head width.

TerraMind pads inputs to multiples of 16 and the composed model resizes/crops predictions using padded_size. Modalities must have aligned spatial grids; matching shapes alone cannot establish geospatial alignment or band identity.

## 7. Changing loss functions and optimization

### Unweighted and weighted cross-entropy

```yaml
training:
  loss:
    name: cross_entropy
    ignore_index: 255
    class_weights: null
```

For weighted CE, use class_weights: [1.0,8.0], for example. This changes the objective, not the IoU formula. Eight is an experimental choice, not an established optimum. For a controlled comparison, keep the loss protocol matched across models.

The criterion is a mean over valid labels:

```python
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction="mean")
```

Do not use sum/none reduction with the current trainer's averaging. Targets are integer classes, not one-hot arrays. Ignore labels stay 255 with this loader.

### Adding a different loss

Changing loss.name alone does not implement a loss. The reference runner rejects unknown names. A criterion must accept logits [B,C,H,W] and integer targets [B,H,W], exclude ignored pixels and return a finite scalar with gradients.

For example, an optional batch water-Dice loss could be defined as follows. This is illustrative code, not an installed GeoTest feature or a claim of better performance:

```python
class WaterDiceLoss(torch.nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        valid = target != self.ignore_index
        probability = logits.softmax(dim=1)[:, 1][valid]
        truth = (target[valid] == 1).to(probability.dtype)
        intersection = (probability * truth).sum()
        return 1 - (2 * intersection + self.smooth) / (
            probability.sum() + truth.sum() + self.smooth
        )
```

Add an explicit water_dice branch in make_loss before the unknown-name rejection to use it. The ignore_index attribute lets the trainer skip all-ignore batches. Dice is non-additive and batch-dependent: the trainer/evaluator's valid-pixel-weighted average of batch Dice losses is NOT dataset-global Dice loss. Use confusion-derived IoU/F1 for comparisons and document loss aggregation. Pure Dice can behave poorly on batches without water; CE+Dice is a separate choice requiring explicit coefficients and implementation. BCEWithLogitsLoss is not a direct replacement for the current two-logit integer-target interface.

### Learning rate and optimizer

```yaml
training:
  optimizer:
    name: adamw
    learning_rate: 0.0001
    weight_decay: 0.01
```

The example accepts adamw, adam or sgd. SGD accepts momentum (default 0.9). Equal numeric weight_decay does not imply the same regularization in Adam and AdamW. Only parameters with requires_grad=True enter the optimizer. For freeze_encoder: false, the example uses the same LR for backbone and head; separate rates require parameter groups.

### Schedulers

Constant LR:

```yaml
training:
  scheduler:
    name: none
```

Cosine warm restarts:

```yaml
training:
  scheduler:
    name: cosine_annealing_warm_restarts
    t0_epochs: 10
    t_mult: 2
    eta_min: 0.0
```

The trainer steps after every optimizer update. Therefore the example converts T_0 to len(train_loader) * t0_epochs updates. Skipped all-ignore batches delay the schedule relative to calendar epochs. lr_start in the history is the epoch's initial rate, not every batch's rate.

For epoch-stepped schedulers, pass scheduler=None to train_one_epoch and call scheduler.step() once after each epoch. For ReduceLROnPlateau call scheduler.step(validation['loss']) only after validation. Do not step the same scheduler both inside and outside the trainer. Adding these policies needs a make_scheduler branch plus corresponding outer-loop logic; YAML alone is insufficient.

### Crop size and memory

Reduce training.batch_size for training memory; reduce evaluation.tile_batch_size for inference memory. Changing crop size also changes spatial context and is a scientific protocol change. TerraMind's 120 -> 128 padding does not rescale the observation. If using 128 for both models, also update CROMA image_resolution and evaluation.tile_size, and choose stride deliberately.

## 8. Saved results and metric interpretation

Use relative paths in YAML and command arguments so the example also records relative paths in its reports. Internal code may resolve paths locally, but that does not require publishing machine-specific paths.

The example creates:

```text
<output.directory>/<timestamp_unique_id>/
  config.yaml
  normalization.json
  provenance.json
  checkpoints/{best.pt,last.pt}
  metrics/
    history.json
    val_001.json
    ...
    evaluation_<unique_id>/{test.json,bolivia.json}
```

history.json contains epoch loss/LR and validation summaries. Per-chip validation data are in the separate val files. best.pt is selected using validation; last.pt can be worse. Metrics are fractions, so multiply by 100 for percentages.

Confusion-matrix rows are target classes and columns predictions:

```text
[[TN, FP],
 [FN, TP]]
```

- Water IoU: TP/(TP+FP+FN), stored in iou_per_class[1].
- Non-water IoU: TN/(TN+FP+FN), stored in iou_per_class[0].
- mIoU: mean of defined per-class IoUs, not mean per-chip IoU.
- Precision: TP/(TP+FP); recall: TP/(TP+FN).
- F1: 2TP/(2TP+FP+FN).
- Omission: 1-recall; commission: 1-precision, if defined.
- Pixel accuracy: (TP+TN)/valid_pixels; can be high when non-water dominates.

Omission/commission are not directly emitted by the current module but can be derived. Zero-denominator ratios become None in Python/null in JSON and are excluded from macro means. All-ignore chips contribute no pooled counts; evaluation with no valid pixels at all raises an error.

Example summary from a report:

```python
import json
from pathlib import Path
report = json.loads(Path("path/to/test.json").read_text())
precision = report["precision_per_class"][1]
recall = report["recall_per_class"][1]
commission = None if precision is None else 1 - precision
omission = None if recall is None else 1 - recall
values = [row["iou_per_class"][1] for row in report["per_sample"]
          if row["iou_per_class"][1] is not None]
mean_chip_iou = sum(values) / len(values) if values else None
print("Pooled water IoU:", report["iou_per_class"][1])
print("Mean per-chip water IoU:", mean_chip_iou)
print("Chips with defined water IoU:", len(values))
```

Pooled IoU weights chips implicitly by their water-union size. Mean per-chip IoU gives each included chip equal influence. Do not interchange them. AW/PW/FW require permanent/flood-water reference masks and are not implemented in the generic evaluator. Test and Bolivia remain separate; combine only deliberately. To pool them, add their confusion matrices and recompute ratios, rather than averaging split mIoUs.

## 9. Checkpoint restoration and scientific comparisons

The example saves adapter/head only for a frozen backbone, avoiding duplicate multi-gigabyte pretrained weights. Full fine-tuning saves the entire model. Component checkpoints require the same pretrained backbone first; its SHA-256 is recorded and checked during separate evaluation. TokenToSpatial currently has no trainable weights.

After constructing matching model, optimizer and scheduler, you can restore:

```python
from training.checkpoints import load_checkpoint
metadata = load_checkpoint(
    "path/to/checkpoints/last.pt", model,
    optimizer=optimizer, scheduler=scheduler, map_location=device,
)
start_epoch = metadata["epoch"] + 1
```

This is a restoration building block, not a --resume option in the reference runner. For resume, retain original preprocessing/configuration, restore prior history and best-validation score, then continue the epoch loop. Current checkpoint functions do not save Python/NumPy/PyTorch RNG states, DataLoader RNG state or an AMP scaler. Optimization can continue, but an exact uninterrupted trajectory is not guaranteed.

Control and record splits, seeds, label budget, crop/augmentation, loss, freezing, duration and checkpoint selection when comparing models. Product-specific normalization can intentionally differ; report it. Repeated seeds quantify variability better than one run. CSV hashes identify splits, but the example does not hash all rasters: retain an immutable/versioned dataset for strict reproducibility. Keep test results out of hyperparameter and checkpoint selection.

The new GeoTest pipeline does not inherit reproducibility of legacy or published results automatically. Those require running the matched protocol and checking preprocessing, labels and evaluation definitions.

## 10. Troubleshooting and checks

| Symptom | Check |
|---|---|
| Missing numpy/torch | Confirm sys.executable points to the populated environment |
| Missing src import | Save the example in the repository root and run from GeoTest |
| Missing checkpoint | Correct model.checkpoint to the actual local path |
| TerraTorch error involving torchgeo.trainers.utils | Preserve/check the working dependency versions instead of upgrading blindly offline |
| Missing/unexpected modality | Match dataset.modalities, model.modalities and input_mapping |
| Wrong channel count | Match product and band_indices; TerraMind L1C uses 13, CROMA optical uses 12 |
| Head dimension mismatch | Use feature width, accounting for Base/Large and concatenation |
| Cannot infer square grid | Use square windows or an explicit correct rectangular grid |
| Validation out of memory | Reduce tile_batch_size independently of training.batch_size |
| No valid training pixels | Inspect masks, crops and ignore_index; ignored is not background |
| null per-chip water IoU | No valid water union; retain a documented undefined-ratio policy |
| Poor test IoU despite low training loss | Check normalization, alignment, protocol, geography and checkpoint provenance |
| integration_test02 cannot import run | It still references removed run.py; this manual does not repair that legacy import |

Existing lightweight checks:

```powershell
python -m unittest discover -s tests -p test_padding_and_metadata.py -v
python -m unittest discover -s tests -p test_terramind_wrapper_unit.py -v
```

The first checks metadata and padded decoding. The second uses a fake backbone, so passing does not verify the real pretrained model's numerical behavior. Code examples in this manual were checked for syntax; no full training or GPU inference was performed for the document.
