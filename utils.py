"""Configuration, data setup and training-component helpers for experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.preprocessing import NormalizeFromStats
from src.data.statistics import get_or_compute_stats
from src.data.transforms import random_crop

# Convert the training.loss settings into a criterion on the selected device.
def build_loss(config, device):
    name = config["name"]

    if name == "cross_entropy":
        # None gives ordinary CE; a list weights the classes in label-ID order.
        weights = config.get("class_weights")
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return torch.nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=config.get("ignore_index", 255),
        )

    raise ValueError(f"Unsupported loss {name!r}. Available: cross_entropy")


# Read YAML and reject configurations incompatible with the current pipeline.
def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}. Run from the project folder.")
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    for section in ("experiment", "dataset", "model", "adapter", "head",
                    "preprocessing", "training", "evaluation", "output"):
        if not isinstance(config, dict) or not isinstance(config.get(section), dict):
            raise ValueError(f"YAML requires a '{section}' mapping")
    # Check required fields before using them, so errors name the YAML location.
    required = (
        "dataset.name", "dataset.modalities", "dataset.train_csv", "dataset.val_csv",
        "model.name", "model.checkpoint", "adapter.name", "head.name",
        "head.in_channels", "head.num_classes", "preprocessing.crop_size",
        "preprocessing.normalization.method", "preprocessing.normalization.statistics",
        "training.epochs", "training.batch_size", "training.loss.name",
        "training.optimizer.name", "training.optimizer.learning_rate", "output.directory",
    )
    for field in required:
        value = config
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Missing required YAML field: {field}")
            value = value[key]
        if value is None:
            raise ValueError(f"YAML field cannot be null: {field}")

    choices = {
        "dataset.name": ("sen1floods11",),
        "model.name": ("croma", "terramind_v1_base", "terramind_v1_large"),
        "adapter.name": ("token_to_spatial",),
        "head.name": ("fcn",),
        "training.loss.name": ("cross_entropy",),
        "training.optimizer.name": ("adamw", "adam", "sgd"),
        "preprocessing.normalization.method": ("zscore", "two_sigma_stretch"),
        "preprocessing.normalization.statistics": ("training_dataset", "pretraining-fixed"),
    }
    for field, available in choices.items():
        value = config
        for key in field.split("."):
            value = value[key]
        if value not in available:
            raise ValueError(f"Unsupported {field}: {value!r}. Available: {', '.join(available)}")
    scheduler = config["training"].get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, dict) or
            scheduler.get("name") not in {"none", "cosine_annealing", "cosine_annealing_warm_restarts"}):
        raise ValueError("training.scheduler.name must be none, cosine_annealing or cosine_annealing_warm_restarts")
    if scheduler and scheduler["name"] == "cosine_annealing":
        eta_min = scheduler.get("eta_min", 0.0)
        if (not isinstance(eta_min, (int, float)) or not np.isfinite(eta_min)
                or not 0 <= eta_min <= config["training"]["optimizer"]["learning_rate"]):
            raise ValueError("training.scheduler.eta_min must be between zero and the initial learning rate")
    if scheduler and scheduler["name"] == "cosine_annealing_warm_restarts":
        if type(scheduler.get("t0_epochs")) is not int or scheduler["t0_epochs"] < 1:
            raise ValueError("training.scheduler.t0_epochs must be a positive integer")
        if type(scheduler.get("t_mult", 1)) is not int or scheduler.get("t_mult", 1) < 1:
            raise ValueError("training.scheduler.t_mult must be a positive integer")
    modalities = config["dataset"]["modalities"]
    if not isinstance(modalities, (list, tuple)) or not modalities or any(
            value not in ("sar", "optical") for value in modalities) or len(set(modalities)) != len(modalities):
        raise ValueError("dataset.modalities must list sar, optical, or both, without duplicates")
    if config["dataset"]["name"] != "sen1floods11":
        raise ValueError("The current dataset loader supports sen1floods11")
    loss = config["training"]["loss"]
    if loss["name"] != "cross_entropy":
        raise ValueError("This runner currently supports cross_entropy; other losses need implementation")
    # The dataset converts ignored ground-truth pixels to 255.
    if loss.get("ignore_index", 255) != 255:
        raise ValueError("Sen1Floods11Dataset maps ignored labels to 255")
    weights = loss.get("class_weights")
    if weights is not None and (len(weights) != config["head"]["num_classes"] or
                               not all(np.isfinite(w) and w > 0 for w in weights)):
        raise ValueError("class_weights must contain one positive finite value per class")
    training = config["training"]
    if training["epochs"] < 1 or training.get("validate_every", 5) < 1:
        raise ValueError("epochs and validate_every must be positive")
    evaluation = config["evaluation"]
    if evaluation.get("checkpoint_metric", "miou") not in {"iou", "miou"}:
        raise ValueError("checkpoint_metric must be iou or miou")
    if not 0 <= evaluation.get("checkpoint_class", 1) < config["head"]["num_classes"]:
        raise ValueError("checkpoint_class is outside the class range")
    # Check that CROMA receives the selected sensors and its configured input size.
    if config["model"]["name"] == "croma":
        model = config["model"]
        expected = {"SAR": {"sar"}, "optical": {"optical"}, "both": {"sar", "optical"}}
        if set(config["dataset"]["modalities"]) != expected.get(model.get("modality", "both")):
            raise ValueError("CROMA model.modality and dataset.modalities disagree")
        if model.get("image_resolution", 120) != config["preprocessing"]["crop_size"] or \
                evaluation.get("tile_size") != model.get("image_resolution", 120):
            raise ValueError("Match CROMA image_resolution, crop_size and evaluation.tile_size")
    # Check every configured split and referenced raster before statistics or GPU work.
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found (model.checkpoint): {checkpoint}")
    water_categories = config["evaluation"].get("aw_pw_fw", False)
    if not isinstance(water_categories, bool):
        raise ValueError("evaluation.aw_pw_fw must be true or false")
    if water_categories and config["head"]["num_classes"] != 2:
        raise ValueError("evaluation.aw_pw_fw requires binary classes 0/1")
    for split in ("train", "val", "test", "bolivia"):
        if split in {"train", "val"} or config["dataset"].get(f"{split}_csv"):
            try:
                build_dataset(config, split)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Missing file for dataset split {split!r}: {exc}") from exc
    return config

# Select chips from a split CSV; modalities can be overridden for statistics.
def build_dataset(config, split, preprocess_func=None, modalities=None):
    return Sen1Floods11Dataset(
        split_csv=config["dataset"][f"{split}_csv"],
        modalities=tuple(modalities or config["dataset"]["modalities"]),
        preprocess_func=preprocess_func,
        include_permanent_water=(split != "train" and config["evaluation"].get("aw_pw_fw", False)),
    )

# Return a reusable normalizer and the actual statistics used by every split.
def build_preprocessing(config):
    statistics = None
    # Estimate data-derived statistics from raw training images only.
    # Fixed pretraining statistics are selected by from_config below.
    if config["preprocessing"]["normalization"]["statistics"] == "training_dataset":
        statistics = get_or_compute_stats(
            dataset_factory=lambda modalities: build_dataset(config, "train", modalities=modalities),
            modalities=config["dataset"]["modalities"],
            split_csv=config["dataset"]["train_csv"],
            dataset_name=config["dataset"]["name"],
            cache_dir="data/statistics",
        )
    normalizer = NormalizeFromStats.from_config(config, training_statistics=statistics)
    return normalizer, normalizer.statistics

# Apply one shared random crop to images and labels, then normalize the images.
# A module-level callable also supports DataLoader worker serialization.
class TrainingPreprocess:
    def __init__(self, normalizer, crop_size):
        self.normalizer = normalizer
        self.crop_size = crop_size

    def __call__(self, sample):
        return self.normalizer(random_crop(sample, self.crop_size))

# Assemble loaders with random training crops and full-chip evaluation inputs.
def build_loaders(config, normalizer):
    loaders, seen = {}, {}
    for split in ("train", "val", "test", "bolivia"):
        if not config["dataset"].get(f"{split}_csv"):
            if split in {"train", "val"}:
                raise ValueError(f"Missing {split}_csv")
            continue
        transform = (TrainingPreprocess(normalizer, config["preprocessing"]["crop_size"])
                     if split == "train" else normalizer)
        dataset = build_dataset(config, split, preprocess_func=transform)
        # Detect chips reused across splits before training starts.
        for paths in dataset.data_list:
            chip = paths["target"].name
            if chip in seen:
                raise ValueError(f"Split overlap: {chip} in {seen[chip]} and {split}")
            seen[chip] = split
        loaders[split] = DataLoader(
            dataset,
            batch_size=(config["training"]["batch_size"] if split == "train"
                        else config["evaluation"].get("batch_size", 1)),
            shuffle=(split == "train"),
            num_workers=config["training"].get("num_workers", 0),
        )
    return loaders

# Only parameters requiring gradients enter the optimizer; frozen weights are excluded.
def build_optimizer(config, model):
    
    factories = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    
    if config["name"] not in factories:
        raise ValueError(f"Unknown optimizer: {config['name']}")
    
    options = {"lr": config["learning_rate"], "weight_decay": config.get("weight_decay", 0)}
    
    if config["name"] == "sgd":
        options["momentum"] = config.get("momentum", 0.9)
  
    return factories[config["name"]]([p for p in model.parameters() if p.requires_grad], **options)

# The trainer steps after optimizer updates, so convert the YAML epoch period to batches.
# Skipped all-ignore batches do not advance the schedule.
def build_scheduler(config, optimizer, batches_per_epoch, epochs=None):
    
    config = config or {"name": "none"}
    
    if config["name"] == "none":
        return None
    
    if config["name"] == "cosine_annealing":
        if type(epochs) is not int or epochs < 1 or batches_per_epoch < 1:
            raise ValueError("Cosine annealing requires positive epochs and batches_per_epoch")
        # Decay over the configured training budget, with no warm restarts.
        # The trainer advances this after each optimizer update; ignored batches do not advance it.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * batches_per_epoch,
            eta_min=config.get("eta_min", 0.0),
        )

    if config["name"] != "cosine_annealing_warm_restarts":
        raise ValueError("Supported schedulers: none, cosine_annealing, cosine_annealing_warm_restarts")
    
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(config["t0_epochs"] * batches_per_epoch),
        T_mult=config.get("t_mult", 1), eta_min=config.get("eta_min", 0.0),
    )

# Read in chunks to identify large checkpoint files without loading them into RAM.
def file_sha256(path):
    
    digest = hashlib.sha256()
    
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
            
    return digest.hexdigest()

# Create the output directory and reject NaN/Infinity values in saved JSON.
def write_json(path, data):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


# Summarize recorded results only; never select checkpoints using test scores.
def build_run_summary(config, history, evaluations=None, checkpoint_sha256=None):
    options = config["evaluation"]
    selection_metric = options.get("checkpoint_metric", "miou")
    selection_class = options.get("checkpoint_class", 1) if selection_metric == "iou" else None
    validated = [row for row in history if row.get("validation") is not None]

    def peak(metric, class_id=None):
        candidates = []
        for row in validated:
            metrics = row["validation"]
            value = metrics["iou_per_class"][class_id] if metric == "iou" else metrics[metric]
            if value is not None:
                candidates.append((row, value))
        if not candidates:
            return None
        # max keeps the first tied epoch, matching the trainer's strict > rule.
        row, value = max(candidates, key=lambda item: item[1])
        return {"epoch": row["epoch"], "value": value}

    selected = peak(selection_metric, selection_class)
    best_checkpoint = None
    if selected is not None:
        row = next(row for row in validated if row["epoch"] == selected["epoch"])
        best_checkpoint = {
            "path": "checkpoints/best.pt", "sha256": checkpoint_sha256,
            "epoch": selected["epoch"], "selection_split": "val",
            "selection_metric": selection_metric, "selection_class": selection_class,
            "selection_value": selected["value"], "train_loss": row["train_loss"],
            "validation": row["validation"],
        }
    return {
        "experiment": config["experiment"], "model": config["model"]["name"],
        "modalities": config["dataset"]["modalities"],
        "training": {
            "epochs_requested": config["training"]["epochs"],
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "validate_every": config["training"].get("validate_every", 5),
            "last_train_loss": history[-1]["train_loss"] if history else None,
            "minimum_train_loss": min((row["train_loss"] for row in history), default=None),
            "history_path": "metrics/history.json",
        },
        "best_validation_observed": {
            "miou": peak("miou"), "water_iou": peak("iou", 1),
        },
        "best_checkpoint": best_checkpoint,
        "evaluation_of_best_checkpoint": {
            split: {key: value for key, value in results.items() if key != "per_sample"}
            for split, results in (evaluations or {}).items()
        },
    }
