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

def build_loss(config, device):
    name = config["name"]

    if name == "cross_entropy":
        weights = config.get("class_weights")
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return torch.nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=config.get("ignore_index", 255),
        )

    if name == "water_dice":
        return WaterDiceLoss(ignore_index=config.get("ignore_index", 255))



def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for section in ("experiment", "dataset", "model", "adapter", "head",
                    "preprocessing", "training", "evaluation", "output"):
        if not isinstance(config, dict) or not isinstance(config.get(section), dict):
            raise ValueError(f"YAML requires a '{section}' mapping")
    if config["dataset"]["name"] != "sen1floods11":
        raise ValueError("The current dataset loader supports sen1floods11")
    loss = config["training"]["loss"]
    if loss["name"] != "cross_entropy":
        raise ValueError("This runner currently supports cross_entropy; other losses need implementation")
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
    if config["model"]["name"] == "croma":
        model = config["model"]
        expected = {"SAR": {"sar"}, "optical": {"optical"}, "both": {"sar", "optical"}}
        if set(config["dataset"]["modalities"]) != expected.get(model["modality"]):
            raise ValueError("CROMA model.modality and dataset.modalities disagree")
        if model["image_resolution"] != config["preprocessing"]["crop_size"] or \
                evaluation.get("tile_size") != model["image_resolution"]:
            raise ValueError("Match CROMA image_resolution, crop_size and evaluation.tile_size")
    return config

def build_dataset(config, split, preprocess_func=None, modalities=None):
    return Sen1Floods11Dataset(
        split_csv=config["dataset"][f"{split}_csv"],
        modalities=tuple(modalities or config["dataset"]["modalities"]),
        preprocess_func=preprocess_func,
    )

def build_preprocessing(config):
    statistics = None
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

class TrainingPreprocess:
    def __init__(self, normalizer, crop_size):
        self.normalizer = normalizer
        self.crop_size = crop_size

    def __call__(self, sample):
        return self.normalizer(random_crop(sample, self.crop_size))

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

def build_optimizer(config, model):
    
    factories = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    
    if config["name"] not in factories:
        raise ValueError(f"Unknown optimizer: {config['name']}")
    
    options = {"lr": config["learning_rate"], "weight_decay": config.get("weight_decay", 0)}
    
    if config["name"] == "sgd":
        options["momentum"] = config.get("momentum", 0.9)
  
    return factories[config["name"]]([p for p in model.parameters() if p.requires_grad], **options)

def build_scheduler(config, optimizer, batches_per_epoch):
    
    config = config or {"name": "none"}
    
    if config["name"] == "none":
        return None
    
    if config["name"] != "cosine_annealing_warm_restarts":
        raise ValueError("Supported schedulers: none, cosine_annealing_warm_restarts")
    
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(config["t0_epochs"] * batches_per_epoch),
        T_mult=config.get("t_mult", 1), eta_min=config.get("eta_min", 0.0),
    )

def file_sha256(path):
    
    digest = hashlib.sha256()
    
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
            
    return digest.hexdigest()

def write_json(path, data):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
