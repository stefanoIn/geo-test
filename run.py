"""Run a YAML-configured segmentation experiment from the project folder.

Edit CONFIG_PATH and run this file, or call run_experiment(path) in Python.
Supported: Sen1Floods11, CROMA/TerraMind, FCN and mean cross-entropy.
"""

import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from src.models.factory import build_model
from src.evaluation.evaluator import evaluate, save_evaluation_report
from training.trainer import train_one_epoch
from training.checkpoints import save_checkpoint, load_checkpoint
from utils import (
    load_config, build_dataset, build_preprocessing, build_loaders,
    build_loss, build_optimizer, build_scheduler, file_sha256, write_json,
)

CONFIG_PATH = Path("configs/croma/croma_s1s2_fcn_all_hand_labels.yaml")


def run_experiment(config_path=CONFIG_PATH, device=None):
    """Train a fresh model, evaluate its best checkpoint, and return its run folder."""
    
    config = load_config(config_path)
    seed = config["experiment"].get("seed", 0)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not Path(config["model"]["checkpoint"]).is_file():
        raise FileNotFoundError(config["model"]["checkpoint"])
    
    normalizer, _ = build_preprocessing(config)
    loaders = build_loaders(config, normalizer)
    model = build_model(config).to(device)
    criterion = build_loss(config["training"]["loss"], device)
    optimizer = build_optimizer(config["training"]["optimizer"], model)
    scheduler = build_scheduler(config["training"].get("scheduler"), optimizer, len(loaders["train"]))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = Path(config["output"]["directory"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "normalization.json", dict(
        statistics=normalizer.statistics, method=normalizer.method,
        band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
    ))
    
    write_json(run_dir / "provenance.json", {
        "pretrained_path": config["model"]["checkpoint"],
        "pretrained_sha256": file_sha256(config["model"]["checkpoint"]),
        "split_sha256": {split: file_sha256(config["dataset"][f"{split}_csv"]) for split in loaders},
        "seed": seed,
    })
    
    options = config["evaluation"]
    
    def evaluate_split(split):
        return evaluate(
            model, loaders[split], device, num_classes=config["head"]["num_classes"],
            criterion=criterion, ignore_index=255,
            tile_size=options.get("tile_size"), stride=options.get("stride"),
            tile_batch_size=options.get("tile_batch_size", 1),
            input_keys=config["dataset"]["modalities"],
            return_per_sample=options.get("return_per_sample", True),
            sample_id_key=options.get("sample_id_key", "chip_id"),
            class_names=options.get("class_names"),
        )

    epochs = config["training"]["epochs"]
    every = config["training"].get("validate_every", 5)
    
    components = ["adapter", "head"] if config["model"].get("freeze_encoder", True) else None
    
    best, history = float("-inf"), []
    best_path = run_dir / "checkpoints/best.pt"
    
    print(f"Device: {device} | Run: {run_dir} | Validate every {every} epochs", flush=True)
    
    for epoch in range(1, epochs + 1):
        
        lr_start = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                               scheduler=scheduler, input_keys=config["dataset"]["modalities"])
        
        row = {"epoch": epoch, "train_loss": loss, "lr_start": lr_start,
               "lr_end": optimizer.param_groups[0]["lr"]}
        
        validation = None
        
        if epoch % every == 0 or epoch == epochs:
            validation = evaluate_split("val")
            
            save_evaluation_report(run_dir / "metrics" / f"validation_{epoch:03d}.json", validation,
                                   metadata={"split": "val", "epoch": epoch})
           
            row["validation"] = {k: v for k, v in validation.items() if k != "per_sample"}
            
            score = (validation["iou_per_class"][options.get("checkpoint_class", 1)]
                     if options.get("checkpoint_metric", "miou") == "iou" else validation["miou"])
            
            if score is not None and score > best:
                best = score
                save_checkpoint(best_path, model, epoch, optimizer=optimizer, scheduler=scheduler,
                                metrics=row["validation"], config=config, components=components)
                
        save_checkpoint(run_dir / "checkpoints/last.pt", model, epoch,
                        optimizer=optimizer, scheduler=scheduler, metrics=row.get("validation"),
                        config=config, components=components)
        
        history.append(row)
        
        write_json(run_dir / "metrics/history.json", history)
        
        message = f"Epoch {epoch:03d}/{epochs} | LR {lr_start:.2e} | Train loss {loss:.4f}"
        if validation is not None:
            message += f" | Water IoU {validation['iou_per_class'][1]:.4f} | mIoU {validation['miou']:.4f}" if validation['iou_per_class'][1] is not None else " | Water IoU undefined"
        print(message, flush=True)

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was selected; inspect validation metrics")
    
    selected = load_checkpoint(best_path, model, map_location=device)
    checkpoint_hash = file_sha256(best_path)
    
    for split in ("test", "bolivia"):
        if split in loaders:
            
            results = evaluate_split(split)
            
            save_evaluation_report(run_dir / "metrics" / f"{split}.json", results, metadata={
                "split": split, "checkpoint": str(best_path), "checkpoint_sha256": checkpoint_hash,
                "selected_epoch": selected["epoch"], "seed": seed,
            })
            
            print(f"{split}: mIoU={results['miou']}, water IoU={results['iou_per_class'][1]}")
    return run_dir


if __name__ == "__main__":
    run_experiment(CONFIG_PATH)
