"""Run a YAML-configured segmentation experiment from the project folder.

Run python run.py --path configs/example.yaml, or call run_experiment(path).
Without --path, CONFIG_PATH is used.
Supported: Sen1Floods11, CROMA/TerraMind, FCN and mean cross-entropy.
"""

import argparse
import json
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
    capture_run_output, report_runtime_setup, report_first_training_batch,
    load_config, build_dataset, build_preprocessing, build_loaders, experiment_random_seeds,
    build_loss, build_optimizer, build_scheduler, file_sha256, write_json, build_run_summary, save_training_plots, save_test_plots, format_iou_metrics,
)

# Select the experiment here; paths are relative to the project folder.
CONFIG_PATH = Path("")


def run_experiment(config_path=CONFIG_PATH, device=None):
    """Train a fresh model, evaluate its best checkpoint, and return its run folder."""
    
    # Validate the YAML before loading data or allocating model memory.
    try:
        config = load_config(config_path, show_config=False)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration rejected: {exc}", flush=True)
        raise
    # Create the run before setup so builder failures have a persistent log.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = Path(config["output"]["directory"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with capture_run_output(run_dir / "run.log"):
        print(f"Configuration: {config_path} | Run: {run_dir}", flush=True)
        print(yaml.safe_dump(config, sort_keys=False).rstrip(), flush=True)
        print("Configuration checks passed.", flush=True)
        return _run_experiment(config, run_dir, device)


def _run_experiment(config, run_dir, device):
    """Execute an already validated experiment inside its logging context."""
    seed = config["experiment"].get("seed", 42)
    random_seeds = experiment_random_seeds(seed)
    
    # Seed sampling and initialization; CUDA operations may still be nondeterministic.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(random_seeds["model"])
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not Path(config["model"]["checkpoint"]).is_file():
        raise FileNotFoundError(config["model"]["checkpoint"])
    
    # Build the data pipeline and trainable components from the same configuration.
    print("[Build] preprocessing", flush=True)
    normalizer, _ = build_preprocessing(config)
    print("[Build] datasets and DataLoaders", flush=True)
    loaders = build_loaders(config, normalizer)
    print("[Build] encoder, adapter and head", flush=True)
    model = build_model(config).to(device)
    print("[Build] loss", flush=True)
    criterion = build_loss(config["training"]["loss"], device)
    print("[Build] optimizer", flush=True)
    optimizer = build_optimizer(config["training"]["optimizer"], model)
    print("[Build] scheduler", flush=True)
    scheduler = build_scheduler(config["training"].get("scheduler"), optimizer,
                                len(loaders["train"]), epochs=config["training"]["epochs"])

    report_runtime_setup(run_dir, config, normalizer, loaders, model, criterion, optimizer, scheduler, device)

    write_json(run_dir / "normalization.json", dict(
        statistics=normalizer.statistics, method=normalizer.method,
        band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
    ))
    
    # Hash weights and split files so their identity can be checked later.
    write_json(run_dir / "provenance.json", {
        "pretrained_path": config["model"]["checkpoint"],
        "pretrained_sha256": file_sha256(config["model"]["checkpoint"]),
        "split_sha256": {split: file_sha256(config["dataset"][f"{split}_csv"]) for split in loaders},
        "seed": seed,
        "random_seeds": random_seeds,
        "randomness_protocol": "independent_streams_v1; worker crop seed = crops + worker seed modulo 2**63",
    })
    
    options = config["evaluation"]
    
    # Use the same inference settings for validation and both held-out splits.
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
            aw_pw_fw=options.get("aw_pw_fw", False),
            prediction_dir=(run_dir / "predictions" / split
                            if split in {"test", "bolivia"} and options.get("save_predictions", False)
                            else None),
        )

    epochs = config["training"]["epochs"]
    every = config["training"].get("validate_every", 5)
    
    # A frozen backbone can be reloaded from pretrained weights; save its head only.
    # Fine-tuning requires saving the full model, including the updated encoder.
    components = ["adapter", "head"] if config["model"].get("freeze_encoder", True) else None
    
    best, history = float("-inf"), []
    best_path = run_dir / "checkpoints/best.pt"
    
    print(f"Device: {device} | Run: {run_dir} | Validate every {every} epochs", flush=True)
    
    def observe_first_batch(images, target, logits):
        report_first_training_batch(run_dir, images, target, logits, criterion.ignore_index)

    for epoch in range(1, epochs + 1):
        
        # The trainer performs optimizer updates and steps the per-update scheduler.
        lr_start = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                               scheduler=scheduler, input_keys=config["dataset"]["modalities"],
                               batch_observer=observe_first_batch if epoch == 1 else None)
        
        row = {"epoch": epoch, "train_loss": loss, "lr_start": lr_start,
               "lr_end": optimizer.param_groups[0]["lr"]}
        
        validation = None
        
        # Always validate the final epoch, even when it is outside the interval.
        if epoch % every == 0 or epoch == epochs:
            validation = evaluate_split("val")
            
            save_evaluation_report(run_dir / "metrics" / f"validation_{epoch:03d}.json", validation,
                                   metadata={"split": "val", "epoch": epoch})
           
            row["validation"] = {k: v for k, v in validation.items() if k != "per_sample"}
            
            # Select checkpoints using validation IoU/mIoU; test data is not used.
            score = (validation["iou_per_class"][options.get("checkpoint_class", 1)]
                     if options.get("checkpoint_metric", "miou") == "iou" else validation["miou"])
            
            if score is not None and score > best:
                best = score
                save_checkpoint(best_path, model, epoch, optimizer=optimizer, scheduler=scheduler,
                                metrics=row["validation"], config=config, components=components)
                
        # Keep the latest state separately from the best validation checkpoint.
        save_checkpoint(run_dir / "checkpoints/last.pt", model, epoch,
                        optimizer=optimizer, scheduler=scheduler, metrics=row.get("validation"),
                        config=config, components=components)
        
        history.append(row)
        
        write_json(run_dir / "metrics/history.json", history)
        # A compact overview stays available even before training finishes.
        write_json(run_dir / "summary.json", build_run_summary(config, history))
        if validation is not None:
            save_training_plots(run_dir, config, history)
        
        message = f"Epoch {epoch:03d}/{epochs} | LR {lr_start:.2e} | Train loss {loss:.4f}"
        if validation is not None:
            message += f" | Val loss {validation['loss']:.4f}"
            message += " | " + format_iou_metrics(config, validation)
        print(message, flush=True)

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was selected; inspect validation metrics")
    
    # Replace the final-epoch weights with the validation-selected weights.
    selected = load_checkpoint(best_path, model, map_location=device)
    checkpoint_hash = file_sha256(best_path)
    evaluations = {}
    write_json(run_dir / "summary.json", build_run_summary(
        config, history, evaluations, checkpoint_hash))
    
    # Report the two held-out splits separately, linked to the evaluated checkpoint.
    for split in ("test", "bolivia"):
        if split in loaders:
            
            results = evaluate_split(split)
            
            save_evaluation_report(run_dir / "metrics" / f"{split}.json", results, metadata={
                "split": split, "checkpoint": str(best_path), "checkpoint_sha256": checkpoint_hash,
                "selected_epoch": selected["epoch"], "seed": seed,
            })
            
            evaluations[split] = results
            write_json(run_dir / "summary.json", build_run_summary(
                config, history, evaluations, checkpoint_hash))

            save_test_plots(run_dir, config, split, results, selected["epoch"])

            print(f"{split}: {format_iou_metrics(config, results)}")
            if "aw_pw_fw" in results:
                for category, values in results["aw_pw_fw"]["summary"].items():
                    print(f"  {category}: mean per-chip IoU={values['mean_chip_iou']}, "
                          f"pooled IoU={values['total_iou']}")
    return run_dir


def run_evaluation(run_dir, device=None, splits=None, save_predictions=None):
    """Evaluate an existing best checkpoint; never train or recompute statistics.

    Results go into a new evaluations/<id> folder inside the original run.
    Relative dataset/weight paths are resolved from the project working directory.
    """
    from torch.utils.data import DataLoader
    from src.data.preprocessing import NormalizeFromStats

    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoints/best.pt"
    for path in (run_dir / "config.yaml", run_dir / "normalization.json", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    options = config["evaluation"]
    if save_predictions is not None:
        options["save_predictions"] = save_predictions
    if not isinstance(options.get("save_predictions", False), bool):
        raise ValueError("evaluation.save_predictions must be true or false")
    splits = list(splits) if splits is not None else [
        split for split in ("test", "bolivia") if config["dataset"].get(f"{split}_csv")]
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Select at least one distinct evaluation split")
    for split in splits:
        if split not in {"val", "test", "bolivia"} or not config["dataset"].get(f"{split}_csv"):
            raise ValueError(f"Unavailable evaluation split: {split!r}")
    normalizer = NormalizeFromStats(**json.loads(
        (run_dir / "normalization.json").read_text(encoding="utf-8")))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = run_dir / "evaluations" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with capture_run_output(output / "evaluation.log"):
        print(f"Evaluation only | Source: {run_dir} | Output: {output} | Device: {device}", flush=True)
        print(yaml.safe_dump(config, sort_keys=False).rstrip(), flush=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json(output / "normalization.json", dict(
            statistics=normalizer.statistics, method=normalizer.method,
            band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit))
        # Partial checkpoints require the original pretrained backbone.
        provenance_path = run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
        pretrained_hash = file_sha256(config["model"]["checkpoint"])
        if provenance.get("pretrained_sha256") and pretrained_hash != provenance["pretrained_sha256"]:
            raise ValueError("Pretrained weights differ from the original run")
        model = build_model(config).to(device)
        selected = load_checkpoint(checkpoint, model, map_location=device)
        # Plot-saving flags may change; the trained architecture must not.
        saved_config = selected.get("config")
        if saved_config and any(config[key] != saved_config[key] for key in ("model", "adapter", "head")):
            raise ValueError("Saved run architecture differs from the checkpoint configuration")
        criterion = build_loss(config["training"]["loss"], device)
        checkpoint_hash = file_sha256(checkpoint)
        summary = {"source_run": str(run_dir), "checkpoint": str(checkpoint),
                   "checkpoint_sha256": checkpoint_hash, "pretrained_sha256": pretrained_hash,
                   "selected_epoch": selected["epoch"], "evaluations": {}}
        seeds = experiment_random_seeds(config["experiment"].get("seed", 42))
        for split in splits:
            split_hash = file_sha256(config["dataset"][f"{split}_csv"])
            expected_hash = provenance.get("split_sha256", {}).get(split)
            if expected_hash and split_hash != expected_hash:
                raise ValueError(f"{split} split differs from the original run")
            dataset = build_dataset(config, split, preprocess_func=normalizer)
            loader = DataLoader(dataset, batch_size=options.get("batch_size", 1),
                                shuffle=False, num_workers=config["training"].get("num_workers", 0),
                                generator=torch.Generator().manual_seed(seeds[f"{split}_loader"]))
            print(f"Evaluating {split}: {len(dataset)} chips | selected epoch {selected['epoch']}", flush=True)
            results = evaluate(
                model, loader, device, num_classes=config["head"]["num_classes"],
                criterion=criterion, ignore_index=criterion.ignore_index,
                tile_size=options.get("tile_size"), stride=options.get("stride"),
                tile_batch_size=options.get("tile_batch_size", 1),
                input_keys=config["dataset"]["modalities"],
                return_per_sample=options.get("return_per_sample", True),
                sample_id_key=options.get("sample_id_key", "chip_id"),
                class_names=options.get("class_names"), aw_pw_fw=options.get("aw_pw_fw", False),
                prediction_dir=output / "predictions" / split if options.get("save_predictions", False) else None)
            save_evaluation_report(output / "metrics" / f"{split}.json", results, metadata={
                "split": split, "split_sha256": split_hash, "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "selected_epoch": selected["epoch"]})
            save_test_plots(output, config, split, results, selected["epoch"])
            summary["evaluations"][split] = {k: v for k, v in results.items() if k != "per_sample"}
            write_json(output / "summary.json", summary)
            print(f"{split}: {format_iou_metrics(config, results)}", flush=True)
    return output


# Importing this module exposes run_experiment without starting training.
if __name__ == "__main__":
    # Command-line selection overrides the default without editing this file.
    parser = argparse.ArgumentParser(description="Train and evaluate an experiment from YAML.")
    parser.add_argument(
        "--path", type=Path, default=CONFIG_PATH, metavar="YAML",
        help=f"Experiment YAML path (default: {CONFIG_PATH}). Run from the project folder.",
    )
    parser.add_argument("--evaluate-only", type=Path, metavar="RUN_DIR",
                        help="Evaluate an existing run's best checkpoint without training.")
    parser.add_argument("--splits", nargs="+", choices=["val", "test", "bolivia"],
                        help="Evaluation-only splits (default: available test and Bolivia).")
    parser.add_argument("--save-predictions", action="store_true", default=None,
                        help="Enable per-chip figures in evaluation-only mode; otherwise use saved YAML.")
    args = parser.parse_args()
    if args.evaluate_only is not None:
        if args.path != CONFIG_PATH:
            parser.error("--evaluate-only uses the run's saved config; omit --path")
        run_evaluation(args.evaluate_only, splits=args.splits, save_predictions=args.save_predictions)
    else:
        if args.splits is not None or args.save_predictions is not None:
            parser.error("--splits and --save-predictions require --evaluate-only")
        run_experiment(args.path)
