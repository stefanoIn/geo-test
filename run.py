"""Run YAML-configured segmentation experiments from the project folder.

Run ``python run.py --path configs/example.yaml`` or call run_experiment(path).
The same runner handles pretrained GeoFMs and from-scratch CNN baselines.
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
    load_config, build_dataset, build_preprocessing, build_loaders,
    experiment_random_seeds, is_supervised_baseline, log_training_crops,
    build_loss, build_optimizer, build_scheduler, file_sha256, write_json,
    build_run_summary, save_training_plots, save_test_plots, format_iou_metrics,
)

# Select the experiment here; paths are relative to the project folder.
CONFIG_PATH = Path("")


def run_experiment(config_path=CONFIG_PATH, device=None):
    """Train a fresh model, evaluate its best checkpoint, and return its run folder."""

    # Validate YAML before loading data or allocating model memory.
    try:
        config = load_config(config_path, show_config=False)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration rejected: {exc}", flush=True)
        raise

    # Create the run before setup so builder failures have a persistent log.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = Path(config["output"]["directory"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with capture_run_output(run_dir / "run.log"):
        print(f"Configuration: {config_path} | Run: {run_dir}", flush=True)
        print(yaml.safe_dump(config, sort_keys=False).rstrip(), flush=True)
        print("Configuration checks passed.", flush=True)
        return _run_experiment(config, run_dir, device)


def resume_experiment(run_dir, device=None):
    """Continue an interrupted run from last.pt using its saved configuration.

    Legacy checkpoints have no RNG state: model/optimizer/scheduler resume,
    but crop and shuffle streams restart from the configured seed.
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml", show_config=False)
    checkpoint_path = run_dir / "checkpoints/last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epoch = checkpoint["epoch"]
    if checkpoint.get("config") != config:
        raise ValueError("Saved configuration differs from last.pt; cannot resume safely")
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        if key == "scheduler_state_dict" and config["training"].get("scheduler", {}).get("name", "none") == "none":
            continue
        if key not in checkpoint:
            raise ValueError(f"Cannot resume: checkpoint lacks {key}")
    if not 0 < epoch < config["training"]["epochs"]:
        raise ValueError("Run has no remaining epochs under its saved budget")
    history = json.loads((run_dir / "metrics/history.json").read_text(encoding="utf-8"))
    if [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("History and last checkpoint disagree; resolve before resuming")
    options = config["evaluation"]
    scores = [
        row["validation"]["iou_per_class"][options.get("checkpoint_class", 1)]
        if options.get("checkpoint_metric", "miou") == "iou"
        else row["validation"]["miou"]
        for row in history if "validation" in row
    ]
    best = max((score for score in scores if score is not None), default=float("-inf"))
    if scores and not (run_dir / "checkpoints/best.pt").is_file():
        raise FileNotFoundError("Existing best.pt is required to retain checkpoint selection")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    for split, expected in provenance["split_sha256"].items():
        if file_sha256(config["dataset"][f"{split}_csv"]) != expected:
            raise ValueError(f"The {split} split changed since training")
    if provenance.get("pretrained_sha256"):
        if file_sha256(config["model"]["checkpoint"]) != provenance["pretrained_sha256"]:
            raise ValueError("Pretrained weights changed since training")
    del checkpoint
    with capture_run_output(run_dir / "run.log"):
        print(f"Resuming {checkpoint_path} at epoch {epoch + 1}.", flush=True)
        print("RNG state was not saved: crop/shuffle streams restart from the seed; "
              "this is not an exact replay of uninterrupted training.", flush=True)
        return _run_experiment(config, run_dir, device, resume={
            "epoch": epoch, "history": history, "best": best,
        })


def _run_experiment(config, run_dir, device, resume=None):
    """Execute an already validated experiment inside its logging context."""
    seed = config["experiment"].get("seed", 42)
    random_seeds = experiment_random_seeds(seed)

    # Seed sampling and initialization; CUDA operations may still be nondeterministic.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(random_seeds["model"])

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    baseline = is_supervised_baseline(config)
    pretrained_path = config["model"].get("checkpoint")
    if not baseline:
        if pretrained_path is None or not Path(pretrained_path).is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    # One data pipeline and one trainable-model contract for all architectures.
    print("[Build] preprocessing", flush=True)
    normalizer, _ = build_preprocessing(config)
    print("[Build] datasets and DataLoaders", flush=True)
    loaders = build_loaders(config, normalizer)
    print("[Build] model", flush=True)
    model = build_model(config).to(device)
    print("[Build] loss", flush=True)
    criterion = build_loss(config["training"]["loss"], device)
    print("[Build] optimizer", flush=True)
    optimizer = build_optimizer(config["training"]["optimizer"], model)
    print("[Build] scheduler", flush=True)
    scheduler = build_scheduler(
        config["training"].get("scheduler"), optimizer,
        len(loaders["train"]), epochs=config["training"]["epochs"],
    )

    if resume is not None:
        saved_normalization = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
        current_normalization = dict(
            statistics=normalizer.statistics, method=normalizer.method,
            band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
        )
        if json.loads(json.dumps(current_normalization)) != saved_normalization:
            raise ValueError("Normalization changed since training; cannot resume")
        load_checkpoint(run_dir / "checkpoints/last.pt", model,
                        optimizer=optimizer, scheduler=scheduler, map_location=device)
        print(f"Restored optimizer LR: {optimizer.param_groups[0]['lr']:.8g}", flush=True)
    else:
        report_runtime_setup(
            run_dir, config, normalizer, loaders, model,
            criterion, optimizer, scheduler, device,
        )

        write_json(run_dir / "normalization.json", dict(
            statistics=normalizer.statistics, method=normalizer.method,
            band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
        ))

        # Record weights (when pretrained) and split hashes for reproducibility.
        write_json(run_dir / "provenance.json", {
            "pretrained_path": pretrained_path,
            "pretrained_sha256": (
                file_sha256(pretrained_path) if pretrained_path is not None else None
            ),
            "initialization": "random" if baseline else "pretrained",
            "split_sha256": {
                split: file_sha256(config["dataset"][f"{split}_csv"])
                for split in loaders
            },
            "seed": seed,
            "random_seeds": random_seeds,
            "randomness_protocol": (
                "independent_streams_v1; worker crop seed = "
                "crops + worker seed modulo 2**63"
            ),
        })

    options = config["evaluation"]

    # Same inference settings for validation and optional held-out splits.
    def evaluate_split(split, epoch=None):
        return evaluate(
            model, loaders[split], device,
            num_classes=config["head"]["num_classes"],
            criterion=criterion, ignore_index=criterion.ignore_index,
            tile_size=options.get("tile_size"), stride=options.get("stride"),
            tile_batch_size=options.get("tile_batch_size", 1),
            input_keys=config["dataset"]["modalities"],
            return_per_sample=options.get("return_per_sample", True),
            sample_id_key=options.get("sample_id_key", "chip_id"),
            class_names=options.get("class_names"),
            aw_pw_fw=options.get("aw_pw_fw", False),
            sample_log_path=(run_dir / "debug" / "validation_chips.jsonl" if split == "val" else None),
            sample_log_metadata={"split": split, "epoch": epoch},
            prediction_dir=(
                run_dir / "predictions" / split
                if split in {"test", "bolivia"} and options.get("save_predictions", False)
                else None
            ),
        )

    epochs = config["training"]["epochs"]
    every = config["training"].get("validate_every", 5)

    # Save full weights for scratch CNNs and for fine-tuned GeoFMs.
    # Only frozen pretrained encoders can be reconstructed from external weights.
    components = (
        ["adapter", "head"]
        if not baseline and config["model"].get("freeze_encoder", True)
        else None
    )

    best = resume["best"] if resume is not None else float("-inf")
    history = resume["history"] if resume is not None else []
    start_epoch = resume["epoch"] + 1 if resume is not None else 1
    best_path = run_dir / "checkpoints/best.pt"

    print(
        f"Device: {device} | Run: {run_dir} | Validate every {every} epochs",
        flush=True,
    )

    def observe_first_batch(images, target, logits):
        report_first_training_batch(
            run_dir, images, target, logits, criterion.ignore_index
        )

    crop_log_path = run_dir / "debug" / "training_crops.jsonl"
    print(f"Crop debug log (epochs 1-{epochs}): {crop_log_path}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        lr_start = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            scheduler=scheduler,
            input_keys=config["dataset"]["modalities"],
            batch_observer=observe_first_batch if epoch == 1 else None,
            batch_logger=(
                lambda batch_index, batch: log_training_crops(
                    crop_log_path, epoch, batch_index, batch, criterion.ignore_index)
            ),
        )
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "lr_start": lr_start,
            "lr_end": optimizer.param_groups[0]["lr"],
        }

        validation = None
        if epoch % every == 0 or epoch == epochs:
            validation = evaluate_split("val", epoch=epoch)
            save_evaluation_report(
                run_dir / "metrics" / f"validation_{epoch:03d}.json",
                validation, metadata={"split": "val", "epoch": epoch},
            )
            row["validation"] = {
                k: v for k, v in validation.items() if k != "per_sample"
            }

            # Validation alone selects the best checkpoint: never test/Bolivia.
            score = (
                validation["iou_per_class"][options.get("checkpoint_class", 1)]
                if options.get("checkpoint_metric", "miou") == "iou"
                else validation["miou"]
            )
            if score is not None and score > best:
                best = score
                save_checkpoint(
                    best_path, model, epoch,
                    optimizer=optimizer, scheduler=scheduler,
                    metrics=row["validation"], config=config,
                    components=components,
                )

        # Keep the last checkpoint separately from the validation-selected best.
        save_checkpoint(
            run_dir / "checkpoints/last.pt", model, epoch,
            optimizer=optimizer, scheduler=scheduler,
            metrics=row.get("validation"), config=config,
            components=components,
        )

        history.append(row)
        write_json(run_dir / "metrics/history.json", history)
        write_json(run_dir / "summary.json", build_run_summary(config, history))
        if validation is not None:
            save_training_plots(run_dir, config, history)

        message = (
            f"Epoch {epoch:03d}/{epochs} | LR {lr_start:.2e} | "
            f"Train loss {loss:.4f}"
        )
        if validation is not None:
            message += f" | Val loss {validation['loss']:.4f}"
            message += " | " + format_iou_metrics(config, validation)
        print(message, flush=True)

    if not best_path.is_file():
        raise RuntimeError(
            "No best checkpoint was selected; inspect validation metrics"
        )

    # Restore validation-selected weights, then optionally evaluate held-out data.
    selected = load_checkpoint(best_path, model, map_location=device)
    checkpoint_hash = file_sha256(best_path)
    evaluations = {}
    write_json(
        run_dir / "summary.json",
        build_run_summary(config, history, evaluations, checkpoint_hash),
    )

    for split in ("test", "bolivia"):
        if split in loaders:
            results = evaluate_split(split)
            save_evaluation_report(
                run_dir / "metrics" / f"{split}.json", results,
                metadata={
                    "split": split, "checkpoint": str(best_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "selected_epoch": selected["epoch"], "seed": seed,
                },
            )
            evaluations[split] = results
            write_json(
                run_dir / "summary.json",
                build_run_summary(config, history, evaluations, checkpoint_hash),
            )
            save_test_plots(run_dir, config, split, results, selected["epoch"])
            print(f"{split}: {format_iou_metrics(config, results)}")
            if "aw_pw_fw" in results:
                for category, values in results["aw_pw_fw"]["summary"].items():
                    print(
                        f"  {category}: mean per-chip IoU={values['mean_chip_iou']}, "
                        f"pooled IoU={values['total_iou']}"
                    )
    return run_dir


def run_evaluation(run_dir, device=None, splits=None, save_predictions=None):
    """Evaluate a saved best checkpoint without training or recomputing stats.

    Results go into a new evaluations/<id> folder inside the original run.
    Relative data and external weight paths use the project working directory.
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
        split for split in ("test", "bolivia")
        if config["dataset"].get(f"{split}_csv")
    ]
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Select at least one distinct evaluation split")
    for split in splits:
        if split not in {"val", "test", "bolivia"} or not config["dataset"].get(f"{split}_csv"):
            raise ValueError(f"Unavailable evaluation split: {split!r}")

    normalizer = NormalizeFromStats(**json.loads(
        (run_dir / "normalization.json").read_text(encoding="utf-8")
    ))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = run_dir / "evaluations" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)

    with capture_run_output(output / "evaluation.log"):
        print(
            f"Evaluation only | Source: {run_dir} | Output: {output} | Device: {device}",
            flush=True,
        )
        print(yaml.safe_dump(config, sort_keys=False).rstrip(), flush=True)
        (output / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        write_json(output / "normalization.json", dict(
            statistics=normalizer.statistics, method=normalizer.method,
            band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit,
        ))

        # A partial GeoFM checkpoint requires exactly the original pretrained weights.
        provenance_path = run_dir / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file() else {}
        )
        baseline = is_supervised_baseline(config)
        pretrained_path = config["model"].get("checkpoint")
        if not baseline and pretrained_path is None:
            raise ValueError("Pretrained GeoFM evaluation requires model.checkpoint")
        pretrained_hash = (
            file_sha256(pretrained_path)
            if pretrained_path is not None else None
        )
        if (provenance.get("pretrained_sha256")
                and pretrained_hash != provenance["pretrained_sha256"]):
            raise ValueError("Pretrained weights differ from the original run")

        model = build_model(config).to(device)
        selected = load_checkpoint(checkpoint, model, map_location=device)
        saved_config = selected.get("config")
        if saved_config:
            architecture_keys = (
                ("model", "head") if baseline else ("model", "adapter", "head")
            )
            if any(config.get(key) != saved_config.get(key) for key in architecture_keys):
                raise ValueError(
                    "Saved run architecture differs from checkpoint configuration"
                )
            # Also pin the input modality ordering and channel selection.
            if (config["dataset"]["modalities"]
                    != saved_config["dataset"]["modalities"]
                    or config["preprocessing"].get("band_indices")
                    != saved_config["preprocessing"].get("band_indices")):
                raise ValueError("Saved run input bands differ from checkpoint configuration")

        criterion = build_loss(config["training"]["loss"], device)
        checkpoint_hash = file_sha256(checkpoint)
        summary = {
            "source_run": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "pretrained_sha256": pretrained_hash,
            "selected_epoch": selected["epoch"],
            "evaluations": {},
        }
        seeds = experiment_random_seeds(config["experiment"].get("seed", 42))
        for split in splits:
            split_hash = file_sha256(config["dataset"][f"{split}_csv"])
            expected_hash = provenance.get("split_sha256", {}).get(split)
            if expected_hash and split_hash != expected_hash:
                raise ValueError(f"{split} split differs from the original run")

            dataset = build_dataset(config, split, preprocess_func=normalizer)
            loader = DataLoader(
                dataset, batch_size=options.get("batch_size", 1),
                shuffle=False,
                num_workers=config["training"].get("num_workers", 0),
                generator=torch.Generator().manual_seed(seeds[f"{split}_loader"]),
            )
            print(
                f"Evaluating {split}: {len(dataset)} chips | "
                f"selected epoch {selected['epoch']}", flush=True,
            )
            results = evaluate(
                model, loader, device,
                num_classes=config["head"]["num_classes"],
                criterion=criterion, ignore_index=criterion.ignore_index,
                tile_size=options.get("tile_size"), stride=options.get("stride"),
                tile_batch_size=options.get("tile_batch_size", 1),
                input_keys=config["dataset"]["modalities"],
                return_per_sample=options.get("return_per_sample", True),
                sample_id_key=options.get("sample_id_key", "chip_id"),
                class_names=options.get("class_names"),
                aw_pw_fw=options.get("aw_pw_fw", False),
                prediction_dir=(
                    output / "predictions" / split
                    if options.get("save_predictions", False) else None
                ),
            )
            save_evaluation_report(
                output / "metrics" / f"{split}.json", results,
                metadata={
                    "split": split, "split_sha256": split_hash,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "selected_epoch": selected["epoch"],
                },
            )
            save_test_plots(output, config, split, results, selected["epoch"])
            summary["evaluations"][split] = {
                k: v for k, v in results.items() if k != "per_sample"
            }
            write_json(output / "summary.json", summary)
            print(f"{split}: {format_iou_metrics(config, results)}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate an experiment from YAML."
    )
    parser.add_argument(
        "--path", type=Path, default=CONFIG_PATH, metavar="YAML",
        help=(
            f"Experiment YAML path (default: {CONFIG_PATH}). "
            "Run from the project folder."
        ),
    )
    parser.add_argument(
        "--evaluate-only", type=Path, metavar="RUN_DIR",
        help="Evaluate an existing best checkpoint without training.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["val", "test", "bolivia"],
        help="Evaluation-only splits (default: available test and Bolivia).",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", default=None,
        help="Enable per-chip figures in evaluation-only mode; otherwise use saved YAML.",
    )
    parser.add_argument(
        "--resume", type=Path, metavar="RUN_DIR",
        help="Continue training from a run's last.pt and saved configuration.",
    )
    args = parser.parse_args()
    if args.resume is not None:
        if (args.path != CONFIG_PATH or args.evaluate_only is not None
                or args.splits is not None or args.save_predictions is not None):
            parser.error("--resume uses the saved run; omit other mode/config options")
        resume_experiment(args.resume)
    elif args.evaluate_only is not None:
        if args.path != CONFIG_PATH:
            parser.error("--evaluate-only uses the run's saved config; omit --path")
        run_evaluation(
            args.evaluate_only, splits=args.splits,
            save_predictions=args.save_predictions,
        )
    else:
        if args.splits is not None or args.save_predictions is not None:
            parser.error("--splits and --save-predictions require --evaluate-only")
        run_experiment(args.path)
