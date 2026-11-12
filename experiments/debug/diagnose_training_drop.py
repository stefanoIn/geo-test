"""Locate a validation drop without changing the normal training loop.

python experiments/debug/diagnose_training_drop.py --path configs/unet/unet_s2_stability_20e.yaml
Use --target-epoch 10 with a GeoFM YAML to investigate a different interval.
The YAML scheduler horizon is preserved; this script stops after target_epoch + 1.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.evaluation.evaluator import evaluate, save_evaluation_report
from src.models.factory import build_model
from training.checkpoints import save_checkpoint
from training.trainer import train_one_epoch
from utils import (load_config, build_preprocessing, build_loaders, build_loss,
                   build_optimizer, build_scheduler, experiment_random_seeds,
                   capture_run_output, log_training_crops, write_json, file_sha256,
                   is_supervised_baseline)


class ObservedBatches:
    """The callback runs after the trainer finishes each batch, including skips."""
    def __init__(self, loader, callback):
        self.loader, self.callback = loader, callback

    def __iter__(self):
        for index, batch in enumerate(self.loader, 1):
            yield batch
            self.callback(index, batch)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batchnorm_buffers(model):
    return {f"{name}.{key}" if name else key: value.detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False) if value is not None}


def copy_buffers(model, values):
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for key, value in values.items():
            buffers[key].copy_(value)


@contextmanager
def preserve_randomness():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def compact(result):
    return {k: result[k] for k in ("miou", "loss", "iou_per_class", "recall_per_class",
                                  "precision_per_class", "confusion_matrix")}


def run_diagnostic(config_path, target_epoch=8, interval=10, device=None):
    config = load_config(config_path)
    stop_epoch = target_epoch + 1
    if target_epoch < 2 or stop_epoch > config["training"]["epochs"] or interval < 1:
        raise ValueError("Require target_epoch >= 2, target_epoch+1 <= YAML epochs, interval >= 1")
    output = Path(config["output"]["directory"]) / (
        "diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with capture_run_output(output / "diagnostic.log"):
        return _execute(config, output, target_epoch, interval, device)


def _execute(config, output, target_epoch, interval, device):
    seed = config["experiment"].get("seed", 42)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(experiment_random_seeds(seed)["model"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizer, _ = build_preprocessing(config)
    loaders = build_loaders(config, normalizer)
    model = build_model(config).to(device)
    criterion = build_loss(config["training"]["loss"], device)
    optimizer = build_optimizer(config["training"]["optimizer"], model)
    scheduler = build_scheduler(config["training"].get("scheduler"), optimizer,
                                len(loaders["train"]), epochs=config["training"]["epochs"])
    options = config["evaluation"]
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / "normalization.json", dict(statistics=normalizer.statistics,
        method=normalizer.method, band_indices=normalizer.band_indices, use_8_bit=normalizer.use_8_bit))
    # Deterministic, model-independent subset: first labelled chip per event in CSV order.
    dataset = loaders["val"].dataset
    indices, events = [], set()
    for i, paths in enumerate(dataset.data_list):
        chip = paths["target"].stem.removesuffix("_LabelHand")
        event = chip.rsplit("_", 1)[0]
        if event not in events:
            import rasterio
            with rasterio.open(paths["target"]) as src:
                mask = src.read(1)
            if np.any((mask != -1) & (mask != criterion.ignore_index)):
                indices.append(i); events.add(event)
    if not indices:
        raise ValueError("No labelled validation chips for fixed probe")
    probe = DataLoader(Subset(dataset, indices), batch_size=options.get("batch_size", 1),
                       shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0))
    probe_ids = [dataset.data_list[i]["target"].stem.removesuffix("_LabelHand") for i in indices]
    write_json(output / "diagnostic_setup.json", {
        "target_epoch": target_epoch, "stop_epoch": target_epoch + 1,
        "scheduler_horizon_epochs": config["training"]["epochs"], "interval_batches": interval,
        "probe_chip_ids": probe_ids, "batchnorm_buffers": list(batchnorm_buffers(model)),
        "split_sha256": {s: file_sha256(config["dataset"][f"{s}_csv"]) for s in loaders},
        "pretrained_sha256": file_sha256(config["model"]["checkpoint"]) if not is_supervised_baseline(config) else None,
        "note": "Diagnostic only; no test evaluation or checkpoint selection. Probe uses validation only."})
    print(f"Diagnostic: {output} | {device} | fixed probe: {len(indices)} chips", flush=True)
    print(f"Stop after epoch {target_epoch+1}; keep YAML scheduler horizon {config['training']['epochs']}", flush=True)
    history, probes = [], []
    anchor_bn = None
    components = ["adapter", "head"] if not is_supervised_baseline(config) and config["model"].get("freeze_encoder", True) else None

    def evaluate_guarded(loader, old_bn=None):
        before = state_digest(model)
        current_bn = batchnorm_buffers(model)
        modes = [(module, module.training) for module in model.modules()]
        try:
            if old_bn is not None:
                copy_buffers(model, old_bn)
            with preserve_randomness():
                result = evaluate(model, loader, device, num_classes=config["head"]["num_classes"],
                    criterion=criterion, ignore_index=criterion.ignore_index,
                    tile_size=options.get("tile_size"), stride=options.get("stride"),
                    tile_batch_size=options.get("tile_batch_size", 1),
                    input_keys=config["dataset"]["modalities"], sample_id_key=options.get("sample_id_key", "chip_id"),
                    class_names=options.get("class_names"), return_per_sample=True)
        finally:
            copy_buffers(model, current_bn)
            for module, mode in modes: module.training = mode
        if state_digest(model) != before:
            raise RuntimeError("Evaluation changed model state")
        return result

    def inspect(epoch, batch_index, repeat=False):
        normal = evaluate_guarded(probe)
        row = {"epoch": epoch, "batch": batch_index, "normal": compact(normal)}
        if repeat:
            second = evaluate_guarded(probe)
            row["repeat_identical_counts"] = normal["confusion_matrix"] == second["confusion_matrix"]
            row["repeat_loss_difference"] = abs(normal["loss"] - second["loss"])
            write_json(output / "repeatability.json", row)
        if anchor_bn:
            row["epoch_anchor_bn"] = compact(evaluate_guarded(probe, anchor_bn))
        probes.append(row)
        write_json(output / "probe_history.json", probes)
        save_evaluation_report(output / "metrics" / f"probe_e{epoch:02d}_b{batch_index:03d}.json", normal)
        # Model snapshots include all buffers; optimizer state is stored at epoch boundaries.
        save_checkpoint(output / "checkpoints" / f"probe_e{epoch:02d}_b{batch_index:03d}.pt",
                        model, epoch, config=config, components=components)
        print(f"Probe e{epoch} b{batch_index}: mIoU={normal['miou']:.4f}" +
              (f" | epoch-{target_epoch-1} BN={row['epoch_anchor_bn']['miou']:.4f}" if anchor_bn else " | no BN swap"), flush=True)

    for epoch in range(1, target_epoch + 2):
        def after_batch(index, batch):
            valid = batch["target"] != criterion.ignore_index
            skipped = not bool(valid.any())
            grad_sq = sum(float(p.grad.detach().float().square().sum()) for p in model.parameters()
                          if p.requires_grad and p.grad is not None) if not skipped else 0.0
            row = {"epoch": epoch, "batch": index, "chip_ids": batch["chip_id"],
                   "skipped": skipped, "gradient_l2": grad_sq**0.5,
                   "learning_rate_after_update": optimizer.param_groups[0]["lr"]}
            with (output / "batch_updates.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if epoch == target_epoch and (index % interval == 0 or index == len(loaders["train"])):
                inspect(epoch, index)
        loss = train_one_epoch(model, ObservedBatches(loaders["train"], after_batch), optimizer,
            criterion, device, scheduler=scheduler, input_keys=config["dataset"]["modalities"],
            batch_logger=lambda index, batch: log_training_crops(output / "debug/training_crops.jsonl",
                epoch, index, batch, criterion.ignore_index))
        result = evaluate_guarded(loaders["val"])
        save_evaluation_report(output / "metrics" / f"validation_{epoch:03d}.json", result)
        history.append({"epoch": epoch, "train_loss": loss, "validation": compact(result)})
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}: train loss={loss:.4f} | full val mIoU={result['miou']:.4f}", flush=True)
        if epoch >= target_epoch - 1:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:02d}.pt", model, epoch,
                optimizer=optimizer, scheduler=scheduler, config=config, components=components)
        if epoch == target_epoch - 1:
            anchor_bn = batchnorm_buffers(model)
            inspect(epoch, len(loaders["train"]), repeat=True)
        if epoch == target_epoch and anchor_bn:
            swapped = evaluate_guarded(loaders["val"], anchor_bn)
            save_evaluation_report(output / "metrics/target_epoch_full_val_anchor_bn.json", swapped)
            print(f"Full validation with anchor BN: {swapped['miou']:.4f}", flush=True)
    write_json(output / "completion.json", {"completed": True, "stop_epoch": target_epoch + 1,
               "batchnorm_comparison_available": bool(anchor_bn)})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("configs/unet/unet_s2_stability_20e.yaml"))
    parser.add_argument("--target-epoch", type=int, default=8)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    path = args.path.resolve()
    os.chdir(PROJECT)
    run_diagnostic(path, args.target_epoch, args.interval, args.device)
