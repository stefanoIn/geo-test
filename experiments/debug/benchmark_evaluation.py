"""Bounded FP32 timing diagnostic. Run from GeoTest; never overwrites source weights.

Default: 12 fixed validation chips, tile batches 8/16/32, two repeats and
8 disposable training batches. Use --val-chips 0 for the full validation set,
or --train-batches 0 for evaluation only. No mixed precision is used.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import copy
import csv
from datetime import datetime
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch
from torch.utils.data import DataLoader, Subset
import yaml
from src.data.preprocessing import NormalizeFromStats
from src.models.factory import build_model
from src.evaluation import evaluator as ev
from src.evaluation.metrics import SegmentationMetrics, WaterCategoryMetrics
from training.checkpoints import load_checkpoint, save_checkpoint
from training.trainer import train_one_epoch
from utils import (build_dataset, build_loaders, build_loss, build_optimizer,
                   build_scheduler, file_sha256, write_json, save_training_plots,
                   capture_run_output, log_training_crops)

DEFAULT_RUN = ("runs/sen1floods11/croma/linear_tuning/s1s2/"
               "lr0.005_e50_warmup5/20261114_150137_b2170c8d")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timings:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.depth = defaultdict(int)

    @contextmanager
    def section(self, name):
        outer = self.depth[name] == 0
        if outer:
            sync()
            start = perf_counter()
        self.depth[name] += 1
        try:
            yield
        finally:
            self.depth[name] -= 1
            if outer:
                sync()
                self.seconds[name] += perf_counter() - start

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)
        return measured


class TimedLoader:
    def __init__(self, loader, timing):
        self.loader, self.timing = loader, timing

    def __iter__(self):
        with self.timing.section("data_loading"):
            iterator = iter(self.loader)
        while True:
            with self.timing.section("data_loading"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch


def run(args):
    os.chdir(PROJECT)
    source = Path(args.run)
    config = yaml.safe_load((source / "config.yaml").read_text())
    normalizer = NormalizeFromStats(**json.loads((source / "normalization.json").read_text()))
    checkpoint = source / "checkpoints" / args.checkpoint
    before = file_sha256(checkpoint)
    provenance = json.loads((source / "provenance.json").read_text())
    for split in ("train", "val"):
        if file_sha256(config["dataset"][f"{split}_csv"]) != provenance["split_sha256"][split]:
            raise ValueError(f"Changed {split} split")
    pretrained = config["model"].get("checkpoint")
    if provenance.get("pretrained_sha256") and file_sha256(pretrained) != provenance["pretrained_sha256"]:
        raise ValueError("Pretrained checkpoint changed")
    output = source / "diagnostics" / datetime.now().strftime("timing_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with capture_run_output(output / "timing.log"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_start = perf_counter()
        model = build_model(config).to(device)
        selected = load_checkpoint(checkpoint, model, map_location=device)
        if selected.get("config") != config:
            raise ValueError("Checkpoint configuration differs from saved YAML")
        criterion = build_loss(config["training"]["loss"], device)
        full_dataset = build_dataset(config, "val", preprocess_func=normalizer)
        count = min(args.val_chips or len(full_dataset), len(full_dataset))
        indices = torch.linspace(0, len(full_dataset) - 1, count).round().long().tolist()
        loader = DataLoader(Subset(full_dataset, indices),
                            batch_size=config["evaluation"].get("batch_size", 1),
                            num_workers=config["training"].get("num_workers", 0), shuffle=False,
                            generator=torch.Generator().manual_seed(42))
        options = config["evaluation"]
        keys = config["dataset"]["modalities"]
        size = options["tile_size"]
        # Warm up on one tile before timing; no optimizer updates.
        batch = next(iter(loader))
        model.eval()
        with torch.inference_mode():
            model({k: batch[k][:1, :, :size, :size].to(device) for k in keys})
        sync()
        setup_seconds = perf_counter() - setup_start
        print(f"FP32 only | checkpoint epoch {selected['epoch']} | {count}/{len(full_dataset)} validation chips")

        def evaluate(tile_batch, name, data=loader):
            return ev.evaluate(
                model, data, device, num_classes=config["head"]["num_classes"],
                criterion=criterion, ignore_index=criterion.ignore_index,
                tile_size=size, stride=options.get("stride"), tile_batch_size=tile_batch,
                input_keys=keys, return_per_sample=options.get("return_per_sample", True),
                sample_id_key=options.get("sample_id_key", "chip_id"),
                class_names=options.get("class_names"), aw_pw_fw=options.get("aw_pw_fw", False),
                sample_log_path=output / f"{name}_chips.jsonl",
                sample_log_metadata={"split": "val", "diagnostic": True},
            )

        baseline_tiles = options.get("tile_batch_size", 8)
        candidates = list(dict.fromkeys([baseline_tiles] + args.tile_batches))
        reference = None
        rows = []
        # Reverse repeat order to reduce systematic cache/order bias.
        for repeat in range(args.repeats):
            for tiles in (candidates if repeat % 2 == 0 else candidates[::-1]):
                name = f"tiles{tiles}_repeat{repeat + 1}"
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    sync()
                    start = perf_counter()
                    result = evaluate(tiles, name)
                    sync()
                    seconds = perf_counter() - start
                    if reference is None:
                        reference = result
                    start = perf_counter()
                    ev.save_evaluation_report(output / f"{name}.json", result)
                    save_seconds = perf_counter() - start
                    row = dict(tile_batch_size=tiles, repeat=repeat + 1,
                               seconds=seconds, report_save_seconds=save_seconds,
                               peak_allocated_mb=(torch.cuda.max_memory_allocated() / 2**20
                                                  if device.type == "cuda" else None),
                               miou=result["miou"], loss=result["loss"],
                               miou_delta_pp=100 * (result["miou"] - reference["miou"]),
                               confusion_matrix_matches=result["confusion_matrix"] == reference["confusion_matrix"],
                               aw_pw_fw_matches=result.get("aw_pw_fw") == reference.get("aw_pw_fw"))
                    rows.append(row)
                    write_json(output / "evaluation_timings.json", rows)
                    print(f"{name}: {seconds:.2f}s | mIoU {100 * result['miou']:.3f}% | "
                          f"delta {row['miou_delta_pp']:+.5f} pp", flush=True)
                    del result
                except torch.cuda.OutOfMemoryError:
                    print(f"{name}: out of GPU memory; skipping", flush=True)
                    torch.cuda.empty_cache()

        # Separate instrumented pass: synchronization adds overhead, so use the
        # uninstrumented rows above for speed comparisons, not this breakdown.
        timing = Timings()
        with ExitStack() as stack:
            stack.enter_context(patch.object(model, "forward", timing.wrap("model_forward", model.forward)))
            stack.enter_context(patch.object(ev, "sliding_window_predict",
                timing.wrap("sliding_total", ev.sliding_window_predict)))
            stack.enter_context(patch.object(criterion, "forward", timing.wrap("loss", criterion.forward)))
            for cls in (SegmentationMetrics, WaterCategoryMetrics):
                for method in ("update", "compute"):
                    stack.enter_context(patch.object(cls, method, timing.wrap("metrics", getattr(cls, method))))
            with timing.section("total"):
                evaluate(baseline_tiles, "profile", TimedLoader(loader, timing))
        profile = dict(timing.seconds)
        profile["tile_assembly_merge"] = profile["sliding_total"] - profile["model_forward"]
        profile["other_transfer_checks_logging"] = profile["total"] - sum(
            profile.get(k, 0) for k in ("sliding_total", "data_loading", "loss", "metrics"))

        # Time real FP32 updates on disposable in-memory weights. Reload the
        # original checkpoint afterwards; never save these updates to the source.
        training = None
        if args.train_batches:
            loaders = build_loaders(config, normalizer)
            optimizer = build_optimizer(config["training"]["optimizer"], model)
            scheduler = build_scheduler(config["training"].get("scheduler"), optimizer,
                                        len(loaders["train"]), epochs=config["training"]["epochs"])
            load_checkpoint(checkpoint, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
            n = min(args.train_batches, len(loaders["train"]))
            timing = Timings()
            with patch.object(model, "forward", timing.wrap("forward", model.forward)):
                with timing.section("total"):
                    train_one_epoch(model, islice(TimedLoader(loaders["train"], timing), n),
                                    optimizer, criterion, device, scheduler=scheduler, input_keys=keys,
                                    batch_logger=lambda i, b: log_training_crops(
                                        output / "training_crops.jsonl", selected["epoch"] + 1,
                                        i, b, criterion.ignore_index))
            training = dict(timing.seconds)
            training.update(batches=n, full_epoch_batches=len(loaders["train"]))
            training["backward_optimizer_transfer_logging"] = training["total"] - training["forward"] - training["data_loading"]
            training["estimated_epoch_seconds"] = training["total"] / n * len(loaders["train"])
            load_checkpoint(checkpoint, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
            model.eval()
            # Measure checkpoint I/O using the same payload format, then remove only
            # the temporary diagnostic file. Source checkpoints are never touched.
            temp = output / "timing_checkpoint.pt"
            start = perf_counter()
            save_checkpoint(temp, model, selected["epoch"], optimizer=optimizer,
                            scheduler=scheduler, config=config, metrics=selected.get("metrics"),
                            components=selected.get("saved_components"))
            training["one_checkpoint_save_seconds"] = perf_counter() - start
            temp.unlink()

        start = perf_counter()
        history = json.loads((source / "metrics/history.json").read_text())
        save_training_plots(output, config, history)
        plot_seconds = perf_counter() - start
        baseline = statistics.median(r["seconds"] for r in rows if r["tile_batch_size"] == baseline_tiles)
        aggregates = []
        for tiles in candidates:
            values = [r["seconds"] for r in rows if r["tile_batch_size"] == tiles]
            if values:
                median = statistics.median(values)
                aggregates.append(dict(tile_batch_size=tiles, median_seconds=median,
                    speedup=baseline / median,
                    estimated_full_validation_seconds=median * len(full_dataset) / count))
        summary = dict(source_run=str(source), checkpoint=str(checkpoint), checkpoint_sha256=before,
            checkpoint_epoch=selected["epoch"], precision="fp32", config=config,
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            torch_version=str(torch.__version__), validation_indices=indices,
            validation_chips=count, full_validation_chips=len(full_dataset),
            setup_seconds=setup_seconds, evaluation=aggregates, instrumented_evaluation=profile,
            training=training, training_plot_save_seconds=plot_seconds,
            notes=["Subset timings are estimates; verify the winner on full validation with --val-chips 0.",
                   "Instrumented timings include synchronization overhead and do not equal clean wall times.",
                   "Training uses new crops and disposable updates, not a replay of the interrupted epoch.",
                   "Tile size, stride, metrics, FP32 and source weights remain unchanged."])
        write_json(output / "summary.json", summary)
        with (output / "evaluation_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert file_sha256(checkpoint) == before, "Source checkpoint changed externally during diagnostic"
        print(json.dumps(summary, indent=2))
        print(f"Results: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", choices=["last.pt", "best.pt"], default="last.pt")
    parser.add_argument("--val-chips", type=int, default=12)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--tile-batches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.val_chips < 0 or args.train_batches < 0 or args.repeats < 1 or any(n < 1 for n in args.tile_batches):
        parser.error("Counts must be nonnegative; repeats and tile batches must be positive")
    run(args)
