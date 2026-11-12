"""Replay epoch 8 from epoch 7, then isolate the effect of batch 114.

Run from GeoTest: python experiments/debug/replay_suspect_batch.py
No test data, new epochs, or changes to the training framework are involved.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from diagnose_training_drop import preserve_randomness, state_digest, compact
from src.data.preprocessing import NormalizeFromStats
from src.evaluation.evaluator import evaluate, save_evaluation_report
from src.models.factory import build_model
from training.checkpoints import load_checkpoint, save_checkpoint
from training.trainer import train_one_epoch
from utils import build_dataset, build_loss, build_optimizer, build_scheduler, capture_run_output, file_sha256, write_json

DEFAULT_RUN = Path("runs/sen1floods11/baselines/unet/unet_s2_stability_20e/diagnostic_20261112_013506_4ebce3d1")


def cpu_copy(obj):
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().clone()
    if isinstance(obj, dict): return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple): return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None: torch.cuda.set_rng_state_all(state[3])


def apply_batch(model, batch, optimizer, criterion, scheduler, device, keys, intervention="normal"):
    """Keep scheduler time aligned even when omitting the suspect update."""
    if intervention == "normal":
        return train_one_epoch(model, [batch], optimizer, criterion, device,
                               scheduler=scheduler, input_keys=keys)
    optimizer.zero_grad(set_to_none=True)
    loss_value = None
    if intervention == "skip_update":
        model.train()
        target = batch["target"].to(device)
        loss = criterion(model({k: batch[k].to(device) for k in keys}), target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite intervention loss")
        loss.backward()  # Retain the forward/BN effect, but do not update weights or Adam state.
        loss_value = float(loss.detach())
    elif intervention != "skip_batch":
        raise ValueError(intervention)
    if scheduler is not None:
        scheduler.step()  # Deliberate control: same LR at subsequent batch indices.
    return loss_value


def replay(source, device=None):
    config = yaml.safe_load((source / "config.yaml").read_text())
    # The original checkpoints omit RNG states. This replay targets the non-dropout U-Net.
    if config["model"]["name"] != "unet":
        raise ValueError("This replay is for the current U-Net case; GeoFM controls require separate setup")
    setup = json.loads((source / "diagnostic_setup.json").read_text())
    if setup["target_epoch"] != 8:
        raise ValueError("Expected the completed epoch-8 diagnostic")
    for split in ("train", "val"):
        if file_sha256(config["dataset"][f"{split}_csv"]) != setup["split_sha256"][split]:
            raise ValueError(f"Changed {split} split")
    output = source / "replays" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with capture_run_output(output / "replay.log"):
        return _execute(source, output, config, setup, device)


def _execute(source, output, config, setup, device):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = config["experiment"].get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    normalizer = NormalizeFromStats(**json.loads((source / "normalization.json").read_text()))
    dataset = build_dataset(config, "train")
    indices = {p["target"].stem.removesuffix("_LabelHand"): i for i, p in enumerate(dataset.data_list)}
    logged = [json.loads(line) for line in (source / "debug/training_crops.jsonl").read_text().splitlines()]
    epoch_rows = [r for r in logged if r["epoch"] == 8]
    groups = {}
    for r in epoch_rows: groups.setdefault(r["batch"], []).append(r)
    if sorted(groups) != list(range(1, len(groups)+1)) or 120 not in groups:
        raise ValueError("Incomplete epoch-8 crop log")

    def batch_at(index):
        samples = []
        for r in sorted(groups[index], key=lambda r: r["position_in_batch"]):
            raw = dataset[indices[r["chip_id"]]]
            c = r["crop"]
            sample = {k: (v[..., c["top"]:c["top"]+c["height"], c["left"]:c["left"]+c["width"]]
                           if isinstance(v, torch.Tensor) and v.ndim in (2, 3) else v)
                      for k, v in raw.items()}
            target = sample["target"]
            valid = target != 255
            labs, counts = torch.unique(target[valid], return_counts=True)
            if int(valid.sum()) != r["valid_pixels"] or {str(int(k)): int(v) for k,v in zip(labs, counts)} != r["class_pixel_counts"]:
                raise ValueError(f"Reconstructed labels differ: {r['chip_id']}")
            samples.append(normalizer(sample))
        return default_collate(samples)

    validation = build_dataset(config, "val", preprocess_func=normalizer)
    val_indices = {p["target"].stem.removesuffix("_LabelHand"): i for i,p in enumerate(validation.data_list)}
    options = config["evaluation"]
    def loader(data):
        return DataLoader(data, batch_size=options.get("batch_size", 1), shuffle=False,
                          num_workers=0, generator=torch.Generator().manual_seed(0))
    probe = loader(Subset(validation, [val_indices[k] for k in setup["probe_chip_ids"]]))
    full = loader(validation)
    model = build_model(config).to(device)
    criterion = build_loss(config["training"]["loss"], device)
    optimizer = build_optimizer(config["training"]["optimizer"], model)
    scheduler = build_scheduler(config["training"].get("scheduler"), optimizer, len(groups), epochs=config["training"]["epochs"])
    load_checkpoint(source / "checkpoints/epoch_07.pt", model, optimizer=optimizer, scheduler=scheduler, map_location=device)
    keys = config["dataset"]["modalities"]

    def assess(data):
        before = state_digest(model)
        with preserve_randomness():
            result = evaluate(model, data, device, num_classes=config["head"]["num_classes"],
                criterion=criterion, ignore_index=criterion.ignore_index, tile_size=options.get("tile_size"),
                stride=options.get("stride"), tile_batch_size=options.get("tile_batch_size",1),
                input_keys=keys, sample_id_key="chip_id", class_names=options.get("class_names"), return_per_sample=True)
        if state_digest(model) != before: raise RuntimeError("Evaluation changed model state")
        return result

    print(f"Replay output: {output} | {device}", flush=True)
    print("Restored epoch 7 weights, optimizer and scheduler; replaying exact logged epoch-8 crops.", flush=True)
    for index in range(1,111):
        batch = batch_at(index)
        if bool((batch["target"] != criterion.ignore_index).any()):
            apply_batch(model,batch,optimizer,criterion,scheduler,device,keys)
        if index % 20 == 0: print(f"Replayed batch {index}/110",flush=True)
    start = assess(probe)
    original = next(r["normal"] for r in json.loads((source/"probe_history.json").read_text()) if r["epoch"]==8 and r["batch"]==110)
    reproduction = {"original":original,"replay":compact(start),
                    "absolute_miou_difference":abs(start["miou"]-original["miou"]),
                    "note":"CUDA replay need not be bitwise identical. Compare normal branch with original before causal interpretation."}
    write_json(output/"reproduction_check.json",reproduction)
    print(f"Batch 110 probe: replay={start['miou']:.4f}, original={original['miou']:.4f}",flush=True)
    # All three branches start from this IDENTICAL weights/Adam/scheduler/RNG state.
    anchor = dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                  scheduler=cpu_copy(scheduler.state_dict()) if scheduler else None,rng=rng_state())
    cached = {index:batch_at(index) for index in range(111,121)}
    summary = {"source_run":str(source),"checkpoint_sha256":file_sha256(source/'checkpoints/epoch_07.pt'),
               "suspect_batch":114,"branches":{},"reproduction":reproduction,
               "control":"Same starting state and subsequent LR; omit batch 114 or only its optimizer update."}
    for branch in ("normal","skip_batch","skip_update"):
        model.load_state_dict(anchor["model"]);optimizer.load_state_dict(anchor["optimizer"])
        if scheduler:scheduler.load_state_dict(anchor["scheduler"])
        restore_rng(anchor["rng"])
        curve=[{"batch":110,**compact(start)}]
        for index,batch in cached.items():
            action=branch if index==114 else "normal"
            loss=apply_batch(model,batch,optimizer,criterion,scheduler,device,keys,action)
            result=assess(probe)
            curve.append({"batch":index,"intervention":action,"batch_loss":loss,
                          "lr_after_batch":optimizer.param_groups[0]["lr"],**compact(result)})
            write_json(output/f"{branch}_curve.json",curve)
            print(f"{branch} batch {index}: probe mIoU={result['miou']:.4f}, recall={result['recall_per_class'][1]:.4f}",flush=True)
        full_result=assess(full)
        save_evaluation_report(output/f"{branch}_full_validation.json",full_result)
        save_checkpoint(output/'checkpoints'/f'{branch}_batch120.pt',model,8,config=config)
        summary["branches"][branch]={"probe":compact(result),"full_validation":compact(full_result)}
        write_json(output/'summary.json',summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for branch in summary['branches']:
        curve=json.loads((output/f'{branch}_curve.json').read_text())
        axes[0].plot([r['batch'] for r in curve],[100*r['miou'] for r in curve],marker='o',label=branch)
        axes[1].plot([r['batch'] for r in curve],[100*r['recall_per_class'][1] for r in curve],marker='o',label=branch)
    for ax,title in zip(axes,['Fixed validation probe mIoU (%)','Fixed validation probe water recall (%)']):
        ax.axvline(114,color='grey',linestyle='--');ax.set(title=title,xlabel='Epoch 8 batch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    write_json(output/'completion.json',{'completed':True})
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--device',default=None)
    args=parser.parse_args();source=args.source_run.resolve();os.chdir(PROJECT)
    replay(source,args.device)
