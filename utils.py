"""Configuration, data setup and training-component helpers for experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, get_worker_info

from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.preprocessing import NormalizeFromStats
from src.data.statistics import get_or_compute_stats
from src.data.transforms import random_crop
from src.models.factory import resolve_croma_modality


BASELINE_MODELS = frozenset({"unet", "resnet_unet"})


def is_supervised_baseline(config):
    """Whether a configuration selects a fully supervised CNN baseline."""
    return (
        isinstance(config, dict)
        and isinstance(config.get("model"), dict)
        and config["model"].get("name") in BASELINE_MODELS
    )


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
def load_config(path, *, show_config=False):
    
    path = Path(path)
    
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}. Run from the project folder.")
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    
    # Print the supplied settings even if a later validation check rejects them.
    if show_config:
        print(f"\nConfiguration: {path}", flush=True)
        print(yaml.safe_dump(config, sort_keys=False).rstrip(), flush=True)

   
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")

    baseline = is_supervised_baseline(config)

    sections = (
        "experiment", "dataset", "model", "head",
        "preprocessing", "training", "evaluation", "output",
    )

    if not baseline:
        sections += ("adapter",)

    for section in sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"YAML requires a '{section}' mapping")

    required = (
        "dataset.name",
        "dataset.modalities",
        "dataset.train_csv",
        "dataset.val_csv",
        "model.name",
        "head.num_classes",
        "preprocessing.crop_size",
        "preprocessing.normalization.method",
        "preprocessing.normalization.statistics",
        "training.epochs",
        "training.batch_size",
        "training.loss.name",
        "training.optimizer.name",
        "training.optimizer.learning_rate",
        "output.directory",
    )

    if baseline:
        required += ("model.in_channels",)
    else:
        required += (
            "model.checkpoint",
            "adapter.name",
            "head.name",
            "head.in_channels",
        )

    for field in required:
        value = config 
        
        for key in field.split("."):
            
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Missing required YAML field: {field}")
            
            value = value[key]
            
        if value is None:
            raise ValueError(f"YAML field cannot be null: {field}")

    # MODIFY THIS WHEN ADDING NEW COMPONENTS
    ###############################################
    choices = {
        "dataset.name": ("sen1floods11",),
        "model.name": (
            "croma",
            "terramind_v1_base",
            "terramind_v1_large",
            "unet",
            "resnet_unet",
        ),
        "training.loss.name": ("cross_entropy",),
        "training.optimizer.name": ("adamw", "adam", "sgd"),
        "preprocessing.normalization.method": (
            "zscore", "two_sigma_stretch",
        ),
        "preprocessing.normalization.statistics": (
            "training_dataset", "pretraining_fixed",
        ),
    }

    if not baseline:
        choices["adapter.name"] = ("token_to_spatial",)
        choices["head.name"] = ("fcn", "linear_segmentation")
    ####################################################
     
    for field, available in choices.items():
        value = config
        
        for key in field.split("."):
            
            value = value[key]
        if value not in available:
            raise ValueError(f"{field}: received {value!r}; expected one of {available!r}. Values are case-sensitive.")
        
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
        raise ValueError(
            f"dataset.modalities: received {modalities!r}; expected ['sar'], ['optical'], "
            "or ['sar', 'optical'] (either order). Use lowercase dataset keys, without duplicates. "
            "Use model.modality='sar' for CROMA radar; native spelling is handled internally."
        )
    
    if config["dataset"]["name"] != "sen1floods11":
        raise ValueError("The current dataset loader supports sen1floods11")
    
    # CNN baselines are fully specified supervised models: no pretrained
    # checkpoint, separate adapter, or pretrained feature head is required.
    if baseline:
        mc = config["model"]
        hc = config["head"]
        if "checkpoint" in mc or "freeze_encoder" in mc:
            raise ValueError(
                "Scratch baselines cannot define model.checkpoint or model.freeze_encoder"
            )
        if "adapter" in config:
            raise ValueError("Scratch baselines do not use an adapter section")
        if "name" in hc or "in_channels" in hc:
            raise ValueError(
                "For baselines, head contains task metadata (num_classes), "
                "not a separately constructed prediction head"
            )
        if (type(hc["num_classes"]) is not int or hc["num_classes"] < 2):
            raise ValueError("head.num_classes must be an integer >= 2")
        if config["preprocessing"]["normalization"]["statistics"] != "training_dataset":
            raise ValueError("Scratch baselines require training_dataset statistics")

        if mc["name"] == "resnet_unet":
            supported = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
            if mc.get("encoder_name", "resnet50") not in supported:
                raise ValueError(
                    f"Unsupported ResNet encoder: {mc.get('encoder_name')!r}"
                )

        # Channel accounting is specific to the Sen1Floods11 loader.
        raw_channels = {"sar": 2, "optical": 13}
        indices_by_modality = config["preprocessing"].get("band_indices") or {}
        if not isinstance(indices_by_modality, dict):
            raise ValueError("preprocessing.band_indices must be a mapping")
        if set(indices_by_modality) - set(modalities):
            raise ValueError("band_indices specifies a modality not selected by dataset")

        expected_channels = 0
        for modality in modalities:
            selected = indices_by_modality.get(modality)
            if selected is None:
                expected_channels += raw_channels[modality]
                continue
            if (not isinstance(selected, list) or not selected
                    or any(type(i) is not int or not 0 <= i < raw_channels[modality]
                           for i in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError(f"Invalid band indices for {modality}")
            expected_channels += len(selected)

        if (type(mc["in_channels"]) is not int
                or mc["in_channels"] != expected_channels):
            raise ValueError(
                f"model.in_channels={mc['in_channels']!r}, "
                f"but preprocessing produces {expected_channels} channels"
            )

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
    if not isinstance(evaluation.get("save_predictions", False), bool):
        raise ValueError("evaluation.save_predictions must be true or false")
    
    if evaluation.get("checkpoint_metric", "miou") not in {"iou", "miou"}:
        raise ValueError("checkpoint_metric must be iou or miou")
    
    if not 0 <= evaluation.get("checkpoint_class", 1) < config["head"]["num_classes"]:
        raise ValueError("checkpoint_class is outside the class range")
    
    # Check that CROMA receives the selected sensors and its configured input size.
    if config["model"]["name"] == "croma":
        
        model = config["model"]
        
        resolve_croma_modality(model.get("modality", "both"), modalities)

        if model.get("image_resolution", 120) != config["preprocessing"]["crop_size"] or \
                evaluation.get("tile_size") != model.get("image_resolution", 120):
            raise ValueError("Match CROMA image_resolution, crop_size and evaluation.tile_size")
        
    # Scratch CNNs have no external weights; pretrained models do.
    if not baseline:
        checkpoint = Path(config["model"]["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found (model.checkpoint): {checkpoint}"
            )
    
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
    
    def __init__(self, normalizer, crop_size, seed=42):
        self.normalizer = normalizer
        self.crop_size = crop_size
        self.seed = seed
        self.generator = None
        self.worker_seed = None

    def __call__(self, sample):
        worker = get_worker_info()
        # Worker seeds come from the training loader's dedicated generator.
        # Recreated workers get new seeds each epoch; the main process keeps
        # advancing its crop generator across epochs when num_workers=0.
        worker_seed = worker.seed if worker is not None else 0
        if self.generator is None or self.worker_seed != worker_seed:
            self.generator = torch.Generator().manual_seed((self.seed + worker_seed) % (2**63))
            self.worker_seed = worker_seed
        return self.normalizer(random_crop(
            sample, self.crop_size, generator=self.generator, include_metadata=True,
            require_valid_target=True,
        ))


def experiment_random_seeds(seed):
    """Independent streams derived reproducibly from experiment.seed."""
    names = ("model", "shuffle", "crops", "train_loader", "val_loader",
             "test_loader", "bolivia_loader")
    return {name: (seed + offset) % (2**63) for offset, name in enumerate(names)}


# Assemble loaders with random training crops and full-chip evaluation inputs.
def build_loaders(config, normalizer):
    
    loaders, seen = {}, {}
    seeds = experiment_random_seeds(config["experiment"].get("seed", 42))
    
    for split in ("train", "val", "test", "bolivia"):
        
        if not config["dataset"].get(f"{split}_csv"):
            if split in {"train", "val"}:
                raise ValueError(f"Missing {split}_csv")
            continue
        
        transform = (TrainingPreprocess(normalizer, config["preprocessing"]["crop_size"],
                                        seed=seeds["crops"])
                     if split == "train" else normalizer)
        
        dataset = build_dataset(config, split, preprocess_func=transform)
        
        # Detect chips reused across splits before training starts.
        for paths in dataset.data_list:
            
            chip = paths["target"].name
            
            if chip in seen:
                raise ValueError(f"Split overlap: {chip} in {seen[chip]} and {split}")
            seen[chip] = split
            
        if split == "train":
            dataset.exclude_all_ignore_chips()
            print(f"Training chips: {len(dataset)} retained; "
                  f"{len(dataset.excluded_all_ignore_chips)} entirely ignored excluded. "
                  "Crops require at least one valid pixel (water-free crops retained).", flush=True)

        loaders[split] = DataLoader(
            dataset,
            batch_size=(config["training"]["batch_size"] if split == "train"
                        else config["evaluation"].get("batch_size", 1)),
            sampler=(RandomSampler(dataset, generator=torch.Generator().manual_seed(seeds["shuffle"]))
                     if split == "train" else None),
            generator=torch.Generator().manual_seed(seeds[f"{split}_loader"]),
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
            "miou": peak("miou"),
            "iou_per_class": [peak("iou", index) for index in range(config["head"]["num_classes"])],
            **({"water_iou": peak("iou", 1)} if config["dataset"]["name"] == "sen1floods11" else {}),
        },
        "best_checkpoint": best_checkpoint,
        "evaluation_of_best_checkpoint": {
            split: {key: value for key, value in results.items() if key != "per_sample"}
            for split, results in (evaluations or {}).items()
        },
    }


# Save curves inside this run; use the same checkpoint selection as summary.json.
def save_training_plots(run_dir, config, history):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not history:
        return
    validated = [row for row in history if row.get("validation") is not None]
    selected = build_run_summary(config, history)["best_checkpoint"]
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    loss_ax, iou_ax = figure.subplots(1, 2)
    loss_ax.plot([row["epoch"] for row in history],
                 [row["train_loss"] for row in history], label="Training loss")
    loss_rows = [row for row in validated if row["validation"].get("loss") is not None]
    loss_ax.plot([row["epoch"] for row in loss_rows],
                 [row["validation"]["loss"] for row in loss_rows],
                 marker="o", label="Validation loss")
    iou_rows = [row for row in validated if row["validation"].get("miou") is not None]
    iou_ax.plot([row["epoch"] for row in iou_rows],
                [100 * row["validation"]["miou"] for row in iou_rows],
                marker="o", label="Validation mIoU")
    loss_ax.set(title="Training and validation loss", ylabel="Loss")
    iou_ax.set(title="Validation mIoU", ylabel="mIoU (%)", ylim=(0, 100))

    for axis in (loss_ax, iou_ax):
        if selected is not None:
            axis.axvline(selected["epoch"], color="tab:red", linestyle="--",
                         label=f"Selected checkpoint: epoch {selected['epoch']}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(config["experiment"]["name"])
    for extension in ("png", "pdf"):
        figure.savefig(plot_dir / f"training_curves.{extension}", dpi=160)
    figure.clear()


# Plot held-out scores without recomputing metrics or selecting on test data.
def save_test_plots(run_dir, config, split, results, selected_epoch):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    categories = (results.get("aw_pw_fw", {}).get("summary", {})
                  if config["dataset"]["name"] == "sen1floods11" else {})
    figure = Figure(figsize=(15 if categories else 5, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3 if categories else 1, squeeze=False)[0]

    def bars(axis, labels, values, title):
        heights = [100 * value if value is not None else 0 for value in values]
        rectangles = axis.bar(labels, heights, color="tab:blue")
        axis.bar_label(rectangles, labels=[f"{100 * value:.2f}%" if value is not None
                                          else "Undefined" for value in values], padding=3)
        axis.set(title=title, ylabel="IoU (%)", ylim=(0, 110))
        axis.set_yticks(range(0, 101, 20))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)

    names = metric_class_names(config, results)
    bars(axes[0], ["mIoU"] + names,
         [results["miou"]] + list(results["iou_per_class"]), "mIoU and per-class IoU")
    axes[0].tick_params(axis="x", labelrotation=45)
    figure.set_size_inches(max(5, 0.8 * (len(names) + 1)) + (10 if categories else 0), 5.5)
    if categories:
        labels = [name for name in ("AW", "PW", "FW") if name in categories]
        bars(axes[1], labels, [categories[name]["mean_chip_iou"] for name in labels],
             "Mean per-chip Water IoU")
        bars(axes[2], labels, [categories[name]["total_iou"] for name in labels],
             "Pooled Water IoU by category")
    figure.suptitle(f"{config['experiment']['name']} | {split} | Selected epoch {selected_epoch}")
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(plot_dir / f"{split}_metrics.{extension}", dpi=160)
    figure.clear()


def metric_class_names(config, metrics):
    """Use saved/YAML class names, falling back to numbered classes."""
    count = len(metrics["iou_per_class"])
    names = metrics.get("class_names") or config["evaluation"].get("class_names")
    if names is None:
        return [f"Class {index}" for index in range(count)]
    if len(names) != count:
        raise ValueError("class_names must contain one name per evaluated class")
    return list(names)


def format_iou_metrics(config, metrics):
    """Format aggregate and per-class IoU without assuming a water class."""
    pairs = [("mIoU", metrics["miou"])] + list(zip(
        [f"{name} IoU" for name in metric_class_names(config, metrics)],
        metrics["iou_per_class"],
    ))
    return " | ".join(f"{name} {value:.4f}" if value is not None
                      else f"{name} undefined" for name, value in pairs)


# Mirror Python console output to the run folder, including setup failures.
class _RunOutput:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def capture_run_output(path):
    from contextlib import contextmanager, redirect_stdout, redirect_stderr
    import sys
    import traceback

    @contextmanager
    def capture():
        with Path(path).open("a", encoding="utf-8", buffering=1) as logfile:
            with redirect_stdout(_RunOutput(sys.stdout, logfile)), redirect_stderr(_RunOutput(sys.stderr, logfile)):
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    raise
    return capture()


def report_runtime_setup(run_dir, config, normalizer, loaders, model, criterion, optimizer, scheduler, device):
    """Report one common runtime contract for both GeoFMs and CNN baselines.

    Architecture-specific submodules are optional in this report; the common
    trainable-parameter, optimizer and data checks are never optional.
    """
    import platform

    def kind(obj):
        return None if obj is None else f"{type(obj).__module__}.{type(obj).__name__}"

    def parameters(module):
        if module is None:
            return None
        return {
            "class": kind(module),
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
        }

    baseline = is_supervised_baseline(config)
    encoder = getattr(model, "encoder", None)
    adapter = getattr(model, "adapter", None)
    head = getattr(model, "head", None)
    network = getattr(model, "network", None)  # SMP ResNetUNet wrapper
    backbone = (getattr(encoder, "backbone", None) if encoder is not None
                else getattr(network, "encoder", None))

    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    checks = {
        "optimizer_matches_trainable_parameters": (
            set(optimized) == trainable and len(optimized) == len(trainable)
        ),
        "has_trainable_parameters": bool(trainable),
    }
    if baseline:
        checks["baseline_all_parameters_trainable"] = all(
            p.requires_grad for p in model.parameters()
        )
    else:
        checks["requested_frozen_encoder_is_frozen"] = (
            not config["model"].get("freeze_encoder", True)
            or (encoder is not None and not any(
                p.requires_grad for p in encoder.parameters()
            ))
        )

    submodules = {
        name: parameters(module) for name, module in model.named_children()
    }
    report = {
        "experiment": config["experiment"],
        "paths": {
            "pretrained_checkpoint": config["model"].get("checkpoint"),
            "splits": {
                split: config["dataset"][f"{split}_csv"] for split in loaders
            },
            "run_directory": str(run_dir),
        },
        "training_settings": {
            "epochs": config["training"]["epochs"],
            "validate_every": config["training"].get("validate_every", 5),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "data": {
            split: {
                "dataset_class": kind(loader.dataset),
                "samples": len(loader.dataset),
                "batches": len(loader),
                "batch_size": loader.batch_size,
                "num_workers": loader.num_workers,
                "drop_last": loader.drop_last,
                "sampler": kind(loader.sampler),
                "modalities": list(loader.dataset.modalities),
                "transform": kind(loader.dataset.preprocess_func),
                "crop_size": getattr(loader.dataset.preprocess_func, "crop_size", None),
                "excluded_all_ignore_chips": loader.dataset.excluded_all_ignore_chips,
                "crop_sampling": ("uniform_valid_windows_v1; 32 retries then exact valid-window sampling"
                                  if split == "train" else None),
                "loader_seed": loader.generator.initial_seed() if loader.generator else None,
            }
            for split, loader in loaders.items()
        },
        "normalization": {
            "class": kind(normalizer),
            "method": normalizer.method,
            "band_indices": normalizer.band_indices,
            "use_8_bit": normalizer.use_8_bit,
            "statistics": normalizer.statistics,
        },
        "model": {
            **parameters(model),
            "name": config["model"]["name"],
            "initialization": "random" if baseline else "pretrained",
            "encoder": parameters(encoder),
            "backbone_class": kind(backbone),
            "adapter": ({**parameters(adapter), "structure": str(adapter)}
                        if adapter is not None else None),
            "head": ({**parameters(head), "structure": str(head)}
                     if head is not None else None),
            "submodules": submodules,
            "native_modality": getattr(encoder, "modality", None),
            "native_modalities": getattr(encoder, "modalities", None),
            "input_mapping": getattr(encoder, "input_mapping", None),
            "merge_method": config["model"].get("merge_method"),
            "input_channels": config["model"].get("in_channels"),
            "modalities": list(config["dataset"]["modalities"]),
            "num_classes": config["head"]["num_classes"],
            "trainable_parameter_names": [
                name for name, p in model.named_parameters() if p.requires_grad
            ],
        },
        "loss": {
            "class": kind(criterion),
            "structure": str(criterion),
            "reduction": getattr(criterion, "reduction", None),
            "ignore_index": getattr(criterion, "ignore_index", None),
            "class_weights": (
                criterion.weight.detach().cpu().tolist()
                if getattr(criterion, "weight", None) is not None else None
            ),
        },
        "optimizer": {
            "class": kind(optimizer),
            "groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ],
        },
        "scheduler": None if scheduler is None else {
            "class": kind(scheduler),
            **{
                key: getattr(scheduler, key)
                for key in ("T_max", "eta_min", "T_0", "T_mult", "last_epoch")
                if hasattr(scheduler, key)
            },
        },
        "random_seeds": experiment_random_seeds(
            config["experiment"].get("seed", 42)
        ),
        "evaluation_options": config["evaluation"],
        "sanity_checks": checks,
    }
    write_json(Path(run_dir) / "runtime_setup.json", report)

    model_report = report["model"]
    lines = [
        "Experiment setup (verified runtime objects)",
        f"Experiment: {config['experiment']['name']}",
        f"Model: {config['model']['name']} | Class: {type(model).__name__}",
        f"Initialization: {model_report['initialization']}",
        f"Model parameters: {model_report['parameters']:,} total, "
        f"{model_report['trainable_parameters']:,} trainable",
        f"Pretrained checkpoint: {config['model'].get('checkpoint')}",
        f"Native input: {model_report['native_modality'] or model_report['native_modalities']}",
        f"Input mapping: {model_report['input_mapping']}",
        f"Configured modalities: {config['dataset']['modalities']}",
        f"Device: {device} | GPU: {report['environment']['gpu']}",
        f"Seed: {config['experiment'].get('seed', 42)} | Streams: {report['random_seeds']}",
        f"Normalization: {normalizer.method} | Bands: {normalizer.band_indices}",
    ]
    for name, component in submodules.items():
        lines.append(
            f"Component {name}: {component['class']} | "
            f"{component['parameters']:,} total, "
            f"{component['trainable_parameters']:,} trainable"
        )
    if head is not None:
        lines.append(f"Head structure: {head}")

    for split, data in report["data"].items():
        lines.append(
            f"Data [{split}]: {config['dataset'][f'{split}_csv']} | "
            f"{data['samples']} samples | inputs={data['modalities']} | "
            f"batch={data['batch_size']} | workers={data['num_workers']} | "
            f"crop={data['crop_size']}"
        )
    lines.append(
        f"Loss: {type(criterion).__name__} | "
        f"reduction={report['loss']['reduction']} | "
        f"weights={report['loss']['class_weights']} | "
        f"ignore_index={report['loss']['ignore_index']}"
    )
    for index, group in enumerate(optimizer.param_groups):
        lines.append(
            f"Optimizer [{index}]: {type(optimizer).__name__} | "
            f"LR={group['lr']} | weight_decay={group.get('weight_decay')} | "
            f"betas={group.get('betas')}"
        )
    lines.extend([
        f"Scheduler: {report['scheduler']}",
        f"Epochs: {config['training']['epochs']} | "
        f"Validate every: {config['training'].get('validate_every', 5)}",
        f"Evaluation: tile={config['evaluation'].get('tile_size')} | "
        f"stride={config['evaluation'].get('stride')} | "
        f"tile_batch={config['evaluation'].get('tile_batch_size', 1)}",
        f"Checkpoint selection: validation "
        f"{config['evaluation'].get('checkpoint_metric', 'miou')}",
        f"Sanity checks: {checks}",
        f"Output: {run_dir}",
    ])
    readable = "\n".join(lines) + "\n"
    (Path(run_dir) / "runtime_setup.txt").write_text(readable, encoding="utf-8")
    print("\n" + readable, flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"Runtime sanity checks failed: {checks}")


def report_first_training_batch(run_dir, images, target, logits, ignore_index=255):
    """Observe the first real forward pass; do not consume extra batches/RNG."""
    def tensor_info(tensor):
        tensor = tensor.detach()
        finite = torch.isfinite(tensor)
        values = tensor[finite]
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                "all_finite": bool(finite.all().item()),
                "min": values.min().item() if values.numel() else None,
                "max": values.max().item() if values.numel() else None}

    valid = target != ignore_index
    labels, counts = torch.unique(target[valid], return_counts=True)
    report = {"inputs": {name: tensor_info(value) for name, value in images.items()},
              "target": {"shape": list(target.shape), "dtype": str(target.dtype),
                         "valid_pixels": int(valid.sum().item()), "ignored_pixels": int((~valid).sum().item()),
                         "valid_label_counts": {str(k): v for k, v in zip(labels.tolist(), counts.tolist())}},
              "logits": tensor_info(logits)}
    report["sanity_checks"] = {
        "finite_inputs": all(value["all_finite"] for value in report["inputs"].values()),
        "finite_logits": report["logits"]["all_finite"],
        "output_matches_target": logits.ndim == 4 and target.ndim == 3 and logits.shape[0] == target.shape[0] and logits.shape[-2:] == target.shape[-2:],
        "valid_label_range": bool(((labels >= 0) & (labels < logits.shape[1])).all().item()) if logits.ndim == 4 else False,
    }
    write_json(Path(run_dir) / "first_training_batch.json", report)
    print("\nFirst valid training batch:", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not all(report["sanity_checks"].values()):
        raise RuntimeError(f"First-batch sanity checks failed: {report['sanity_checks']}")


def log_training_crops(path, epoch, batch_index, batch, ignore_index=255):
    """Append one JSONL record per sampled crop, including skipped batches.

    Called by the trainer in the main process, after worker results are collated.
    Coordinates are zero-based row/column offsets in the original chip.
    Records describe batches received, not proof of completed optimizer updates.
    """
    metadata = batch.get("crop_metadata")
    if metadata is None:
        raise ValueError("Crop logging requires crop_metadata from TrainingPreprocess")
    targets = batch["target"].detach().cpu()
    ids = batch["chip_id"]
    batch_valid = int((targets != ignore_index).sum())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for i, target in enumerate(targets):
            valid = target != ignore_index
            labels, counts = torch.unique(target[valid], return_counts=True)
            record = {
                "epoch": epoch, "batch": batch_index, "position_in_batch": i,
                "batch_size": len(targets), "chip_id": str(ids[i]),
                "crop": {key: int(value[i]) for key, value in metadata.items()},
                "valid_pixels": int(valid.sum()),
                "ignored_pixels": int((~valid).sum()),
                "class_pixel_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
                "batch_valid_pixels": batch_valid,
                "batch_skipped_all_ignore": batch_valid == 0,
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
