
"""Smoke test: real Sen1Floods11 training crop and full-image validation."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run import build_dataset, build_preprocessing, load_config
from src.adapters.token_to_spatial import TokenToSpatial
from src.data.transforms import random_crop
from src.encoders.croma.croma import CROMAEncoder
from src.evaluation.sliding_window import sliding_window_predict
from src.heads.fcn import FCNHead
from src.models.dense_prediction import DensePredictionModel


CONFIG = "configs/croma/croma_s1s2_fcn_all_hand_labels.yaml"


def main():
    cfg = load_config(CONFIG)
    torch.manual_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = cfg["dataset"]["modalities"]
    ignore_index = 255
    print(f"Device: {device}")

    # 1. Reuse the existing normalisation pipeline.
    normalizer, _ = build_preprocessing(cfg)
    crop_size = cfg["preprocessing"]["crop_size"]

    def train_preprocess(sample):
        return random_crop(normalizer(sample), size=crop_size)

    # Cropping is applied only to the training dataset.
    train_data = build_dataset(cfg, "train", preprocess_func=train_preprocess)
    val_data = build_dataset(cfg, "val", preprocess_func=normalizer)

    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=0
    )

    # Find a crop containing at least one labelled pixel.
    train_batch = next(
        (batch for batch in train_loader
         if (batch["target"] != ignore_index).any().item()),
        None,
    )
    if train_batch is None:
        raise RuntimeError("No training crop contains valid labels")

    # 2. Build the complete model.
    encoder_cfg = cfg["model"]
    checkpoint = Path(encoder_cfg["checkpoint"])

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = DensePredictionModel(
        encoder=CROMAEncoder(
            checkpoint_path=checkpoint,
            size=encoder_cfg["size"],
            modality=encoder_cfg["modality"],
            image_resolution=encoder_cfg["image_resolution"],
            freeze=encoder_cfg["freeze_encoder"],
        ),
        adapter=TokenToSpatial(
            feature_key=cfg["adapter"]["feature_key"],
            grid_size=cfg["adapter"]["grid_size"],
        ),
        head=FCNHead(
            in_channels=cfg["head"]["in_channels"],
            num_classes=cfg["head"]["num_classes"],
        ),
    ).to(device)

    # 3. Training forward and backward on a real crop.
    model.train()
    train_images = {
        name: train_batch[name].to(device) for name in modalities
    }
    train_target = train_batch["target"].to(device)

    expected_hw = (
        (crop_size, crop_size)
        if isinstance(crop_size, int)
        else tuple(crop_size)
    )

    assert train_target.shape[-2:] == expected_hw
    assert all(x.shape[-2:] == expected_hw for x in train_images.values())
    assert set(torch.unique(train_target).tolist()) <= {0, 1, ignore_index}

    train_logits = model(train_images)

    assert train_logits.shape == (
        1, cfg["head"]["num_classes"], *expected_hw
    )
    assert torch.isfinite(train_logits).all()

    train_loss = F.cross_entropy(
        train_logits, train_target, ignore_index=ignore_index
    )
    assert torch.isfinite(train_loss)
    train_loss.backward()

    assert all(p.grad is None for p in model.encoder.parameters())
    assert not model.encoder.backbone.training
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.head.parameters()
    )

    print(
        f"TRAIN | SAR {tuple(train_images['sar'].shape)} | "
        f"Optical {tuple(train_images['optical'].shape)} | "
        f"Logits {tuple(train_logits.shape)} | "
        f"Loss {train_loss.item():.4f}"
    )

    # No optimizer.step(): we are testing, not training.
    del train_logits, train_loss, train_images, train_target, train_batch
    model.zero_grad(set_to_none=True)

    # 4. Validation on a complete image.
    val_batch = next(
        (
            batch for batch in val_loader
            if (batch["target"] != ignore_index).any().item()
        ),
        None,
    )

    if val_batch is None:
        raise RuntimeError(
            "No validation image contains valid labels. "
            "Check the split and mask decoding."
        )

    labels, counts = torch.unique(
        val_batch["target"], return_counts=True
    )
    print("Validation labels:", dict(zip(labels.tolist(), counts.tolist())))

    val_images = {
        name: val_batch[name].to(device) for name in modalities
    }
    val_target = val_batch["target"].to(device)
    height, width = val_target.shape[-2:]

    assert all(
        x.shape[-2:] == (height, width)
        for x in val_images.values()
    )
    assert set(torch.unique(val_target).tolist()) <= {0, 1, ignore_index}

    model.eval()

    with torch.inference_mode():
        val_logits = sliding_window_predict(
            model,
            val_images,
            tile_size=cfg["evaluation"]["tile_size"],
            tile_batch_size=1,
        )

        assert val_logits.shape == (
            1, cfg["head"]["num_classes"], height, width
        )
        assert torch.isfinite(val_logits).all()

        val_predictions = val_logits.argmax(dim=1)

        if (val_target != ignore_index).any().item():
            val_loss = F.cross_entropy(
                val_logits, val_target, ignore_index=ignore_index
            )
            assert torch.isfinite(val_loss)
            print(
                f"VAL   | Logits {tuple(val_logits.shape)} | "
                f"Predictions {tuple(val_predictions.shape)} | "
                f"Loss {val_loss.item():.4f}"
            )
        else:
            print(
                f"VAL   | Logits {tuple(val_logits.shape)} | "
                "all labels ignored, loss skipped"
            )

    print("Integration test 02 passed!")


if __name__ == "__main__":
    main()
