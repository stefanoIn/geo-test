"""Integration test: local TerraMind v1 Large -> adapter -> FCN."""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.adapters.token_to_spatial import TokenToSpatial
from src.encoders.terramind.terramind import TerraMindEncoder
from src.heads.fcn import FCNHead
from src.models.dense_prediction import DensePredictionModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/terramind/terramind_v1_large_s1_fcn_test.yaml"


def main():
    with CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_cfg = cfg["model"]
    encoder = TerraMindEncoder(
        checkpoint_path=ROOT / model_cfg["checkpoint"],
        freeze=model_cfg["freeze_encoder"],
    )
    adapter = TokenToSpatial(
        feature_key=cfg["adapter"]["feature_key"],
        grid_size=cfg["adapter"]["grid_size"],
    )
    head = FCNHead(
        in_channels=cfg["head"]["in_channels"],
        num_classes=cfg["head"]["num_classes"],
    )
    model = DensePredictionModel(encoder, adapter, head).to(device)
    model.train()

    size = cfg["test"]["input_size"]
    batch_size = cfg["test"]["batch_size"]
    images = {"sar": torch.randn(batch_size, 2, size, size, device=device)}

    # Check the actual representation independently of the wrapper.
    with torch.no_grad():
        features = model.encoder(images)
        spatial = model.adapter(features)
    print("Tokens:", tuple(features["tokens"].shape))
    print("Spatial features:", tuple(spatial.shape))

    grid_side = (size + 15) // 16
    assert features["tokens"].shape == (batch_size, grid_side ** 2, 1024)
    assert spatial.shape == (batch_size, 1024, grid_side, grid_side)

    logits = model(images)
    print("Logits:", tuple(logits.shape))
    assert logits.shape == (batch_size, cfg["head"]["num_classes"], size, size)
    assert torch.isfinite(logits).all()

    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.backbone.training

    target = torch.randint(0, cfg["head"]["num_classes"],
                           (batch_size, size, size), device=device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum().item() > 0
        for p in model.head.parameters()
    )
    print(f"Dummy loss: {loss.item():.4f}")
    print("Encoder frozen: OK")
    print("FCN gradients: OK")
    print("Integration test 03 passed!")


if __name__ == "__main__":
    main()
