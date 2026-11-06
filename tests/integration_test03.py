"""TerraMind v1 Large -> adapter -> FCN (requires real TerraTorch/checkpoint)."""

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

    mc = cfg["model"]
    encoder = TerraMindEncoder(
        checkpoint_path=ROOT / mc["checkpoint"],
        size=mc["size"],
        modalities=mc["modalities"],
        input_mapping=mc["input_mapping"],
        merge_method=mc["merge_method"],
        freeze=mc["freeze_encoder"],
    )
    model = DensePredictionModel(
        encoder,
        TokenToSpatial(feature_key=cfg["adapter"]["feature_key"],
                       grid_size=cfg["adapter"]["grid_size"]),
        FCNHead(in_channels=cfg["head"]["in_channels"],
                num_classes=cfg["head"]["num_classes"]),
    ).to(device)
    model.train()

    size = cfg["test"]["input_size"]
    bs = cfg["test"]["batch_size"]
    images = {"sar": torch.randn(bs, 2, size, size, device=device)}
    with torch.no_grad():
        features = model.encoder(images)
        spatial = model.adapter(features)
    side = (size + 15) // 16
    assert features["tokens"].shape == (bs, side * side, 1024)
    assert spatial.shape == (bs, 1024, side, side)
    logits = model(images)
    assert logits.shape == (bs, cfg["head"]["num_classes"], size, size)
    target = torch.randint(0, cfg["head"]["num_classes"],
                           (bs, size, size), device=device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
    print("Integration test 03 passed!")


if __name__ == "__main__":
    main()
