
import torch
import torch.nn.functional as F
import yaml

from src.models.factory import build_model

CONFIG = "configs/terramind/linear_segmentation/s2.yaml"

with open(CONFIG, encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_model(config).to(device)
model.train()

# TerraMind S2: 13 bande, 128x128
images = {
    "optical": torch.randn(1, 13, 128, 128, device=device)
}

target = torch.randint(
    0, 2, (1, 128, 128), device=device
)

# Forward
logits = model(images)

assert logits.shape == (1, 2, 128, 128)
assert torch.isfinite(logits).all()

# Backward
loss = F.cross_entropy(logits, target)
loss.backward()

# Encoder congelato
assert all(p.grad is None for p in model.encoder.parameters())

# Head allenabile
assert sum(p.numel() for p in model.head.parameters()) == 2050
assert all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.head.parameters()
)

print("TerraMind linear segmentation integration: PASSED")
