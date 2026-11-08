
import torch
import torch.nn.functional as F
import yaml

from src.models.factory import build_model
from utils import build_optimizer

CONFIG = "configs/croma/linear_segmentation/s2.yaml"

with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_model(cfg).to(device)
model.train()

# 1. Controlla il congelamento dell'encoder.
assert not any(p.requires_grad for p in model.encoder.parameters())
assert not model.encoder.backbone.training

# 2. Controlla la dimensione della linear head.
assert sum(p.numel() for p in model.head.parameters()) == 2050

# 3. Simula un batch Sentinel-2 di CROMA.
images = {
    "optical": torch.randn(1, 12, 120, 120, device=device)
}

target = torch.randint(
    0, 2, (1, 120, 120), device=device
)

# 4. Esegui il modello completo.
logits = model(images)

assert logits.shape == (1, 2, 120, 120)
assert torch.isfinite(logits).all()

# 5. Verifica che la loss propaghi gradienti alla head.
optimizer = build_optimizer(cfg["training"]["optimizer"], model)

old_weights = model.head.classifier.weight.detach().clone()

optimizer.zero_grad(set_to_none=True)
loss = F.cross_entropy(logits, target)
loss.backward()

assert all(p.grad is None for p in model.encoder.parameters())

assert all(
    p.grad is not None
    and torch.isfinite(p.grad).all()
    and p.grad.abs().sum().item() > 0
    for p in model.head.parameters()
)

# 6. Verifica che l'optimizer aggiorni effettivamente la head.
optimizer.step()

assert not torch.equal(
    old_weights,
    model.head.classifier.weight.detach()
)

print("Full model integration: PASSED")
