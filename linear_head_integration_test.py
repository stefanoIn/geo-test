import torch
from src.heads.linear import LinearSegmentationHead

head = LinearSegmentationHead(1024, 2)

features = torch.randn(2,1024,15,15)
logits = head(features)

assert logits.shape == (2,2,15,15)
assert sum(p.numel() for p in head.parameters()) == 2050

target = torch.randint(0, 2, (2, 15, 15))

loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()

assert all(p.grad is not None for p in head.parameters())

print("Linear head:ok")

