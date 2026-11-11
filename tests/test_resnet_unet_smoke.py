
import torch

from src.baselines.resnet_unet import ResNetUNet


model = ResNetUNet(
    in_channels=15,
    num_classes=2,
    modalities=("sar", "optical"),
    encoder_name="resnet50",
)

images = {
    "sar": torch.randn(1, 2, 64, 64),
    "optical": torch.randn(1, 13, 64, 64),
}

target = torch.randint(0, 2, (1, 64, 64))

model.train()

logits = model(images)

assert logits.shape == (1, 2, 64, 64)

loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()

assert all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.parameters() if p.requires_grad
)

print("Output shape:", logits.shape)
print("Loss:", loss.item())
print("Trainable parameters:", sum(
    p.numel() for p in model.parameters() if p.requires_grad
))
print("SMOKE TEST PASSED")
