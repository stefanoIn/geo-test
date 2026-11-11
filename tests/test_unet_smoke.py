
"""
Smoke test for the fully supervised U-Net baseline.

Tests:
    1. Model construction
    2. Correct output shape
    3. Finite predictions
    4. Cross-entropy loss
    5. Gradient propagation
    6. Parameter updates
"""

import torch
import torch.nn.functional as F

from src.baselines.unet import UNet


def main():

    torch.manual_seed(234)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Model
    # -------------------------

    model = UNet(
        in_channels=15,
        num_classes=2,
        modalities=("sar", "optical"),
    ).to(device)

    model.train()

    # -------------------------
    # Synthetic input
    # -------------------------

    images = {
        "sar": torch.randn(
            1, 2, 120, 120, device=device
        ),
        "optical": torch.randn(
            1, 13, 120, 120, device=device
        ),
    }

    target = torch.randint(
        low=0,
        high=2,
        size=(1, 120, 120),
        device=device,
    )

    # Test the ignored label convention.
    target[:, 0, 0] = 255

    # -------------------------
    # Forward pass
    # -------------------------

    logits = model(images)

    assert logits.shape == (1, 2, 120, 120), (
        f"Unexpected output shape: {logits.shape}"
    )

    assert torch.isfinite(logits).all(), (
        "Model outputs contain NaN or Inf"
    )

    # -------------------------
    # Loss
    # -------------------------

    loss = F.cross_entropy(
        logits,
        target,
        ignore_index=255,
    )

    assert torch.isfinite(loss), (
        "Loss contains NaN or Inf"
    )

    # -------------------------
    # Backward pass
    # -------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    for name, parameter in model.named_parameters():

        assert parameter.requires_grad, (
            f"Unexpected frozen parameter: {name}"
        )

        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )

        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )

    # -------------------------
    # Optimizer update
    # -------------------------

    weights_before = (
        model.classifier.weight.detach().clone()
    )

    optimizer.step()

    weights_after = (
        model.classifier.weight.detach()
    )

    assert not torch.equal(
        weights_before,
        weights_after,
    ), "Optimizer did not update the classifier"

    # -------------------------
    # Results
    # -------------------------

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Device:", device)
    print("Output shape:", logits.shape)
    print("Loss:", loss.item())
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)
    print("Optimizer update: PASSED")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
