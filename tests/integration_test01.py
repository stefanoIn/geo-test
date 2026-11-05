"""Check that the configured CROMA encoder, token-to-spatial adapter and FCN
head work together as a dense prediction model.

Loads the local pretrained checkpoint and passes one synthetic SAR/optical
sample through the model. Checks the output shape and finite logits, then
backpropagates a cross-entropy loss against random labels to verify that the
encoder stays frozen and the FCN head receives finite, nonzero gradients.

Uses no real dataset or preprocessing pipeline and performs no optimizer step.
This checks model integration, not segmentation accuracy or full training.
Run from the project root with: python -m tests.integration_test01
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.encoders.croma.croma import CROMAEncoder
from src.adapters.token_to_spatial import TokenToSpatial
from src.heads.fcn import FCNHead
from src.models.dense_prediction import DensePredictionModel


ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. Load experiment configuration
    config_path = ROOT / "configs/croma/croma_s1s2_fcn_all_hand_labels.yaml"

    with config_path.open("r") as file:
        cfg = yaml.safe_load(file)

    torch.manual_seed(cfg["experiment"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # 2. Construct the encoder
    encoder_cfg = cfg["model"]
    checkpoint = ROOT / encoder_cfg["checkpoint"]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    encoder = CROMAEncoder(
        checkpoint_path=checkpoint,
        size=encoder_cfg["size"],
        modality=encoder_cfg["modality"],
        image_resolution=encoder_cfg["image_resolution"],
        freeze=encoder_cfg["freeze_encoder"],
    )

    # 3. Construct adapter and head
    adapter_cfg = cfg["adapter"]

    adapter = TokenToSpatial(
        feature_key=adapter_cfg["feature_key"],
        grid_size=adapter_cfg["grid_size"],
    )

    head = FCNHead(
        in_channels=cfg["head"]["in_channels"],
        num_classes=cfg["head"]["num_classes"],
    )

    # 4. Assemble the complete model
    model = DensePredictionModel(
        encoder=encoder,
        adapter=adapter,
        head=head,
    ).to(device)

    model.train()

    # 5. Create synthetic Sentinel-1 and Sentinel-2 inputs
    batch_size = 1
    size = encoder_cfg["image_resolution"]
    bands = cfg["preprocessing"]["band_indices"]

    images = {
        "sar": torch.rand(
            batch_size, len(bands["sar"]), size, size,
            device=device
        ),
        "optical": torch.rand(
            batch_size, len(bands["optical"]), size, size,
            device=device
        ),
    }

    # 6. Forward pass
    logits = model(images)

    print(f"SAR shape: {tuple(images['sar'].shape)}")
    print(f"Optical shape: {tuple(images['optical'].shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

    expected_shape = (
        batch_size,
        cfg["head"]["num_classes"],
        size,
        size,
    )

    assert logits.shape == expected_shape
    assert torch.isfinite(logits).all()

    # 7. Check that the encoder is frozen
    assert all(
        not param.requires_grad
        for param in model.encoder.parameters()
    )

    # 8. Compute a dummy loss and test backward propagation
    target = torch.randint(
        0,
        cfg["head"]["num_classes"],
        (batch_size, size, size),
        device=device,
    )

    loss = F.cross_entropy(logits, target)
    loss.backward()

    assert all(
        param.grad is None
        for param in model.encoder.parameters()
    )

    assert any(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum().item() > 0
        for param in model.head.parameters()
    )

    print(f"Dummy loss: {loss.item():.4f}")
    print("Encoder frozen: OK")
    print("FCN gradients: OK")
    print("Integration test passed!")


if __name__ == "__main__":
    main()
