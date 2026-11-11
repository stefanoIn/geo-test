
import torch

from src.models.factory import build_model


def test_model(name):

    config = {
        "model": {
            "name": name,
            "in_channels": 15,
            "encoder_name": "resnet50",
        },
        "dataset": {
            "modalities": ["sar", "optical"],
        },
        "head": {
            "num_classes": 2,
        },
    }

    model = build_model(config)
    model.eval()

    images = {
        "sar": torch.randn(1, 2, 64, 64),
        "optical": torch.randn(1, 13, 64, 64),
    }

    with torch.inference_mode():
        output = model(images)

    assert output.shape == (1, 2, 64, 64)

    print(f"{name}: PASSED")


if __name__ == "__main__":
    test_model("unet")
    test_model("resnet_unet")

    print("BASELINE FACTORY TEST PASSED")
