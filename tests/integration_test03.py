"""Inspect TerraMind v1's S1, S2 or fused S1+S2 forward pass.

This is a model-only integration test. Inputs/labels are synthetic: the test
checks shapes, actual padding reaching the patch embeddings, token fusion,
logits and gradient flow. It does NOT validate radiometric preprocessing or
segmentation accuracy on Sen1Floods11.

Examples (from the repository root):
    python -m tests.integration_test03 --modality S1GRD
    python -m tests.integration_test03 --modality S2L1C
    python -m tests.integration_test03 --modality both
    python -m tests.integration_test03 --modality both --input-size 224
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.adapters.token_to_spatial import TokenToSpatial
from src.encoders.terramind.terramind import TerraMindEncoder
from src.heads.fcn import FCNHead
from src.models.dense_prediction import DensePredictionModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/terramind/terramind_v1_large_s1_fcn_test.yaml"

# Only the *test data* are specialised here. TerraMindEncoder remains
# dataset-agnostic and receives native modality names + a mapping from input
# keys to native names.
CASES = {
    "S1GRD": {"sar": ("S1GRD", 2)},
    "S2L1C": {"optical": ("S2L1C", 13)},
    "both": {"sar": ("S1GRD", 2), "optical": ("S2L1C", 13)},
}


def _shape(value):
    """Display a compact tuple of tensor dimensions."""
    return tuple(value.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modality", choices=tuple(CASES), default="S1GRD",
        help="Input modality/modes to exercise (default: S1GRD)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="YAML supplying checkpoint, model size, adapter and head",
    )
    parser.add_argument(
        "--input-size", type=int, default=None,
        help="Override synthetic image side length from the YAML",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    torch.manual_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = cfg["model"]
    case = CASES[args.modality]
    modalities = [native_name for native_name, _ in case.values()]
    input_mapping = {data_name: native for data_name, (native, _) in case.items()}
    side = args.input_size if args.input_size is not None else cfg["test"]["input_size"]
    batch_size = cfg["test"]["batch_size"]
    if side <= 0 or batch_size <= 0:
        raise ValueError("input size and batch size must be positive")

    checkpoint = Path(mc["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    print("\n" + "=" * 74)
    print(f"TerraMind integration test | case: {args.modality}")
    print("=" * 74)
    print(f"Device: {device} | Model: {mc['size']} | Checkpoint: {checkpoint}")
    print(f"Native modalities: {modalities}")
    print(f"Dataset key -> TerraMind modality: {input_mapping}")
    print(f"Merge method: {mc.get('merge_method', 'mean')}")
    print(f"Encoder frozen: {mc['freeze_encoder']}")

    encoder = TerraMindEncoder(
        checkpoint_path=checkpoint,
        size=mc["size"],
        modalities=modalities,
        input_mapping=input_mapping,
        merge_method=mc.get("merge_method", "mean"),
        freeze=mc["freeze_encoder"],
    )
    model = DensePredictionModel(
        encoder=encoder,
        adapter=TokenToSpatial(
            feature_key=cfg["adapter"]["feature_key"],
            grid_size=cfg["adapter"].get("grid_size"),
        ),
        head=FCNHead(
            in_channels=cfg["head"]["in_channels"],
            num_classes=cfg["head"]["num_classes"],
        ),
    ).to(device)
    model.train()

    images = {
        key: torch.randn(batch_size, channels, side, side, device=device)
        for key, (_native, channels) in case.items()
    }

    # Observe the *real* tensors that arrive at TerraTorch's patch embeddings.
    # This is more reliable than merely printing the padding we expect the
    # wrapper to perform. Hooks are removed after one forward pass.
    captured = {"embeddings": {}}
    hooks = []

    def pre_hook_for(modality):
        def hook(_module, args):
            value = args[0]
            if isinstance(value, dict):
                value = value["tensor"]
            captured["embeddings"].setdefault(modality, {})["input"] = _shape(value)
        return hook

    def post_hook_for(modality):
        def hook(_module, _args, output):
            captured["embeddings"].setdefault(modality, {})["patch_tokens"] = _shape(output["x"])
            captured["embeddings"][modality]["positional_tokens"] = _shape(output["emb"])
        return hook

    for native_name in modalities:
        internal_name = model.encoder.backbone.mod_name_mapping[native_name]
        embedding = model.encoder.backbone.encoder_embeddings[internal_name]
        hooks.append(embedding.register_forward_pre_hook(pre_hook_for(native_name)))
        hooks.append(embedding.register_forward_hook(post_hook_for(native_name)))

    def encoder_hook(_module, _args, output):
        captured["tokens"] = _shape(output["tokens"])
        captured["pooled"] = _shape(output["pooled"])

    def adapter_hook(_module, _args, output):
        captured["spatial"] = _shape(output)

    def head_hook(_module, _args, output):
        captured["head_logits"] = _shape(output)

    hooks.extend([
        model.encoder.register_forward_hook(encoder_hook),
        model.adapter.register_forward_hook(adapter_hook),
        model.head.register_forward_hook(head_hook),
    ])

    model.zero_grad(set_to_none=True)
    try:
        # ONE model forward: this goes through encoder, adapter and FCN.
        logits = model(images)
    finally:
        for hook in hooks:
            hook.remove()

    print("\n[1] RAW INPUT -> ACTUAL INPUT RECEIVED BY PATCH EMBEDDING")
    patch_counts = []
    for data_key, (native_name, channels) in case.items():
        raw_shape = _shape(images[data_key])
        observation = captured["embeddings"][native_name]
        actual_shape = observation["input"]
        internal_name = model.encoder.backbone.mod_name_mapping[native_name]
        embedding = model.encoder.backbone.encoder_embeddings[internal_name]
        patch_h, patch_w = embedding.patch_size
        height, width = raw_shape[-2:]
        expected_pad_h = (-height) % patch_h
        expected_pad_w = (-width) % patch_w
        expected_shape = (batch_size, channels,
                          height + expected_pad_h, width + expected_pad_w)

        print(f"  {data_key} -> {native_name}")
        print(f"    Original tensor:    {raw_shape}")
        print(f"    Patch size:         {patch_h} x {patch_w}")
        print("    Padding (left, right, top, bottom): "
              f"(0, {expected_pad_w}, 0, {expected_pad_h})")
        print(f"    Actual padded tensor at embedding: {actual_shape}")
        assert actual_shape == expected_shape, (
            f"{native_name}: padding mismatch: expected {expected_shape}, got {actual_shape}"
        )
        print("    Padding check: OK")

        patch_shape = observation["patch_tokens"]
        pos_shape = observation["positional_tokens"]
        expected_n = (expected_shape[-2] // patch_h) * (expected_shape[-1] // patch_w)
        assert patch_shape[:2] == (batch_size, expected_n), (
            f"{native_name}: unexpected projected patch count {patch_shape}"
        )
        assert pos_shape == patch_shape, (
            f"{native_name}: positional/modality embeddings mismatch"
        )
        print(f"    Projected patch tokens: {patch_shape}")
        print(f"    Position + modality embeddings: {pos_shape}")
        print(f"    Spatial patch grid: {expected_shape[-2] // patch_h} x "
              f"{expected_shape[-1] // patch_w} = {expected_n} tokens")
        patch_counts.append(expected_n)

    print("\n[2] SHARED TRANSFORMER -> MODALITY MERGE")
    print(f"    Tokens supplied by embeddings (total): {sum(patch_counts)}")
    print(f"    Merge method: {mc.get('merge_method', 'mean')}")
    print(f"    Final encoder tokens: {captured['tokens']}")
    print(f"    Spatially pooled vector: {captured['pooled']}")
    assert mc.get("merge_method", "mean") == "mean", (
        "This test's shape expectations assume mean fusion"
    )
    assert len(set(patch_counts)) == 1, "Multimodal inputs must produce the same patch grid"
    num_tokens = patch_counts[0]
    expected_channels = cfg["head"]["in_channels"]
    assert captured["tokens"] == (batch_size, num_tokens, expected_channels), (
        f"Final token shape mismatch: {captured['tokens']}"
    )
    assert captured["pooled"] == (batch_size, expected_channels)

    print("\n[3] TOKEN ADAPTER -> FCN -> UPSAMPLING")
    print(f"    TokenToSpatial output: {captured['spatial']}")
    print(f"    FCN logits before upsampling: {captured['head_logits']}")
    print(f"    Final logits after upsampling: {_shape(logits)}")
    num_classes = cfg["head"]["num_classes"]
    assert captured["spatial"] == (
        batch_size, expected_channels,
        (side + 15) // 16, (side + 15) // 16,
    )
    assert _shape(logits) == (batch_size, num_classes, side, side)
    assert torch.isfinite(logits).all(), "Non-finite output logits"

    print("\n[4] BACKWARD / FROZEN ENCODER")
    target = torch.randint(0, num_classes, (batch_size, side, side), device=device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    print(f"    Synthetic target: {_shape(target)}")
    print(f"    Cross-entropy loss: {loss.item():.6f}")
    print(f"    Encoder training mode: {model.encoder.backbone.training}")
    print(f"    Encoder parameters with gradients: {len(encoder_grads)}")
    print(f"    FCN parameters with gradients: {len(head_grads)}")
    assert torch.isfinite(loss), "Non-finite loss"
    assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
    if mc["freeze_encoder"]:
        assert not model.encoder.backbone.training, "Frozen backbone must be in eval mode"
        assert not encoder_grads, "Frozen backbone must not receive gradients"

    print(f"\nIntegration test 03 PASSED for {args.modality}!\n")


if __name__ == "__main__":
    main()
