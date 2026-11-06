"""Dataset-agnostic TerraMind v1 encoder for pretrained spatial representations.

The caller selects TerraMind modality identifiers and, optionally, maps dataset
keys to those identifiers. Radiometric preprocessing belongs to the data pipeline;
segmentation logits belong to the downstream head, not to this encoder.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TerraMindEncoder(nn.Module):
    """TerraMind v1 Base/Large with one or more TerraTorch input modalities.

    Args:
        checkpoint_path: Local TerraMind v1 checkpoint.
        size: 'base' (768 features) or 'large' (1024 features).
        modalities: Native TerraTorch identifiers, e.g. ['S1GRD', 'S2L1C'].
        input_mapping: Optional {dataset_key: TerraTorch_modality}, e.g.
            {'sar': 'S1GRD', 'optical': 'S2L1C'}.
        merge_method: 'mean', 'max', or 'concat'. With 'concat', output feature
            dimension changes and must match the downstream head.
        freeze: Keep the pretrained backbone frozen and in eval mode.

    Returns:
        {'tokens': Tensor[B, N, D], 'pooled': Tensor[B, D],
         'padded_size': (H_padded, W_padded)}.

    Notes:
        - Raw image inputs use [B, C, H, W]; the channel count is checked
          against the instantiated TerraTorch patch embedding.
        - Raw image inputs are padded to multiples of the backbone patch size.
        - Sequence/tokenized modalities (e.g. Coords/NDVI/LULC) are passed to
          TerraTorch without conversion. Their input formats and tokenizer
          checkpoints must be handled and tested separately. At least one
          spatial image modality is required for spatial outputs.
        - This wrapper does NOT normalise, resample or atmospherically correct
          input products. Provide data in the model-appropriate format.
    """

    def __init__(
        self,
        checkpoint_path,
        *,
        size="large",
        modalities=("S1GRD",),
        input_mapping=None,
        merge_method="mean",
        freeze=True,
    ):
        super().__init__()

        if size not in {"base", "large"}:
            raise ValueError("TerraMind size must be 'base' or 'large'")
        if isinstance(modalities, str) or not modalities:
            raise ValueError("modalities must be a non-empty sequence of names")
        modalities = tuple(modalities)
        if any(not isinstance(m, str) or not m for m in modalities):
            raise ValueError("Each modality must be a non-empty TerraTorch name")
        if len(set(modalities)) != len(modalities):
            raise ValueError("Duplicate TerraMind modalities")
        if merge_method not in {"mean", "max", "concat"}:
            raise ValueError("Use merge_method='mean', 'max' or 'concat' for spatial tokens")

        self.modalities = modalities
        self.input_mapping = dict(input_mapping or {})
        if set(self.input_mapping.values()) - set(modalities):
            raise ValueError("input_mapping values must be present in modalities")
        if len(set(self.input_mapping.values())) != len(self.input_mapping):
            raise ValueError("Two dataset keys cannot map to the same modality")

        from terratorch.models.backbones.terramind.model.encoder_embeddings import (
            ImageEncoderEmbedding,
        )
        from terratorch.models.backbones.terramind.model.terramind_register import (
            terramind_v1_base,
            terramind_v1_large,
        )

        self._image_embedding_type = ImageEncoderEmbedding
        factory = {"base": terramind_v1_base, "large": terramind_v1_large}[size]
        self.backbone = factory(
            pretrained=False,
            modalities=list(modalities),
            img_size=224,  # Reference positional embeddings of the checkpoint.
            merge_method=merge_method,
        )

        # Check the chosen modalities against what this TerraTorch version built.
        if set(self.backbone.mod_name_mapping) != set(modalities):
            raise RuntimeError("TerraTorch did not initialise all requested modalities")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        expected = self.backbone.state_dict()
        missing = [k for k in expected if k not in state]
        mismatched = [
            k for k, v in expected.items()
            if k in state and state[k].shape != v.shape
        ]
        if missing or mismatched:
            raise RuntimeError(
                f"Incompatible TerraMind checkpoint: missing={missing[:5]}, "
                f"shape mismatches={mismatched[:5]}"
            )
        # The checkpoint can contain other, unused modality weights.
        self.backbone.load_state_dict(
            {key: state[key] for key in expected}, strict=True
        )

        self.freeze = bool(freeze)
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | tuple[int, int]]:
        if not isinstance(images, dict) or not images:
            raise ValueError("Expected a non-empty dictionary of modality tensors")

        prepared = {}
        raw_image_hw = None
        padded_hw = None
        spatial_input_found = False
        for input_key, tensor in images.items():
            modality = self.input_mapping.get(input_key, input_key)
            if modality not in self.modalities:
                raise ValueError(f"Unexpected input modality {input_key!r} -> {modality!r}")
            if modality in prepared:
                raise ValueError(f"Duplicate input for modality {modality!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{modality}: expected torch.Tensor")

            internal_name = self.backbone.mod_name_mapping[modality]
            embedding = self.backbone.encoder_embeddings[internal_name]
            if isinstance(embedding, self._image_embedding_type):
                if tensor.ndim != 4:
                    raise ValueError(f"{modality}: expected [B, C, H, W]")
                ph, pw = embedding.patch_size
                expected_channels = embedding.proj.in_features // (ph * pw)
                if tensor.shape[1] != expected_channels:
                    raise ValueError(
                        f"{modality}: expected {expected_channels} channels, "
                        f"got {tensor.shape[1]}"
                    )
                height, width = tensor.shape[-2:]
                if raw_image_hw is None:
                    raw_image_hw = (height, width)
                elif raw_image_hw != (height, width):
                    raise ValueError("Spatial image modalities must be aligned")

                pad_h = (-height) % ph
                pad_w = (-width) % pw
                if pad_h or pad_w:
                    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
                if padded_hw is not None and padded_hw != tensor.shape[-2:]:
                    raise ValueError("Spatial modalities have incompatible padded sizes")
                padded_hw = tuple(tensor.shape[-2:])
                spatial_input_found = True

            # Non-raw modalities are passed through to TerraTorch unchanged;
            # their representation is determined by its registered embedding.
            prepared[modality] = tensor

        missing = set(self.modalities) - set(prepared)
        if missing:
            raise ValueError(f"Missing modalities: {sorted(missing)}")
        if not spatial_input_found:
            raise ValueError(
                "At least one raw spatial image modality is required by this "
                "token-to-spatial encoder wrapper"
            )

        features = self.backbone(prepared)
        tokens = features[-1]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError("Expected TerraMind merged spatial tokens [B, N, D]")
        # For mean/max/concat with raw spatial inputs, each patch is represented
        # once; extra non-image modalities participate in attention, not output.
        internal_name = next(
            m for m in self.modalities
            if isinstance(
                self.backbone.encoder_embeddings[self.backbone.mod_name_mapping[m]],
                self._image_embedding_type,
            )
        )
        ph, pw = self.backbone.encoder_embeddings[
            self.backbone.mod_name_mapping[internal_name]
        ].patch_size
        expected_tokens = (padded_hw[0] // ph) * (padded_hw[1] // pw)
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected TerraMind spatial token count: {tokens.shape[1]} "
                f"(expected {expected_tokens})"
            )

        # The decoder needs the actual patch-grid extent to avoid stretching
        # padded pixels into the original image when restoring logits.
        return {
            "tokens": tokens,
            "pooled": tokens.mean(dim=1),
            "padded_size": padded_hw,
        }
