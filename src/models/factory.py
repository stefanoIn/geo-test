"""Construct a dense-prediction model from YAML config. No dataset-specific logic."""
from torch import nn
from pathlib import Path

from src.adapters.token_to_spatial import TokenToSpatial
from src.models.dense_prediction import DensePredictionModel


# heads
from src.heads.fcn import FCNHead
from src.heads.linear import LinearSegmentationHead


def resolve_croma_modality(value, dataset_modalities=None):
    """Translate GeoTest modality names to CROMA's native spelling."""
    
    inputs = {"sar": {"sar"}, "optical": {"optical"}, "both": {"sar", "optical"}}
    
    canonical = value.strip().lower() if isinstance(value, str) else None
    
    if canonical not in inputs:
        raise ValueError(
            f"model.modality: received {value!r}; expected 'sar', 'optical', or 'both'. "
            "Capitalization is accepted; spelling must match."
        )
        
    if dataset_modalities is not None and set(dataset_modalities) != inputs[canonical]:
        required = next((name for name, keys in inputs.items() if keys == set(dataset_modalities)), None)
        raise ValueError(
            f"model.modality: received {value!r}; expected {required!r} "
            f"for dataset.modalities={dataset_modalities!r}. "
            "Dataset keys must be lowercase 'sar' and/or 'optical'."
        )
        
    return "SAR" if canonical == "sar" else canonical


def build_model(config: dict) -> nn.Module:
    """Build a pretrained GeoFM or a supervised baseline."""

    mc = config["model"]
    name = mc["name"].lower()

    # ----------------------------------------
    # Supervised CNN baselines (from scratch)
    # ----------------------------------------

    if name in {"unet", "resnet_unet"}:

        modalities = tuple(config["dataset"]["modalities"])
        in_channels = mc["in_channels"]
        num_classes = config["head"]["num_classes"]

        if name == "unet":
            from src.baselines.unet import UNet

            return UNet(
                in_channels=in_channels,
                num_classes=num_classes,
                modalities=modalities,
            )

        from src.baselines.resnet_unet import ResNetUNet

        return ResNetUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            modalities=modalities,
            encoder_name=mc.get("encoder_name", "resnet50"),
        )

    # ----------------------------------------
    # Pretrained GeoFMs
    # ----------------------------------------

    ac = config["adapter"]
    hc = config["head"]

    path = Path(mc["checkpoint"]).expanduser()

    if name == "croma":
        from src.encoders.croma.croma import CROMAEncoder

        modality = resolve_croma_modality(
            mc.get("modality", "both"), config.get("dataset", {}).get("modalities")
        )
        
        encoder = CROMAEncoder(
            checkpoint_path=path,
            size=mc.get("size", "large"),
            modality=modality,
            image_resolution=mc.get("image_resolution", 120),
            freeze=mc.get("freeze_encoder", True),
        )

    elif name in {"terramind_v1_base", "terramind_v1_large"}:
        
        from src.encoders.terramind.terramind import TerraMindEncoder

        name_size = name.removeprefix("terramind_v1_")
        size = mc.get("size", name_size)
        
        if size != name_size:
            raise ValueError(f"Model name {name!r} disagrees with size {size!r}")
        
        native_modalities = mc["modalities"]
        
        mapping = mc.get("input_mapping")
        
        if "dataset" in config:
            expected_keys = set(mapping) if mapping else set(native_modalities)
            
            if set(config["dataset"]["modalities"]) != expected_keys:
                raise ValueError("TerraMind input_mapping and dataset modalities disagree")
            
        encoder = TerraMindEncoder(
            checkpoint_path=path,
            size=size,
            modalities=native_modalities,
            input_mapping=mapping,
            merge_method=mc.get("merge_method", "mean"),
            freeze=mc.get("freeze_encoder", True),
        )
        
        expected_channels = encoder.backbone.out_channels[-1]
        
        if hc["in_channels"] != expected_channels:
            raise ValueError(
                f"TerraMind produces {expected_channels} channels with this merge_method, "
                f"but head.in_channels={hc['in_channels']}"
            )

    else:
        raise ValueError(f"Unknown encoder model: {mc['name']!r}")

    if ac["name"] != "token_to_spatial":
        raise ValueError(f"Unknown feature adapter: {ac['name']!r}")
    
    adapter = TokenToSpatial(
        feature_key=ac.get("feature_key", "tokens"),
        grid_size=ac.get("grid_size"),
    )

    if hc["name"] == "fcn":
        
        head = FCNHead(
            in_channels=hc["in_channels"],
            num_classes=hc["num_classes"],
        )        
        
    elif hc["name"] == "linear_segmentation":
        
        head = LinearSegmentationHead(
            in_channels=hc["in_channels"],
            num_classes=hc["num_classes"],
        )
        
    else:
        raise ValueError(f"Unknown prediction head: {hc['name']!r}")
    
    return DensePredictionModel(encoder=encoder, adapter=adapter, head=head)
