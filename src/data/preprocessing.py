"""Reusable image normalisation; no statistic computation or file I/O."""

import torch

# TerraMind v1 statistics from TerraTorch terramind_register.py.
# S1: VV/VH in dB. S2: native DN, in the product default band order.
TERRAMIND_STATS = {
    "S1GRD": {
        "mean": [-12.599, -20.293],
        "std": [5.195, 5.89],
    },
    "S1RTC": {
        "mean": [-10.93, -17.329],
        "std": [4.391, 4.459],
    },
    "S2L1C": {
        "mean": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.56, 3306.481, 1473.847, 506.07, 2472.825, 1838.929],
        "std": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
    },
    "S2L2A": {
        "mean": [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
        "std": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    },
}

class NormalizeFromStats:
    """Normalise each image modality with precomputed per-band statistics.

    Methods:
      - 'zscore': (x - mean) / std
      - 'two_sigma_stretch': clip((x - (mean - 2*std)) / (4*std), 0, 1)

    The optional band_indices are zero-based positions in the raw dataset.
    They select/reorder image bands AND the corresponding statistics.
    """

    def __init__(
        self,
        statistics,
        method="two_sigma_stretch",
        band_indices=None,
        use_8_bit=False,
    ):
        if method not in {"zscore", "two_sigma_stretch"}:
            raise ValueError(f"Unknown normalization method: {method}")
        if use_8_bit and method != "two_sigma_stretch":
            raise ValueError("use_8_bit is supported only for two_sigma_stretch")
        self.statistics = statistics
        self.method = method
        self.band_indices = band_indices or {}
        self.use_8_bit = use_8_bit

    @classmethod
    def from_config(cls, config, training_statistics=None):
        """Build normalization from YAML; dataset statistics are supplied by the caller."""
        preprocessing = config["preprocessing"]
        normalization = preprocessing["normalization"]
        policy = normalization["statistics"]
        if policy == "pretraining_fixed":
            model = config["model"]
            if model["name"] not in {"terramind_v1_base", "terramind_v1_large"}:
                raise ValueError("pretraining_fixed currently supports TerraMind v1 only")
            method = normalization.get("method", "zscore")
            if method != "zscore":
                raise ValueError("pretraining_fixed requires method: zscore")
            mapping = model.get("input_mapping") or {
                name: name for name in model["modalities"]
            }
            if set(mapping.values()) != set(model["modalities"]):
                raise ValueError("input_mapping must match model modalities")
            if set(mapping) != set(config["dataset"]["modalities"]):
                raise ValueError("input_mapping must match dataset modalities")
            statistics = {}
            for name, product in mapping.items():
                if product not in TERRAMIND_STATS:
                    raise ValueError(f"No fixed statistics for {product!r}")
                statistics[name] = {
                    key: list(values) for key, values in TERRAMIND_STATS[product].items()
                }
        elif policy == "training_dataset":
            if training_statistics is None:
                raise ValueError("Provide training_statistics for training_dataset normalization")
            statistics = training_statistics
            method = normalization.get("method", "two_sigma_stretch")
        else:
            raise ValueError(f"Unknown statistics policy: {policy!r}")
        return cls(
            statistics=statistics,
            method=method,
            band_indices=preprocessing.get("band_indices"),
            use_8_bit=normalization.get("use_8_bit", False),
        )

    def __call__(self, sample):
        result = sample.copy()

        for modality, stats in self.statistics.items():
            if modality not in sample:
                continue

            image = sample[modality].float()
            if image.ndim != 3:
                raise ValueError(f"{modality}: expected [C,H,W]")
            raw_channels = image.shape[0]
            mean = image.new_tensor(stats["mean"])
            std = image.new_tensor(stats["std"])

            indices = self.band_indices.get(modality)
            if indices is not None:
                image = image[indices]
                # Dataset-derived statistics cover all raw channels. External
                # model statistics may already cover only the selected channels.
                if mean.numel() == raw_channels:
                    mean = mean[indices]
                    std = std[indices]

            if image.shape[0] != mean.numel() or mean.numel() != std.numel():
                raise ValueError(
                    f"{modality}: {image.shape[0]} image bands but "
                    f"{mean.numel()} statistics entries"
                )
            if not bool(torch.all(torch.isfinite(std) & (std > 0))):
                raise ValueError(f"{modality}: invalid standard deviations")

            valid = torch.isfinite(image)
            safe_image = torch.where(valid, image, 0.0)
            mean = mean[:, None, None]
            std = std[:, None, None]

            if self.method == "zscore":
                normalized = (safe_image - mean) / std
            else:
                normalized = ((safe_image - (mean - 2 * std)) / (4 * std)).clamp(0, 1)
                if self.use_8_bit:
                    normalized = (normalized * 255).to(torch.uint8).float() / 255

            result[modality] = torch.where(valid, normalized, 0.0)

        return result
