from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
from src.data.statistics import get_or_compute_stats


train_csv = config["dataset"]["train_csv"]

statistics = get_or_compute_stats(
    dataset_factory=lambda modalities: Sen1Floods11Dataset(
        split_csv=train_csv,
        modalities=modalities,
        preprocess_func=None,
    ),
    modalities=config["dataset"]["modalities"],
    split_csv=train_csv,
    dataset_name=config["dataset"]["name"],
)