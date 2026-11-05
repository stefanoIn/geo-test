from src.data.datasets.sen1floods11 import Sen1Floods11Dataset
# from src.data.so2sat import So2SatDataset  # Later


def build_dataset(config, split, preprocess_func=None):

    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]

    if dataset_name == "sen1floods11":

        csv_path = dataset_cfg[f"{split}_csv"]

        return Sen1Floods11Dataset(
            split_csv=csv_path,
            modalities=dataset_cfg["modalities"],
            preprocess_func=preprocess_func,
        )

    # When you add So2Sat:
    #
    # elif dataset_name == "so2sat":
    #     return So2SatDataset(...)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    
from torch.utils.data import DataLoader

train_dataset = build_dataset(config, split="train")
val_dataset = build_dataset(config, split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["training"]["num_workers"],
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)