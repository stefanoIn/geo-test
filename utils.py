# training/losses.py
import torch 

def build_loss(config, device):
    name = config["name"]

    if name == "cross_entropy":
        weights = config.get("class_weights")
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return torch.nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=config.get("ignore_index", 255),
        )

    if name == "water_dice":
        return WaterDiceLoss(ignore_index=config.get("ignore_index", 255))

    raise ValueError(f"Unknown loss: {name}")

'''


TO BE COMPLETED!!!!!!!!!!!!!!!!!


'''
