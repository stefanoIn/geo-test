import torch
import torch.nn as nn
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import json

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation.fcn import FCNHead
from use_croma import PretrainedCROMA


CROP_SIZE = 120
IGNORE_INDEX = 255

def load_split_paths(csv_file, x_dir, y_dir):
    data = []

    with open(csv_file, newline="") as f:
        
        for x_name, y_name in csv.reader(f):
            x_path = Path(x_dir) / x_name.strip()
            y_path = Path(y_dir) / y_name.strip()
            
            if not x_path.is_file() or not y_path.is_file():
                print(f"Missing pair: {x_path}, {y_path}")
                continue

            data.append((x_path, y_path))

    return data

def load_image_pair(pair):
    x_path, y_path = pair

    with rasterio.open(x_path) as src:
        x = src.read().astype(np.float32)

    with rasterio.open(y_path) as src:
        y = src.read(1).astype(np.int16)

    # Original FCN preprocessing
    x = np.nan_to_num(x)
    x = np.clip(x, -50, 1)
    x = (x + 50) / 51

    # Original ignore-label convention
    y[y == -1] = 255

    return x, y


def normalize_croma(image, eps=1e-6):
    """
    Normalize one SAR image using mean ± 2 standard deviations.

    Input:
        image: [C, H, W]

    Output:
        image: [C, H, W], values in [0, 1]

    Statistics are computed independently for each channel
    using the entire image.
    """

    image = image.float()

    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True)

    min_value = mean - 2 * std
    max_value = mean + 2 * std

    denominator = (max_value - min_value).clamp_min(eps)

    image = (image - min_value) / denominator

    return image.clamp(0, 1)

def normalize_croma_reference(x, use_8_bit=True):
    """
    x: [B, 2, 120, 120] raw SAR patches.
    Channel statistics are computed over B, H and W.
    """
    x = x.float()
    channels = []

    for c in range(x.shape[1]):
        values = x[:, c, :, :]

        mean = values.mean()
        std = values.std()

        low = mean - 2 * std
        high = mean + 2 * std
        denom = (high - low).clamp_min(1e-6)

        normalized = (values - low) / denom

        if use_8_bit:
            normalized = (
                (normalized * 255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .float() / 255.0
            )
        else:
            normalized = normalized.clamp(0, 1)

        channels.append(normalized.unsqueeze(1))

    return torch.cat(channels, dim=1)

def prepare_sample(data):
    """
    Convert one sample to PyTorch tensors,
    normalize the image and prepare its mask.

    Assumes data contains:
        image: [2, 512, 512]
        mask:  [1, 512, 512] or [512, 512]
    """

    image, mask = data

    image = torch.as_tensor(image, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.long).clone()

    if mask.ndim == 3:
        assert mask.shape[0] == 1
        mask = mask.squeeze(0)

    # Sen1Floods11 invalid pixels
    mask[mask == -1] = IGNORE_INDEX

    assert image.ndim == 3 and image.shape[0] == 2
    assert mask.shape == image.shape[-2:]
    assert torch.isfinite(image).all(), (
        "SAR image contains NaN or infinite values"
    )

    assert set(torch.unique(mask).tolist()).issubset(
        {0, 1, IGNORE_INDEX}
    ), "Unexpected mask values"

    # Normalize the complete 512x512 image
    image = normalize_croma(image)

    return image, mask


class Sen1Floods11Dataset(Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = load_image_pair(self.data_list[idx])
        return self.preprocess_func(data)


class CROMAFloodSegmentation(nn.Module):
    def __init__(self, croma, head):
        super().__init__()
        
        self.croma = croma
        self.head = head
    
    def train(self, mode=True):
        super().train(mode)
        self.croma.eval()
        return self
    
    def forward(self, images):
        
        # 1. Extract pretrained CROMA features
        with torch.no_grad():
            outputs = self.croma(SAR_images=images)
            features = outputs["SAR_encodings"]
        
        # features: [B, 225, 1024]

        B, N, D = features.shape
        H = W = int(N ** 0.5)
        
        assert H * W == N
        
        # 2. Restore spatial structure
        spatial_features = features.transpose(1, 2).reshape(B, D, H, W)
        
        logits = self.head(spatial_features)
        # features: [B, 1024, 15, 15]
        
        # 4. Upsample to original image size
        logits = F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # logits: [B, 2, 120, 120]

        return logits