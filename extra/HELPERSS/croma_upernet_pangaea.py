# Adapted from VMarsocci/pangaea-bench (GPL-3.0), pinned below.
# Upstream source and license: https://github.com/VMarsocci/pangaea-bench
from pathlib import Path
import csv
import json
import math
import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from einops import rearrange

PANGAEA_REVISION = "ca19b837a16797cd809225225b671473c7c93768"
SCHEMA = "croma_sar_upernet_pangaea_v2"
PREPROCESSING = {
    "name": "pangaea_sen1floods11_sar_mean_std",
    "bands": ["VV", "VH"],
    "mean": [-10.184408, -16.895273],
    "std": [4.255339, 5.290568],
    "nan_to_num": "numpy_default_before_normalization",
    "clip": None,
    "ignore_index": 255,
    "source_revision": PANGAEA_REVISION,
}


def load_split_paths(csv_file, x_dir, y_dir):
    with open(csv_file, newline="") as handle:
        pairs = [(Path(x_dir) / x.strip(), Path(y_dir) / y.strip()) for x, y in csv.reader(handle)]
    missing = [str(p) for pair in pairs for p in pair if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing[:5]}")
    return pairs



def filter_labeled_training_pairs(pairs):
    kept, excluded = [], []
    for pair in pairs:
        with rasterio.open(pair[1]) as src:
            mask = src.read(1)
        if not np.isin(mask, [-1, 0, 1, 255]).all():
            raise ValueError(f"Unexpected labels in {pair[1]}")
        (kept if np.isin(mask, [0, 1]).any() else excluded).append(pair)
    if not kept:
        raise ValueError("No labeled training images remain.")
    return kept, excluded


def random_labeled_crop(image, mask, crop_size, max_attempts=32):
    height, width = image.shape[-2:]
    if min(height, width) < crop_size or mask.shape != (height, width):
        raise ValueError("Expected aligned image/mask at least as large as the crop.")
    valid = np.isin(mask, [0, 1])
    if not valid.any():
        raise ValueError("Fully ignored image must be excluded from training.")
    for _ in range(max_attempts):
        top = torch.randint(0, height - crop_size + 1, (1,)).item()
        left = torch.randint(0, width - crop_size + 1, (1,)).item()
        if valid[top:top+crop_size, left:left+crop_size].any():
            break
    else:
        # Bounded fallback: choose a valid pixel and a random crop containing it.
        coords = np.argwhere(valid)
        y, x = coords[torch.randint(len(coords), (1,)).item()]
        top = torch.randint(max(0, int(y)-crop_size+1),
                            min(int(y), height-crop_size)+1, (1,)).item()
        left = torch.randint(max(0, int(x)-crop_size+1),
                             min(int(x), width-crop_size)+1, (1,)).item()
    return image[:, top:top+crop_size, left:left+crop_size], mask[top:top+crop_size, left:left+crop_size]


class Sen1Floods11Dataset(Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        x_path, y_path = self.data_list[index]
        with rasterio.open(x_path) as src:
            image = np.nan_to_num(src.read()).astype(np.float32)
        with rasterio.open(y_path) as src:
            mask = src.read(1).astype(np.int64)
        # No FCN clipping/scaling and no per-image normalization.
        return self.preprocess_func((image, mask))


def prepare_sample(data):
    image, mask = data
    image = torch.as_tensor(image, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.long).clone()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask[mask == -1] = 255
    if image.ndim != 3 or image.shape[0] != 2 or image.shape[-2:] != mask.shape:
        raise ValueError("Expected aligned VV/VH image and label raster.")
    if not set(torch.unique(mask).tolist()).issubset({0, 1, 255}):
        raise ValueError("Unexpected label values.")
    mean = image.new_tensor(PREPROCESSING["mean"])[:, None, None]
    std = image.new_tensor(PREPROCESSING["std"])[:, None, None]
    image = (image - mean) / std
    if not torch.isfinite(image).all():
        raise ValueError("Nonfinite normalized input; inspect the source SAR raster.")
    return image, mask


class UPerNetHead(nn.Module):
    def __init__(self, in_channels=1024, channels=512, num_classes=2,
                 out_indices=(3, 5, 7, 11), pool_scales=(1, 2, 3, 6)):
        super().__init__()
        if len(out_indices) != 4 or list(out_indices) != sorted(set(out_indices)):
            raise ValueError("Select four distinct ordered encoder blocks.")
        self.out_indices = tuple(out_indices)
        self.config = dict(in_channels=in_channels, channels=channels, num_classes=num_classes,
                           out_indices=list(out_indices), pool_scales=list(pool_scales))
        self.in_channels = [in_channels] * 4
        rescales = [4, 2, 1, 0.5]
        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.align_corners = False

        self.channels = channels
        self.num_classes = num_classes

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def psp_forward(self, inputs):
            """Forward function of PSP module."""
            x = inputs[-1]
            psp_outs = [x]
            psp_outs.extend(self.psp_modules(x))
            psp_outs = torch.cat(psp_outs, dim=1)
            output = self.bottleneck(psp_outs)
    
            return output

    def _forward_feature(self, inputs):
            """Forward function for feature maps before classifying each pixel with
            ``self.cls_seg`` fc.
    
            Args:
                inputs (list[Tensor]): List of multi-level img features.
    
            Returns:
                feats (Tensor): A tensor of shape (batch_size, self.channels,
                    H, W) which is feature map for last layer of decoder head.
            """
            # inputs = self._transform_inputs(inputs)
    
            # build laterals
            laterals = [
                lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
            ]
    
            laterals.append(self.psp_forward(inputs))
    
            # build top-down path
            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=prev_shape,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
    
            # build outputs
            fpn_outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
            ]
            # append psp feature
            fpn_outs.append(laterals[-1])
    
            for i in range(used_backbone_levels - 1, 0, -1):
                fpn_outs[i] = F.interpolate(
                    fpn_outs[i],
                    size=fpn_outs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            fpn_outs = torch.cat(fpn_outs, dim=1)
            feats = self.fpn_bottleneck(fpn_outs)
            return feats

    def forward(self, features):
        features = self.neck(features)
        features = self._forward_feature(features)
        return self.conv_seg(self.dropout(features))


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
    """

    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        nn.BatchNorm2d(embed_dim[i]),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)

class CROMAUPerNet(nn.Module):
    def __init__(self, croma, head):
        super().__init__()
        self.croma = croma.requires_grad_(False)
        self.head = head
        self.croma.eval()
        if max(head.out_indices) >= len(croma.s1_encoder.transformer.layers):
            raise ValueError('Requested blocks exceed the SAR encoder depth.')
        if head.config['in_channels'] != croma.encoder_dim:
            raise ValueError('Decoder channel count does not match CROMA.')

    def train(self, mode=True):
        super().train(mode)
        self.croma.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images):
        encoder = self.croma.s1_encoder
        p = encoder.patch_size
        h, w = images.shape[-2:]
        if h != w or h % p or (h // p) * (w // p) != self.croma.num_patches:
            raise ValueError('Input tile must match the square CROMA image_resolution.')
        x = rearrange(images, 'b c (h i) (w j) -> b (h w) (c i j)', i=p, j=p)
        x = encoder.linear_input(x)
        bias = self.croma.attn_bias.to(device=x.device, dtype=x.dtype)
        features = []
        # Capture the full residual stream AFTER attention and FFN, not FFN outputs alone.
        # Match PANGAEA: early selected streams remain raw; normalize only the final selected stream.
        for index, (attention, ffn) in enumerate(encoder.transformer.layers):
            x = x + attention(x, bias)
            x = x + ffn(x)
            if index in self.head.out_indices:
                selected = encoder.transformer.norm_out(x) if index == self.head.out_indices[-1] else x
                features.append(selected.transpose(1, 2).reshape(x.shape[0], x.shape[2], h // p, w // p))
            if index == self.head.out_indices[-1]:
                break
        return features

    def forward(self, images):
        logits = self.head(self.extract_features(images))
        return F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)

def latest_completed_checkpoint(run_root):
    candidates = []
    for manifest in Path(run_root).glob('train_*/run.json'):
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('schema') != SCHEMA:
            continue
        history_path = manifest.parent / 'croma_training_history.json'
        if not history_path.is_file():
            continue
        history = json.loads(history_path.read_text(encoding='utf-8'))
        checkpoint = manifest.parent / Path(metadata['checkpoint']).name
        if history.get('epochs_completed') == metadata.get('epochs') and checkpoint.is_file():
            candidates.append((metadata['started_at'], checkpoint))
    if not candidates:
        raise FileNotFoundError('No completed UPerNet run found. Run the training notebook first.')
    return max(candidates, key=lambda item: item[0])[1]
