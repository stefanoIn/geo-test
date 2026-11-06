"""Regression checks for chip_id batches and padded spatial decoding.

No pretrained checkpoint or dataset required.
Run: python -m unittest discover -s tests -p test_padding_and_metadata.py -v
"""
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from training.trainer import train_one_epoch
from src.adapters.token_to_spatial import TokenToSpatial
from src.models.dense_prediction import DensePredictionModel


class InspectInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.seen = []

    def forward(self, images):
        self.seen.append(tuple(images.keys()))
        return self.conv(images["sar"])


class SyntheticEncoder(nn.Module):
    def __init__(self, padded_size=None):
        super().__init__()
        self.padded_size = padded_size

    def forward(self, images):
        b = next(iter(images.values())).shape[0]
        # Four spatial tokens in a 2x2 grid, with non-constant features.
        tokens = torch.tensor([[0., 1.], [1., 0.], [2., 0.], [4., 1.]])
        result = {"tokens": tokens.unsqueeze(0).expand(b, -1, -1)}
        if self.padded_size is not None:
            result["padded_size"] = self.padded_size
        return result


class PatchTests(unittest.TestCase):
    def test_trainer_skips_chip_id_and_non_image_metadata(self):
        model = InspectInputs()
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batches = [{"sar": torch.randn(2, 2, 4, 4),
                    "target": torch.randint(0, 2, (2, 4, 4)),
                    "chip_id": ["A", "B"],
                    "validity": torch.ones(2, 4, 4, dtype=torch.bool)}]
        before = model.conv.weight.detach().clone()
        loss = train_one_epoch(model, batches, optimizer, criterion, "cpu")
        self.assertTrue(loss >= 0)
        self.assertEqual(model.seen, [("sar",)])
        self.assertFalse(torch.equal(model.conv.weight, before))

    def test_trainer_explicit_input_keys(self):
        model = InspectInputs()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batches = [{"sar": torch.randn(1, 2, 4, 4),
                    "extra_image": torch.randn(1, 1, 4, 4),
                    "target": torch.zeros(1, 4, 4, dtype=torch.long),
                    "chip_id": ["chip001"]}]
        train_one_epoch(model, batches, optimizer, criterion, "cpu", input_keys=["sar"])
        self.assertEqual(model.seen, [("sar",)])

    def test_padded_model_resizes_then_crops(self):
        model = DensePredictionModel(
            encoder=SyntheticEncoder(padded_size=(4, 4)),
            adapter=TokenToSpatial(), head=nn.Identity())
        images = {"sar": torch.empty(1, 2, 3, 3)}
        logits = model(images)
        grid = model.adapter(model.encoder(images))
        correct = F.interpolate(grid, size=(4, 4), mode="bilinear", align_corners=False)[..., :3, :3]
        incorrect = F.interpolate(grid, size=(3, 3), mode="bilinear", align_corners=False)
        self.assertEqual(logits.shape, (1, 2, 3, 3))
        self.assertTrue(torch.allclose(logits, correct))
        self.assertFalse(torch.allclose(logits, incorrect))

    def test_unpadded_model_backward_compatible(self):
        model = DensePredictionModel(
            encoder=SyntheticEncoder(), adapter=TokenToSpatial(), head=nn.Identity())
        images = {"sar": torch.empty(1, 2, 3, 3)}
        logits = model(images)
        grid = model.adapter(model.encoder(images))
        expected = F.interpolate(grid, size=(3, 3), mode="bilinear", align_corners=False)
        self.assertTrue(torch.allclose(logits, expected))

    def test_unpadded_terramind_metadata(self):
        model = DensePredictionModel(
            encoder=SyntheticEncoder(padded_size=(4, 4)), adapter=TokenToSpatial(), head=nn.Identity())
        images = {"sar": torch.empty(1, 2, 4, 4)}
        logits = model(images)
        self.assertEqual(logits.shape, (1, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
