"""Lightweight TerraMind wrapper tests with a fake TerraTorch backbone.

No pretrained model is needed. These tests validate the input-to-backbone
contract, NOT real TerraMind checkpoint compatibility or accuracy.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


WRAPPER = Path(__file__).resolve().parents[1] / "src/encoders/terramind/terramind.py"
CHANNELS = {"S1GRD": 2, "S1RTC": 2, "S2L1C": 13, "S2L2A": 12, "RGB": 3, "DEM": 1}


class FakeImageEncoderEmbedding(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.patch_size = (16, 16)
        self.proj = nn.Linear(16 * 16 * channels, dim, bias=False)


class FakeTerraMind(nn.Module):
    def __init__(self, modalities, dim, merge_method):
        super().__init__()
        self.mod_name_mapping = {m: m for m in modalities}
        self.encoder_embeddings = nn.ModuleDict({
            m: FakeImageEncoderEmbedding(CHANNELS[m], dim) for m in modalities
        })
        self.merge_method = merge_method

    def forward(self, d):
        encoded = []
        for m in self.mod_name_mapping:
            x = d[m]
            batch, channels, height, width = x.shape
            x = x.reshape(batch, channels, height // 16, 16, width // 16, 16)
            x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, 16 * 16 * channels)
            encoded.append(self.encoder_embeddings[m].proj(x))
        if self.merge_method == "mean":
            result = torch.stack(encoded, dim=0).mean(dim=0)
        elif self.merge_method == "max":
            result = torch.stack(encoded, dim=0).amax(dim=0)
        else:
            result = torch.cat(encoded, dim=-1)
        return [result]


def factory(dim):
    def make(pretrained, modalities, img_size, merge_method):
        assert not pretrained and img_size == 224
        return FakeTerraMind(modalities, dim, merge_method)
    return make


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = Path(self.tmp.name) / "weights.pt"
        self.modules = {
            "terratorch": types.ModuleType("terratorch"),
            "terratorch.models": types.ModuleType("terratorch.models"),
            "terratorch.models.backbones": types.ModuleType("terratorch.models.backbones"),
            "terratorch.models.backbones.terramind": types.ModuleType("terratorch.models.backbones.terramind"),
            "terratorch.models.backbones.terramind.model": types.ModuleType("terratorch.models.backbones.terramind.model"),
        }
        prefix = "terratorch.models.backbones.terramind.model."
        embed = types.ModuleType(prefix + "encoder_embeddings")
        embed.ImageEncoderEmbedding = FakeImageEncoderEmbedding
        register = types.ModuleType(prefix + "terramind_register")
        register.terramind_v1_base = factory(8)
        register.terramind_v1_large = factory(16)
        self.modules[prefix + "encoder_embeddings"] = embed
        self.modules[prefix + "terramind_register"] = register
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        spec = importlib.util.spec_from_file_location("terramind_wrapper_for_tests", WRAPPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.TerraMindEncoder = module.TerraMindEncoder

    def make(self, modalities, *, size="large", mapping=None, merge="mean"):
        dim = 16 if size == "large" else 8
        torch.save(FakeTerraMind(modalities, dim, merge).state_dict(), self.ckpt)
        return self.TerraMindEncoder(
            checkpoint_path=self.ckpt,
            size=size,
            modalities=modalities,
            input_mapping=mapping,
            merge_method=merge,
        )

    def test_s1_s2_and_both(self):
        examples = [
            (["S1GRD"], {"sar": "S1GRD"}, {"sar": 2}),
            (["S2L1C"], {"optical": "S2L1C"}, {"optical": 13}),
            (["S1GRD", "S2L1C"], {"sar": "S1GRD", "optical": "S2L1C"},
             {"sar": 2, "optical": 13}),
        ]
        for modalities, mapping, inputs in examples:
            with self.subTest(modalities=modalities):
                model = self.make(modalities, mapping=mapping)
                images = {k: torch.randn(1, ch, 120, 120) for k, ch in inputs.items()}
                out = model(images)
                self.assertEqual(tuple(out["tokens"].shape), (1, 64, 16))
                self.assertEqual(tuple(out["pooled"].shape), (1, 16))
                model.train()
                self.assertFalse(model.backbone.training)
                self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_native_s2l2a_base(self):
        model = self.make(["S2L2A"], size="base")
        out = model({"S2L2A": torch.randn(2, 12, 224, 224)})
        self.assertEqual(tuple(out["tokens"].shape), (2, 196, 8))

    def test_wrong_channels(self):
        model = self.make(["S2L1C"])
        with self.assertRaisesRegex(ValueError, "13 channels"):
            model({"S2L1C": torch.randn(1, 12, 120, 120)})

    def test_missing_modality(self):
        model = self.make(["S1GRD", "S2L1C"])
        with self.assertRaisesRegex(ValueError, "Missing modalities"):
            model({"S1GRD": torch.randn(1, 2, 120, 120)})

    def test_wrong_checkpoint(self):
        torch.save({"not_a_weight": torch.ones(1)}, self.ckpt)
        with self.assertRaisesRegex(RuntimeError, "Incompatible TerraMind checkpoint"):
            self.TerraMindEncoder(checkpoint_path=self.ckpt)

    def test_concat(self):
        model = self.make(["S1GRD", "S2L1C"], merge="concat")
        out = model({"S1GRD": torch.randn(1, 2, 128, 128),
                     "S2L1C": torch.randn(1, 13, 128, 128)})
        self.assertEqual(tuple(out["tokens"].shape), (1, 64, 32))


if __name__ == "__main__":
    unittest.main()
