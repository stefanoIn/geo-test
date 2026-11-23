"""CPU-only regression checks for epoch-boundary checkpoint resumption.

Run: python -m unittest tests.test_resume_rng -v
No dataset files, pretrained weights or GPU training are required.
"""
import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from training.checkpoints import load_checkpoint, save_checkpoint
from training.trainer import train_one_epoch
from utils import TrainingPreprocess, capture_rng_state, restore_rng_state


class IdentityNormalizer:
    statistics = {}
    method = "zscore"
    band_indices = {}
    use_8_bit = False

    def __call__(self, sample):
        return sample


class TinyDataset(Dataset):
    def __init__(self, training=True):
        self.preprocess_func = TrainingPreprocess(IdentityNormalizer(), 4, seed=44) if training else IdentityNormalizer()

    def __len__(self):
        return 5

    def __getitem__(self, index):
        image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 64 + index / 10
        sample = {"sar": image, "target": (torch.arange(64).reshape(8, 8) % 2).long(),
                  "chip_id": str(index)}
        sample = self.preprocess_func(sample)
        # Also exercise Python/NumPy RNGs in worker processes.
        sample["worker_draws"] = torch.tensor([random.random(), np.random.random()])
        return sample


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Dropout(.25), torch.nn.Conv2d(1, 2, 1))

    def forward(self, images):
        return self.head(images["sar"])


def make_loaders(workers=0):
    loaders = {}
    for i, split in enumerate(("train", "val")):
        dataset = TinyDataset(split == "train")
        loaders[split] = DataLoader(
            dataset, batch_size=2, num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
            sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(43)) if split == "train" else None,
            generator=torch.Generator().manual_seed(45 + i),
        )
    return loaders


def setup(workers=0):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Preserve the cached Gaussian value as well as the NumPy key array.
    np.random.normal()
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=9)
    return model, optimizer, scheduler, make_loaders(workers)


def epoch(model, optimizer, scheduler, loaders):
    seen = []
    def record(_, batch):
        seen.append((batch["chip_id"], {k: v.tolist() for k, v in batch["crop_metadata"].items()},
                     batch["worker_draws"].tolist()))
    loss = train_one_epoch(model, loaders["train"], optimizer, torch.nn.CrossEntropyLoss(),
                           "cpu", scheduler=scheduler, input_keys=["sar"], batch_logger=record)
    list(loaders["val"])
    draws = (random.random(), float(np.random.normal()), torch.rand(3).tolist())
    return loss, seen, draws


class ResumeRNGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_continuation(self, workers):
        with tempfile.TemporaryDirectory() as folder:
            model, opt, sched, loaders = setup(workers)
            epoch(model, opt, sched, loaders)
            state = capture_rng_state(loaders)
            path = Path(folder) / "last.pt"
            save_checkpoint(path, model, 1, optimizer=opt, scheduler=sched, rng_state=state)
            expected = epoch(model, opt, sched, loaders)
            weights = copy.deepcopy(model.state_dict())
            lr = opt.param_groups[0]["lr"]
            model, opt, sched, loaders = setup(workers)
            random.random(); np.random.random(); torch.rand(7)
            restored = load_checkpoint(path, model, optimizer=opt, scheduler=sched)
            self.assertTrue(restore_rng_state(restored["rng_state"], loaders))
            actual = epoch(model, opt, sched, loaders)
            self.assertEqual(actual, expected)
            self.assertEqual(opt.param_groups[0]["lr"], lr)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, weights[key]), key)

    def test_continuation_main_process(self):
        self.assert_continuation(0)

    def test_continuation_two_spawn_workers(self):
        self.assert_continuation(2)

    def test_legacy_and_weights_only_loading(self):
        model, opt, sched, loaders = setup()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            save_checkpoint(path, model, 1, optimizer=opt, scheduler=sched, components=["head"])
            state = capture_rng_state(loaders)
            loaded = load_checkpoint(path, model, optimizer=opt, scheduler=sched)
            self.assertIsNone(loaded["rng_state"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertFalse(restore_rng_state(None, loaders))
            self.assertIn("no RNG state", output.getvalue())
            self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_cpu"]))
            save_checkpoint(path, model, 1, components=["head"], rng_state=state)
            torch.load(path, weights_only=True)
            torch.rand(5)
            before_load = torch.get_rng_state()
            load_checkpoint(path, model)  # Evaluation must not restore RNGs implicitly.
            self.assertTrue(torch.equal(torch.get_rng_state(), before_load))

    def test_loader_mismatch_is_rejected(self):
        _, _, _, loaders = setup()
        state = capture_rng_state(loaders)
        state["loaders"]["train"]["num_workers"] = 2
        before = torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "Loader settings changed"):
            restore_rng_state(state, loaders)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_states_and_cuda_map_location(self):
        _, _, _, loaders = setup()
        state = capture_rng_state(loaders)
        expected = [torch.rand(4, device=i) for i in range(torch.cuda.device_count())]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rng.pt"
            torch.save(state, path)
            restored = torch.load(path, weights_only=True, map_location="cuda")
            restore_rng_state(restored, loaders)
        for i, value in enumerate(expected):
            self.assertTrue(torch.equal(value, torch.rand(4, device=i)))

    def test_real_runner_resume_matches_uninterrupted(self):
        self.assert_runner_resume_matches_uninterrupted(1)

    def test_resume_after_epoch_without_validation(self):
        self.assert_runner_resume_matches_uninterrupted(2)

    def assert_runner_resume_matches_uninterrupted(self, validate_every):
        import run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "split.csv"
            split.write_text("synthetic,label\n")
            config = {
                "experiment": {"name": "rng_test", "seed": 42},
                "dataset": {"name": "sen1floods11", "modalities": ["sar"],
                            "train_csv": str(split), "val_csv": str(split)},
                "model": {"name": "unet", "in_channels": 1},
                "head": {"num_classes": 2},
                "training": {"epochs": 3, "validate_every": validate_every,
                             "loss": {"name": "cross_entropy", "ignore_index": 255},
                             "optimizer": {"name": "adamw", "learning_rate": .003},
                             "scheduler": {"name": "cosine_annealing", "warmup_epochs": 1}},
                "evaluation": {"checkpoint_metric": "miou", "class_names": ["land", "water"],
                               "sample_id_key": "chip_id", "return_per_sample": True},
            }
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(run, "build_preprocessing", return_value=(IdentityNormalizer(), {})))
                stack.enter_context(patch.object(run, "build_loaders", side_effect=lambda *a: make_loaders()))
                stack.enter_context(patch.object(run, "build_model", side_effect=lambda *a: TinyModel()))
                stack.enter_context(patch.object(run, "load_config", return_value=config))
                for name in ("report_runtime_setup", "report_first_training_batch", "save_training_plots"):
                    stack.enter_context(patch.object(run, name))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                full, resumed = root / "full", root / "resumed"
                full.mkdir(); resumed.mkdir()
                run._run_experiment(config, full, "cpu")
                calls = 0
                original = run.train_one_epoch
                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise InterruptedError("Synthetic stop between epochs")
                    return original(*args, **kwargs)
                with patch.object(run, "train_one_epoch", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        run._run_experiment(config, resumed, "cpu")
                run.resume_experiment(resumed, "cpu")
                a = json.loads((full / "metrics/history.json").read_text())
                b = json.loads((resumed / "metrics/history.json").read_text())
                self.assertEqual(a, b)
                for name in ("best.pt", "last.pt"):
                    a = torch.load(full / "checkpoints" / name, weights_only=True)
                    b = torch.load(resumed / "checkpoints" / name, weights_only=True)
                    self.assertEqual(a["epoch"], b["epoch"])
                    self.assertIn("rng_state", a)
                    for key in a["model_state_dict"]:
                        self.assertTrue(torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]))


if __name__ == "__main__":
    unittest.main()
