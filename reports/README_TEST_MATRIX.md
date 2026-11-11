# GeoTest | CNN baseline integration tests

Ten ready-to-use YAML configurations targeting the current GeoTest `run.py` and
`utils.py` with scratch baselines. Copy the `configs/` directory into the
root of `D:\UNI\GeoTest`. Files are split into `configs/unet/` (5) and
`configs/resnet_unet/` (5).

These are **functional tests, not a defensible hyperparameter search**.
All use `experiment.seed: 234`, the same published HandLabeled splits,
120x120 training crops, validation-based checkpoint selection, and random initialization.

## Recommended order

1. Run **01_unet_s1s2_full_pipeline_2e.yaml** to validate the full pipeline.
2. Run **01_resnet50_s1s2_full_pipeline_2e.yaml** to validate the SMP encoder pathway.
3. Run the eight remaining configs to exercise modality, band selection,
   normalisation, backbone options, and optimizer/scheduler paths.

Only the two `01_*_full_pipeline_2e.yaml` configs include test and Bolivia splits.
They also enable AW/PW/FW. They evaluate the held-out data *only to exercise the
reporting code*. Never use those scores to choose hyperparameters; for tuning,
use the other eight configs' validation data. The other eight configs do not
include held-out paths, so their `run.py` calls do not evaluate them.

`save_predictions: false` keeps disk usage controlled; set to `true` on one
full-pipeline trial if you specifically want to exercise generation of all
per-chip prediction images (beware of a large number of files).

`preprocessing.band_indices.optical: [0, ..., 9, 11, 12]` excludes raw
band index 10 (B10) in the 14-channel trials. This is the optical band subset
used by the CROMA YAML in the current repository. The 15-channel configs use
all 13 optical bands. The 8-bit trial tests the two-sigma input path; it is
not claimed to be better scientifically.

## Run a single YAML

```powershell
python run.py --path configs/unet/01_unet_s1s2_full_pipeline_2e.yaml
python run.py --path configs/resnet_unet/01_resnet50_s1s2_full_pipeline_2e.yaml
```

## Run all 10 sequentially (stop on first failure)

```powershell
python run_batch_baselines.py
```

Each run creates a unique subdirectory under `runs/sen1floods11/baselines/`.
Inspect `run.log`, `runtime_setup.json`, `first_training_batch.json`,
`metrics/history.json`, `summary.json`, `plots/`, and `checkpoints/best.pt`.
Full-pipeline trials also create `metrics/test.json`, `metrics/bolivia.json`
and the corresponding plots.

If GPU out-of-memory occurs on the RTX 3070 8GB, lower
`training.batch_size: 4` to `2` (or 1, if needed) on that YAML.
`evaluation.tile_batch_size` is already 1. Lowering batch size changes the
optimization run and should be logged. These YAMLs have not been executed on
your local dataset or GPU and their elapsed times are not guaranteed.

## Important before actual thesis comparisons

Use a separate, fixed experimental protocol: equal budgets/splits, controlled
band availability and normalisation, validation-only tuning, several seeds,
and documented convergence criteria. A 2-4 epoch smoke test is **not** a
sample-efficiency result. Different backbones, channels, and learning rates
mean the trials are not controlled head-to-head comparisons.
