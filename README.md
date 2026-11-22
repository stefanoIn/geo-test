# GeoTest

**Work in progress:** GeoTest is under active development and is not yet ready for others to reproduce the experiments. Setup requirements, interfaces and experimental workflows are still being consolidated. Installation and reproduction instructions will be added later.

GeoTest is a YAML-configured project for comparing geospatial foundation models and supervised baselines. It is developed as part of my MSc thesis on benchmarking GeoFMs, focusing on limited labelled data and geographical generalisation.

The current experiments focus on **water segmentation with the hand-labelled Sen1Floods11 dataset**, using Sentinel-1, Sentinel-2 or both. Training, validation, checkpoint selection and evaluation share a common runner. Model-specific preprocessing and feature extraction remain explicit in each configuration.

## Current scope

| Component | Implemented options |
| --- | --- |
| Dataset | Sen1Floods11 hand-labelled chips, selected through split CSVs |
| Foundation models | CROMA Large; TerraMind v1 Large |
| GeoFM segmentation heads | Linear segmentation (`1x1` convolution) and FCN |
| Supervised baselines | U-Net and ResNet U-Net |
| Inputs | Sentinel-1 SAR, Sentinel-2 optical, or paired SAR/optical |
| Execution | Training, evaluation only, checkpoint resume and sequential batch runs |
| Supporting analysis | Crop diagnostics, evaluation timing and geographical pretraining exposure |

The model wrappers expose more options than have been experimentally benchmarked. The recent tuning configurations use frozen Large GeoFM encoders with linear heads, and fully trained U-Net/ResNet-18 U-Net baselines on S1+S2. Regression, change detection and additional datasets are future extensions.

## Repository layout

```text
GeoTest/
|-- configs/                 # Training, tuning and evaluation YAMLs
|   |-- STANDARD_PROTOCOL.md # Reference protocol; not loaded automatically
|   `-- template.yaml        # Configuration starting point
|-- src/
|   |-- data/                # Dataset loading, transforms and normalization
|   |-- encoders/            # CROMA and TerraMind wrappers
|   |-- adapters/            # Token sequences to spatial feature maps
|   |-- heads/               # Linear and FCN segmentation heads
|   |-- baselines/           # Supervised U-Net variants
|   |-- models/              # Model assembly and factory
|   `-- evaluation/          # Tiled inference, metrics and prediction figures
|-- training/                 # Training loop and checkpoint serialization
|-- experiments/              # Notebooks and supporting analysis scripts
|-- tests/                    # Unit checks and manual model integration checks
|-- reports/                  # Analysis outputs and archived tuning records
|-- runs/                     # Per-run settings, metrics, logs and plots
|-- data/                     # Local datasets and statistics cache; Git-ignored
|-- pretrained_models/        # Local pretrained weights and metadata; Git-ignored
|-- run.py                    # Training, evaluation and resume entry point
|-- run_batch.py              # Sequential execution of an editable YAML list
`-- utils.py                  # Configuration validation, builders and reporting
```

Experiment settings are stored in YAML files. Datasets and pretrained weights are maintained locally and are excluded from Git.

## Experiment design

The shared pipeline loads split CSVs, selects the input modalities, applies model-specific preprocessing, constructs the model and trains the selected parameters. Validation determines the best checkpoint; held-out evaluation is a separate step or follows training when configured.

For GeoFMs, an encoder produces token representations, an adapter converts them into spatial feature maps, and a linear or FCN head produces segmentation logits. The supervised baselines train the complete network from scratch.

Configurations specify the seed, input bands, normalization, crop size, batch size, learning rate, optimizer, scheduler, validation interval and evaluation tiling. The current reference settings are recorded in [configs/STANDARD_PROTOCOL.md](configs/STANDARD_PROTOCOL.md); older configurations may use different protocols. These files document the ongoing experiments and do not yet constitute a complete reproduction package.

Training uses aligned crops across each sample's modalities and target. Entirely ignored training chips are excluded, and crops without valid labels are resampled. Valid crops without water are retained.

## Outputs and metrics

Each training invocation creates a timestamped run directory containing:

```text
<run>/
|-- config.yaml
|-- normalization.json
|-- provenance.json
|-- runtime_setup.json
|-- runtime_setup.txt
|-- run.log
|-- summary.json
|-- checkpoints/              # best.pt and last.pt; Git-ignored
|-- metrics/                  # history.json, validation reports and optional test reports
|-- plots/                    # Training/validation curves and evaluation metric figures
|-- debug/                    # Crop coordinates and validation sample logs
`-- predictions/              # Optional per-chip figures; Git-ignored
```

`summary.json` records checkpoint selection and summary metrics. Detailed reports include confusion counts, pixel accuracy, per-class IoU, mIoU, precision, recall and F1. Validation loss and training loss are recorded in the history. Plot files are saved inside each run's `plots/` directory.

For Sen1Floods11, `evaluation.aw_pw_fw: true` adds all-water (AW), permanent-water (PW) and flood-water (FW) IoU using the JRC references. Reports distinguish:

- **Pooled Water IoU:** calculated after summing TP, FP and FN across evaluated pixels.
- **Mean per-chip Water IoU:** the arithmetic mean of the included chips' Water IoUs.
- **mIoU:** the mean of the defined per-class IoUs from the pooled confusion matrix.

Ignored pixels are excluded. Entirely ignored chips are excluded from per-chip averages. The AW/PW/FW implementation retains background, masks the other water category, and applies category-specific chip inclusion rules. Valid empty unions score 1 in the per-chip calculation. Report these conventions when comparing results with other implementations; PW and FW averages do not generally average to AW.

## Geographical pretraining exposure

The standalone study compares Sen1Floods11 footprints with CROMA's SSL4EO-S12 locations and TerraMind's TerraMesh metadata. It produces per-chip overlap/proximity summaries and local maps.

CROMA footprints are reconstructed approximations. Geographical exposure does not establish that a model saw the same acquisition or labels, nor prove downstream data leakage. The output provenance records the analysis assumptions and limitations.

## Development status

The repository currently documents the implementation and ongoing experiments for the thesis. Dependency management, tests, examples and instructions still need further work before the experiments can be reproduced reliably from a fresh clone.

Saved configurations, metrics, hashes and logs retain information about completed runs, but they are not a complete reproduction package. A documented setup and step-by-step reproduction workflow will be provided later.

Current comparisons also need to be interpreted in context: frozen GeoFM linear probing and fully trained supervised baselines use different adaptation regimes, and single-seed tuning does not establish a general ranking of models.

## Attribution and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream code attribution, retained licence texts and unresolved provenance questions. GeoTest currently has no project-wide licence assigned. Dataset and checkpoint terms are separate from source-code terms; local availability does not establish redistribution permission.

Relevant upstream work includes [CROMA](https://github.com/antofuller/CROMA), [TerraMind](https://github.com/IBM/terramind), [TerraTorch](https://github.com/IBM/terratorch), [PANGAEA](https://github.com/VMarsocci/pangaea-bench), and [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). The supplied Sen1Floods11 data should be cited as Bonafilia, Tellman, Anderson and Issenberg (2020), *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops.
