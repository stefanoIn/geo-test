# GeoTest

**Research project — under active development**

GeoTest is a configuration-driven research framework for evaluating and benchmarking geospatial foundation models (GeoFMs) against supervised baselines for Earth observation.

The project is being developed as part of my MSc thesis in Computer Science (Artificial Intelligence) at the University of Genoa.

The research focuses on downstream performance, label efficiency and geographical generalisation, with particular attention to how pretrained models behave under limited labelled data, different input modalities and geographical distribution shifts.

The current experimental work uses the hand-labelled Sen1Floods11 dataset for water segmentation with Sentinel-1 SAR, Sentinel-2 optical imagery and paired SAR/optical inputs.

The framework, experimental protocols and documentation are under active development. A complete installation and reproduction workflow will be provided once the experimental infrastructure has been consolidated.

## Research objectives

The principal research directions are:

- Evaluation of geospatial foundation models on downstream Earth observation tasks.
- Comparison with supervised architectures trained from scratch.
- Downstream sample efficiency under progressively reduced label availability.
- Geographical generalisation and distribution shift.
- Evaluation across different input modalities.
- Investigation of potential associations between geographical pretraining exposure, environmental representation and downstream performance.

The geographical and ecological components of the research are maintained in the companion repository [GeoTest-Atlas](https://github.com/stefanoIn/geo-test-atlas).

## Current scope

| Component | Supported options |
| --- | --- |
| Downstream dataset | Sen1Floods11 — hand-labelled subset |
| Downstream task | Semantic segmentation |
| Foundation models | CROMA Large, TerraMind v1 Large |
| Segmentation heads | Linear segmentation (1×1 convolution), FCN |
| Supervised baselines | U-Net, ResNet-18 U-Net |
| Input modalities | Sentinel-1, Sentinel-2, Sentinel-1 + Sentinel-2 |
| Model adaptation | Frozen GeoFM encoders with trainable segmentation heads |
| Experimental execution | Training, validation, independent evaluation, checkpoint resume and sequential batch execution |
| Evaluation | Aggregate and per-chip segmentation metrics |
| Visualisation | Learning curves, evaluation metrics and qualitative prediction figures |
| Configuration | YAML-based experimental settings |

The model wrappers expose additional configuration options that have not yet been systematically benchmarked.

Current preliminary experiments include frozen GeoFM encoders with linear segmentation heads and fully trained supervised baselines. Additional datasets, adaptation regimes and downstream tasks may be introduced as the research progresses.

## Repository layout

```text
GeoTest/
├── configs/
│   ├── STANDARD_PROTOCOL.md
│   ├── template.yaml
│   └── ...                   # Model-specific and evaluation configurations
├── src/
│   ├── data/                 # Dataset loading, transforms and normalisation
│   ├── encoders/             # Foundation model wrappers
│   ├── adapters/             # Spatial feature adaptation
│   ├── heads/                # Segmentation heads
│   ├── baselines/            # Supervised architectures
│   ├── models/               # Model assembly and factory
│   ├── evaluation/           # Inference, metrics and prediction visualisation
│   └── plotting.py           # Scientific plotting utilities
├── training/                 # Training and checkpoint management
├── experiments/              # Notebooks and supporting analysis scripts
├── tests/                    # Unit and model integration checks
├── reports/                  # Analysis outputs and archived experimental records
├── runs/                     # Experimental configurations, metrics and figures
├── run.py                    # Training, evaluation and resume entry point
├── run_batch.py              # Sequential execution of configured experiments
├── utils.py                  # Configuration, construction and reporting utilities
└── THIRD_PARTY_NOTICES.md
```

Datasets, pretrained weights and trained checkpoints are stored locally and are excluded from version control.

## Experimental framework

GeoTest provides a common experimental pipeline while retaining explicit model-specific preprocessing and architectural requirements.

Experiment configurations define:

- Dataset and split CSVs.
- Input modalities and spectral bands.
- Input normalisation.
- Model architecture and adaptation regime.
- Segmentation head.
- Training crop dimensions.
- Optimiser and learning-rate schedule.
- Batch size and random seed.
- Validation frequency.
- Evaluation tiling and inference settings.

For foundation models, encoder representations are converted into spatial feature maps through dedicated adapters and processed by trainable segmentation heads.

Supervised baselines are trained from scratch.

The current reference experimental settings are documented in [configs/STANDARD_PROTOCOL.md](configs/STANDARD_PROTOCOL.md). Individual experiments may use different settings, particularly during hyperparameter selection.

### Training and validation

The training pipeline uses aligned crops across the input modalities and segmentation targets.

Entirely ignored training chips are excluded, and crops containing no valid labels are resampled. Valid crops without water are retained.

Validation is used to monitor learning, compare configurations and select model checkpoints.

### Held-out evaluation

Saved checkpoints can be evaluated independently of training using the evaluation-only workflow.

The framework supports tiled inference, aggregate segmentation metrics, per-chip evaluation and optional qualitative prediction visualisation.

Held-out evaluation must not be used for hyperparameter or checkpoint selection.

The experimental configuration and selected checkpoint are retained alongside evaluation results to support subsequent analysis.

## Evaluation metrics

The framework records confusion counts and segmentation metrics, including:

- Intersection over Union (IoU).
- Class-wise pooled IoU.
- Mean Intersection over Union (mIoU).
- Pixel accuracy.
- Precision, recall and F1-score.
- Per-chip performance metrics.

Class-wise IoU is calculated from the confusion matrix accumulated across the evaluated samples.

The mIoU is the arithmetic mean of the defined pooled class-wise IoUs.

For Sen1Floods11, additional water-specific evaluation distinguishes:

- All water (AW).
- Permanent water (PW).
- Flood water (FW).

For each water category, the framework distinguishes two aggregation methods:

| Metric | Definition |
| --- | --- |
| Pooled Water IoU | IoU computed after accumulating TP, FP and FN across the included samples |
| Mean per-chip Water IoU | Arithmetic mean of the IoUs computed independently for the included chips |

Ignored pixels are excluded from metric calculations.

The AW/PW/FW implementation retains background pixels, masks the alternative water category and applies category-specific chip inclusion rules. Entirely ignored chips are excluded, while valid empty unions receive a score of one in the per-chip calculation.

These conventions should be considered when comparing results against other implementations.

## Experimental outputs

Training runs produce timestamped directories containing experiment configurations, runtime information, provenance, checkpoints, metrics and diagnostic figures.

A typical training run contains:

```text
<run>/
├── config.yaml
├── normalization.json
├── provenance.json
├── runtime_setup.json
├── runtime_setup.txt
├── run.log
├── summary.json
├── checkpoints/
├── metrics/
├── plots/
└── debug/
```

Independent evaluations are stored under the corresponding run's `evaluations/` directory.

Depending on the configuration, evaluation outputs may include aggregate metrics, per-chip results, prediction figures and additional diagnostic information.

Numerical results are retained independently of their visual representations, allowing subsequent statistical analysis and figure regeneration.

## Scientific visualisation

GeoTest uses Matplotlib and SciencePlots to produce consistent scientific figures suitable for research reports and publications.

The default visual style combines:

```python
["science", "no-latex", "vibrant"]
```

The plotting infrastructure supports shared styling conventions while retaining figure-specific settings.

Current visualisations include:

- Training and validation loss.
- Validation mIoU.
- Selected checkpoint indicators.
- Class-wise pooled IoU.
- Water-category IoU comparisons.
- Ground-truth, prediction and segmentation error maps.

Figures are exported in PNG and PDF formats.

Existing metric figures can be regenerated from saved experimental results without repeating training or inference:

```bash
python experiments/plot_saved_results.py --run-dir <run_directory>
```

Qualitative prediction figures require the corresponding predicted segmentation masks.

## Companion repository: GeoTest-Atlas

[GeoTest-Atlas](https://github.com/stefanoIn/geo-test-atlas) is a standalone companion project for the geographical and ecological components of this research.

The two repositories have separate responsibilities:

| Repository | Responsibilities |
| --- | --- |
| GeoTest | Model integration, training, validation, inference and downstream evaluation |
| GeoTest-Atlas | Geographical metadata, ecological characterisation, pretraining exposure and geographical analysis |

GeoTest-Atlas currently provides functionality for:

- Geographical metadata extraction.
- Centre-based biome and ecoregion assignment using RESOLVE Ecoregions 2017.
- Ecological characterisation of downstream and pretraining-related inventories.
- Spatial footprint overlap and coverage calculations.
- Nearest-pretraining-footprint distances.
- Geographical neighbourhood statistics.
- Preliminary geographical and ecological visualisations.

The next research stage will use these outputs to characterise downstream geographical domains, compare their representation in available pretraining inventories and investigate potential associations with downstream performance.

Further event-level analyses, comparative geographical figures and integration with downstream performance metrics remain ongoing work.

Geographical exposure is treated as a potential explanatory variable, rather than direct evidence of observation reuse or supervised label leakage.

GeoTest does not depend on or import GeoTest-Atlas for model training or evaluation.

## Data and pretrained weights

This repository does not distribute downstream satellite imagery, segmentation labels, pretrained model weights or trained checkpoints.

The datasets and pretrained checkpoints required by the experiments must be obtained separately from their respective providers and configured locally.

The current downstream dataset is [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), using its hand-labelled subset.

Dataset and checkpoint availability, usage and redistribution remain subject to their respective upstream licences and terms.

## Research status

### Implemented

- Integration of CROMA and TerraMind.
- Supervised segmentation baselines.
- Model-specific preprocessing and feature adaptation.
- YAML-based experimental configuration.
- Training, validation and checkpoint management.
- Independent evaluation and tiled inference.
- Aggregate and per-chip segmentation metrics.
- Scientific plotting and figure regeneration.
- Geographical exposure and ecological enrichment infrastructure through GeoTest-Atlas.

### Ongoing research

- Consolidation of preliminary experimental results.
- Hyperparameter selection.
- Reproducible qualitative prediction comparisons across models.
- Event-level geographical and ecological characterisation.
- Comparison of downstream and pretraining environmental distributions.
- Extended geographical exposure visualisations.
- Integration of geographical metadata with downstream performance.

### Planned experiments

- Geographically independent evaluation protocols.
- Controlled downstream label-efficiency experiments.
- Evaluation across multiple random seeds.
- Systematic comparisons between GeoFMs and supervised baselines.
- Investigation of additional downstream datasets and adaptation regimes.

Preliminary results should be interpreted in the context of their experimental configurations. In particular, frozen GeoFM linear probing and fully trained supervised baselines represent different adaptation regimes.

Individual runs and single-seed experiments do not establish general model rankings.

## Reproducibility

Experiment configurations, checkpoint hashes, provenance information, logs and numerical results are retained to support experimental traceability.

The repository is currently a research codebase rather than a complete reproduction package.

Dependency management, environment specification, automated checks and step-by-step reproduction instructions will be consolidated as the research progresses.

## Attribution and licensing

Related projects and resources:

- [CROMA](https://github.com/antofuller/CROMA)
- [TerraMind](https://github.com/IBM/terramind)
- [TerraTorch](https://github.com/IBM/terratorch)
- [PANGAEA](https://github.com/VMarsocci/pangaea-bench)
- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)

For upstream code attribution, retained licence texts and outstanding provenance considerations, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
