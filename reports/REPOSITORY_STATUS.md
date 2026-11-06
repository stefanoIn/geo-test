# GeoTest repository: structure, status and implementation review

Reviewed: 6 November 2026. Scope: the current GeoTest source and configurations, not the older CROMA project.

This report describes a code-based experiment framework under development. It is not evidence that a complete benchmark has run successfully in this repository. This review inspected code; it did not run training, inference or the test suite. No implementation files were changed for this report.

## Purpose and current execution flow

GeoTest separates dataset loading, preprocessing, pretrained encoders, feature adaptation and segmentation heads so experiments can be selected through YAML instead of duplicated notebooks. The extra directory intentionally preserves previous implementations for reference.

The intended flow is:

```text
YAML -> dataset + preprocessing -> DataLoader
     -> encoder -> token-to-spatial adapter -> FCN -> segmentation logits
     -> training / validation / testing -> checkpoints and saved metrics
```

The current run.py only implements the data-pipeline portion: configuration loading, seeding, training-statistics calculation/cache, loaders and inspection of one batch per configured split. It does not construct a model, apply the configured crop, train, evaluate or write experimental results. Integration scripts exercise some later components separately.

## Folder guide

| Location | Responsibility and current status |
|---|---|
| configs/croma/ | CROMA experiment settings, including splits, normalization, bands, crop size and optimization settings. Several settings await runner integration. |
| configs/terramind/ | A model integration-test configuration and an illustrative S1+S2 configuration. Neither is currently a complete training configuration for run.py. |
| data/ | Sen1Floods11 imagery, labels, catalog and split files; data/statistics stores cached training statistics. |
| pretrained_models/ | Local foundation-model checkpoints, separate from future trained-run checkpoints. |
| src/data/ | Dataset reading, normalization, statistics and spatial transforms. |
| src/encoders/ | CROMA and TerraMind wrappers around their backbone implementations. |
| src/adapters/ | Conversion of spatial patch tokens into feature maps. |
| src/heads/ | Current FCN segmentation head. |
| src/models/ | Composition of encoder, adapter and head into a dense prediction model. |
| src/evaluation/ | Sliding-window inference; metrics and evaluator are still planned. |
| src/baselines/ | Placeholder package; no baseline implementation here yet. |
| training/ | One-epoch training function. |
| tests/ | Component/integration checks, rather than benchmark results. |
| extra/experiments and extra/HELPERSS/ | Previous implementations retained as reference; not the main execution path. |
| outputs/ | Existing output location. The CROMA YAML instead proposes runs/...; settle on one convention when connecting saving. |
| reports/ | Human-readable documentation, including this report. |
| run.py | Current data-pipeline smoke-test entry point. |

## Source modules

### Dataset and preprocessing

- src/data/datasets/sen1floods11.py: reads headerless image/label CSV splits, verifies paired chip names, rejects duplicate chips within a split, checks files exist and lazily reads selected SAR/optical rasters. Input nodata becomes NaN; label -1 becomes 255. A custom CSV selects chips from the existing HandLabeled directory. The loader currently does not load weak labels or JRC permanent-water references.
- src/data/preprocessing.py: NormalizeFromStats selects/reorders bands, applies z-score or a clipped two-standard-deviation stretch, optionally quantizes the stretch to 8 bits and fills invalid image values with zero after normalization. It consumes supplied statistics rather than computing them.
- src/data/statistics.py: computes per-channel statistics from finite image values and caches them independently by modality. Cache identity includes the split and source-file metadata. These are image statistics, not statistics restricted to valid ground-truth pixels.
- src/data/transforms.py: random_crop applies identical coordinates to every image modality and the mask. It exists but is not connected to run.py training loaders.

### Encoders and prediction

- src/encoders/croma/use_croma.py: underlying CROMA implementation used by the wrapper.
- src/encoders/croma/croma.py: constructs the backbone, supports SAR/optical/both selection and returns spatial tokens plus pooled features. The selected modality determines whether the returned representation is SAR, optical or joint. Frozen encoders remain in evaluation mode when the composed model enters training mode.
- src/encoders/terramind/terramind.py: configurable TerraMind v1 Base/Large wrapper using a local checkpoint. Accepts native product names and an optional dataset-key mapping, such as sar -> S1GRD and optical -> S2L1C. It checks checkpoint tensors, selected inputs and image channel counts, pads raw images to patch-compatible sizes and returns last-layer tokens plus their spatial mean. Mean/max/concatenation fusion are configurable. Normalization and product conversion are deliberately outside this wrapper. Non-image modalities are not established as working by this review; the wrapper itself says they require separate handling and testing.
- src/adapters/token_to_spatial.py: reshapes [B,N,D] spatial tokens to [B,D,H,W]. Infers a square grid unless an explicit grid is supplied. It does not change feature channels or generate a feature pyramid.
- src/heads/fcn.py: torchvision FCN head with BatchNorm replaced by 16-group GroupNorm. It takes one spatial feature map and emits class logits. This is not UPerNet.
- src/models/dense_prediction.py: combines the three components and bilinearly resizes head outputs to the input size.

### Training and inference

- training/trainer.py: train_one_epoch handles device transfer, backward propagation, optimizer updates and optional per-update scheduler steps. It skips batches with only ignored targets and aggregates mean cross-entropy using valid-pixel counts or class-weight sums. It does not own the epoch loop, validation or checkpointing.
- src/evaluation/sliding_window.py: predicts tiles, averages overlapping logits and returns full-image logits. It checks spatial shapes and uses inference mode. The caller must set model.eval(). Window positions are evenly distributed using ceil(image_size/tile_size); there is no independently configured stride.

### Existing checks

integration_test01 checks the CROMA composition with synthetic inputs. integration_test02 exercises data/preprocessing and model integration. integration_test03 inspects TerraMind shapes and gradient behavior. test_terramind_wrapper_unit.py uses a fake backbone to check wrapper logic, including modality selection, channel errors, checkpoint compatibility and concatenation. Mock-based tests do not prove numerical agreement with the real pretrained model. Tests were not executed during this review.

## Verified issues and limitations

### 1. Padding is not accounted for when restoring prediction geometry

TerraMind can pad a 120 x 120 input to 128 x 128. Its tokens then cover the padded extent. DensePredictionModel currently interpolates those predictions directly to 120 x 120, compressing the padded extent into the original image rather than removing it. For padded inputs, propagate the padded dimensions, resize logits to that extent, then crop to the original dimensions. Shapes alone will not detect this alignment problem. Inputs already divisible by the patch size avoid this particular padding case.

### 2. Configuration is not yet an executed experiment protocol

run.py does not consume the model settings to build the model, nor execute optimizer, epoch, output or evaluation settings. CROMA feature_output is present in YAML, but the current wrapper selects its output from modality and has no feature_output argument. Either wire supported options explicitly or remove redundant fields to avoid implying they affect execution.

TerraMind product selection now exists in the wrapper, but run.py does not connect it. Dataset modality names, model modalities and input_mapping must agree. Product names are declarations, not conversion operations: selecting S1RTC does not turn a GRD raster into RTC. Channel-count checks cannot establish band order, physical units or processing level.

### 3. Fixed normalization is not connected

NormalizeFromStats can accept externally supplied statistics, but build_preprocessing currently rejects anything except statistics: training_dataset. Add an explicit fixed-statistics branch for the selected TerraMind products. Keep band order and statistics order synchronized. Also validate finite means; the normalizer currently explicitly validates standard deviations only.

### 4. Trainer assumptions need an explicit contract

The current aggregation is intended for mean cross-entropy on hard labels. A scalar check alone does not reject reduction='sum' or establish correct averaging for Dice or other losses. Select supported loss/reduction combinations explicitly.

The scheduler is stepped after every optimizer update. This can suit an iteration-based schedule, but an epoch-based scheduler belongs in the outer loop, and ReduceLROnPlateau requires a validation metric. Do not pass all scheduler types through the current optional argument unchanged.

### 5. Split and data validity checks are partial

Duplicate IDs are rejected within each CSV, but there is no cross-split leakage check in run.py. Add a check that the intended train/validation/test chip sets do not overlap. Pixel-array shape equality does not prove geospatial alignment; a dataset audit should check CRS and transforms as well.

HandLabeled paths are fixed inside the dataset module. Custom split CSVs work, but moving the dataset elsewhere requires a code change. The source-location comment above PROJECT_ROOT is stale: the file is now under src/data/datasets/. The current parents[3] expression matches that actual location.

The CSV reader expects exactly two fields on every row; blank lines and malformed rows produce generic unpacking errors. Supporting clearer row-numbered errors is a small usability improvement, not a prerequisite to training with the existing valid files.

### 6. Missing-data and ignore-label policy need consistency

The dataset hardcodes ignore label 255, while run.py permits an ignore_index configuration for its checks. Use a single convention throughout dataset, criterion and metrics.

Invalid input pixels are filled after normalization without automatically changing target validity. This is a policy choice rather than automatically a bug: decide and record how missing sensor observations are handled, particularly for multimodal comparisons. Do not silently remove evaluation samples to improve scores. All-ignore training batches are skipped, but the evaluator still needs a documented all-ignore and zero-union policy.

### 7. Shape generality is limited

TokenToSpatial assumes a square grid when none is provided. Rectangular token grids need explicit dimensions; even a perfect-square token count does not prove the original grid was square. TerraMind concatenation changes feature width, and Base/Large use different widths; the configured head must match. FCN GroupNorm additionally requires its hidden width to be divisible by 16.

### 8. Reproducibility and output management are unfinished

There is no root dependency manifest or README in the inspected top-level listing. Record the tested Python, PyTorch, torchvision, TerraTorch and other required versions when making the project runnable for others. Dependency compatibility has mattered previously.

The current .gitignore excludes pretrained_models and data, but does not exclude future run/checkpoint outputs. Its __pycache/ entry also misses the actual __pycache__ spelling, although *.pyc excludes compiled Python files. Choose output exclusions before generating large trained checkpoints.

No saved score in the legacy implementation should be represented as a newly reproduced GeoTest result until the corresponding experiment actually runs here.

## Planned next modules

The following are responsibilities to implement, not files created by this report.

### Metrics

Accumulate confusion counts over valid ground-truth pixels and derive per-class IoU, mIoU, water precision/recall/F1, accuracy, omission and commission. Define zero-denominator handling and log evaluated pixel/chip counts. Keep pooled water IoU distinct from mean per-chip water IoU. AW/PW/FW analyses require the corresponding reference-mask inputs and explicit definitions; AW is not generally the arithmetic mean of PW and FW.

### Evaluator

Set evaluation mode, disable gradients, separate targets from image inputs, use direct or sliding-window inference consistently, and aggregate loss and metrics. Preserve chip identity for per-chip exports. The dataset currently returns tensors without an explicit chip ID, so provide identity deliberately rather than relying on assumptions about iteration order. If metadata is added to batches, update the trainer's current assumption that every non-target entry is an image tensor.

### Checkpointing

Save a run's resolved configuration, epoch, selected validation metric and model state. For resumable training also save optimizer/scheduler states and random-generator states, plus scaler state if mixed precision is introduced. Record pretrained checkpoint identity and split identity. Distinguish best-validation and latest checkpoints, and explicitly link evaluation outputs to the checkpoint evaluated. Select checkpoints using validation, not test results.

### Runner integration

Extend the existing run.py to construct the components, apply training transforms, select criterion/optimizer, call train_one_epoch, validate at the chosen interval and invoke checkpointing. Reload the selected checkpoint for final testing. Complete the TerraMind training YAML. Small builder functions in run.py are sufficient initially; a registry or additional framework is not required.

## Recommended implementation order

1. Implement metrics, evaluator and checkpointing with clear input/output contracts.
2. Correct padded-output geometry and connect fixed normalization and training transforms.
3. Complete YAML validation and model construction in run.py.
4. Check metric formulas with small known confusion matrices, split separation and checkpoint save/reload behavior.
5. Run a short end-to-end experiment before committing to full training.

This preserves the current understandable separation of responsibilities while completing the missing execution path.
