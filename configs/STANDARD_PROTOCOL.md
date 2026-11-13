# Standard experiment protocol

Reference for future Sen1Floods11 runs. This document is not loaded automatically: copy the settings into each run's YAML configuration. It does not modify previous experiments.

## Common settings

| Parameter | Standard |
| --- | --- |
| Seed | 42 |
| Epochs | 50, including warmup |
| Training batch size | 8 |
| DataLoader workers | 0 |
| Validation | Every epoch |
| Loss | Unweighted cross entropy; ignore index 255 |
| Optimizer | AdamW, weight decay 0.01, standard betas (0.9, 0.999) |
| Learning rate | Tuned for each model, head and protocol |
| Scheduler | Cosine annealing, no warm restarts, eta_min 0 |
| Optional warmup | Currently 5 epochs, from 10% of the LR to the configured LR |
| Best checkpoint | Highest validation mIoU |
| Evaluation batch size | 1 image |
| Tiles processed together | 8, subject to available memory |

50 epochs is the current budget, not a guarantee of convergence. The final batch may contain fewer than 8 samples. Image batch size and tile batch size are separate parameters.

## YAML snippet to copy

Merge these sections into the complete model configuration. LR 0.003 is an example from the current CROMA linear tuning setup, not a universal value selected for all models.

```yaml
experiment:
  name: unique_experiment_name
  seed: 42

training:
  epochs: 50
  batch_size: 8
  num_workers: 0
  validate_every: 1
  loss:
    name: cross_entropy
    ignore_index: 255
    class_weights: null
  optimizer:
    name: adamw
    learning_rate: 0.003  # Replace with the trial LR or the selected LR.
    weight_decay: 0.01
  scheduler:
    name: cosine_annealing
    warmup_epochs: 5
    warmup_start_factor: 0.1
    eta_min: 0.0

evaluation:
  checkpoint_metric: miou
  batch_size: 1
  tile_batch_size: 8
  class_names: [non_water, water]
  return_per_sample: true
  sample_id_key: chip_id
  aw_pw_fw: true
```

This snippet is not a complete runnable configuration: retain dataset, model, adapter, head, preprocessing, tile_size, stride and output from the specific YAML. AdamW uses its default betas; no additional YAML parameter is needed.

## Warmup

With 50 total epochs and warmup_epochs: 5, the first 5 increase the LR linearly from 0.1 * learning_rate to learning_rate; the remaining 45 apply cosine decay. The scheduler advances with each training update.

To disable warmup, set warmup_epochs: 0 or omit the warmup fields. If the budget changes, explicitly choose the number of warmup epochs (approximately 5-10%); it is not calculated automatically as a percentage. Keep the same policy across trials within a comparison. Tuning without warmup does not automatically identify the best LR with warmup.

## Data and model-specific settings

- Use the same train/validation/test splits for comparisons under the same protocol. Current splits are in data/sen1floods11/splits/all_hand_labels/.
- During tuning, use only train_csv and val_csv: exclude test_csv and bolivia_csv. Use the test sets only after hyperparameter selection.
- Exclude entirely ignored chips from training and resample crops with zero valid pixels. Keep valid crops without water. No minimum water fraction is imposed.
- Exclude ignored pixels from metrics and entirely ignored chips from per-chip averages. Distinguish Water IoU pooled across pixels from mean per-chip Water IoU.
- Normalization, bands, modalities, pretrained checkpoints and input dimensions remain model-specific. Do not blindly copy CROMA preprocessing to TerraMind.
- Current CROMA setup: crop/tile size 120, stride 80; linear probing with a frozen encoder. TerraMind: crop/tile size 128; check stride and normalization in its YAML.
- Supervised baselines train the entire network: do not automatically apply the linear probing freeze_encoder setting.
- AW/PW/FW and the non_water/water classes are specific to this Sen1Floods11 evaluation; adapt them for other datasets.

A common seed supports reproducibility but does not, by itself, guarantee identical crops across configurations with different dimensions or worker handling. Preserve the actual YAML and logs for each run. Repeat promising configurations with additional seeds to quantify variability.

## Currently prepared CROMA linear tuning

S1+S2, 50 epochs, batch size 8, seed 42, validation every epoch, 5 warmup epochs and cosine decay without restarts. LRs to compare: 0.0003, 0.001, 0.003, 0.005.

The four YAML files are in croma/linear_tuning/ (s1s2_lr*.yaml). These are search configurations: select the winner using validation results. Do not change the protocol between trials without recording it as a separate experiment.
