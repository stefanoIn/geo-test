# CROMA and TerraMind: learning-rate selection

Recorded: 7 November 2026.

These choices come from completed 20-epoch FCN-head tuning runs using unweighted cross-entropy. Checkpoints were selected by validation Water IoU. These are validation results, not held-out test results.

| Input | Best tested CROMA LR | CROMA Water IoU | CROMA mIoU | Best tested TerraMind LR | TerraMind Water IoU | TerraMind mIoU |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.0003 | 66.14% | 80.59% | 0.005 | 65.45% | 80.38% |
| S2 | 0.0003 | 74.62% | 85.64% | 0.0001 | 73.26% | 84.87% |
| S1+S2 | 0.0003 | 76.73% | 86.72% | 0.0001 | 73.78% | 85.17% |

## Rates covered

- CROMA: completed unweighted trials across all three modalities at 0.0003, 0.0005 and 0.005. An incomplete S1 trial at 0.0001 is not included.
- TerraMind: completed unweighted trials across all three modalities at 0.0001, 0.0003, 0.0005, 0.001 and 0.005.

The searches were not identical. The selected rates are the best among the completed trials, not proven global optima. In particular, CROMA's unweighted search did not fully cover TerraMind's five-rate grid.

## Choices for the next longer runs

- CROMA S1, S2 and S1+S2: learning rate 0.0003.
- TerraMind S1: learning rate 0.005.
- TerraMind S2 and S1+S2: learning rate 0.0001.
- Keep unweighted cross-entropy for these selected configurations.

These are provisional choices from one seed and a short training budget. Longer training and repeated seeds are needed to establish stability. Model-specific learning rates are appropriate; comparisons should provide comparable tuning budgets and use the same held-out evaluation protocol. This report does not modify any configurations or launch training.

## Saved evidence

- [CROMA validation ranking](../runs/sen1floods11/croma/tuning/validation_ranking.csv)
- [TerraMind trial summaries](../runs/sen1floods11/terramind/tuning_manual/)

For each model, mIoU in the table is measured at the checkpoint selected by Water IoU, rather than independently selecting the highest mIoU.
