# Real 5m vs 15m Training Experiment Comparison

## Scope

This note records one real side-by-side training run for the existing BTC/USDT pipeline on:

- `5m` horizon
- `15m` horizon

The goal is to compare the current production-style `5m` setup with the newly added `15m` setup under the same data source, model family, and validation protocol.

## Data And Protocol

- Input dataset: `artifacts/datasets/binance_btc_usdt_1m.parquet`
- Config: `config/settings.yaml`
- Model plugins: `stage1=lightgbm_stage1`, `stage2=lightgbm_stage2`
- Validation window: last `30` days
- Calibration fraction: `0.15`
- Purge rows: `1`
- Derivatives features: disabled in both runs
- Feature count: `126` in both runs

Commands used:

```powershell
rtk proxy python scripts/run_model_experiments.py `
  --input artifacts/datasets/binance_btc_usdt_1m.parquet `
  --output-dir artifacts/models/horizon_compare/experiments_5m `
  --config config/settings.yaml `
  --horizon 5m `
  --stage1-model-plugins lightgbm_stage1 `
  --stage2-model-plugins lightgbm_stage2 `
  --ablation-variants baseline `
  --validation-window-days 30 `
  --purge-rows 1

rtk proxy python scripts/run_model_experiments.py `
  --input artifacts/datasets/binance_btc_usdt_1m.parquet `
  --output-dir artifacts/models/horizon_compare/experiments_15m `
  --config config/settings.yaml `
  --horizon 15m `
  --stage1-model-plugins lightgbm_stage1 `
  --stage2-model-plugins lightgbm_stage2 `
  --ablation-variants baseline `
  --validation-window-days 30 `
  --purge-rows 1
```

## Side-By-Side Metrics

| Metric | 5m | 15m | Delta (15m - 5m) |
| --- | ---: | ---: | ---: |
| Train accuracy | 0.572329 | 0.628701 | +0.056372 |
| Train log loss | 0.685688 | 0.679531 | -0.006157 |
| Train ROC AUC | 0.612795 | 0.680142 | +0.067347 |
| Valid accuracy | 0.518257 | 0.530271 | +0.012015 |
| Valid log loss | 0.691730 | 0.691566 | -0.000164 |
| Valid ROC AUC | 0.531100 | 0.539170 | +0.008070 |
| Overfit gap accuracy | 0.054073 | 0.098430 | +0.044357 |
| Overfit gap log loss | 0.006042 | 0.012035 | +0.005993 |
| Overfit gap ROC AUC | 0.081695 | 0.140972 | +0.059277 |
| Train sample count | 228110 | 75823 | -152287 |
| Valid sample count | 8627 | 2874 | -5753 |

## Market Window Details

### 5m

- Train rows: `228110`
- Train window: `2024-01-01 02:00:00+00:00` to `2026-03-09 23:45:00+00:00`
- Valid rows: `8627`
- Valid window: `2026-03-09 23:55:00+00:00` to `2026-04-08 23:55:00+00:00`

### 15m

- Train rows: `75823`
- Train window: `2024-01-01 02:00:00+00:00` to `2026-03-09 23:15:00+00:00`
- Valid rows: `2874`
- Valid window: `2026-03-09 23:45:00+00:00` to `2026-04-08 23:45:00+00:00`

## 15m Metrics To Record

The requested `15m` market train and valid metrics are:

### 15m Train

- Accuracy: `0.628701`
- Log loss: `0.679531`
- ROC AUC: `0.680142`
- Sample count: `75823`

### 15m Valid

- Accuracy: `0.530271`
- Log loss: `0.691566`
- ROC AUC: `0.539170`
- Sample count: `2874`

## Interpretation

- `15m` is better than `5m` on all three validation metrics in this run.
- The absolute improvement on validation is real but still modest:
  - ROC AUC improves from `0.531100` to `0.539170`
  - Accuracy improves from `0.518257` to `0.530271`
  - Log loss improves from `0.691730` to `0.691566`
- `15m` also fits the training set much more strongly than `5m`, which shows up in the larger overfit gap.
- The larger overfit gap on `15m` means the current baseline is more expressive relative to the available `15m` sample size.
- `15m` has only about one third of the samples of `5m`, because the grid is three times coarser.

## Practical Conclusion

- If the decision criterion is purely current validation quality, `15m` is ahead of `5m` in this experiment.
- If the decision criterion includes robustness, `5m` is currently the more conservative baseline because its train-valid gap is materially smaller.
- The current evidence supports keeping `5m` as the main baseline while treating `15m` as a promising parallel horizon that needs the same downstream evaluation stack:
  - threshold tuning
  - signal density comparison
  - cost-sensitive backtest
  - live/offline parity verification

## Artifact Locations

- `artifacts/models/horizon_compare/experiments_5m/summary.json`
- `artifacts/models/horizon_compare/experiments_5m/summary.md`
- `artifacts/models/horizon_compare/experiments_5m/baseline/lightgbm_stage1__lightgbm_stage2/experiment_report.json`
- `artifacts/models/horizon_compare/experiments_15m/summary.json`
- `artifacts/models/horizon_compare/experiments_15m/summary.md`
- `artifacts/models/horizon_compare/experiments_15m/baseline/lightgbm_stage1__lightgbm_stage2/experiment_report.json`
