# Model Experiment Log

## Scope

This file records the model experiments that are still available in the repo after cleanup.

Data / task baseline:

- Market: `BTC/USDT`
- Exchange: `binance`
- Base timeframe: `1m`
- Horizon: `5m`
- Label: `1{close[t0+5m] > open[t0]}`
- Train start: `2024-01-01`
- Validation scheme: last `30` days
- Purge rows: `1`
- Feature count: `126`
- Main metrics: `roc_auc`, `log_loss`, `accuracy`

Notes:

- Historical artifact directories from earlier manual experiments were deleted during repo cleanup, so only currently retained experiment outputs are listed here.
- Current retained manual baseline artifact: [artifacts/models/manual/btc_usdt_5m_recent30d](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\manual\btc_usdt_5m_recent30d)
- Current retained model-family comparison artifact: [artifacts/models/experiments/model_family_baseline](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline)

## Current Ranking

| Model | Train ROC AUC | Valid ROC AUC | Train logloss | Valid logloss | Train acc | Valid acc | Comment |
|---|---:|---:|---:|---:|---:|---:|---|
| `catboost` | 0.584590 | 0.539404 | 0.686263 | 0.691130 | 0.559362 | 0.529153 | Best current validation AUC among retained runs |
| `logistic` | 0.530948 | 0.535921 | 0.691757 | 0.691338 | 0.521301 | 0.524284 | Most stable, essentially no overfit, but slightly lower ceiling |
| `lightgbm` | 0.612795 | 0.531100 | 0.685688 | 0.691730 | 0.572329 | 0.518257 | Clear overfit relative to validation |

## Experiment Records

### 1. Manual baseline rerun with recent-30d validation

- Model: `lightgbm`
- Artifact path: [artifacts/models/manual/btc_usdt_5m_recent30d](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\manual\btc_usdt_5m_recent30d)
- Report source: [training_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\manual\btc_usdt_5m_recent30d\training_report.json)
- Calibration: `isotonic`
- Validation window: `30` days
- Train window:
  - Rows: `228110`
  - Start: `2024-01-01 02:00:00+00:00`
  - End: `2026-03-09 23:45:00+00:00`
- Validation window:
  - Rows: `8627`
  - Start: `2026-03-09 23:55:00+00:00`
  - End: `2026-04-08 23:55:00+00:00`
- Parameters:
  - `n_estimators=700`
  - `learning_rate=0.03`
  - `num_leaves=15`
  - `min_child_samples=100`
  - `subsample=1.0`
  - `colsample_bytree=0.6`
  - `reg_alpha=0.5`
  - `reg_lambda=1.0`
  - `max_depth=4`
  - `random_state=42`
  - `class_weight=balanced`
- Train metrics:
  - `roc_auc=0.612795`
  - `log_loss=0.685688`
  - `accuracy=0.572329`
- Validation metrics:
  - `roc_auc=0.531100`
  - `log_loss=0.691730`
  - `accuracy=0.518257`
- Read:
  - The model learns on train, but validation drops noticeably.
  - This is the current reference point for overfit.

### 2. Model-family baseline comparison

- Experiment path: [artifacts/models/experiments/model_family_baseline](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline)
- Summary source: [summary.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline\summary.json)
- Ranking priority:
  - `roc_auc`
  - `log_loss`
  - `accuracy`

#### 2.1 CatBoost baseline

- Model: `catboost`
- Artifact path: [catboost](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline\catboost)
- Report source: [experiment_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline\catboost\experiment_report.json)
- Calibration: `isotonic`
- Duration: `184.79s`
- Parameters:
  - `iterations=700`
  - `learning_rate=0.03`
  - `depth=5`
  - `l2_leaf_reg=5.0`
  - `random_seed=42`
  - `loss_function=Logloss`
  - `eval_metric=Logloss`
  - `verbose=false`
- Train metrics:
  - `roc_auc=0.584590`
  - `log_loss=0.686263`
  - `accuracy=0.559362`
- Validation metrics:
  - `roc_auc=0.539404`
  - `log_loss=0.691130`
  - `accuracy=0.529153`
- Overfit gap:
  - `roc_auc_gap=0.045186`
  - `log_loss_gap=0.004867`
  - `accuracy_gap=0.030209`
- Read:
  - Best current validation AUC among retained experiments.
  - Overfit is present, but much smaller than the retained LightGBM baseline.

#### 2.2 Logistic regression baseline

- Model: `logistic`
- Artifact path: [logistic](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline\logistic)
- Report source: [experiment_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline\logistic\experiment_report.json)
- Calibration: `isotonic`
- Duration: `19.63s`
- Parameters:
  - `C=0.5`
  - `max_iter=2000`
  - `solver=lbfgs`
  - `random_state=42`
- Train metrics:
  - `roc_auc=0.530948`
  - `log_loss=0.691757`
  - `accuracy=0.521301`
- Validation metrics:
  - `roc_auc=0.535921`
  - `log_loss=0.691338`
  - `accuracy=0.524284`
- Overfit gap:
  - `roc_auc_gap=-0.004973`
  - `log_loss_gap=-0.000419`
  - `accuracy_gap=-0.002983`
- Read:
  - Almost no overfit.
  - Strong low-variance baseline.
  - Lower ceiling than CatBoost so far, but useful as a sanity baseline for future experiments.

### 3. Real derivatives ablation on recent 30-day public window

- Experiment path: [artifacts/models/experiments/derivatives_ablation_2026-03-10_2026-04-08](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\derivatives_ablation_2026-03-10_2026-04-08)
- Summary source: [summary.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\derivatives_ablation_2026-03-10_2026-04-08\summary.json)
- Readable summary: [summary.md](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\derivatives_ablation_2026-03-10_2026-04-08\summary.md)
- Window:
  - Spot input span: `2026-03-10` to `2026-04-08`
  - Validation window: `7` days
  - Purge rows: `1`
- Data sources:
  - Spot: [binance_btc_usdt_1m.parquet](C:\Users\ROG\Desktop\crypto_engine\artifacts\datasets\binance_btc_usdt_1m.parquet)
  - Funding: [binance_btcusdt_funding.parquet](C:\Users\ROG\Desktop\crypto_engine\artifacts\data\derivatives\public_2026-03-10_2026-04-08\binance_btcusdt_funding.parquet)
  - Basis: [binance_btcusdt_basis.parquet](C:\Users\ROG\Desktop\crypto_engine\artifacts\data\derivatives\public_2026-03-10_2026-04-08\binance_btcusdt_basis.parquet)
  - OI: [binance_btcusdt_oi.parquet](C:\Users\ROG\Desktop\crypto_engine\artifacts\data\derivatives\public_2026-03-10_2026-04-08\binance_btcusdt_oi.parquet)
  - Options proxy: [deribit_btc_volatility_index.parquet](C:\Users\ROG\Desktop\crypto_engine\artifacts\data\derivatives\public_2026-03-10_2026-04-08\deribit_btc_volatility_index.parquet)
- Important caveat:
  - This is a short-window public-data experiment, not a full-history replacement for the long baseline above.
  - The options input is a Deribit public volatility-index proxy for `atm_iv_near`, not a full option-chain implementation.

#### 3.1 Best overall result

- Variant: `funding_basis_oi_options`
- Model: `lightgbm`
- Feature count: `153`
- Train metrics:
  - `roc_auc=0.922087`
  - `log_loss=0.646900`
  - `accuracy=0.845797`
- Validation metrics:
  - `roc_auc=0.547549`
  - `log_loss=0.689804`
  - `accuracy=0.533367`
- Overfit gap:
  - `roc_auc_gap=0.374538`
  - `log_loss_gap=0.042904`
  - `accuracy_gap=0.312431`
- Read:
  - Best validation ROC AUC in this recent-window ablation.
  - Also slightly beats the recent-window baseline on logloss and accuracy.

#### 3.2 Best by variant

| Variant | Best model | Valid ROC AUC | Valid logloss | Valid acc | Delta AUC vs baseline |
|---|---|---:|---:|---:|---:|
| `baseline` | `catboost` | 0.537451 | 0.690975 | 0.525398 | 0.000000 |
| `funding` | `catboost` | 0.514721 | 1.684620 | 0.504980 | -0.022729 |
| `funding_basis` | `catboost` | 0.530917 | 1.860132 | 0.526892 | -0.006534 |
| `funding_basis_oi` | `catboost` | 0.527972 | 1.748203 | 0.520418 | -0.009478 |
| `funding_basis_oi_options` | `lightgbm` | 0.547549 | 0.689804 | 0.533367 | 0.010098 |

#### 3.3 Read

- `funding` alone was clearly harmful on this window.
- `funding + basis` recovered some signal, but still did not beat baseline on ROC AUC.
- Adding `oi` on top of `funding + basis` did not produce a net gain.
- Adding the phase-3 options proxy produced the first clear improvement over the recent-window baseline.
- The winning setup changed model family from `catboost` to `lightgbm`, which suggests the options-related signals interact differently with model capacity than the earlier derivatives subsets.

## Practical Takeaways

- Current best retained validation result is `catboost`, not `lightgbm`.
- `logistic` should stay in every comparison run as the low-variance baseline.
- `lightgbm` is still useful as a benchmark, but its current retained setup is not the best candidate for deployment.
- On the recent public-data window, the best derivatives stack was `funding_basis_oi_options`, but this result should be treated as provisional until it repeats on another window.
- Future experiments should append to this file instead of creating more loosely named artifact folders.

## 4. 5m Label V2 benchmark with thresholded direction target

- Artifact path: [artifacts/models/benchmarks/catboost_5m_label_v2_threshold_1p001](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\benchmarks\catboost_5m_label_v2_threshold_1p001)
- Report source: [training_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\benchmarks\catboost_5m_label_v2_threshold_1p001\training_report.json)
- Label definition:

```text
y = 1{close[t0+5m] > 1.001 * open[t0]}
```

- Horizon config:
  - `5m.label_params.threshold_multiplier = 1.001`
  - `5m.label_params.label_version = v2`
- Model: `catboost`
- Feature count: `126`
- Validation window: `30` days
- Purge rows: `1`
- Train metrics:
  - `roc_auc = 0.715525`
  - `log_loss = 0.414985`
  - `accuracy = 0.829442`
- Raw validation metrics:
  - `roc_auc = 0.665833`
  - `log_loss = 0.418549`
  - `accuracy = 0.837835`
- Calibrated validation metrics:
  - `roc_auc = 0.665350`
  - `log_loss = 0.418819`
  - `accuracy = 0.837835`
- Label balance snapshot:
  - full sample positive rate: `0.170080`
  - train window positive rate: `0.170388`
  - validation window positive rate: `0.161903`
- Read:
  - 引入阈值后，方向任务从接近随机噪声的 `5m v1` 显著提升为可分任务，validation ROC AUC 提升到 `0.665` 左右。
  - 但该任务已经不再是接近平衡二分类，而是正类占比约 `16%` 到 `17%` 的偏斜标签，因此 `accuracy` 不能与 `v1` 直接横向比较。
  - 这个 benchmark 说明“提高标签信噪比”比继续堆更多 feature 更有效。

## Public-facing interpretation

The long-running baseline and the recent derivatives ablation answer different questions and should not be compared as if they were trained on the same data span.

- The long-running baseline uses the full BTC/USDT 1m history available in the repo, starting from `2024-01-01`, with a `30`-day validation window.
- The recent derivatives ablation uses a short public-data window from `2026-03-10` to `2026-04-08`, with a `7`-day validation window, because the public derivatives sources available here do not reliably support a full-history backfill to `2024-01-01`.
- The recent ablation therefore tests whether the new derivatives stack adds signal on a constrained, realistic short window. It does not replace the long baseline as the canonical full-history benchmark.
- In that short window, the best stack was `funding_basis_oi_options + lightgbm`, with validation `roc_auc=0.547549`, beating the same-window baseline `roc_auc=0.537451` by `0.010098`.
- The result is encouraging but provisional. It is evidence that the derivatives layer can help under at least one recent market regime, not proof that the same gain will hold over the full 2024-to-present history.
- The correct external summary is: long-history baseline remains the reference benchmark, while the recent derivatives ablation is a separate short-window validation showing incremental value from the added derivatives features.
