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

## Practical Takeaways

- Current best retained validation result is `catboost`, not `lightgbm`.
- `logistic` should stay in every comparison run as the low-variance baseline.
- `lightgbm` is still useful as a benchmark, but its current retained setup is not the best candidate for deployment.
- Future experiments should append to this file instead of creating more loosely named artifact folders.
