# Scripts

This directory is organized by workflow stage. Mainline scripts live in `data/`, `model/`, and `runtime/`. Research and one-off experiment runners live in `experiments/`.

## Data Workflow

Run these groups in order when rebuilding data from raw sources.

| Order | Directory | Scripts | Purpose |
|---|---|---|---|
| 1 | `data/step1_acquire/` | `backfill_binance_public_history.py`, `backfill_derivatives_history.py`, `download_derivatives_public_data.py` | Download or backfill raw Binance and derivatives inputs. |
| 2 | `data/step2_normalize/` | `normalize_binance_public_history.py`, `normalize_aggtrades_daily.py` | Convert raw archives into normalized Parquet datasets. |
| 3 | `data/step3_quality/` | `qa_binance_public_history.py` | Validate normalized data before feature generation. |
| 4 | `data/step4_features/` | `build_second_level_feature_store.py`, `build_dataset.py` | Build reusable feature stores and final training frames. |
| 5 | `src/quality_check/data_quality_report.py` | called by `model/train_model.py` | Generate train/validation DQC after feature frames are built or loaded from a cached split. |

## Model Workflow

| Script | Purpose |
|---|---|
| `model/train_model.py` | Train the configured binary selective model and write artifacts. |
| `model/run_binary_rolling_validation.py` | Run chronological rolling validation for the binary model. |

## Runtime Workflow

| Script | Purpose |
|---|---|
| `runtime/run_live_signal.py` | Run protected live or paper signal flow. |
| `runtime/run_shadow.py` | Run shadow inference and decision audit flow without order submission. |

## Experiments

`experiments/` contains research and one-off runners. These scripts are useful for reproducing past investigations, but they are not the default production workflow:

- `run_auc_model_family_experiments.py`
- `run_auc_optimization_experiments.py`
- `run_balanced_precision_holdout_experiment.py`
- `run_bp_feature_sample_model_experiments.py`
- `run_bp_targeted_experiments.py`
- `run_model_experiments.py`
- `run_validation_optimization_experiments.py`

Prefer adding new recurring workflows under `data/`, `model/`, or `runtime/`. Put short-lived analysis runners under `experiments/`.
