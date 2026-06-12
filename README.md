# crypto_engine

Agent-oriented overview for the current `crypto_engine` repo.

`crypto_engine` predicts BTC 5-minute Polymarket UP/DOWN settlement outcomes from Binance BTCUSDT market data, then passes the calibrated decision to a separate execution layer that can paper trade, audit, or submit Polymarket orders.

The repo is intentionally narrow: one asset, one base timeframe, one horizon, one primary resolved-market label, one shared feature pipeline, and one production deploy artifact. The priority is offline/online parity and reproducible validation, not a generic trading framework.

See also: [AGENTS.md](AGENTS.md), [DATA_PIPELINE.md](DATA_PIPELINE.md), [scripts/README.md](scripts/README.md), and [docs/project_architecture_overview.md](docs/project_architecture_overview.md).

---

## Current Baseline

Current accepted validation baseline:

```yaml
experiment_id: 20260611_catboost_calendar_coordinate_search
config_path: experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
report_path: artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_calendar_coordinate_search/report.json
label_source: polymarket_resolved
model_plugin: catboost
calibration_plugin: none
threshold_policy: utc_day_session_coordinate
selection_score: 0.6413740846
coverage: 0.7000535619
accepted_sample_accuracy: 0.7073450650
utility: 0.2903053026
accepted_count: 5228
```

Current deploy artifact:

```yaml
artifact_dir: execution_engine/deploy/baseline
training_mode: calendar_coordinate_offline_train
threshold_source: offline_validation_calendar_coordinate
offline_validation_selection_score: 0.6413740846
limit_config_source: execution_engine/limit_configs.py
```

Previous accepted validation baseline for comparison:

```yaml
experiment_id: 20260520_polymarket_resolved_extended_history_baseline
selection_score: 0.5748509217
coverage: 0.7001874665
accepted_sample_accuracy: 0.6909542934
utility: 0.2674076058
accepted_count: 5229
```

Future experiments should use `20260611_catboost_calendar_coordinate_search` as the baseline unless explicitly stated otherwise.

---

## What This System Does

- Ingests BTCUSDT market history from Binance.
- Builds a shared feature frame on 5-minute grid decision rows.
- Joins resolved Polymarket BTC 5-minute UP/DOWN labels.
- Trains a binary selective model with validation threshold search.
- Retrains the accepted configuration on all split rows for deployment.
- Loads the deploy artifact in `execution_engine` for paper, shadow, or live Polymarket execution.

---

## Prediction Target

Current default target:

```text
y_t = resolved Polymarket BTC 5m UP/DOWN settlement outcome
```

Key settings:

```yaml
asset: BTC/USDT
base_timeframe: 1m
horizon: 5m
label_builder: polymarket_resolved
label_version: polymarket_resolved_gamma_v1
label_store_path: artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet
```

Historical BTC OHLCV direction, `1{close[t0 + 4m] >= open[t0]}`, is diagnostic only unless explicitly selected for an experiment.

---

## High-Level Flow

```text
Binance BTCUSDT 1m data
    -> normalized Parquet
    -> shared feature builder
    -> Polymarket resolved label join
    -> chronological split training and validation threshold search
    -> accepted thresholds and offline validation metrics
    -> online_full_train deploy artifact
    -> execution_engine/deploy/baseline
    -> Polymarket execution or shadow/audit output
```

Acceptance and deployment are intentionally separate:

- `scripts/model/train_model.py` performs split train/validation training and threshold search. Validation is the acceptance set.
- `scripts/model/train_online_full_train.py` retrains on development + validation rows and writes the deploy artifact.
- The deploy manifest copies `offline_validation_metrics` from the accepted split artifact.
- Full-train metrics are not acceptance metrics.

---

## Non-Negotiable Rules

1. Offline and online logic must share the same feature and label builders.
2. Business parameters live in `config/settings.yaml` or an experiment-specific config copied from it.
3. Do not duplicate feature or label logic in scripts, execution code, or strategy adapters.
4. Keep Freqtrade and execution adapters thin.
5. The execution layer must not recompute BTC features.
6. Thresholds must come from config or artifact, never hard-coded `0.5`.
7. Validation `coverage` must be at least `0.70`; reject lower-coverage results even if `selection_score` is higher.
8. Use `rtk` for verbose shell commands.

---

## Architecture

```text
config/settings.yaml
    -> src/core/        schemas, timegrid, constants, validation
    -> src/features/    shared FeaturePacks and registry
    -> src/labels/      polymarket_resolved primary label, grid_direction diagnostics
    -> src/data/        loaders, preprocessing, dataset_builder.TrainingFrame
    -> src/model/       plugin models and training functions
    -> src/calibration/ platt, isotonic, none
    -> src/services/    shared signal service
    -> execution_engine deploy/runtime adapter
```

Dependency direction is one-way:

```text
core -> features/labels/horizons -> data -> model/calibration -> services -> execution/strategies
```

---

## Main Entry Points

| Task | Path |
|---|---|
| Feature builder | `src/features/builder.py` |
| Feature registry | `src/features/registry.py` |
| Polymarket label builder | `src/labels/polymarket_resolved.py` |
| BTC direction diagnostic label | `src/labels/grid_direction.py` |
| Training frame assembly | `src/data/dataset_builder.py` |
| Dataset build script | `scripts/data/step4_features/build_dataset.py` |
| Split training and threshold search | `scripts/model/train_model.py` |
| Full-train deploy generation | `scripts/model/train_online_full_train.py` |
| Execution artifact loader | `execution_engine/artifacts.py` |
| Runtime config example | `execution_engine/config.example.yaml` |
| Execution docs | `execution_engine/README.md` |

---

## Quick Commands

Build the current extended training frame:

```powershell
rtk python scripts/data/step4_features/build_dataset.py `
  --input artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet `
  --output artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/polymarket_resolved_extended_training_frame.parquet `
  --config config/settings.yaml `
  --horizon 5m
```

Run accepted split training:

```powershell
rtk python scripts/analysis/catboost_calendar_coordinate_search.py `
  --config experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
```

Regenerate the deploy artifact:

```powershell
rtk python scripts/analysis/catboost_calendar_coordinate_search.py `
  --config experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
```

Run focused verification:

```powershell
rtk python -m pytest -q `
  tests/test_model_pipeline.py::test_online_full_train_script_uses_accepted_thresholds_and_writes_deploy_artifacts `
  tests/test_model_artifacts.py::test_load_binary_selective_artifacts_from_manifest_and_directory `
  tests/test_execution_engine.py::test_execution_config_example_loads
```

---

## Before Changing Code

- Feature changes: edit the relevant `src/features/` pack, update `src/features/registry.py`, and bump `CORE_FEATURE_VERSION` when semantics change.
- Label changes: edit the relevant `src/labels/` builder and bump `CORE_LABEL_VERSION` when semantics change.
- Model changes: implement `src/model/base.py` and register in `src/model/registry.py`.
- Business parameter changes: copy `config/settings.yaml` to `experiments/configs/<timestamp>_<description>.yaml` for experiments.
- Deploy changes: regenerate `execution_engine/deploy/baseline` only after a split artifact is accepted.

---

## Do Not

- Do not recompute BTC features inside `execution_engine` or `src/strategies`.
- Do not silently switch labels between Polymarket resolved outcomes and BTC OHLCV direction.
- Do not use full-train metrics as acceptance metrics.
- Do not bypass `src/core/timegrid.py`.
- Do not hard-code thresholds in execution code.
