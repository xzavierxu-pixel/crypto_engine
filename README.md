# crypto_engine

Agent-oriented overview for the `crypto_engine` repo.

`crypto_engine` exists to predict the next 5-minute BTC/USDT direction from Binance 1-minute market data, then use that prediction in a separate execution layer that can place or simulate actions on Polymarket.

The repo is intentionally narrow in V1: one asset, one base timeframe, one horizon, one primary label, one shared feature pipeline, one model family baseline. The point is not to be a generic trading framework. The point is to keep training, backtesting, and live inference on the same logic so model behavior stays explainable and operationally safe.

> Full design: [docs/workflow_guide.md](docs/workflow_guide.md). Architecture context: [docs/project_architecture_overview.md](docs/project_architecture_overview.md). Working rules: [AGENTS.md](AGENTS.md). This file is the fast orientation guide for coding agents.

---

## What this system does

- Ingests BTC/USDT market history from Binance as the modeling source of truth.
- Builds a shared feature frame and a shared label frame for both offline training and online inference.
- Trains a binary classifier that answers: "will the 5-minute candle close above its open?"
- Optionally calibrates model outputs and converts them into a decision.
- Hands the final decision to a thin execution layer that submits, simulates, or audits actions for Polymarket.

## What matters most for agents

- This is a **shared-core prediction system** with an execution adapter, not an execution-first bot.
- The most important invariant is parity: offline dataset construction and online inference must use the same feature and label logic.
- Most changes should happen in `src/core/`, `src/features/`, `src/labels/`, `src/data/`, `src/model/`, or `src/services/`.
- `src/strategies/` and `src/execution/` are downstream consumers. They should stay thin and should not invent their own BTC feature logic.
- `config/settings.yaml` is the only place for business parameters such as thresholds, horizons, and feature settings.

---

## Prediction target

$$y_t = \mathbb{1}\{\text{close}_{t_0+5m} > \text{open}_{t_0}\}$$

- Asset: `BTC/USDT`, base timeframe `1m`, horizon `5m`.
- `t₀` must sit on the 5-minute grid (`minute % 5 == 0`).
- Baseline model: LightGBM (swappable: CatBoost / LogReg).

## High-level flow

```text
Binance 1m data
    -> shared preprocessing and time-grid alignment
    -> shared feature builder
    -> shared label builder
    -> model training or live inference
    -> calibration and decisioning
    -> Polymarket execution or shadow/audit output
```

If you are changing anything that affects the meaning of the training frame, assume it can affect both historical experiments and live behavior.

## Non-negotiable rules (from [AGENTS.md](AGENTS.md))

1. Online and offline logic must be identical — share the same feature builder and label builder.
2. All business parameters live **only** in [config/settings.yaml](config/settings.yaml).
3. Never duplicate feature or label logic.
4. Keep the Freqtrade strategy thin — adapter only, no business logic.
5. Single source of truth for time grid, labels, features and schemas is `src/core/`.
6. The execution layer must not recompute BTC features.
7. Prefer `rtk` for shell commands (see [RTK.md](RTK.md)).

## Architecture (condensed)

```
config/settings.yaml              ← single source of business parameters
        │
        ▼
src/core/        schemas / timegrid / versioning / validation   (zero deps)
src/features/    13 FeaturePacks + registry
src/labels/      grid_direction (only implementation of y)
src/horizons/    HorizonSpec (metadata for "5m")
src/data/        loaders → preprocess → dataset_builder.TrainingFrame
src/model/       plugin arch: lightgbm / catboost / logreg + train.py
src/calibration/ platt / isotonic / none
src/signal/      decision_engine (edge / threshold)
src/sizing/      fixed_fraction
src/services/    SignalService (features + model + calibration, shared online/offline)
src/execution/   guards / idempotency / order_router / adapters/polymarket
src/strategies/  thin Freqtrade adapter
```

Dependency direction is strictly one-way:
`core ← features/labels/horizons ← data ← model/calibration ← services ← execution/strategies`.

## Where to start by task

- Feature engineering: [src/features/base.py](src/features/base.py), [src/features/registry.py](src/features/registry.py), then the specific feature pack file.
- Label logic: [src/labels/grid_direction.py](src/labels/grid_direction.py).
- Dataset assembly: [src/data/](src/data/) and [scripts/build_dataset.py](scripts/build_dataset.py).
- Model training and artifact persistence: [src/model/](src/model/) and [scripts/train_model.py](scripts/train_model.py).
- Shared online inference path: [src/services/](src/services/) and [scripts/run_live_signal.py](scripts/run_live_signal.py).
- Execution and order routing: [src/execution/](src/execution/).
- Tests to read first: [tests/](tests/), especially the closest `test_*.py` file for the area you are touching.

## Core schemas ([src/core/schemas.py](src/core/schemas.py))

`Signal` · `Decision` · `MarketQuote` · `OrderRequest` · `GuardResult` · `AuditEvent` · `RiskState` — all frozen dataclasses, passed across module boundaries.

## Script entry points ([scripts/](scripts/))

| Script | Purpose |
|---|---|
| `download_binance_vision_data.py` | Pull historical 1m data from Binance Vision |
| `build_dataset.py` | OHLCV → features + labels + sample weights (`TrainingFrame`) |
| `train_model.py` | Train and persist model artifacts |
| `run_live_signal.py` | Online inference + Polymarket order submission |
| `run_shadow.py` | Shadow mode (no orders, audit only) |
| `run_model_experiments.py` | Batch experiments |

All scripts share: `--input`, `--output/--output-dir`, `--config config/settings.yaml`, `--horizon 5m`.

## Dev quickstart

```bash
# Run tests
rtk pytest -q

# Build dataset locally
rtk python scripts/build_dataset.py \
    --input data/raw/BTCUSDT_1m.parquet \
    --output data/training/BTCUSDT_5m.parquet \
    --config config/settings.yaml --horizon 5m

# Train
rtk python scripts/train_model.py \
    --input data/training/BTCUSDT_5m.parquet \
    --output-dir artifacts/models/local \
    --config config/settings.yaml --horizon 5m
```

## Before you change code

- Add/modify a **feature** → edit [src/features/base.py](src/features/base.py) + register in [src/features/registry.py](src/features/registry.py); bump `CORE_FEATURE_VERSION` in [src/core/versioning.py](src/core/versioning.py).
- Modify a **label** → edit [src/labels/grid_direction.py](src/labels/grid_direction.py); bump `CORE_LABEL_VERSION`.
- Add a **model** → implement the [src/model/base.py](src/model/base.py) interface and register in [src/model/registry.py](src/model/registry.py).
- Add a **calibrator / horizon / label / execution adapter** → use the matching `*/registry.py`.
- Change **business parameters** → edit `config/settings.yaml` only; never hard-code thresholds in code.
- For any change: first read the matching `tests/test_*.py`, then run `rtk pytest -q` after editing.

## Do NOT

- Do NOT recompute features inside `src/strategies/` or `src/execution/`.
- Do NOT duplicate feature/label logic in scripts — always go through `build_feature_frame` and the registered label builder.
- Do NOT bypass `src/core/timegrid.py` grid checks.
- Do NOT add speculative abstractions, comments, or type annotations that were not requested.
