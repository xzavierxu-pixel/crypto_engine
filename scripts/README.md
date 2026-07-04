# Scripts

Run scripts from the repository root. Prefer `rtk python ...` for long commands.
This directory contains offline data/model utilities and protected runtime entry points. The live execution adapter and deploy configs live under `execution_engine/`.

## Current Workflow

| Step | Command family | Purpose |
| ---:|---|--- |
| 1 | `data/step1_acquire/` | Download or backfill Binance public market data. |
| 2 | `data/step2_normalize/` | Normalize raw Binance files into stable Parquet tables. |
| 3 | `data/step3_quality/` | QA normalized public-history tables. |
| 4 | `data/step4_features/` | Build Polymarket labels, second-level feature stores, L2 products, and final training frames. |
| 5 | `model/train_model.py` or accepted analysis runners | Train and validate the binary selective direction model. |
| 6 | `model/train_online_full_train.py` | Retrain an accepted split artifact on all split rows for deploy, when the experiment flow requires it. |
| 7 | `runtime/run_live_signal.py`, `runtime/run_shadow.py` | Run protected signal/shadow flows. Live order submission is configured in `execution_engine/config.example.yaml`. |

Current classifier deploy artifact: `execution_engine/deploy/baseline`, experiment `20260611_catboost_calendar_coordinate_search`.

Current price estimator in the execution template: `expected_return_h14` under `execution_engine/deploy/price_estimator_expected_return_h14`.

## Data Scripts

Data outputs are rooted under `artifacts/data_v2`:

| Output area | Purpose |
| ---|--- |
| `raw/` | Extracted Binance public archives. |
| `normalized/` | Stable source Parquet tables. |
| `labels/` | Polymarket resolved label stores. |
| `second_level/` | Materialized 1-second / microstructure feature stores. |
| `datasets/` | Final model training frames. |
| `manifests/` | Download, schema, and QA manifests. |

Main data scripts:

| Script | Purpose |
| ---|--- |
| `data/step0_pmdata/fetch_fill_data.py` | Fetch Polymarket fill/trade data used by execution and PnL analysis. |
| `data/step1_acquire/backfill_binance_public_history.py` | Download Binance Vision public archives into `artifacts/data_v2/raw`. |
| `data/step2_normalize/normalize_binance_public_history.py` | Normalize Binance public raw files and write schema/QA manifests. |
| `data/step2_normalize/normalize_aggtrades_daily.py` | Normalize Spot aggTrades CSVs into daily partitions. |
| `data/step3_quality/qa_binance_public_history.py` | Re-run QA over normalized Binance public tables. |
| `data/step4_features/build_polymarket_resolved_label_store.py` | Build the canonical BTC 5m resolved Polymarket label store. |
| `data/step4_features/build_second_level_feature_store.py` | Build the materialized second-level feature store from 1s data and optional aggTrades/book inputs. |
| `data/step4_features/build_decision_second_level_store_from_binance_archives.py` | Build compact decision-row second-level features directly from Binance daily archives. |
| `data/step4_features/build_dataset.py` | Build the final 5-minute training frame from OHLCV, labels, configured features, derivatives, and second-level stores. |
| `data/build_l2_eligible_markets.py`, `data/freeze_l2_splits.py`, `data/build_required_feature_packs.py` | Freeze leakage-safe Polymarket L2 universes, splits, and feature-pack requirements. |
| `data/step4_features/build_polymarket_l2_features.py` | Build cutoff-safe Polymarket L2 data products. |
| `data/audit_polymarket_l2_products.py`, `data/audit_polymarket_l2_side_mapping.py` | Audit L2 product reproducibility and UP/DOWN side semantics. |

Minimal rebuild shape:

```bash
rtk python scripts/data/step1_acquire/backfill_binance_public_history.py --settings config/settings.yaml --output-root artifacts/data_v2
rtk python scripts/data/step2_normalize/normalize_binance_public_history.py --settings config/settings.yaml --output-root artifacts/data_v2
rtk python scripts/data/step4_features/build_polymarket_resolved_label_store.py --fetch-mode time-slices --start <utc-start> --end <utc-end> --report-json artifacts/data_v2/manifests/polymarket_resolved_label_store_report.json
rtk python scripts/data/step4_features/build_second_level_feature_store.py --config config/settings.yaml --kline-1s-input artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1s.parquet --agg-trades-input artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades --data-root artifacts/data_v2
rtk python scripts/data/step4_features/build_dataset.py --input artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet --config config/settings.yaml --horizon 5m --data-root artifacts/data_v2
```

Use `--help` on each script for exact required inputs. Some feature-store and L2 commands need source-specific paths that are intentionally not hard-coded here.

## Model Scripts

| Script | Purpose |
| ---|--- |
| `model/train_model.py` | Train the configured binary selective model and write split artifacts. Also runs train/validation DQC. |
| `model/train_online_full_train.py` | Retrain the accepted offline model on all split rows for deploy. |
| `model/run_binary_rolling_validation.py` | Run chronological rolling validation. |
| `model/run_l2_direction_experiment.py` | Run L2 direction-model experiments. |

The current accepted direction baseline is reproduced through `analysis/catboost_calendar_coordinate_search.py` with `experiments/configs/20260611_catboost_calendar_coordinate_search.yaml`.

## Analysis Scripts

`analysis/` contains one-off diagnostics, replay summaries, Polymarket price-edge analysis, and accepted-experiment runners. These are useful for reproducing investigations, but recurring production workflows should live under `data/`, `model/`, `runtime/`, or `execution_engine/`.

## Runtime Scripts

| Script | Purpose |
| ---|--- |
| `runtime/run_live_signal.py` | Run a protected live or paper signal flow. |
| `runtime/run_shadow.py` | Run shadow inference and decision audit without order submission. |

Runtime order behavior is controlled by `execution_engine/config.example.yaml` and local server config, not by hard-coded thresholds in these scripts.
