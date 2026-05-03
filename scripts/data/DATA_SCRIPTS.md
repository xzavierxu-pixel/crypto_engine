# Data Scripts

This document explains the scripts under `scripts/data/`, their inputs, their outputs, and the purpose of each generated file. Paths are shown relative to the repository root unless explicitly marked as absolute.

## Workflow Order

Run the scripts in this order when rebuilding data from raw public sources:

1. `step1_acquire/`: download or backfill raw source data.
2. `step2_normalize/`: normalize raw files into stable Parquet tables.
3. `step3_quality/`: validate normalized Binance public tables.
4. `step4_features/`: build second-level feature stores and final model training frames.
5. `src/quality_check/data_quality_report.py`: called by `scripts/model/train_model.py` after feature frames are built or loaded as cached splits.

Current data outputs are rooted under `artifacts/data_v2`.

- `artifacts/data_v2/binance_public`: default root for Binance public archive download, normalization, and QA scripts.
- `artifacts/data_v2/normalized`: normalized derivatives and other normalized source tables.
- `artifacts/data_v2/second_level`: second-level feature stores.
- `artifacts/data_v2/datasets`: model training frames.
- `artifacts/data_v2/experiments`: training/evaluation outputs.

## Step 1: Acquire

### `step1_acquire/backfill_binance_public_history.py`

Purpose: Download Binance Vision public history archives, extract raw CSV files, and write download manifests.

Typical command:

```powershell
rtk python scripts/data/step1_acquire/backfill_binance_public_history.py `
  --settings config/settings.yaml `
  --output-root artifacts/data_v2/binance_public `
  --as-of-date 2026-05-03
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--settings` | no | `config/settings.yaml` | Reads `data_backfill` for enabled market families, symbols, data types, intervals, start date, and checksum behavior. |
| `--output-root` | no | `settings.second_level.data_root/binance_public` | Root where raw extracted files and manifests are written. |
| `--as-of-date` | no | current UTC date | Determines which months are complete monthly downloads and which open-month days should be downloaded daily. |

Main config inputs:

- `data_backfill.start_date`
- `data_backfill.use_monthly_for_full_months`
- `data_backfill.use_daily_for_open_month_tail`
- `data_backfill.verify_checksum`
- `data_backfill.spot`, `futures_um`, `futures_cm`, `option`

Outputs:

| Output path pattern | File type | Purpose |
|---|---|---|
| `<output-root>/raw/<market_family>/<data_type>/<symbol>/<interval>/<granularity>/<period_label>/*.csv` | CSV | Extracted raw Binance files. For interval data such as klines, interval is included in the path. |
| `<output-root>/raw/<market_family>/<data_type>/<symbol>/<granularity>/<period_label>/*.csv` | CSV | Extracted raw Binance files for data types without interval, such as funding or option sources. |
| `<output-root>/manifests/download_manifest.json` | JSON | Full request manifest: generated timestamp, provider, as-of date, summary counts, successful downloads, unavailable files, failures, and skipped existing files. |
| `<output-root>/manifests/file_checksums.json` | JSON | Checksum manifest for downloaded/extracted files. Used by normalization to carry source checksum metadata forward. |

Important output examples:

```text
artifacts/data_v2/binance_public/raw/spot/klines/BTCUSDT/1m/monthly/2026-02/BTCUSDT-1m-2026-02.csv
artifacts/data_v2/binance_public/raw/spot/klines/BTCUSDT/1s/daily/2026-04-30/BTCUSDT-1s-2026-04-30.csv
artifacts/data_v2/binance_public/raw/futures_um/fundingRate/BTCUSDT/daily/2026-04-30/BTCUSDT-fundingRate-2026-04-30.csv
artifacts/data_v2/binance_public/manifests/download_manifest.json
artifacts/data_v2/binance_public/manifests/file_checksums.json
```

Primary role of each output:

- Raw CSV files are the immutable local copy of Binance public data.
- `download_manifest.json` is the audit trail for what was requested and what succeeded.
- `file_checksums.json` supports source integrity checks and reproducibility.

### `step1_acquire/backfill_derivatives_history.py`

Purpose: Backfill long-range derivatives data in chunks, then merge chunks into final source-specific Parquet files.

Typical command:

```powershell
rtk python scripts/data/step1_acquire/backfill_derivatives_history.py `
  --start-date 2026-02-01 `
  --end-date 2026-05-01 `
  --output-root artifacts/data_v2/normalized/binance/futures_um/BTCUSDT/derivatives `
  --chunk-days 30 `
  --include-options `
  --skip-existing-chunks
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--start-date` | no | `2024-01-01` | Inclusive UTC start date. |
| `--end-date` | yes | none | Inclusive UTC end date. |
| `--output-root` | no | `artifacts/data_v2/normalized/binance/futures_um/BTCUSDT/derivatives` | Directory for chunk files, final merged files, and manifest. |
| `--chunk-days` | no | `30` | Backfill window size. |
| `--basis-period` | no | `5m` | Binance basis period. |
| `--oi-period` | no | `5m` | Binance open interest statistics period. |
| `--include-options` | no | false | Also fetch the options proxy source. |
| `--options-resolution-seconds` | no | script default | Resolution for options volatility proxy. |
| `--skip-existing-chunks` | no | false | Reuse existing chunk Parquet files when available. |

Outputs:

| Output path pattern | File type | Purpose |
|---|---|---|
| `<output-root>/chunks/funding/funding_<start>_<end>.parquet` | Parquet | Per-window funding-rate chunk. |
| `<output-root>/chunks/basis/basis_<start>_<end>.parquet` | Parquet | Per-window basis/mark/index/premium chunk. |
| `<output-root>/chunks/oi/oi_<start>_<end>.parquet` | Parquet | Per-window open-interest chunk. |
| `<output-root>/chunks/options/options_<start>_<end>.parquet` | Parquet | Per-window options proxy chunk, only when `--include-options` is used. |
| `<output-root>/binance_btcusdt_funding_<start>_<end>.parquet` | Parquet | Final merged funding table for the full requested date range. |
| `<output-root>/binance_btcusdt_basis_<start>_<end>.parquet` | Parquet | Final merged basis table for the full requested date range. |
| `<output-root>/binance_btcusdt_oi_<start>_<end>.parquet` | Parquet | Final merged open-interest table for the full requested date range. |
| `<output-root>/deribit_btc_volatility_index_<start>_<end>.parquet` | Parquet | Final merged options volatility proxy table, only when `--include-options` is used. |
| `<output-root>/manifests/binance_btcusdt_derivatives_manifest.json` | JSON | Backfill manifest: requested range, chunk paths, final paths, source row counts, and generation metadata. |

Primary role of each output:

- `chunks/*/*.parquet` lets long backfills resume and avoid re-downloading completed windows.
- Final merged Parquet files are intended as canonical derivative inputs for feature generation.
- The manifest records provenance and date coverage.

## Step 2: Normalize

### `step2_normalize/normalize_binance_public_history.py`

Purpose: Normalize raw Binance public CSV files under `raw/` into stable Parquet tables and immediately run Binance public QA.

Typical command:

```powershell
rtk python scripts/data/step2_normalize/normalize_binance_public_history.py `
  --settings config/settings.yaml `
  --output-root artifacts/data_v2/binance_public
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--settings` | no | `config/settings.yaml` | Used to resolve default artifact root. |
| `--output-root` | no | `settings.second_level.data_root/binance_public` | Root containing `raw/`; also receives `normalized/` and `manifests/`. |

Expected input layout:

```text
<output-root>/raw/<market_family>/<data_type>/<symbol>/...
<output-root>/manifests/download_manifest.json
<output-root>/manifests/file_checksums.json
```

Outputs:

| Output path pattern | File type | Purpose |
|---|---|---|
| `<output-root>/normalized/<market_family>/<data_type>/<symbol>-<interval>.parquet` | Parquet | Normalized interval table, for example 1m or 1s klines. |
| `<output-root>/normalized/<market_family>/<data_type>/<symbol>.parquet` | Parquet | Normalized non-interval table, for example funding rate or option source. |
| `<output-root>/manifests/schema_manifest.json` | JSON | Normalization manifest: output tables, schemas, unsupported files, source metadata. |
| `<output-root>/manifests/qa_manifest.json` | JSON | QA results generated immediately after normalization. |

Output examples:

```text
artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1m.parquet
artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1s.parquet
artifacts/data_v2/binance_public/normalized/futures_um/markPriceKlines/BTCUSDT-1m.parquet
artifacts/data_v2/binance_public/normalized/futures_um/fundingRate/BTCUSDT.parquet
artifacts/data_v2/binance_public/manifests/schema_manifest.json
artifacts/data_v2/binance_public/manifests/qa_manifest.json
```

Primary role of each output:

- Normalized Parquet tables are stable local tables for later feature-store construction and analysis.
- `schema_manifest.json` documents what normalized tables exist and what schema was written.
- `qa_manifest.json` documents table-level and cross-table data quality status.

### `step2_normalize/normalize_aggtrades_daily.py`

Purpose: Normalize one or more raw Binance Spot aggTrades CSV files into daily Parquet partitions.

Typical command:

```powershell
rtk python scripts/data/step2_normalize/normalize_aggtrades_daily.py `
  --input artifacts/data_v2/binance_public/raw/spot/aggTrades/BTCUSDT/daily/2026-04-30/BTCUSDT-aggTrades-2026-04-30.csv `
  --data-root artifacts/data_v2 `
  --start 2026-04-01 `
  --end 2026-05-01
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--input` | yes | none | Raw aggTrades CSV path. Can be repeated. |
| `--data-root` | no | `artifacts/data_v2` | Root used when `--output-dir` is omitted. |
| `--output-dir` | no | `<data-root>/normalized/binance/spot/BTCUSDT/aggTrades` | Partition root for normalized aggTrades. |
| `--start` | no | none | Inclusive UTC timestamp/date filter. |
| `--end` | no | none | Exclusive UTC timestamp/date filter. |
| `--chunksize` | no | script default | CSV streaming chunk size. |

Outputs:

| Output path pattern | File type | Purpose |
|---|---|---|
| `<output-dir>/date=YYYY-MM-DD/agg_trades.parquet` | Parquet | Daily normalized aggTrades partition. |
| `<output-dir>/manifest.json` | JSON | Source files, output directory, partition count, row count, and time coverage. |

Default output examples:

```text
artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades/date=2026-04-30/agg_trades.parquet
artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades/manifest.json
```

Primary role of each output:

- Daily `agg_trades.parquet` partitions are optional microstructure enrichment inputs for second-level features.
- `manifest.json` summarizes ingest coverage and row counts.

## Step 3: Quality

### `step3_quality/qa_binance_public_history.py`

Purpose: Run QA over normalized Binance public history outputs.

Typical command:

```powershell
rtk python scripts/data/step3_quality/qa_binance_public_history.py `
  --settings config/settings.yaml `
  --output-root artifacts/data_v2/binance_public
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--settings` | no | `config/settings.yaml` | Used to resolve default artifact root. |
| `--output-root` | no | `settings.second_level.data_root/binance_public` | Root containing `normalized/`. |

Expected input layout:

```text
<output-root>/normalized/**/*.parquet
```

Outputs:

| Output path | File type | Purpose |
|---|---|---|
| `<output-root>/manifests/qa_manifest.json` | JSON | Table-level checks, cross-table checks, row counts, timestamp checks, schema checks, and failure counts. |

Primary role:

- Confirms normalized public-history tables are usable before feature generation.
- Can be run independently after normalization, even though `normalize_binance_public_history.py` already calls it.

## Step 4: Features and Training Frames

### `step4_features/build_second_level_feature_store.py`

Purpose: Build the materialized 1-second feature store used by the 5-minute training frame.

Typical single-file command:

```powershell
rtk python scripts/data/step4_features/build_second_level_feature_store.py `
  --config config/settings.yaml `
  --kline-1s-input artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1s.parquet `
  --agg-trades-input artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades `
  --data-root artifacts/data_v2
```

Typical partitioned command:

```powershell
rtk python scripts/data/step4_features/build_second_level_feature_store.py `
  --config config/settings.yaml `
  --kline-1s-input artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1s.parquet `
  --agg-trades-input artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades `
  --data-root artifacts/data_v2 `
  --partition-frequency daily `
  --resume
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--config` | no | `config/settings.yaml` | Reads `second_level` profile, version, market, and feature settings. |
| `--kline-1s-input` | yes | none | Canonical Binance Spot 1s kline input. |
| `--agg-trades-input` | no | none | Optional Spot aggTrades enrichment input. May be a file or partitioned directory. |
| `--book-ticker-input` | no | none | Optional Spot bookTicker liquidity input. |
| `--depth-input` | no | none | Optional multi-level depth snapshot input. |
| `--perp-kline-1s-input` | no | none | Optional perp 1s kline input for cross-market features. |
| `--perp-book-ticker-input` | no | none | Optional perp bookTicker input for cross-market quote-state features. |
| `--eth-kline-1s-input` | no | none | Optional ETH 1s kline input for beta/residual features. |
| `--data-root` | no | `settings.second_level.data_root` | Root used for default output. |
| `--output` | no | `<data-root>/second_level/version=<version>/market=<market>/second_features.parquet` | Output path or partition base path. |
| `--write-source-tables` | no | false | Write source-normalized 1s tables next to the feature store when using non-partitioned inputs. |
| `--partition-frequency` | no | `none` | `none`, `daily`, or `monthly`. |
| `--warmup-seconds` | no | script default | Lookback prepended to each partition before trimming. |
| `--resume` | no | false | Reuse readable existing partition outputs. |

Outputs when `--partition-frequency none`:

| Output path | File type | Purpose |
|---|---|---|
| `<output>` | Parquet | Wide second-level feature store. Includes timestamp metadata, source-state flags, and `sl_*` features. |
| `<output-dir>/manifest.json` | JSON | Feature-store metadata: version, row count, time range, feature count, schema, source paths, and profile. |
| `<output-dir>/qa_report.json` | JSON | Missingness and row-level QA summary for the feature store. |
| `<output-dir>/source_tables/*.parquet` | Parquet | Optional normalized source-state tables when `--write-source-tables` is used. |

Default single-file output examples:

```text
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/second_features.parquet
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/manifest.json
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/qa_report.json
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/source_tables/kline.parquet
```

Outputs when partitioned:

| Output path pattern | File type | Purpose |
|---|---|---|
| `<output-without-.parquet>/date=YYYY-MM-DD/second_features.parquet` | Parquet | Daily feature-store partition. |
| `<output-without-.parquet>/date=YYYY-MM/second_features.parquet` | Parquet | Monthly feature-store partition. |
| `<output-without-.parquet>/manifest.json` | JSON | Partition manifest with partition paths, date coverage, row counts, schema, and source metadata. |
| `<output-without-.parquet>/qa_report.json` | JSON | Partition count, row count, and duplicate-label checks. |

Partitioned output examples:

```text
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/second_features/date=2026-04-30/second_features.parquet
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/second_features/manifest.json
artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/second_features/qa_report.json
```

Primary role of each output:

- `second_features.parquet` or partitioned `second_features.parquet` files are sampled later onto the 5-minute decision grid by `build_dataset.py`.
- `manifest.json` records schema, feature version, source paths, and partition coverage.
- `qa_report.json` identifies missingness and partition quality issues.
- `source_tables/*.parquet` is for debugging source alignment, not required for training.

### `step4_features/build_dataset.py`

Purpose: Build the final 5-minute training frame by combining OHLCV, labels, feature packs, derivatives features, optional second-level features, and sample weights.

Typical command:

```powershell
rtk python scripts/data/step4_features/build_dataset.py `
  --input artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1m.parquet `
  --output artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/training_frame.parquet `
  --config config/settings.yaml `
  --horizon 5m
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--input` | yes | none | OHLCV CSV, Feather, or Parquet input. Usually normalized 1m BTCUSDT klines. |
| `--output` | no | `<data-root>/datasets/market=<market>/horizon=<horizon>/training_frame.parquet` | Output training frame path. |
| `--data-root` | no | `settings.second_level.data_root` | Root used when `--output` is omitted. |
| `--config` | no | `config/settings.yaml` | Feature profiles, labels, dataset rules, derivatives paths, second-level settings, sample weighting. |
| `--horizon` | no | `5m` | Horizon spec to build. |
| `--funding-input` | no | config path resolution | Optional funding override. |
| `--basis-input` | no | config path resolution | Optional basis override. |
| `--oi-input` | no | config path resolution | Optional open-interest override. |
| `--options-input` | no | config path resolution | Optional options override. |
| `--second-level-feature-store` | no | `settings.second_level.feature_store_path` | Materialized second-level feature store file or partitioned directory. |
| `--derivatives-path-mode` | no | `settings.derivatives.path_mode` | `latest` or `archive`; controls derivative input path resolution. |

Outputs:

| Output path | File type | Purpose |
|---|---|---|
| `<output>` | Parquet | Final training frame. Contains timestamps, OHLCV/base fields, labels, returns, feature columns, derivatives features, second-level sampled features, and sample weights. |
| `<output-dir>/training_frame_qa.json` | JSON | Feature schema QA: row count, column count, feature count, target presence, null feature checks, leakage feature checks, and pass/fail status. |
| `<output-dir>/training_frame_summary.json` | JSON | Build summary: output path, config path, horizon, row count, feature count, target rate, sample-weight status, second-level settings, and data availability. |

Default output examples:

```text
artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/training_frame.parquet
artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/training_frame_qa.json
artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/training_frame_summary.json
```

Primary role of each output:

- `training_frame.parquet` is the canonical offline model input used by `scripts/model/train_model.py`.
- `training_frame_qa.json` verifies feature legality and catches target/leakage/raw metadata columns before training.
- `training_frame_summary.json` records build provenance and high-level dataset statistics.

Important columns in `training_frame.parquet`:

- `timestamp`: decision-grid timestamp.
- `target`: binary settlement direction label.
- `signed_return`, `abs_return`: label-window return diagnostics and sample-weight inputs.
- `stage1_sample_weight`: sample weight used by training when enabled.
- Feature columns inferred by `src.data.dataset_builder.infer_feature_columns`.
- Optional `sl_*` columns from second-level feature store.
- Optional derivative columns such as funding, basis, OI, and options features.

## Post-Feature DQC

### `src/quality_check/data_quality_report.py`

Purpose: Generate train/validation data quality reports after feature frames are built. This script is not under `scripts/data/`, but it is part of the data workflow and is called by `scripts/model/train_model.py`.

Typical direct command:

```powershell
rtk python src/quality_check/data_quality_report.py `
  --train artifacts/data_v2/experiments/<run>/development_frame.parquet `
  --valid artifacts/data_v2/experiments/<run>/validation_frame.parquet `
  --output-dir artifacts/data_v2/experiments/<run>/data_quality
```

Inputs:

| Input | Required | Default | Meaning |
|---|---:|---|---|
| `--train` | yes | none | Development/train split Parquet. |
| `--valid` | no | none | Validation split Parquet. |
| `--output-dir` | yes | none | Directory for DQC output. |
| `--full-frame` | no | false | If omitted, inspect model feature columns only. If set, inspect all columns. |

Outputs:

| Output path | File type | Purpose |
|---|---|---|
| `<output-dir>/dqc_summary.txt` | Text | Human-readable train/valid data quality summary: dimensions, duplicate rows/columns, missing/inf columns, constant features, near-constant features, high-cardinality columns, and train/valid schema/missingness comparison. |

Current training behavior:

- When `scripts/model/train_model.py` builds a fresh training frame, it writes `development_frame.parquet` and `validation_frame.parquet`, then calls this DQC script.
- When `scripts/model/train_model.py --cached-split-dir <dir>` is used, it now also calls this DQC script using the cached split as input and writes the report into the current output directory.

Output example:

```text
artifacts/data_v2/experiments/20260503_settings_train_20260201_20260501_wide_thresholds/data_quality/dqc_summary.txt
```

Primary role:

- Gives a readable feature-quality report after train/validation split creation.
- Catches duplicate columns, missingness drift, constant features, and schema mismatch before interpreting model results.

## Recommended End-to-End Path

For the current project, a typical full rebuild looks like this:

```powershell
# 1. Download Binance public raw archives.
rtk python scripts/data/step1_acquire/backfill_binance_public_history.py `
  --settings config/settings.yaml `
  --output-root artifacts/data_v2/binance_public

# 2. Normalize Binance public raw archives and run QA.
rtk python scripts/data/step2_normalize/normalize_binance_public_history.py `
  --settings config/settings.yaml `
  --output-root artifacts/data_v2/binance_public

# 3. Optionally normalize aggTrades into daily partitions.
rtk python scripts/data/step2_normalize/normalize_aggtrades_daily.py `
  --input <raw-aggTrades-csv> `
  --data-root artifacts/data_v2

# 4. Build second-level feature store.
rtk python scripts/data/step4_features/build_second_level_feature_store.py `
  --config config/settings.yaml `
  --kline-1s-input artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1s.parquet `
  --agg-trades-input artifacts/data_v2/normalized/binance/spot/BTCUSDT/aggTrades `
  --data-root artifacts/data_v2

# 5. Build final training frame.
rtk python scripts/data/step4_features/build_dataset.py `
  --input artifacts/data_v2/binance_public/normalized/spot/klines/BTCUSDT-1m.parquet `
  --config config/settings.yaml `
  --horizon 5m `
  --data-root artifacts/data_v2

# 6. Train model; this creates cached splits and DQC under the output dir.
rtk python scripts/model/train_model.py `
  --input artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/training_frame.parquet `
  --output-dir artifacts/data_v2/experiments/<run-name> `
  --config config/settings.yaml `
  --horizon 5m
```
