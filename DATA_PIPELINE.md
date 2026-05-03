# Crypto Engine — Data Pipeline Reference
End-to-end guide covering raw data ingestion, normalization, feature engineering, label generation, and training-frame assembly.
> Architecture overview: [docs/project_architecture_overview.md](docs/project_architecture_overview.md). Agent rules: [AGENTS.md](AGENTS.md).
---
## 1. Design Principles
- **Immutability** — Raw downloads are never modified; all cleaning outputs go to `normalized/`.- **Schema stability** — `_stabilize_dtypes()` enforces canonical types (`float64`, `Int64`, `category`) to prevent Pandas auto-inference drift.- **Plug-and-play features** — Each feature pack is an independent class implementing `FeaturePack.transform()`.- **Traceability** — Every row carries `source_file`, `ingested_at`, and `checksum_status` metadata.- **Online/offline parity** — `build_feature_frame()` and `GridDirectionLabelBuilder.build()` are shared by both training scripts and the live `SignalService`.
---
## 2. High-Level Architecture
```text[ Binance Vision ] | | scripts/backfill_binance_public_history.py (spot, futures, option public archives) v[ artifacts/data_v2/raw/ ] — raw ZIP / CSV files | | scripts/normalize_binance_public_history.py | scripts/normalize_aggtrades_daily.py | src/data/binance_public/normalizer.py v[ artifacts/data_v2/normalized/ ] — stabilized Parquet (dtype-enforced + metadata) | | src/data/derivatives/feature_store.py → attach_to_spot() | src/data/second_level_features.py → 1-second microstructure features | src/features/builder.py → build_feature_frame() | └─ 28 registered FeaturePacks (src/features/registry.py) v[ Feature Matrix ] — per-row: timestamp + all computed features + grid columns | | src/labels/grid_direction.py → GridDirectionLabelBuilder.build() v[ Label Vector ] — target column on 5-min grid points | | src/data/dataset_builder.py → build_training_frame() | ├─ merge features + labels + abs_return | ├─ filter_by_timerange() | ├─ drop_incomplete_samples() | └─ compute_sample_weight() v[ TrainingFrame ] — final Parquet ready for model training```
---
## 3. Phase 1 — Raw Data Ingestion
### 3.1 Spot Kline History
**Script:** `scripts/backfill_binance_public_history.py`
Downloads BTC/USDT 1-minute klines from [Binance Vision](https://data.binance.vision/).
- Automatically manages monthly vs. daily download windows.- Supports multiple market families: `spot`, `um` (USDT-M Futures), `cm` (COIN-M Futures).- Validates checksums when `.CHECKSUM` files are available.
### 3.2 Derivatives History
**Scripts:**
| Script | Use Case ||---|---|| `backfill_binance_public_history.py` | Long-range public archive backfill for spot, futures, and option inputs || `download_derivatives_public_data.py` | Latest snapshots for experiments |
**Data sources collected:**
| Indicator | Source | Typical Frequency ||---|---|---|| Funding Rate | Binance FAPI | Every 8 hours || Open Interest (OI) | Binance FAPI | 5 min || Basis (mark / index / premium) | Binance FAPI | 1 min || Book Ticker (best bid/ask) | Binance FAPI | Tick-level || Options (DVOL, ATM IV) | Deribit API | Variable |
### 3.3 Second-Level Tick Data
**Script:** `scripts/normalize_aggtrades_daily.py`
Normalizes raw aggTrades CSVs into daily Parquet files, feeding into the 1-second microstructure feature store (`scripts/build_second_level_feature_store.py`).
---
## 4. Phase 2 — Normalization
**Script:** `scripts/normalize_binance_public_history.py`**Core module:** `src/data/binance_public/normalizer.py`
### 4.1 Schema Enforcement (`_stabilize_dtypes`)
| Column Group | Examples | Target dtype ||---|---|---|| `FLOAT_LIKE_COLUMNS` | `open`, `high`, `low`, `close`, `volume`, `price`, `quantity`, `bid_price`, `ask_price` | `float64` || `INT_LIKE_COLUMNS` | `open_time`, `close_time`, `count`, `trade_id`, `update_id`, `calc_time` | `Int64` (nullable) || `CATEGORY_LIKE_COLUMNS` | `symbol`, `market_family`, `data_type`, `source_file`, `checksum_status` | `category` |
### 4.2 Timestamp Parsing (`_parse_timestamp_series`)
Automatically infers millisecond vs. microsecond unit from value magnitude:
```pythonunit = "us" if max_value >= 10**15 else "ms"```
All timestamps are converted to UTC-aware `datetime64[ns, UTC]`.
### 4.3 Metadata Injection
Each row receives:
| Column | Content ||---|---|| `source_file` | Original filename || `ingested_at` | Ingestion timestamp || `checksum_status` | `passed` / `failed` / `unknown` || `raw_timestamp` | Original numeric timestamp preserved as string |
### 4.4 Data-Type Routing
`_normalize_file()` routes to the appropriate reader based on `data_type`:
| Data Type | Reader | Timestamp Source ||---|---|---|| `klines` / `*Klines` | `_read_kline_frame()` | `open_time` || `fundingRate` | `_read_funding_rate_frame()` | `calc_time` || `aggTrades` | `_read_agg_trades_frame()` | `transact_time` || `trades` | `_read_trades_frame()` | `transact_time` || `bookTicker` | `_read_book_ticker_frame()` | `transaction_time` (fallback: `event_time`) |
### 4.5 OHLCV Validation (`src/core/validation.py`)
`normalize_ohlcv_frame()` performs:
1. **Required columns check** — `timestamp`, `open`, `high`, `low`, `close` (optionally `volume`).2. **Timestamp normalization** — Convert to UTC datetime.3. **Sort + deduplicate** — `sort_values(timestamp).drop_duplicates(keep='last')`.4. **Monotonic check** — Raises `ValueError` if timestamps are not strictly increasing after sorting.
### 4.6 Quality Assurance
**Script:** `scripts/qa_binance_public_history.py`
Runs `_strict_1m_continuity()` to detect gaps in normalized kline data.
---
## 5. Phase 3 — Feature Engineering
**Core module:** `src/features/builder.py` → `build_feature_frame()`
### 5.1 Feature Build Flow
```Input: normalized OHLCV DataFrame │ ├─ 1. DerivativesFeatureStore.attach_to_spot() (if derivatives.enabled) │ merge_derivatives_frames() → outer-join funding/basis/OI/options/book_ticker │ align_derivatives_to_spot() → pd.merge_asof(direction="backward") │ → adds raw_* columns to DataFrame │ ├─ 2. Merge second_level_features_frame (if provided) │ → left-join on timestamp, validate one_to_one │ ├─ 3. For each pack in profile.packs: │ pack.transform(df, settings, profile) → new feature columns │ pd.concat([df, features], axis=1) │ ├─ 4. Drop derivatives helper columns (raw_*, exchange, symbol, etc.) │ ├─ 5. add_grid_columns() → grid_t0, grid_id, is_grid_t0 │ ├─ 6. Add metadata: asset, horizon, feature_version (currently "v5") │ └─ 7. select_grid_rows() if strict_grid_only=True → keep only rows where minute % grid_minutes == 0```
### 5.2 Registered Feature Packs (28 total)
The `core_5m` profile activates 25 packs. Full registry in `src/features/registry.py`:
| Category | Packs ||---|---|| **Spot price** | `momentum`, `momentum_acceleration`, `volatility`, `path_structure`, `regime`, `candle_structure`, `compression_breakout`, `asymmetry` || **Volume** | `volume`, `flow_proxy`, `market_quality` || **Structure** | `htf_context` (15m aggregation), `intra_5m_structure` || **Microstructure** | `completed_bar_microstructure`, `flow_pressure`, `book_pressure`, `second_level_microstructure`, `event_window_burst`, `side_specific_transforms` || **Derivatives** | `derivatives_funding`, `derivatives_basis`, `derivatives_book_ticker`, `derivatives_oi`, `derivatives_options` || **Cross-feature** | `interaction_bank`, `regime_interactions`, `lagged`, `time` |
### 5.3 Look-Ahead Bias Prevention
- All spot features use `shift(1)` — feature at time $t$ only uses data up to $t-1$.- Derivatives alignment: `pd.merge_asof(direction="backward")` — only the most recent known value at or before $t$.- Derivatives internal merge: `ffill()` propagates the last known value forward (no future peek).- HTF context: only uses *completed* higher-timeframe candles (bucket must have ≥ N bars).
### 5.4 Derivatives Alignment Detail
**Module:** `src/data/derivatives/aligner.py`
1. `merge_derivatives_frames()` — Outer-joins up to 5 sources (funding, basis, OI, options, book_ticker) on timestamp, then `ffill()` all value columns.2. `align_derivatives_to_spot()` — `pd.merge_asof(spot, derivatives, on=timestamp, direction="backward")` ensures each spot bar gets the most recent derivatives reading *at or before* that bar's timestamp.
### 5.5 Window Configuration (`core_5m` profile)
| Parameter | Values ||---|---|| `momentum_windows` | `[1, 3, 5, 10, 15]` || `vol_windows` | `[3, 5, 10, 30]` || `volume_windows` | `[3, 5, 10, 20]` || `market_quality_windows` | `[5, 20]` || `slope_windows` | `[3, 5]` || `range_windows` | `[3, 5, 10]` || `htf_context_timeframes` | `[15]` (minutes) || `compression_window` | `20` || `compression_rank_window` | `100` || `asymmetry_rv_windows` | `[5, 20]` || `asymmetry_skew_windows` | `[10, 20]` || `lagged_feature_lags` | `[1, 2, 3, 6, 12]` |
---
## 6. Phase 4 — Label Generation
**Module:** `src/labels/grid_direction.py` → `GridDirectionLabelBuilder`
### 6.1 Primary Label: Grid Direction (Binary)
$$y_t = \mathbb{1}\{\text{close}_{t_0+4} \geq \text{open}_{t_0}\}$$
Where $t_0$ is on the 5-minute grid (`minute % 5 == 0`), and `+4` means the close of the 4th subsequent 1-minute bar.
**Implementation:**
```pythonfuture_close = df["close"].shift(-horizon.future_close_offset) # offset = 4 for 5mtarget = (future_close >= df["open"]).astype("float64")target[future_close.isna()] = pd.NA # end-of-data: no future bar availabletarget[~df["is_grid_t0"]] = pd.NA # non-grid rows: not a valid prediction point```
Labels are only defined at grid timestamps. Non-grid rows are set to `NaN` and dropped during dataset assembly.
### 6.2 Auxiliary Label Modules
| Module | Purpose | Registered as LabelBuilder? ||---|---|---|| `abs_return.py` | `build_abs_return_frame()` computes `abs_return` and `signed_return` for sample weighting | No || `three_class_direction.py` | `build_three_class_direction_target()` for two-stage training `stage2_target` | No |
---
## 7. Phase 5 — Dataset Assembly
**Module:** `src/data/dataset_builder.py` → `build_training_frame()`
### 7.1 Assembly Pipeline
```Step 1: normalize_ohlcv_frame(raw_df)Step 2: build_feature_frame(normalized, settings, derivatives_frame, second_level_features_frame)Step 3: label_builder.build(normalized, settings, horizon)Step 4: Merge features ← labels (left join on timestamp, validate one_to_one)Step 5: Merge ← abs_return_frame (left join on timestamp)Step 6: filter_by_timerange(train_start, train_end)Step 7: infer_feature_columns() + assert_feature_schema()Step 8: drop_incomplete_samples() (if drop_incomplete_candles=True)Step 9: compute_sample_weight(abs_return)Step 10: Return TrainingFrame(frame, feature_columns, target_column, sample_weight_column)```
### 7.2 Feature Column Inference
`infer_feature_columns()` selects all columns that are **not** in:- `BASE_DATASET_COLUMNS` (timestamp, OHLCV, target, grid_id, sample_weight, etc.)- `RAW_METADATA_FEATURE_COLUMNS` (source_file, ingested_at, checksum_status, etc.)- Columns starting with `raw_` or `source_` prefixes
### 7.3 Sample Weighting (`compute_sample_weight`)
Mode: `linear_ramp` (the only supported mode).
$$w = \text{clip}\!\Big(\text{min\_weight} + (\text{max\_weight} - \text{min\_weight}) \times \frac{|\text{return}|}{\text{full\_weight\_abs\_return}},\;\text{min\_weight},\;\text{max\_weight}\Big)$$
Samples with $|\text{return}| < \text{min\_abs\_return}$ are forced to `min_weight`.
**Current config values:**
```yamlsample_weighting: enabled: true mode: linear_ramp min_abs_return: 0.0001 full_weight_abs_return: 0.0003 min_weight: 0.35 max_weight: 1.00```
### 7.4 TrainingFrame Dataclass
```python@dataclass(frozen=True)class TrainingFrame: frame: pd.DataFrame # complete DataFrame feature_columns: list[str] # pure feature column names target_column: str # "target" sample_weight_column: str | None # "stage1_sample_weight" or None
 X -> frame[feature_columns] y -> frame[target_column] sample_weight -> frame[sample_weight_column]```
---
## 8. Key File Reference
| File | Responsibility ||---|---|| `scripts/backfill_binance_public_history.py` | Download spot, futures, and option public archives from Binance Vision || `scripts/normalize_binance_public_history.py` | Normalize raw CSVs → stabilized Parquet || `scripts/normalize_aggtrades_daily.py` | Normalize aggTrades into daily Parquet || `scripts/qa_binance_public_history.py` | Quality checks on normalized data || `scripts/build_second_level_feature_store.py` | Build split 1-second feature stores; default project storage is `second_features_kline` + `second_features_agg` || `scripts/build_dataset.py` | End-to-end dataset build (features + labels + weights) || `scripts/train_model.py` | Train single-stage model || `scripts/train_two_stage.py` | Train two-stage (direction + selective) model || `scripts/run_live_signal.py` | Online inference + Polymarket order submission || `scripts/run_shadow.py` | Shadow mode (audit only, no real orders) || `scripts/run_model_experiments.py` | Batch model experiments || `scripts/run_binary_rolling_validation.py` | Walk-forward rolling validation || `src/data/binance_public/normalizer.py` | Schema enforcement, dtype stabilization, metadata injection || `src/data/derivatives/aligner.py` | Merge + backward-asof alignment for derivatives || `src/data/derivatives/feature_store.py` | Load and attach derivatives to spot frame || `src/data/dataset_builder.py` | `build_training_frame()`, `compute_sample_weight()` || `src/features/builder.py` | `build_feature_frame()` — central feature orchestrator || `src/features/registry.py` | 28 FeaturePack registrations || `src/labels/grid_direction.py` | Primary binary label builder || `src/core/validation.py` | `normalize_ohlcv_frame()` — sort, dedupe, monotonic check || `src/core/timegrid.py` | Grid alignment utilities (`floor_to_grid`, `add_grid_columns`) || `src/core/constants.py` | Version constants (`CORE_FEATURE_VERSION = "v5"`, `CORE_LABEL_VERSION = "v1"`) |
---
## 9. Configuration Reference (`config/settings.yaml`)
### 9.1 Dataset
```yamldataset: train_start: "2024-01-01" train_end: "2026-12-31" validation_window_days: 30 train_window_days: 30 strict_grid_only: true drop_incomplete_candles: true walk_forward: enabled: false```
### 9.2 Horizons
```yamlhorizons: active: ["5m"] specs: "5m": minutes: 5 grid_minutes: 5 label_builder: grid_direction feature_profile: core_5m signal_policy: selective_binary_policy sizing_plugin: fixed_fraction```
### 9.3 Sample Weighting
```yamlsample_weighting: enabled: true mode: linear_ramp min_abs_return: 0.0001 full_weight_abs_return: 0.0003 min_weight: 0.35 max_weight: 1.00```
### 9.4 Derivatives
```yamlderivatives: enabled: true funding: enabled: true zscore_window: 720 basis: enabled: true use_mark_price: true use_index_price: true use_premium_index: true zscore_window: 720 oi: enabled: false # available but not active in core_5m frequency: 5m zscore_window: 288 options: enabled: false # available but not active in core_5m zscore_window: 288 book_ticker: enabled: true zscore_window: 288```
---
## 10. Quickstart Commands
```powershell# 1. Download spot kline historyrtk python scripts/backfill_binance_public_history.py
# 2. Normalize raw datartk python scripts/normalize_binance_public_history.py
# 3. (Optional) Build split second-level feature stores; loader joins kline + agg storesrtk python scripts/build_second_level_feature_store.py
# 5. Build training datasetrtk python scripts/build_dataset.py --config config/settings.yaml --horizon 5m
# 6. Train modelrtk python scripts/train_model.py --config config/settings.yaml --horizon 5m
# 7. Run QA checksrtk python scripts/qa_binance_public_history.py```
---
## 11. How to Add a New Feature Pack
1. Create `src/features/my_feature.py`:
```pythonfrom src.features.base import FeaturePack
class MyFeaturePack(FeaturePack): name = "my_feature"
 def transform(self, df, settings, profile): past_close = df["close"].shift(1) features = pd.DataFrame(index=df.index) features["my_signal"] = past_close.pct_change(5) return features```
2. Register in `src/features/registry.py`:
```pythonfrom src.features.my_feature import MyFeaturePack
FEATURE_PACKS: dict[str, FeaturePack] = { # ... existing packs ... "my_feature": MyFeaturePack(),}```
3. Add to the feature profile in `config/settings.yaml`:
```yamlfeatures: profiles: core_5m: packs: # ... existing packs ... - my_feature```
4. Bump `CORE_FEATURE_VERSION` in `src/core/constants.py`.5. Run `rtk pytest -q` to verify.
---
## 12. Troubleshooting
### Data Gaps
If QA reports `strict_1m_continuity: false`, there are missing minutes in the kline data.
**Fix:** Run `scripts/qa_binance_public_history.py` to identify the gap timestamps, then re-run `backfill_binance_public_history.py` for those date ranges.
### Feature Parity Drift
If feature values differ between backtest and live inference:
1. Both paths must call `build_feature_frame()` — never compute features independently.2. All features must use `shift(1)` on raw OHLCV so that feature at $t$ only sees data through $t-1$.3. Run `tests/test_train_live_feature_parity_*.py` to verify numeric equivalence.
### Out of Memory
Second-level features on 1-second data can consume 15–30 GB for a full year.
**Mitigations:**- Process in monthly chunks via `build_second_level_feature_store.py`.- Use `float32` instead of `float64` for feature columns where precision is sufficient.- Increase swap space or use a machine with ≥ 32 GB RAM.
