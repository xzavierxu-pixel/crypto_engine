# Second-Level Directional Feature Implementation Guide
Date: 2026-05-01 Project: `crypto_engine` Language: English Scope: Additional second-level features likely to improve short-horizon directional prediction for BTC 5m settlement direction
---
## 1. Objective
This document proposes additional second-level features that are likely to improve **directional prediction quality** on top of the current second-level pipeline.
The goal is not to increase label frequency. The target remains unchanged:
```textone decision timestamp -> one feature row -> one 5-minute settlement-direction label```
The purpose of new second-level features is to improve:
- directional separability near the decision timestamp,- ranking quality of `p_up`,- directional precision at selective thresholds,- robustness under noisy microstructure conditions.
---
## 2. Current State
The current code already supports real second-level feature construction in:
- `src/data/second_level_features.py`
The pipeline can consume:
- trade-level or pre-aggregated second data,- second-level `bookTicker` data,- and merge the resulting `sl_*` features into training via `scripts/train_model.py`.
The currently implemented `sl_*` families already cover:
- short-horizon returns and log returns,- realized volatility,- directional efficiency and choppiness,- taker buy/sell imbalance,- signed dollar flow,- trade intensity,- volume burst flags,- basic top-of-book imbalance,- microprice premium,- spread dynamics.
This is already a strong foundation.
The next gains are likely to come from **microstructure state transitions**, not from adding more generic return windows.
---
## 2.1 How `aggTrades` and `1s klines` Should Be Combined
These two sources should not be treated as interchangeable copies of the same second-level input.
They should be combined using a **backbone + enrichment** design:
- `1s klines` should be the **canonical second-level backbone**.- `aggTrades` should be the **event-structure enrichment layer**.- second-level `bookTicker` should be the **liquidity-state layer**.
The reason is architectural, not only statistical:
- `1s klines` already provide one stable row per second and are much cheaper to store, backfill, QA, and load during training.- `aggTrades` retain microstructure information that `1s klines` cannot express, especially intrasecond clustering, trade-size distribution, burstiness, and event ordering.- `bookTicker` is needed for queue state, spread state, microprice, OFI, depletion, and replenishment.
The correct target is therefore:
```text1s kline backbone + aggTrades-only enrichment features + bookTicker-only liquidity features = one materialized second-level feature table```
This is better than either extreme:
- using only `1s klines`, which loses intrasecond trade-structure information,- or using only `aggTrades`, which is more expensive and unnecessary for fields already represented well by 1-second candles.
---
## 2.2 Which Features Should Come Only From `1s klines`
`1s klines` should own the features whose natural granularity is already one second and whose value does not materially depend on intrasecond event ordering.
Recommended `1s klines`-owned families:
- second-level OHLC path summaries,- close-to-close returns and log returns,- range and realized-volatility features,- 1-second trade count, total volume, and quote volume,- 1-second taker buy base volume and taker buy quote volume,- taker buy ratio and taker imbalance from candle aggregates,- VWAP-like features derived from candle quote volume and base volume,- short-vs-long contrasts built from the above, such as `5s - 60s` and `30s / 300s`.
Concretely, the following families should default to `1s klines` as the canonical source:
- `sl_return_*`- `sl_log_return_*`- `sl_rv_*`- `sl_rvar_*`- `sl_mean_abs_return_*`- `sl_max_abs_return_*`- `sl_range_*`- `sl_price_slope_*`- `sl_log_price_slope_*`- `sl_trade_count_*`- `sl_trades_per_second_*`- `sl_total_volume_*`- `sl_dollar_volume_*`- `sl_taker_buy_volume_*`- `sl_taker_sell_volume_*`- `sl_taker_buy_ratio_*`- `sl_taker_sell_ratio_*`- `sl_taker_imbalance_*`- `sl_signed_dollar_flow_*` if the sign convention can be reconstructed reliably from candle-side aggregates- `sl_vwap_*` and `sl_price_minus_vwap_*`
Why these belong to `1s klines`:
- they are already second-aligned,- they are compact and deterministic,- they reduce training-time memory pressure,- and they can become the canonical second-level table shared by training, backtests, and QA.
If both `1s klines` and `aggTrades` can produce a similar field, the default rule should be:
```textuse `1s klines` as the production feature source,use `aggTrades` only for QA cross-checks or residual/enrichment features.```
This avoids having two nearly-duplicate versions of volume, trade-count, and basic flow features competing inside the model.
---
## 2.3 Which Features Must Depend On `aggTrades`
`aggTrades` should be reserved for signals that genuinely require event-level trade structure and cannot be reconstructed faithfully from 1-second candles.
These are the highest-value `aggTrades`-dependent families:
- inter-arrival time features,- trade clustering / burstiness,- trade-size distribution statistics,- large-trade detection and large-trade imbalance,- streak or run-length features for aggressive buying or selling,- intrasecond execution concentration,- trade-ordering features such as last-few-trades dominance,- flow segmentation within the second,- more precise absorption proxies that compare event flow and subsequent micro-movement.
These should be considered `aggTrades`-only or `aggTrades`-first:
- `sl_mean_interarrival_ms_*`- `sl_min_interarrival_ms_*`- `sl_interarrival_cv_*`- `sl_trade_cluster_score_*`- `sl_buy_trade_cluster_score_*`- `sl_sell_trade_cluster_score_*`- `sl_median_trade_size_*`- `sl_large_trade_count_*`- `sl_large_trade_volume_share_*`- `sl_large_buy_trade_count_*`- `sl_large_sell_trade_count_*`- `sl_large_buy_volume_share_*`- `sl_large_sell_volume_share_*`- `sl_large_trade_imbalance_*`- `sl_last_n_trades_buy_share`- `sl_last_n_trades_sell_share`- `sl_buy_run_length_*`- `sl_sell_run_length_*`- `sl_intrasecond_flow_concentration_*`- `sl_trade_arrival_burst_flag_*`
Reasoning:
- `1s klines` can tell us how much traded in a second.- `aggTrades` can tell us **how that second happened**.
That distinction matters because directional precision often improves when the model sees whether flow arrived as:
- many tiny alternating trades,- a few large aggressive prints,- a compressed burst right before the second closed,- or a one-sided execution run.
Those are not candle features. They are event-structure features.
---
## 2.4 Final Rule For Source Ownership
To keep the feature space clean, every second-level field should have one canonical owner.
Recommended rule set:
1. `1s klines` own all canonical second-level price, volume, and baseline flow features.2. `aggTrades` own event-structure and intrasecond execution features.3. `bookTicker` owns quote-state and liquidity-state features.4. cross-interaction features are built only after the three source families have been normalized into a single second-level table.
This prevents source duplication and makes later debugging much easier.
---
## 3. What Is Missing Today
The highest-value missing pieces are not raw momentum features. They are mostly:
1. **Queue depletion / replenishment asymmetry**2. **Large-trade and informed-flow features**3. **Trade-book coupling features**4. **Absorption vs exhaustion features**5. **Shock continuation vs shock reversal features**6. **Multi-level depth structure** if deeper snapshots are available7. **Cross-market leading signals** from perp and related assets
These are the feature families most likely to improve directional precision because they describe **who is pushing the market, whether liquidity is yielding, and whether the last move is likely to continue or fail**.
---
## 4. Highest-Priority New Features
## 4.1 Order Flow Imbalance (OFI) at the Top of Book
### Why it matters
Top-of-book imbalance alone is often too static. Directional prediction improves when we measure **how the best bid and best ask change through time**, not only their current snapshot.
A classical microstructure signal is **order flow imbalance (OFI)**:
- increases in bid price or bid size are bullish,- decreases in ask price or ask size are bullish,- the reverse is bearish.
This is usually stronger than a raw `bid_qty - ask_qty` snapshot.
### Proposed features
For windows `5s`, `10s`, `30s`:
- `sl_ofi_5s`, `sl_ofi_10s`, `sl_ofi_30s`- `sl_ofi_z_300s`- `sl_ofi_persistence_30s`- `sl_ofi_acceleration`
### Implementation idea
Given second-level best quotes:
```textOFI_t = delta_bid_size * 1{bid_price_t >= bid_price_{t-1}} - delta_bid_size * 1{bid_price_t < bid_price_{t-1}} - delta_ask_size * 1{ask_price_t <= ask_price_{t-1}} + delta_ask_size * 1{ask_price_t > ask_price_{t-1}}```
Then aggregate OFI over rolling windows and sample at the decision timestamp.
### Expected value
This should improve directional prediction when markets move because of **active liquidity pressure**, not just because of recent returns.
---
## 4.2 Queue Replenishment and Queue Depletion Rates
### Why it matters
Directional continuation is much more likely when one side of the book is repeatedly being consumed **without replenishment**.
The current code already includes:
- `sl_bid_qty_depletion_*`- `sl_ask_qty_depletion_*`
But it does not yet measure the opposite side of the event: **how fast queues refill after being hit**.
### Proposed features
- `sl_bid_replenishment_rate_5s`, `sl_bid_replenishment_rate_10s`, `sl_bid_replenishment_rate_30s`- `sl_ask_replenishment_rate_5s`, `sl_ask_replenishment_rate_10s`, `sl_ask_replenishment_rate_30s`- `sl_bid_depletion_minus_replenishment_10s`- `sl_ask_depletion_minus_replenishment_10s`- `sl_bid_wall_disappear_flag`- `sl_ask_wall_disappear_flag`
### Implementation idea
For bid replenishment:
```textreplenishment = positive changes in bid_qty while bid_price stays unchanged or improves```
For ask replenishment:
```textreplenishment = positive changes in ask_qty while ask_price stays unchanged or improves```
Then compare depletion vs replenishment on each side.
### Expected value
These features help separate:
- true directional pressure,- from fake moves where liquidity immediately refills and absorbs impact.
---
## 4.3 Large-Trade / Informed-Flow Features
### Why it matters
Not all flow is equal. Direction often follows **a small number of unusually large aggressive trades**, especially near short-horizon decision boundaries.
The current pipeline captures total flow and counts, but not enough about **trade-size distribution**.
### Proposed features
Per window `10s`, `30s`, `60s`:
- `sl_median_trade_size_30s`- `sl_large_trade_count_30s`- `sl_large_trade_volume_share_30s`- `sl_large_buy_trade_count_30s`- `sl_large_sell_trade_count_30s`- `sl_large_buy_volume_share_30s`- `sl_large_sell_volume_share_30s`- `sl_large_trade_imbalance_30s`
### Threshold definition
A large trade can be defined dynamically as:
```texttrade notional > rolling 95th percentile of the last 300s```
or initially:
```texttrade notional > fixed threshold in quote currency```
### Expected value
These features should be especially useful in distinguishing:
- broad retail churn,- from concentrated, directional, informed-looking flow.
---
## 4.4 Inter-Arrival Time and Trade Clustering Features
### Why it matters
Directional urgency is often visible in **time compression of trades** before it is visible in price.
A burst of very short inter-arrival times often indicates information arrival or aggressive execution.
### Proposed features
- `sl_mean_interarrival_ms_10s`- `sl_mean_interarrival_ms_30s`- `sl_min_interarrival_ms_10s`- `sl_interarrival_cv_30s`- `sl_trade_cluster_score_10s`- `sl_trade_cluster_score_30s`- `sl_buy_trade_cluster_score_10s`- `sl_sell_trade_cluster_score_10s`
### Implementation idea
From raw trade timestamps, compute time gaps between consecutive trades, then aggregate:
```textcluster_score = count(interarrival < threshold_ms) / trade_count```
### Expected value
Useful for detecting:
- urgency,- informed participation,- pre-breakout crowding.
---
## 4.5 Trade-to-Book Coupling Features
### Why it matters
A directional move is more reliable when **aggressive flow and book state agree**.
Examples:
- taker buy flow is positive while book imbalance is also positive,- signed dollar flow is positive while microprice premium is rising,- buy pressure continues while ask queues are being depleted.
The current pipeline computes the components separately, but not enough cross-interactions between them.
### Proposed features
- `sl_flow_x_book_imbalance_30s`- `sl_flow_x_microprice_premium_30s`- `sl_taker_imbalance_x_spread_30s`- `sl_signed_flow_x_ask_depletion_10s`- `sl_signed_flow_x_bid_replenishment_10s`- `sl_buy_pressure_with_tight_spread_flag`- `sl_sell_pressure_with_tight_spread_flag`
### Implementation idea
Examples:
```textflow_x_book_imbalance = sl_signed_dollar_flow_30s * sl_bid_ask_qty_imbalanceflow_x_microprice_premium = sl_signed_dollar_flow_30s * sl_microprice_premium```
### Expected value
These features should improve precision by distinguishing:
- clean directional pressure,- from noisy bursts unsupported by liquidity structure.
---
## 4.6 Absorption and Exhaustion Features
### Why it matters
Some aggressive order flow should predict continuation. But sometimes strong buy flow produces little price movement because a large passive seller is absorbing it.
That is a major directional clue.
### Proposed features
- `sl_buy_absorption_30s`- `sl_sell_absorption_30s`- `sl_flow_to_price_efficiency_30s`- `sl_price_to_flow_divergence_30s`- `sl_exhaustion_after_buy_burst_flag`- `sl_exhaustion_after_sell_burst_flag`
### Implementation idea
A simple proxy:
```textbuy_absorption = positive signed flow / abs(price return)```
when positive flow is large but price return is small.
Similarly:
```textflow_to_price_efficiency = abs(return_30s) / abs(signed_dollar_flow_30s)```
Low efficiency suggests absorption. Very high efficiency suggests thin-liquidity price displacement.
### Expected value
This is one of the most promising families for improving **directional reversal detection**.
---
## 4.7 Shock Continuation vs Shock Reversal Features
### Why it matters
The current pipeline already captures max returns and some flip flags, but it does not yet explicitly distinguish:
- “jump then continue”- from “jump then fade”
This distinction is critical at sub-minute horizons.
### Proposed features
- `sl_time_since_last_positive_jump`- `sl_time_since_last_negative_jump`- `sl_positive_jump_count_30s`- `sl_negative_jump_count_30s`- `sl_shock_continuation_flag`- `sl_shock_reversal_flag`- `sl_post_jump_pullback_ratio`- `sl_post_jump_followthrough_ratio`
### Implementation idea
Define a jump using a threshold such as:
```textabs(second_return) > rolling 95th percentile of abs(second_return) over 300s```
Then measure what happened after that jump over the next few seconds.
### Expected value
These should help discriminate between:
- impulse continuation,- liquidity vacuum jumps,- and mean-reverting spikes.
---
## 4.8 VWAP Deviation and Execution Footprint Features
### Why it matters
In second-level directional prediction, the distance between the last traded price and short-horizon VWAP is often useful.
If price is above VWAP with strong buy flow, continuation is more likely. If price is far above VWAP but flow has weakened, reversal risk rises.
### Proposed features
- `sl_vwap_10s`, `sl_vwap_30s`, `sl_vwap_60s`- `sl_price_minus_vwap_10s`- `sl_price_minus_vwap_30s`- `sl_price_minus_vwap_z_300s`- `sl_buy_flow_with_positive_vwap_deviation`- `sl_sell_flow_with_negative_vwap_deviation`
### Implementation idea
```textvwap_w = dollar_volume_w / total_volume_wprice_minus_vwap_w = mid_or_last_price - vwap_w```
### Expected value
Useful for separating:
- genuine directional re-pricing,- from overextended short-term displacement.
---
## 4.9 Quote Update Asymmetry Features
### Why it matters
The current pipeline includes quote update count, but not enough detail about **who is updating more aggressively**.
Directional information often appears in:
- more frequent ask revisions than bid revisions,- or repeated microprice improvements with limited spread widening.
### Proposed features
- `sl_bid_uptick_count_10s`, `sl_bid_downtick_count_10s`- `sl_ask_uptick_count_10s`, `sl_ask_downtick_count_10s`- `sl_bid_revision_pressure_10s`- `sl_ask_revision_pressure_10s`- `sl_quote_update_asymmetry_10s`- `sl_microprice_improvement_persistence_30s`
### Expected value
This should improve precision in thin but actively repricing markets.
---
## 4.10 Multi-Level Book Features (If Depth Snapshots Exist)
### Why it matters
Top-of-book is noisy. Short-horizon direction often depends on whether deeper liquidity supports the move.
### Proposed features
For `N = 5` or `10` depth levels:
- `sl_bid_depth_5`- `sl_ask_depth_5`- `sl_depth_imbalance_5`- `sl_weighted_depth_imbalance_5`- `sl_near_mid_depth_imbalance`- `sl_distance_to_large_bid_wall`- `sl_distance_to_large_ask_wall`- `sl_bid_wall_size`- `sl_ask_wall_size`- `sl_book_slope_bid`- `sl_book_slope_ask`- `sl_book_convexity`
### Expected value
High, but only if reliable depth snapshots are available and storage/latency costs are acceptable.
This should be a **Phase 2** item, not the first addition.
---
## 4.11 Cross-Market Second-Level Features
### Why it matters
BTC short-horizon direction is often led by related markets:
- perp before spot,- futures book before spot tape,- BTC before Polymarket,- sometimes ETH or crypto beta before BTC idiosyncratic movement stabilizes.
### Proposed features
If available at second frequency:
- `sl_perp_return_5s`, `sl_perp_return_30s`- `sl_perp_book_imbalance_30s`- `sl_perp_microprice_premium_10s`- `sl_spot_minus_perp_return_10s`- `sl_basis_change_10s`- `sl_btc_minus_eth_return_30s`- `sl_crypto_beta_residual_return_30s`
### Expected value
Potentially very strong, especially for direction ranking, but requires more data plumbing than the first-priority additions.
---
## 5. Recommended Priority Order
The highest expected directional value per engineering hour is likely:
### Tier 1: Implement immediately
1. OFI features2. queue replenishment / depletion asymmetry3. large-trade features4. trade-book coupling features5. absorption / exhaustion features
### Tier 2: Implement next
6. inter-arrival / clustering features7. shock continuation vs reversal features8. VWAP deviation features9. quote update asymmetry features
### Tier 3: Implement only if the data exists reliably
10. multi-level depth features11. cross-market second-level features
---
## 6. Recommended Second-Level Feature Store Architecture
If second-level features materially improve prediction quality, they should stop being a training-time side path and become a first-class data product.
The target architecture should be:
```textraw second-level inputs -> source-specific normalization -> 1-second source tables -> unified second-level feature table -> decision-time feature sampling -> training / backtest / live reuse```
This is not the same as a normal feature pack.
Regular feature packs in `src/features/` assume that the main bar-level frame already exists in memory. Real second-level processing is different because it must:
- read raw high-frequency inputs,- normalize source-specific schemas,- aggregate to 1-second state,- compute rolling windows,- and only then sample backward to the decision timeline.
That makes second-level processing a **feature-store layer**, not just another pack transform.
### 6.1 Layered Design
The cleanest design is a four-layer pipeline.
#### Layer 1: Raw source layer
Inputs:
- raw `aggTrades`- raw `trades` if available- raw `1s klines`- raw second-level `bookTicker`- optional depth snapshots later
Responsibilities:
- archive raw files,- preserve source timestamps,- preserve source metadata,- record source version and path manifest,- run source-level QA such as duplicates, gaps, and schema validation.
#### Layer 2: Source-normalized 1-second layer
Outputs:
- one normalized `1s kline` table,- one normalized `aggTrades -> 1s event summary` table,- one normalized `bookTicker -> 1s quote state` table.
Responsibilities:
- normalize column names and dtypes,- align all timestamps to UTC second boundaries,- preserve source-specific coverage flags,- compute source-local second-level summaries before cross-source joins.
This is the layer where duplicate information should be resolved. For example:
- `trade_count`, `volume`, and `quote_volume` should usually come from `1s klines` as the production source,- while `aggTrades` can still compute the same fields for QA delta checks.
#### Layer 3: Unified second-level feature store
Output:
- one materialized per-second table keyed by `timestamp`- containing canonical second-level features and metadata
Responsibilities:
- join `1s klines`, `aggTrades`-derived event features, and `bookTicker` features,- compute rolling second-level windows,- compute cross-source interactions,- store feature version, source coverage, and QA metadata,- write partitioned parquet artifacts.
This is the main reusable asset for:
- model training,- offline evaluation,- walk-forward backtests,- error analysis,- and future online feature serving.
#### Layer 4: Decision-time sampler
Output:
- one row per decision timestamp
Responsibilities:
- sample from the per-second feature store using backward `asof`,- enforce the anti-leakage rule,- align with the 5-minute prediction horizon,- hand off a compact `sl_*` frame to `build_training_frame(...)`.
This keeps training code simple. Training should load second-level features, not build them.
### 6.2 Why This Is Better Than Training-Time Construction
This architecture is better on every important axis:
- lower peak memory during training,- less duplicated computation across experiments,- more reproducible training data,- easier QA and anomaly investigation,- consistent second-level features across training and backtests,- clearer ownership between data engineering and model experimentation.
Once second-level features matter materially, recomputing them inside the training script becomes the wrong abstraction boundary.
### 6.3 Join Strategy For `1s klines`, `aggTrades`, and `bookTicker`
The recommended join strategy is:
1. treat `1s klines` as the canonical second grid,2. left join `aggTrades`-derived enrichment features on the same second,3. left join `bookTicker`-derived quote-state features on the same second,4. compute cross-source interactions only after canonical fields are resolved,5. sample this unified per-second table to decision timestamps.
That means the pipeline should prefer:
```textcanonical second grid = `1s klines````
not:
```textrebuild the second grid from raw aggTrades every time training runs```
The latter is more expensive and also blurs the difference between baseline candle facts and event-structure enrichments.
### 6.4 Suggested Artifact Layout
Suggested output layout under `artifacts/` or a dedicated feature-store path:
```textartifacts/ second_level/ version=YYYYMMDD_or_semver/ market=BTCUSDT/ date=2026-05-01/ second_features.parquet manifest.json qa_report.json```
The manifest should record at least:
- feature version,- generation timestamp,- source file paths,- source date range,- source coverage ratios,- missing-data policy,- column schema,- aggregation rules,- any source-precedence decisions.
---
## 7. Proposed Final Feature Store Schema
The final per-second feature store should be narrow enough to load efficiently but explicit enough to debug provenance.
Recommended column groups:
### 7.1 Primary key and metadata
- `timestamp`- `market`- `exchange`- `second_level_feature_version`- `source_manifest_id`- `decision_grid_name` if multiple decision calendars may exist later
### 7.2 Source coverage and QA columns
- `has_1s_kline`- `has_agg_trade_enrichment`- `has_book_ticker`- `agg_trade_gap_flag`- `book_gap_flag`- `kline_gap_flag`- `cross_source_volume_delta`- `cross_source_trade_count_delta`
These columns are useful for both QA and model-side filtering if a period is structurally degraded.
### 7.3 Canonical 1-second backbone columns
- `sec_open`- `sec_high`- `sec_low`- `sec_close`- `sec_volume`- `sec_quote_volume`- `sec_trade_count`- `sec_taker_buy_base_volume`- `sec_taker_buy_quote_volume`- `sec_taker_sell_base_volume`- `sec_taker_sell_quote_volume`- `sec_vwap`
These are not necessarily all training features directly. They are the canonical second-level facts from which many rolling `sl_*` features are derived.
### 7.4 `aggTrades` enrichment columns
- `sec_median_trade_size`- `sec_large_trade_count`- `sec_large_trade_volume_share`- `sec_large_buy_trade_count`- `sec_large_sell_trade_count`- `sec_buy_run_length`- `sec_sell_run_length`- `sec_mean_interarrival_ms`- `sec_min_interarrival_ms`- `sec_interarrival_cv`- `sec_trade_cluster_score`- `sec_intrasecond_flow_concentration`- `sec_last_trades_buy_share`- `sec_last_trades_sell_share`
These should remain clearly identifiable as event-structure fields.
### 7.5 `bookTicker` liquidity-state columns
- `sec_bid_price`- `sec_bid_qty`- `sec_ask_price`- `sec_ask_qty`- `sec_spread_bps`- `sec_microprice`- `sec_microprice_premium`- `sec_bid_ask_qty_imbalance`- `sec_quote_update_count`- `sec_ofi`- `sec_bid_depletion`- `sec_ask_depletion`- `sec_bid_replenishment`- `sec_ask_replenishment`
### 7.6 Final training-facing derived columns
These are the rolling and interaction features ultimately merged into the training frame, for example:
- `sl_return_5s`- `sl_rv_30s`- `sl_taker_imbalance_30s`- `sl_ofi_10s`- `sl_bid_replenishment_rate_30s`- `sl_large_trade_imbalance_30s`- `sl_trade_cluster_score_10s`- `sl_flow_x_book_imbalance_30s`- `sl_buy_absorption_30s`
The important distinction is:
- `sec_*` columns are source-level or second-level state columns,- `sl_*` columns are model-facing rolling features sampled at decision timestamps.
This distinction makes the feature store much easier to reason about.
---
## 8. Recommended Rollout Sequence For This Architecture
To reduce migration risk, the second-level architecture should be upgraded in phases.
### Phase A: Introduce `1s klines` backbone
- build a materialized per-second backbone from `1s klines`- move canonical second-level price/volume features onto that backbone- keep current trade-side logic only for features not reproducible from candles
### Phase B: Restrict `aggTrades` to enrichment-only features
- remove duplicated baseline candle metrics from the `aggTrades` production path- keep only event-structure signals as true `aggTrades` features- add cross-source QA checks between `1s klines` and `aggTrades`
### Phase C: Materialize a reusable second-level feature store
- write the unified per-second table to parquet- add manifest and QA metadata- make training read from this artifact instead of recomputing from raw inputs
### Phase D: Move decision-time sampling out of the training script
- make dataset-building and training share the same second-level asset- ensure walk-forward, offline evaluation, and live shadow runs all use the same feature definition
This staged rollout keeps the architecture change controlled while preserving current research velocity.
---
## 9. Where to Add Them in the Current Codebase
### 9.1 Extend trade-side second-level features
Primary function:
- `build_trade_second_level_features(...)`
This is the correct place for:
- large-trade metrics,- inter-arrival metrics,- VWAP features,- absorption proxies,- shock continuation / reversal,- trade clustering,- additional signed flow transforms.
### 9.2 Extend book-side second-level features
Primary function:
- `build_book_second_level_features(...)`
This is the correct place for:
- OFI,- replenishment rates,- quote revision asymmetry,- bid/ask wall disappearance flags,- stronger microprice persistence features,- multi-level depth features if snapshots become available.
### 9.3 Add coupled trade-book features after both blocks exist
Best place:
- after both trade and book pieces are constructed,- inside `build_second_level_feature_frame(...)`,- before final concat is returned.
This is where to build cross-interactions such as:
- signed flow × book imbalance,- signed flow × ask depletion,- buy burst × spread regime,- shock flag × microprice premium.
---
## 10. Naming Convention
Keep the current naming convention:
- prefix all real second-level features with `sl_`- suffix by horizon where relevant: `_5s`, `_10s`, `_30s`, `_60s`, `_300s`- use explicit names for derived transforms: - `_z_300s` - `_minus_60s` - `_x_microprice_premium` - `_persistence_30s` - `_flag`
This keeps feature provenance obvious and helps downstream selection and debugging.
---
## 11. Anti-Leakage Rule
All new features must obey the same rule already used by the current pipeline:
```textOnly data at or before the decision timestamp may enter the sampled feature row.```
That means:
- build all second-level series on the second timeline,- aggregate only backward-looking windows,- sample using backward `asof` logic,- never use any data after the decision timestamp.
---
## 12. Practical Rollout Plan
### Phase 1
Add the five highest-priority families:
- OFI- replenishment / depletion asymmetry- large-trade flow- trade-book coupling- absorption / exhaustion
### Phase 2
Retrain the binary selective model using the same threshold workflow and compare:
- ROC AUC- balanced precision- top-decile UP rate- false-UP slices and false-DOWN slices- walk-forward stability
### Phase 3
Keep only the second-level additions that improve:
- validation/test ranking quality,- precision at the chosen coverage band,- and walk-forward consistency.
Do not keep second-level features solely because they improve in-sample feature importance.
---
## 13. Expected Impact
The most likely gains will not come from “more micro returns”. They will come from features that explain:
- whether aggressive flow is real,- whether liquidity is yielding or absorbing,- whether the last micro-move is clean or noisy,- and whether book state confirms or contradicts recent trade flow.
If second-level features are already improving accuracy materially in this project, the next step should be to make them more **microstructure-native**, not merely more numerous.
The strongest candidates for directional precision improvement are therefore:
1. OFI2. depletion vs replenishment asymmetry3. large-trade imbalance4. flow-book coupling5. absorption / exhaustion
These should be implemented first.