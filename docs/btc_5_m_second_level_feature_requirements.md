# BTC 5m Second-Level Feature Requirements

## 1. Purpose

This document defines the second-level feature requirements for the BTC 5-minute Polymarket direction prediction task.

The goal is to improve short-horizon directional separability by using sub-minute market information while keeping the modeling target unchanged:

```text
one decision timestamp = one feature row = one settlement label
```

Second-level data should increase feature granularity, not label frequency.

The main model remains:

```text
weighted binary settlement model
+ asymmetric threshold policy
+ coverage >= 60%
```

---

## 2. Core Principle

Second-level data should be aggregated into fixed features at each decision timestamp.

Correct structure:

```text
raw second-level data before decision time
  ↓
rolling / window aggregation
  ↓
one feature row at decision timestamp
  ↓
one label based on expiry price
```

Example:

```text
decision_time = 10:00:00
expiry_time   = 10:05:00
label         = 1 if price_10:05:00 > price_10:00:00 else 0
features      = summaries of data up to 10:00:00 only
```

Important leakage rule:

```text
No data after decision_time can be used in features.
```

---

## 3. Recommended Lookback Windows

Use multiple short windows because different signals decay at different speeds.

Required windows:

```text
1s, 3s, 5s, 10s, 15s, 30s, 60s, 120s, 300s
```

Recommended initial subset if feature count must be controlled:

```text
5s, 10s, 30s, 60s, 300s
```

For each feature family, calculate values over selected windows and also calculate short-vs-long contrasts, such as:

```text
feature_5s - feature_60s
feature_10s / feature_60s
feature_30s z-score versus 300s baseline
```

---

## 4. Price Action Features

### 4.1 Return features

For each lookback window `w`:

```text
ret_w = price_t / price_{t-w} - 1
log_ret_w = log(price_t / price_{t-w})
```

Required:

- return_1s;
- return_3s;
- return_5s;
- return_10s;
- return_30s;
- return_60s;
- return_300s;
- return_5s_minus_60s;
- return_10s_minus_60s;
- return_30s_minus_300s.

Purpose:

Capture immediate momentum, reversal, and acceleration into the decision timestamp.

### 4.2 Price slope and acceleration

Required:

- linear price slope over 5s / 10s / 30s / 60s;
- log-price slope over 5s / 10s / 30s / 60s;
- slope_5s_minus_slope_30s;
- slope_10s_minus_slope_60s;
- return acceleration: return_5s - return_30s scaled to same horizon;
- late-window acceleration flag.

Purpose:

Differentiate steady momentum from late sudden moves.

### 4.3 Reversal and flip features

Required:

- sign of return_5s;
- sign of return_30s;
- sign disagreement between 5s and 30s;
- number of price direction flips in 10s / 30s / 60s;
- last-second reversal flag;
- max adverse move before decision;
- rebound from local low;
- pullback from local high.

Purpose:

Detect when apparent momentum is weakening or reversing before the decision.

---

## 5. Realized Volatility and Noise Features

For each window:

- realized volatility from second returns;
- realized variance;
- mean absolute second return;
- max absolute second return;
- return range high-low;
- volatility burst versus 300s baseline;
- current volatility percentile versus recent rolling history.

Derived features:

```text
rv_10s / rv_60s
rv_30s / rv_300s
abs(return_10s) / rv_60s
abs(return_30s) / rv_300s
```

Purpose:

Separate directional movement from noisy choppy movement. High return with low noise is more useful than high return inside chaotic volatility.

---

## 6. Trade Flow Features

Use aggregate trades or trade-level data where available.

### 6.1 Taker buy / sell pressure

For each window:

- taker_buy_volume;
- taker_sell_volume;
- taker_buy_count;
- taker_sell_count;
- taker_buy_ratio;
- taker_sell_ratio;
- taker_volume_imbalance:

```text
(buy_volume - sell_volume) / (buy_volume + sell_volume)
```

- taker_count_imbalance:

```text
(buy_count - sell_count) / (buy_count + sell_count)
```

- taker_imbalance_slope;
- taker_imbalance_zscore versus 300s baseline.

Purpose:

Capture aggressive buyer or seller dominance immediately before prediction.

### 6.2 Signed dollar flow

For each window:

```text
signed_dollar_flow = sum(price * quantity * trade_sign)
```

Required:

- signed_dollar_flow_5s;
- signed_dollar_flow_10s;
- signed_dollar_flow_30s;
- signed_dollar_flow_60s;
- signed_dollar_flow_zscore;
- signed_dollar_flow_acceleration;
- signed_dollar_flow / total_dollar_volume.

Purpose:

Capture directional capital pressure rather than only trade counts.

### 6.3 Trade intensity

For each window:

- number of trades;
- trades per second;
- total traded volume;
- dollar volume;
- average trade size;
- median trade size;
- large trade count;
- large trade volume share;
- trade intensity z-score versus 300s baseline.

Purpose:

Identify bursts of activity that may precede short-term continuation or reversal.

### 6.4 Buy / sell burst features

Required:

- buy_volume_burst_flag;
- sell_volume_burst_flag;
- buy_trade_count_burst_flag;
- sell_trade_count_burst_flag;
- consecutive buy-dominant seconds;
- consecutive sell-dominant seconds;
- last 3s buy dominance;
- last 3s sell dominance.

Purpose:

Capture sudden one-sided activity right before decision time.

---

## 7. Order Book and Top-of-Book Features

Use bookTicker or order book snapshots where available.

### 7.1 Basic top-of-book state

Required:

- best_bid;
- best_ask;
- mid_price;
- spread;
- spread_bps;
- bid_qty;
- ask_qty;
- bid_ask_qty_imbalance:

```text
(bid_qty - ask_qty) / (bid_qty + ask_qty)
```

- microprice:

```text
microprice = (ask_price * bid_qty + bid_price * ask_qty) / (bid_qty + ask_qty)
```

- microprice premium:

```text
(microprice - mid_price) / mid_price
```

Purpose:

Measure immediate liquidity pressure at the best bid and ask.

### 7.2 Book dynamics

For each window:

- change in bid_qty;
- change in ask_qty;
- change in imbalance;
- change in spread_bps;
- microprice drift;
- mid-price drift;
- quote update count;
- bid price up-tick count;
- bid price down-tick count;
- ask price up-tick count;
- ask price down-tick count;
- spread widening flag;
- spread tightening flag.

Purpose:

Detect whether liquidity is moving in favor of UP or DOWN.

### 7.3 Liquidity depletion and replenishment

Required:

- bid_qty_depletion over 5s / 10s / 30s;
- ask_qty_depletion over 5s / 10s / 30s;
- bid_replenishment_rate;
- ask_replenishment_rate;
- bid_wall_disappear_flag;
- ask_wall_disappear_flag;
- imbalance_persistence.

Purpose:

Capture whether one side of the book is being consumed or replenished.

---

## 8. Multi-Level Order Book Features

If depth snapshots are available beyond best bid / ask, add these features for top N levels.

Recommended depth levels:

```text
N = 5, 10, 20
```

Required:

- bid_depth_N;
- ask_depth_N;
- depth_imbalance_N;
- weighted_depth_imbalance_N;
- near_mid_depth_imbalance;
- distance_to_large_bid_wall;
- distance_to_large_ask_wall;
- bid_wall_size;
- ask_wall_size;
- order_book_slope_bid;
- order_book_slope_ask;
- book_convexity;
- depth_change_N over 5s / 10s / 30s.

Purpose:

Top-of-book can be noisy. Multi-level book features help identify deeper support / resistance and liquidity asymmetry.

Initial priority:

- implement top-of-book features first;
- add multi-level depth only if reliable depth snapshots are available and storage is manageable.

---

## 9. Liquidity and Market Quality Features

Required:

- spread_bps current;
- spread_bps mean over 10s / 30s / 60s;
- spread_bps max over 10s / 30s / 60s;
- spread volatility;
- quote update intensity;
- trade-to-quote ratio;
- volume per quote update;
- realized spread proxy;
- price impact proxy:

```text
abs(return_w) / dollar_volume_w
```

- illiquidity score;
- market_quality_regime flag.

Purpose:

Avoid over-trusting direction signals during poor liquidity or unstable quote conditions.

---

## 10. Momentum Quality Features

Not all momentum is equally useful. Add features that describe momentum quality.

Required:

- momentum_persistence: share of seconds with same return sign;
- directional_efficiency:

```text
abs(price_t - price_{t-w}) / sum(abs(second_returns))
```

- choppiness score;
- trend consistency over 10s / 30s / 60s;
- pullback depth after recent move;
- continuation after burst flag;
- failed breakout flag;
- compression before breakout flag.

Purpose:

Differentiate clean directional movement from noisy movement that may reverse.

---

## 11. Extreme Move and Shock Features

Required:

- largest 1s return in last 10s / 30s / 60s;
- smallest 1s return in last 10s / 30s / 60s;
- max positive jump;
- max negative jump;
- jump count above threshold;
- time since last positive jump;
- time since last negative jump;
- shock reversal flag;
- shock continuation flag.

Purpose:

Short-horizon markets can be dominated by sudden jumps. These features help identify whether a recent jump is likely to continue or mean-revert.

---

## 12. Time-to-Decision and Time-of-Day Features

For each decision timestamp:

- second within minute;
- minute within hour;
- hour of day;
- trading session flag: Asia / Europe / US;
- weekend flag;
- funding-time proximity;
- major scheduled event proximity if available;
- liquidity session regime.

Purpose:

BTC microstructure can vary by session and time-of-day. Directional features may have different reliability across sessions.

---

## 13. Cross-Market Features

If available, add second-level features from related markets.

### 13.1 BTC spot / futures basis

Required:

- spot return over 5s / 10s / 30s / 60s;
- futures return over 5s / 10s / 30s / 60s;
- futures minus spot return;
- basis change;
- basis z-score;
- basis acceleration.

Purpose:

Capture whether futures are leading spot or vice versa.

### 13.2 Perpetual futures features

Required:

- perp mid return;
- perp book imbalance;
- perp taker imbalance;
- funding rate level;
- funding z-score;
- open interest change, if available;
- long / short liquidation bursts, if available.

Purpose:

Perp market pressure may lead short-horizon spot movement.

### 13.3 ETH and market beta features

Optional:

- ETH return over 5s / 30s / 60s;
- BTC minus ETH return;
- crypto basket return;
- BTC idiosyncratic return after removing ETH / market beta.

Purpose:

Separate BTC-specific pressure from broad crypto market movement.

---

## 14. Polymarket-Specific Features

If Polymarket order book / trade data is available, add market-specific features.

Required if available:

- UP contract best bid;
- UP contract best ask;
- DOWN contract best bid;
- DOWN contract best ask;
- UP / DOWN mid price;
- UP / DOWN spread;
- UP order book imbalance;
- DOWN order book imbalance;
- Polymarket implied probability change over 5s / 10s / 30s;
- Polymarket volume burst;
- Polymarket trade imbalance;
- deviation between model BTC signal and Polymarket price.

Purpose:

Polymarket price may embed crowd information and liquidity constraints. These features can help decide whether the BTC signal is already priced in.

Caution:

Polymarket data must be timestamp-aligned carefully. Do not use any trade or quote after the decision timestamp.

---

## 15. Relative / Normalized Features

Raw values may not be stable across regimes. Add normalized versions.

For each major signal, calculate:

- raw value;
- z-score versus 5m rolling baseline;
- percentile rank versus 1h rolling baseline;
- short-window value minus long-window value;
- short-window value divided by long-window value;
- regime-normalized value.

Examples:

```text
taker_imbalance_10s_z_300s
volume_30s_pct_rank_1h
spread_10s_minus_spread_300s
rv_30s_div_rv_300s
```

Purpose:

A 30s volume burst only matters relative to the current market baseline.

---

## 16. Interaction Features

Add controlled interactions, not unlimited combinations.

Priority interactions:

- taker_imbalance × volatility regime;
- taker_imbalance × spread regime;
- taker_imbalance × trend strength;
- book_imbalance × taker_imbalance;
- microprice_premium × spread_bps;
- signed_flow × book_imbalance;
- return_10s × volume_burst;
- return_30s × choppiness;
- momentum_persistence × volatility regime;
- basis_change × taker_imbalance;
- perp_pressure × spot_pressure.

Purpose:

Many microstructure signals are useful only in specific conditions.

---

## 17. Directional Symmetry Features

Keep one binary model, but include directional transforms.

For a signal `x`, add:

```text
x_pos = max(x, 0)
x_neg = min(x, 0)
abs_x = abs(x)
sign_x = sign(x)
```

Apply to:

- return;
- taker imbalance;
- signed flow;
- book imbalance;
- microprice premium;
- basis change;
- price slope;
- volatility-adjusted return.

Purpose:

Let one binary model learn different UP and DOWN structures without using two independent probability models.

---

## 18. Feature Availability Priority

### Tier 1: Must-have

Use data that is easiest to obtain and most likely useful:

- second-level price returns;
- realized volatility;
- taker buy / sell imbalance;
- signed dollar flow;
- trade intensity;
- volume burst;
- direction flips;
- momentum persistence.

### Tier 2: High priority

Requires reliable bookTicker / top-of-book data:

- spread_bps;
- bid / ask quantity imbalance;
- microprice premium;
- book imbalance change;
- spread widening / tightening;
- quote update intensity.

### Tier 3: Advanced

Requires heavier data or more careful alignment:

- multi-level order book depth;
- liquidity wall features;
- Polymarket order book features;
- cross-exchange basis;
- liquidation bursts;
- open interest change;
- event proximity features.

---

## 19. Data Quality Requirements

For every second-level feature pack, validate:

- timestamp timezone consistency;
- monotonic timestamps;
- no duplicate timestamps after aggregation;
- missing second ratio;
- stale quote ratio;
- trade / quote alignment delay;
- outlier price filter;
- no use of data after decision timestamp;
- feature availability rate by day;
- feature availability rate by validation sample.

If a feature has poor availability or unstable alignment, exclude it from the first baseline.

---

## 20. Aggregation Rules

For each decision timestamp `t` and lookback window `w`:

```text
feature uses data in (t - w, t]
```

Do not use:

```text
[t, t + w]
```

or any data after `t`.

For missing data:

- use NaN indicators where useful;
- avoid forward-filling stale book data for too long;
- cap stale quote forward-fill duration;
- add stale_data_flag if book updates are missing.

Recommended stale thresholds:

```text
bookTicker stale if no update for > 2s
trade data sparse flag if no trade for > 5s
```

---

## 21. Feature Count Control

Second-level features can explode quickly. Start with Tier 1 and selected Tier 2.

Recommended first implementation:

- windows: 5s, 10s, 30s, 60s, 300s;
- feature families: price, volatility, taker flow, trade intensity, top-of-book;
- add normalized versions only for the strongest signals;
- add interactions only after baseline feature importance is reviewed.

Do not add every possible interaction in the first run.

---

## 22. Required Experiment Report

For each feature run, report:

- feature packs enabled;
- number of features;
- feature availability rate;
- training sample count;
- validation sample count;
- coverage;
- precision_up;
- precision_down;
- balanced_precision;
- selected t_up;
- selected t_down;
- accepted-sample accuracy;
- precision versus coverage frontier;
- top feature importance;
- boundary-sliced metrics;
- regime-sliced metrics.

Compare at least:

```text
Run A: minute-level baseline
Run B: + Tier 1 second-level features
Run C: + Tier 2 top-of-book features
Run D: + selected interactions
```

---

## 23. Initial Feature Pack Recommendation

Start with this compact but high-value feature set:

### Price / momentum

- return_5s;
- return_10s;
- return_30s;
- return_60s;
- return_300s;
- slope_10s;
- slope_30s;
- return_10s_minus_60s;
- direction_flips_30s;
- momentum_persistence_30s.

### Volatility / noise

- rv_10s;
- rv_30s;
- rv_60s;
- rv_30s_div_300s;
- range_30s;
- choppiness_30s;
- directional_efficiency_30s.

### Taker flow

- taker_imbalance_5s;
- taker_imbalance_10s;
- taker_imbalance_30s;
- taker_imbalance_60s;
- taker_imbalance_30s_z_300s;
- signed_dollar_flow_10s;
- signed_dollar_flow_30s;
- signed_dollar_flow_60s;
- signed_dollar_flow_30s_z_300s.

### Trade intensity

- trade_count_10s;
- trade_count_30s;
- dollar_volume_10s;
- dollar_volume_30s;
- volume_burst_30s;
- large_trade_share_30s;
- avg_trade_size_30s.

### Top-of-book

- spread_bps;
- spread_bps_mean_30s;
- spread_bps_max_30s;
- bid_ask_qty_imbalance;
- book_imbalance_change_10s;
- book_imbalance_change_30s;
- microprice_premium;
- microprice_drift_10s;
- spread_widening_10s.

### Directional transforms

- positive_taker_imbalance_30s;
- negative_taker_imbalance_30s;
- positive_signed_flow_30s;
- negative_signed_flow_30s;
- positive_microprice_premium;
- negative_microprice_premium;
- bullish_burst_score;
- bearish_burst_score.

This should be the first second-level feature baseline before expanding into heavier order book depth or Polymarket-specific features.

---

## 24. Non-Goals

This feature layer should not:

- create multiple rows for the same decision timestamp;
- copy the same label across every second inside a 5-minute window;
- use data after decision time;
- rely on raw tick data directly as model rows;
- add uncontrolled feature interactions;
- prioritize feature count over timestamp alignment and validation quality.

---

## 25. Final Summary

Second-level features should help the model capture information that 1-minute bars may miss:

- late momentum;
- reversal;
- taker flow pressure;
- liquidity imbalance;
- spread changes;
- volume bursts;
- microprice drift;
- volatility and noise quality;
- regime-dependent signal strength.

The first implementation should stay simple:

```text
one decision timestamp
one feature row
one settlement label
weighted binary model
asymmetric threshold se