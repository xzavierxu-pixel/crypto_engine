# Execution Engine Flow Timeline

This note describes the current live/paper execution path for the BTC 5-minute Polymarket workflow. It is intentionally short and only covers runtime behavior, not training.

## 1. Runtime Role

`execution_engine` is a thin runtime layer:

```text
run_once.py -> BinanceRealtimeClient -> RuntimeInferenceEngine -> build_second_level_feature_store -> sample_second_level_feature_store -> src.features.builder.build_feature_frame -> src.model.infer.predict_frame -> src.signal.policies.evaluate_selective_binary_signal -> PolymarketV2Adapter -> build_two_limit_order_plan
```

It reuses the shared feature builder, model inference path, and signal policy. It should not duplicate BTC feature formulas or label logic.

## 2. Current Configuration Anchors

From `execution_engine/config.example.yaml`:

```yaml
runtime.mode: live
schedule.trigger_delay_seconds: 68
schedule.max_data_wait_seconds: 20
price_estimator.enabled: true
price_estimator.active_artifact: expected_return_h14
orders.enabled: true
orders.first.price_mode: reference_multiplier_offset_and_cap
orders.second.enabled: false
execution_edge.enabled: false
```

From `config/settings.yaml`:

```yaml
decision_alignment.enabled: true
decision_alignment.mode: delayed_feature_offset
decision_alignment.feature_offset_minutes: 1
decision_alignment.row_policy: delayed_1m_synthetic_decision_row
second_level.enabled: true
second_level.feature_profile: expanded_v2
second_level.require_agg_trade_through_last_second: true
second_level.max_agg_trade_lag_seconds: 2.0
```

Current baseline artifact:

```yaml
experiment_id: 20260611_catboost_calendar_coordinate_search
model_plugin: catboost
calibration_plugin: none
feature_count: 569
t_up: 0.5792857142857143
t_down: 0.43142857142857144
validation_selection_score: 0.6413740846017104
validation_coverage: 0.7000535618639528
validation_accepted_sample_accuracy: 0.70734506503443
```

The baseline includes `sl_` features, so second-level 1s and aggTrades inputs are part of the current model input path.

## 3. Market-Time Semantics

For a Polymarket window `[T,T+5m)`:

```text
market_t0 / signal.t0 = T
decision_time = T+1m
feature_timestamp = T+1m
market window = [T,T+5m)
```

The runtime trades the still-open `[T,T+5m)` market after seeing data through the first minute of that market. The synthetic feature row at `T+1m` is a decision anchor, not a real Binance candle.
Runtime does not compute labels. The accepted baseline was trained against the resolved Polymarket BTC up/down label source recorded in the artifact and settings.

## 4. Scheduling

Production timers are aligned to the delayed decision flow:

```text
T+00:23 prewarm runtime data/cache
T+01:08 run execution for market window [T,T+5m)
```

`run_once.py` also waits internally until `current_5m_window_start() + schedule.trigger_delay_seconds` when no explicit `target_window_start` is passed. The systemd timer should still fire near `T+01:08`; a very late service start can still target the current window because there is no explicit max-window-age guard.

## 5. Binance Data Gate

For `feature_offset_minutes=1`, `wait_for_signal_runtime_frames()` requires:

```text
required_latest_closed_minute = T
required_latest_closed_second = T+59s
required_latest_agg_trade = T+57s by default
```

`finalize_runtime_frames_for_signal()` then filters inputs to the decision cutoff:

```text
minute rows: timestamp <= T
second rows: timestamp < T+1m
aggTrades rows: timestamp < T+1m
```

It appends:

```text
timestamp = T+1m
OHLCV = NaN
close_time = NaT
```

Healthy summaries should show:

```text
signal.t0 == T
signal.decision_time == T+1m
signal.feature_timestamp == T+1m
signal.minute_latest == T
signal.second_latest >= T+59s and < T+1m
signal.agg_trade_latest >= T+57s and < T+1m
```

## 6. Feature Build and Inference

Runtime feature building uses the same shared builder as research code:

```text
decision_frame = minute_frame[["timestamp"]]
second_store = build_second_level_feature_store(second_frame, agg_trades_frame, expanded_v2)
sampled_second = sample_second_level_feature_store(decision_frame, second_store)
feature_frame = build_feature_frame(..., second_level_features_frame=sampled_second)
```

`RuntimeInferenceEngine.predict()` selects the row where:

```text
feature_frame.timestamp == runtime_context["feature_timestamp"] == T+1m
```

The resulting `Signal` keeps:

```text
signal.t0 = T
signal.decision_context["market_t0"] = T
signal.decision_context["decision_time"] = T+1m
signal.decision_context["feature_timestamp"] = T+1m
```

## 7. Thresholds and Decisions

Effective thresholds come from config overrides when present, otherwise from the baseline artifact. The current artifact also contains a UTC day/session threshold policy, so summaries should be checked for both effective and artifact thresholds.
Decision rule:

```text
p_up >= t_up -> YES
p_up <= t_down -> NO
otherwise -> NO-SIGNAL
```

`NO-SIGNAL` stops the cycle before market lookup and order planning.

## 8. Price Estimator and Orders

When the classifier emits YES or NO, runtime evaluates the active H14 expected-return estimator for the selected side and records fields such as:

```text
price_estimator_expected_return_bid
price_estimator_expected_return_ev
price_estimator_expected_return_fill_probability
price_estimator_expected_return_eligible
```

Current default order pricing does not directly use that bid. The enabled first leg uses:

```text
price_mode = reference_multiplier_offset_and_cap
raw_price = min(best_bid * reference_multiplier + offset, price_cap)
```

With current defaults:

```text
raw_price = min(best_bid + 0.01, 0.75)
price = floor_to_tick(max(raw_price, 0.10), tick_size)
```

If a leg is changed to `expected_return_optimal_bid`, the order price becomes:

```text
min(best_ask - 0.01, expected_return_bid)
```

and the leg is skipped unless the H14 policy is eligible, the bid is positive, and `best_ask` exists.
Orders are submitted only when:

```text
mode == "live"
orders.enabled == true
```

The idempotency key uses market window start, token id, side, and leg, not `decision_time`.

## 9. Main Timeline Example

For `[12:00:00Z,12:05:00Z)`:

```text
12:00:00Z Market starts. signal.t0 = 12:00:00Z.
12:00:00Z to 12:00:59Z Binance 1m, 1s, and aggTrades for the first market minute arrive.
12:01:08Z Execution timer fires. run_once 
targets the 12:00 window.
Data gate Requires 1m candle T, 1s through T+59s, aggTrades through about T+57s.
Feature phase Appends synthetic row at 12:01:00Z and builds shared 5m features.
Prediction phase Selects feature row timestamp 12:01:00Z and emits p_up.
Decision phase Applies artifact/config thresholds to produce YES, NO, or NO-SIGNAL.
Market/order phase Maps signal.t0 to btc-updown-5m-<12:00 epoch> and plans BUY orders on the selected token.
```

## 10. Audit Checklist

Every live or paper summary should satisfy:

```text
signal.t0 == market.window_start
market.window_end == signal.t0 + 5m
signal.feature_offset_minutes == 1
signal.row_policy == delayed_1m_synthetic_decision_row
signal.decision_time == signal.t0 + 1m
signal.feature_timestamp == signal.t0 + 1m
signal.required_latest_closed_minute == signal.t0
signal.minute_latest == signal.t0
signal.required_latest_closed_second == signal.t0 + 59s
signal.second_latest >= signal.t0 + 59s
signal.second_latest < signal.t0 + 1m
signal.agg_trade_latest >= signal.required_latest_agg_trade
signal.agg_trade_latest < signal.t0 + 1m
```

For submitted or planned orders, also check:

```text
decision.side in {YES, NO}
YES uses yes_token_id
NO uses no_token_id
order side is BUY
price respects tick_size, min_price, max_price, and guards
idempotency key uses market window_start
```

## 11. Known Runtime Risks

There is still no explicit max-window-age guard. A late service start can target a market near settlement.
The synthetic decision row has NaN OHLCV by design. Monitor runtime feature NaN rates, especially for any feature pack that might use the current row without a historical shift.
AggTrades freshness is strict. If aggTrades lag past `max_agg_trade_lag_seconds`, the cycle raises or skips instead of silently degrading the feature set.
Explicit `thresholds.t_up` or `thresholds.t_down` in `execution_engine/config.yaml` override artifact thresholds. Treat threshold overrides as a live experiment.

## 12. Summary

Current runtime semantics:

```text
For each 5-minute market window T:
- execute around T+01:08;
- use closed 1m data through T and second-level data through T+59s;
- build a synthetic decision feature row at T+1m;
- score the 20260611 CatBoost baseline;
- optionally evaluate the H14 expected-return price estimator;
- map signal.t0 back to the [T,T+5m) Polymarket market; and
- submit BUY limit orders only in live mode with orders enabled.
```
