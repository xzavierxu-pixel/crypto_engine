# Live Reversal Risk Optimization PRD - 2026-05-21

## 1. Background

This PRD is based on the live replay report:

```text
source: artifacts/reports/execution_engine/live_loss_analysis_last_9h.json
window_start_utc: 2026-05-20T15:23:40+00:00
window_end_utc: 2026-05-21T00:23:40+00:00
cycle_count: 107
resolved_cycle_count: 106
accepted_resolved_count: 71
abstain_count: 35
submitted_resolved_cycles: 71
artifact: artifacts/data_v2/experiments/20260520_polymarket_resolved_extended_history_baseline
```

The server timers have been stopped and disabled. This document is for product and experiment design only. The 9-hour live window must not be treated as enough evidence to deploy a fitted rule.

## 2. Problem Statement

The last 9 hours were not a total model failure. The problem was expected value after market price, combined with first-minute reversal risk:

```text
accepted fills: 71
wins / losses: 45 / 26
accuracy: 63.38%
avg_price: 0.6392
estimated pnl: -1.90
roi_on_cost: -0.84%
```

The model nearly always followed the first-minute move:

```text
model side agrees with first-minute side: 70 / 71 = 98.59%
first-minute continuation windows: 45
first-minute reversal windows: 26
```

Performance by whether the first-minute side continued into the 5-minute BTC direction:

```text
continuation:
  count: 45
  accuracy: 95.56%
  pnl: +67.45
  roi: +45.71%

reversal:
  count: 26
  accuracy: 7.69%
  pnl: -69.35
  roi: -87.40%
```

The highest-value problem is not detecting the first-minute direction. The system already does that. The missing capability is detecting when a strong first-minute move is likely to fail during the remaining four minutes.

## 3. Goal

Reduce loss caused by post-first-minute reversals without changing:

```text
label: Polymarket resolved BTC 5m UP/DOWN
decision timing: delayed T0+1m synthetic decision row
feature semantics: only information available at decision time
```

Primary optimization target remains:

```text
maximize validation selection_score
subject to coverage >= 0.70
```

Every candidate must report:

```text
coverage
accepted_sample_accuracy
utility
downside_risk
selection_score
accepted_count
up_prediction_count
down_prediction_count
share_up_predictions
share_down_predictions
```

## 4. Non-Goals

- Do not switch the target to Binance 5m OHLCV direction.
- Do not hardcode thresholds or filters in execution code.
- Do not deploy a rule fitted only on the last 9 live hours.
- Do not reimplement feature logic in the execution layer.
- Do not add future-visible features, such as T0+2m or T0+3m prices.

## 5. Are Current Features Enough?

Current answer: partially enough for the next experiment, not enough to claim a deployable reversal detector.

The active split-validation artifact has 1016 features. It already includes several signals that may help detect reversal risk:

```text
ret_1
ret_1_rolling_z_12 / 24
ret_vol_ratio__ret_1__rv_3 / 5 / 10 / 30
ret_vol_product__ret_1__rv_3 / 10 / 30
second_level_price_slope
second_level_signed_dollar_flow
second_level_volume_burst
price_direction_flips_30s
last_second_reversal_flag
late_window_acceleration_flag
taker_imbalance
taker_imbalance_slope
prev_bar_taker_imbalance
wick_pressure_1
momentum_reversal
```

Limitations:

1. Most features describe existing momentum, volatility, flow, or candle state. They do not explicitly model first-minute failure risk.
2. Feature importance is still dominated by `ret_1` and `ret_1` interactions, so live behavior is close to a first-minute follower.
3. Reversal is a conditional event: `P(final_side != first_minute_side | features at T0+1m)`, not just `P(UP)`.
4. The live report proves a failure mode exists, but does not prove that a stable feature rule has been found.

Recommendation: first train and evaluate a reversal-risk layer using existing features. Add new features only if this does not improve validation selection_score under the coverage constraint.

## 6. Optimization Options

### 6.1 Reversal Diagnostics Report

Add an offline report that splits validation samples into:

```text
first_minute_side = sign(ret_1)
final_side = Polymarket resolved side
reversal = final_side != first_minute_side
continuation = final_side == first_minute_side
```

Required outputs:

```text
reversal_rate
continuation_accuracy
reversal_accuracy
pnl_by_reversal_flag
selection_score_by_bucket
ret_1_abs_bucket
confidence_bucket
price_bucket
volatility_bucket
taker_imbalance_bucket
second_level_price_slope_bucket
```

Acceptance:

```text
report_path: artifacts/data_v2/reports/reversal_diagnostics/<run_id>.json
validation metrics included
live replay diagnostics included
no feature, label, or timestamp semantics changed
```

### 6.2 Two-Stage Reversal Risk Model

Train a separate risk head with target:

```text
target_reversal = 1{resolved_side != first_minute_side}
```

Input:

```text
same feature columns available to the main model at T0+1m
```

Runtime behavior:

```text
main model predicts UP/DOWN
risk head predicts reversal_risk = P(main side will be reversed)
trade only if reversal_risk <= configured_max_reversal_risk
```

Config:

```yaml
execution_filters:
  reversal_risk:
    enabled: false
    artifact_dir: null
    max_risk: null
```

Acceptance:

```text
validation coverage >= 0.70
validation selection_score > 0.5748509217
utility > 0
accepted_sample_accuracy > 0.50
live replay shows lower reversal-loss contribution
filter defaults to disabled
```

### 6.3 Ret_1 Dependency Ablation

Run ablations instead of directly removing `ret_1`:

```text
A: current baseline
B: remove raw ret_1 only
C: remove ret_1 family interactions
D: keep ret_1 but increase model regularization if supported
E: keep baseline model and add two-stage reversal-risk filter
```

Compare:

```text
validation selection_score
coverage
accepted_sample_accuracy
reversal bucket accuracy
continuation bucket accuracy
feature importance share of ret_1 family
```

Priority: E, then D, then B, then C. Directly removing `ret_1` is risky because first-minute continuation is a real profitable regime.

### 6.4 Confidence Bucket Guardrail

The 9-hour report showed a weak confidence bucket:

```text
0.50-0.60: count 4,  accuracy 75.00%, pnl +5.15
0.60-0.70: count 31, accuracy 61.29%, pnl +2.15
0.70-0.80: count 19, accuracy 47.37%, pnl -18.15
0.80-0.90: count 16, accuracy 81.25%, pnl +7.70
0.90-1.01: count 1,  accuracy 100.00%, pnl +1.25
```

Do not deploy a hard ban on `0.70-0.80` based only on this sample. Use it as a calibration diagnostic:

```text
calibration_by_confidence_bucket
expected_value_by_confidence_and_price
non_monotonic_confidence_alert
```

If validation and rolling time splits confirm it, expose a config-only guardrail:

```yaml
execution_filters:
  confidence_bucket:
    enabled: false
    blocked_ranges: []
```

### 6.5 Edge Filter

The average fill price was 0.6392, so the break-even win rate was about 63.92%. Actual win rate was 63.38%.

Add a disabled-by-default config filter:

```yaml
execution_filters:
  min_edge:
    enabled: false
    min_edge: null
```

Definitions:

```text
YES edge = p_up - ask_price
NO edge = (1 - p_up) - ask_price
```

This should be validated together with the reversal-risk filter. It must not reduce validation coverage below 0.70.

## 7. How To Capture Post-First-Minute Reversals

The key question is the quality of the first-minute impulse:

```text
1. Was the move supported by real taker flow?
2. Did the last part of the first minute already stall or reject?
3. Did order book pressure diverge from price direction?
4. Was volatility high enough for mean reversion?
5. Did the move happen near prior range boundaries or wick rejection zones?
```

Validate these existing features first:

```text
ret_1 magnitude and z-score
ret_1 / rv_x and ret_1 * rv_x
taker_imbalance and taker_imbalance_slope
second_level_signed_dollar_flow
second_level_volume_burst
price_direction_flips_30s
last_second_reversal_flag
late_window_acceleration_flag
wick_pressure_1
range_pos_x
efficiency_x
```

If current features are not enough, add online-available features built only from T0 through T0+1m:

```text
first_minute_ret_abs
first_minute_ret_vs_prev_5m_range
first_minute_close_location_in_1m_range
first_minute_upper_wick_ratio
first_minute_lower_wick_ratio
first_minute_taker_buy_ratio
first_minute_taker_imbalance_slope_10s
first_minute_last_10s_ret
first_minute_last_10s_ret_opposes_ret_1
first_minute_microprice_divergence
first_minute_book_imbalance_divergence
first_minute_volume_climax_z
first_minute_price_stall_after_impulse
```

Any new feature must be built through the shared feature pipeline and must pass offline-online consistency tests.

## 8. Implementation Plan

### Phase 1: Diagnostics

Files:

```text
scripts/analysis/reversal_diagnostics.py
artifacts/data_v2/reports/reversal_diagnostics/<run_id>.json
docs/live_reversal_risk_prd_20260521.md
```

Acceptance:

```text
validation and live replay both produce reversal / continuation splits
required metrics are included
no training or execution behavior changes
```

### Phase 2: Reversal Risk Head

Files:

```text
src/models/plugins/reversal_risk_*.py
scripts/model/train_reversal_risk.py
execution_engine/config.py
execution_engine/run_once.py
tests/
```

Acceptance:

```text
risk artifact can be saved and loaded
execution only reads artifact and config
filter defaults to disabled
validation coverage >= 0.70
selection_score beats 0.5748509217 before enabling
```

### Phase 3: Optional New Reversal Features

Files:

```text
src/features/
src/data/
experiments/configs/<run_id>_reversal_features.yaml
tests/test_*feature*
```

Acceptance:

```text
online availability test passes
offline-online feature consistency test passes
no future information
ablation beats current baseline
```

### Phase 4: Limited Live Observation

Requirements:

```text
paper or minimal-size live mode
second leg remains disabled by default
reversal filter enabled only if offline validation passes
kill switch documented
monitor reports reversal risk and reject reason
```

## 9. Risks

1. The 9-hour live sample is small and can overfit easily.
2. Polymarket resolved outcomes can differ from Binance 5m direction, so Polymarket labels remain primary.
3. Reversal filtering may reduce coverage below 0.70.
4. If thresholds are tuned on the same validation set, report metrics must be marked as threshold-tuned and optimistic.
5. New second-level or aggTrade features must fail closed when online data completeness cannot be proven.

## 10. Recommended Next Step

Build Phase 1 and Phase 2 offline:

```text
1. Generate validation reversal diagnostics.
2. Train a reversal_risk head using the existing 1016 features.
3. Search max_reversal_risk with coverage >= 0.70 as a hard constraint.
4. Compare selection_score, utility, accepted_sample_accuracy, and reversal-loss contribution against the baseline.
5. Deploy only as a disabled-by-default execution filter if validation improves.
```

Current judgement: existing features are enough to start a reversal-risk experiment. They are not enough to justify directly deploying a stable live reversal filter.
