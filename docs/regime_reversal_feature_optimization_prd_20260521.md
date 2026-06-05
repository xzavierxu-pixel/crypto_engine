# Regime and Reversal Feature Optimization PRD - 2026-05-21

## 1. Background

Recent live and replay analysis showed:

```text
1. Current artifact is not directionally worse than the old artifact on the old profitable window.
2. Current thresholds are too conservative for live coverage.
3. 2026-05-20/21 was a worse first-minute-continuation regime.
4. Both old and current artifacts mostly follow the first-minute direction.
5. The largest model-side weakness is not first-minute detection; it is detecting when first-minute continuation is likely vs when the remaining four minutes are likely to reverse.
```

Relevant reports:

```text
docs/replay_old_artifact_on_20260520_window.md
docs/replay_current_artifact_on_20260515_window.md
docs/model_signal_regression_127776a_vs_current_20260521.md
docs/live_reversal_risk_prd_20260521.md
docs/live_profit_window_comparison_20260515_20260521.md
```

This PRD defines the next offline optimization work. It does not claim improvement yet.

## 1.1 Implementation Status Snapshot

Status source:

```text
python scripts/analysis/regime_reversal_prd_audit.py
```

Audit output:

```text
artifacts/data_v2/reports/regime_reversal_prd_audit_20260521.json
```

Overall status:

```text
PARTIAL / NOT COMPLETE
```

Completed:

```text
[DONE] Baseline coverage>=0.90 config exists:
       experiments/configs/20260521_polymarket_resolved_baseline_coverage_090.yaml

[DONE] Regime/reversal feature experiment config exists:
       experiments/configs/20260521_regime_reversal_second_agg_features.yaml

[DONE] Baseline validation report exists and satisfies coverage>=0.90.
       coverage: 0.9027852169
       selection_score: 0.5144593007
       utility: 0.2857525442
       accepted_sample_accuracy: 0.6582616434
       accepted_count: 6742

[DONE] Baseline train report exists and satisfies coverage>=0.90.
       coverage: 0.9093158481
       selection_score: 0.6210403444
       utility: 0.3332920945
       accepted_sample_accuracy: 0.6832653061
       accepted_count: 22050

[DONE] Baseline replay reports exist for the mandatory 2026-05-15/16 and 2026-05-20/21 windows.

[DONE] Baseline reversal/trend diagnostic artifact exists:
       artifacts/data_v2/reports/reversal_diagnostics/baseline_reversal_trend_slices_090.json
```

Incomplete or blocked:

```text
[BLOCKED] Materialized second-level feature store is missing:
          artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT

[TODO] New feature experiment report is missing:
       artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features/report.json

[TODO] New feature replay-window reports are missing:
       artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260515_window.json
       artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260520_window.json

[INCOMPLETE] Baseline replay 2026-05-15/16 exists but does not satisfy coverage>=0.90.
             coverage: 0.8725490196
             accepted_sample_accuracy: 0.7752808989
             selection_score: 1.0848782348
             reversal/trend metrics available: false

[INCOMPLETE] Baseline replay 2026-05-20/21 exists but does not satisfy coverage>=0.90.
             coverage: 0.8971962617
             accepted_sample_accuracy: 0.59375
             selection_score: 0.2786431126
             reversal/trend metrics available: false

[BLOCKED] Local normalized BTCUSDT 1m data does not cover the mandatory replay windows.
          available start: 2025-10-01T00:00:00+00:00
          available end:   2026-05-10T23:55:00+00:00
          required end:    2026-05-21T00:23:40+00:00

[TODO] Feature availability, leakage, and offline-online consistency checks are not complete for the new feature artifact.

[TODO] First-leg execution policy change has not been separately tested as an execution policy.

[TODO] No completed experiment commit hash is recorded for the new feature experiment.
```

Current gating conclusion:

```text
Do not claim regime/reversal feature improvement yet.
The baseline coverage>=0.90 acceptance report is complete.
The new feature artifact, feature replay reports, second-level feature store, and mandatory replay data coverage are not complete.
```

## 1.2 Latest Implementation Result

Status source:

```text
python scripts/analysis/regime_reversal_prd_audit.py
```

Current status:

```text
IMPLEMENTED ARTIFACTS / FINAL REPLAY GATE NOT PASSED
```

Completed implementation artifacts:

```text
[DONE] Local normalized BTCUSDT 1m data now covers mandatory replay windows.
       start: 2025-10-01T00:00:00+00:00
       end:   2026-05-21T00:30:00+00:00

[DONE] Second-level split feature store exists:
       artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT

[DONE] New feature training frame was built with second-level and first-minute impulse features.
       path: artifacts/data_v2/datasets/market=BTCUSDT/horizon=5m/regime_reversal_second_agg_training_frame.parquet
       rows: 31718
       feature_count: 1805
       sl_* feature count: 719
       fm_* feature count: 70
       feature schema QA: passed

[DONE] New feature experiment report exists:
       artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features/report.json

[DONE] Mandatory new feature replay reports exist:
       artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260515_window.json
       artifacts/reports/execution_engine/replay_20260521_regime_reversal_second_agg_features_20260520_window.json
```

Validation result:

```text
Baseline coverage>=0.90 validation:
  coverage:                 0.9027852169
  selection_score:          0.5144593007
  utility:                  0.2857525442
  accepted_sample_accuracy: 0.6582616434
  accepted_count:           6742

New feature validation:
  coverage:                 0.9019817890
  selection_score:          0.5400093403
  utility:                  0.2970005356
  accepted_sample_accuracy: 0.6646377672
  accepted_count:           6736

Validation comparison:
  selection_score_delta:          +0.0255500396
  utility_delta:                  +0.0112479914
  accepted_sample_accuracy_delta: +0.0053761238
  coverage_delta:                 -0.0008034280
  coverage>=0.90:                 yes
```

Mandatory replay result:

```text
2026-05-15/16 baseline replay:
  coverage:                 0.8725490196
  accepted_sample_accuracy: 0.7752808989
  selection_score:          1.0848782348
  coverage>=0.90:           no

2026-05-15/16 new feature replay:
  coverage:                 0.8921568627
  accepted_sample_accuracy: 0.7582417582
  selection_score:          0.9921707777
  coverage>=0.90:           no

2026-05-20/21 baseline replay:
  coverage:                 0.8971962617
  accepted_sample_accuracy: 0.5937500000
  selection_score:          0.2786431126
  coverage>=0.90:           no

2026-05-20/21 new feature replay:
  coverage:                 0.8971962617
  accepted_sample_accuracy: 0.5625000000
  selection_score:          0.1790048145
  coverage>=0.90:           no
```

Final gate conclusion:

```text
Do not claim full PRD acceptance.
The validation artifact improves selection_score while satisfying coverage>=0.90.
The mandatory replay windows are reported, but replay coverage remains below 0.90 and replay rows do not include reversal/trend slice fields.
The experiment is useful as an offline validation improvement, but it is not accepted for deployment under the full PRD replay gate.
```

## 2. Primary Objective

Build feature improvements that help the main binary model distinguish:

```text
trend-following / first-minute-continuation regime
vs
post-first-minute-reversal regime
```

The optimization target changes to:

```text
maximize validation selection_score
subject to coverage >= 0.90
```

This replaces the previous live/offline target of `coverage >= 0.70` for the next experiment family.

Required reporting:

```text
sample_count
coverage
accepted_sample_accuracy
precision_up
precision_down
balanced_precision
all_sample_accuracy
selected_t_up
selected_t_down
accepted_count
up_prediction_count
down_prediction_count
share_up_predictions
share_down_predictions
roc_auc
brier_score
log_loss
utility
downside_risk
selection_score
```

Additional required slicing:

```text
continuation_sample_count
continuation_coverage
continuation_accepted_accuracy
continuation_accepted_count
reversal_sample_count
reversal_coverage
reversal_accepted_accuracy
reversal_accepted_count
reversal_loss_contribution
trend_following_loss_contribution
```

## 3. Non-Goals

Do not add a separate reversal model.

Do not change the label away from:

```text
Polymarket resolved BTC 5m UP/DOWN
```

Do not add future-looking features from after the decision time.

Do not duplicate feature logic in execution.

Do not enable the second execution leg in this experiment.

Do not claim improvement unless the offline replay/report proves:

```text
coverage >= 0.90
selection_score improves vs the current baseline recomputed at coverage >= 0.90
continuation and reversal slice metrics are reported
```

## 4. Current Hypothesis

The current feature set has enough first-minute direction signal, but not enough market-regime signal.

The model needs better features that answer:

```text
1. Is the first-minute move supported by real aggressive flow?
2. Is liquidity following the move or fading it?
3. Is the move a continuation impulse or a local exhaustion move?
4. Is the last part of the first minute already rejecting the move?
5. Is volatility/volume state favorable for follow-through or mean reversion?
```

Second-level features and aggTrade features are likely useful because they can describe the quality of the first-minute move, not just its net return.

## 5. Feature Plan

### 5.1 Enable / Validate Existing Second-Level and AggTrade Inputs

The first priority is not inventing many new features. It is ensuring the existing second-level and aggTrade-derived features are fully enabled, available offline, and available online with the same semantics.

Candidate existing feature families:

```text
second_level_price_slope
second_level_signed_dollar_flow
second_level_volume_burst
price_direction_flips_30s
last_second_reversal_flag
late_window_acceleration_flag
micro_ret_5s / 10s / 30s / 60s
micro_rv_5s / 10s / 30s / 60s
taker_imbalance
taker_imbalance_slope
prev_bar_taker_imbalance
prev_bar_taker_buy_ratio
```

Acceptance requirements:

```text
offline feature availability: pass
online feature availability: pass
offline-online consistency: pass
no missing feature columns in runtime artifact
no future timestamps after T0+1m
```

### 5.2 Add First-Minute Impulse Quality Features

These features should be built from data available from `T0` through `T0+1m`, evaluated at the synthetic decision row `T0+1m`.

Price path:

```text
fm_ret
fm_abs_ret
fm_close_location_in_1m_range
fm_upper_wick_ratio
fm_lower_wick_ratio
fm_body_to_range
fm_last_10s_ret
fm_last_15s_ret
fm_last_10s_opposes_ret
fm_last_15s_opposes_ret
fm_intraminute_max_adverse_move
fm_intraminute_max_favorable_move
fm_stall_after_impulse
```

Flow:

```text
fm_taker_buy_ratio
fm_taker_sell_ratio
fm_taker_imbalance
fm_taker_imbalance_last_10s
fm_taker_imbalance_slope_10s
fm_signed_dollar_flow
fm_signed_dollar_flow_last_10s
fm_trade_count_z
fm_volume_z
fm_large_trade_share
fm_aggressive_flow_confirms_price
fm_aggressive_flow_diverges_from_price
```

Book / microstructure:

```text
fm_book_imbalance_mean
fm_book_imbalance_last
fm_book_imbalance_slope
fm_microprice_premium_mean
fm_microprice_premium_last
fm_microprice_divergence_from_ret
fm_spread_bps_mean
fm_spread_bps_max
fm_depth_thinning
fm_bid_replenishment_after_up_move
fm_ask_replenishment_after_down_move
```

Regime context:

```text
fm_ret_vs_rv_30
fm_ret_vs_prev_5m_range
fm_ret_vs_htf_rv_15m
fm_volume_climax_z
fm_range_breakout_distance
fm_prior_trend_alignment
fm_prior_chop_score
fm_continuation_pressure_score
fm_reversal_pressure_score
```

These are features, not a separate reversal head. They should be consumed by the same main binary model.

## 6. Reversal and Trend Slice Definitions

For diagnostics only:

```text
first_minute_side = YES if ret_1 >= 0 else NO
resolved_side = Polymarket resolved YES/NO
trend_following = first_minute_side == resolved_side
post_first_minute_reversal = first_minute_side != resolved_side
```

These diagnostic labels must not become the primary training label.

For each experiment, report:

```text
overall accepted accuracy
overall coverage
trend_following accepted accuracy
trend_following coverage
post_first_minute_reversal accepted accuracy
post_first_minute_reversal coverage
accepted count by bucket
false positives by bucket
confidence bucket metrics
side bucket metrics
hour bucket metrics
```

Required comparison:

```text
baseline at coverage >= 0.90
new feature artifact at coverage >= 0.90
```

If baseline cannot satisfy `coverage >= 0.90` with positive utility, that must be stated explicitly.

## 7. Offline Experiment Design

### 7.1 Baseline Recompute

Before testing new features, recompute the current accepted artifact under the new coverage constraint:

```yaml
objective:
  min_coverage: 0.90

threshold_search:
  hard_constraint: coverage_only
```

Output:

```text
baseline_coverage_090_report.json
baseline_reversal_trend_slices_090.json
```

This becomes the correct comparison baseline for the new objective.

### 7.2 Feature Experiment

Create a new experiment config:

```text
experiments/configs/20260521_regime_reversal_second_agg_features.yaml
```

Experiment variants:

```text
A: current feature set, coverage >= 0.90
B: enable/validate existing second-level + aggTrade families, coverage >= 0.90
C: B + first-minute impulse quality features, coverage >= 0.90
D: C + feature selection / regularization if overfitting is observed
```

Primary acceptance:

```text
validation selection_score improves vs A
coverage >= 0.90
utility > 0
accepted_sample_accuracy > 0.50
```

Secondary acceptance:

```text
post_first_minute_reversal accepted accuracy improves
trend_following accepted accuracy does not materially degrade
confidence bucket monotonicity improves
live-window replay improves on 2026-05-15/16 and 2026-05-20/21
```

### 7.3 Replay Windows

Mandatory replay windows:

```text
2026-05-15T15:00:00Z to 2026-05-16T01:00:00Z
2026-05-20T15:23:40Z to 2026-05-21T00:23:40Z
```

Optional future rolling windows:

```text
last 24h resolved
last 72h resolved
hourly rolling slices
```

## 8. Execution Policy Changes

### 8.1 First Leg

Change first-leg price policy to:

```text
first_leg_price = min(best_bid - 0.05, 0.65)
```

This must be config-driven:

```yaml
orders:
  first:
    enabled: true
    price_mode: min_best_bid_offset_and_cap
    best_bid_offset: -0.05
    price_cap: 0.65
```

Rationale:

```text
1. Avoid paying overly high prices.
2. Still loosen the first leg compared with overly conservative low fixed prices.
3. Keep the order maker-oriented when best_bid is high.
```

This is an execution change and must be evaluated separately from model accuracy.

### 8.2 Second Leg

Keep second leg disabled:

```yaml
orders:
  second:
    enabled: false
```

Rationale:

```text
1. Previous second leg had positive optionality, but it confounds model evaluation.
2. Re-enable only after model/regime feature evaluation is complete.
3. Keep live blast radius small.
```

## 9. Files Expected To Change

Likely implementation files:

```text
src/data/second_level_features.py
src/features/prd_microstructure.py
src/features/registry.py
src/core/config.py
src/data/dataset_builder.py
scripts/model/train_model.py
scripts/analysis/reversal_diagnostics.py
execution_engine/config.py
execution_engine/order_plan.py
tests/
experiments/configs/20260521_regime_reversal_second_agg_features.yaml
```

Reports:

```text
artifacts/data_v2/experiments/<run_id>/report.json
artifacts/data_v2/experiments/<run_id>/feature_importance.csv
artifacts/data_v2/reports/reversal_diagnostics/<run_id>.json
artifacts/reports/execution_engine/replay_<run_id>_20260515_window.json
artifacts/reports/execution_engine/replay_<run_id>_20260520_window.json
```

## 10. Risks

### 10.1 Coverage 0.90 May Force Lower Accuracy

Moving from coverage 0.70 to 0.90 is a major objective change. Expected accepted accuracy will likely fall. This is acceptable only if utility and selection_score improve under the new objective.

### 10.2 Second-Level/AggTrade Online Completeness

If aggTrade or second-level data is incomplete online, inference must fail closed or avoid using those features.

Required guard:

```text
require_agg_trade_through_last_second: true
max_agg_trade_lag_seconds configured
online feature completeness audit included in summary
```

### 10.3 Feature Overfitting

First-minute impulse features can overfit recent live windows. Validation must use chronological splits and replay multiple windows.

### 10.4 Polymarket-vs-Binance Mismatch

Diagnostic BTC direction can differ from Polymarket resolved outcome. Polymarket resolved label remains primary.

## 11. Definition Of Done

An experiment is complete only when:

```text
1. exact config is saved
2. coverage >= 0.90 is enforced
3. report includes required objective metrics
4. trend-following and reversal slice metrics are included
5. feature availability and leakage checks pass
6. offline-online consistency checks pass
7. replay windows are reported
8. before/after metrics are compared against the coverage>=0.90 baseline
9. no second leg is enabled
10. first-leg price rule is separately tested as execution policy
```

## 12. Recommended Implementation Order

```text
[DONE] Step 1: Recompute current baseline with coverage >= 0.90.

[DONE] Step 2: Add baseline reversal/trend diagnostic report.

[DONE] Step 3: Enable and validate existing second-level + aggTrade feature families.
       Materialized second-level feature store exists.

[DONE] Step 4: Add first-minute impulse quality features.

[DONE] Step 5: Train and validate feature artifact.

[PARTIAL] Step 6: Replay old and bad live windows.
          Baseline and new feature replay reports exist.
          Both mandatory replay windows remain below coverage>=0.90 and lack reversal/trend metrics.

[DONE] Step 7: Only if metrics improve, prepare execution config change:
        first_leg_price = min(best_bid - 0.05, 0.65)
        second_leg.enabled = false
```

Current recommendation:

```text
Do not revert the model.
Do not add a separate reversal model.
Use richer online-available microstructure features to let the main model learn continuation vs reversal regimes.
Unify the next offline objective at coverage >= 0.90.
```
