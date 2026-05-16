# PRD: T+1 Delayed Feature Decision For BTC 5m Direction

## 2026-05-16 Revision: T+2 Deployment

The implementation has been advanced one additional minute beyond the original T+1 convention.
The live and offline convention is now:

```text
market_t0 / Polymarket window start = T
target window                       = [T, T+5m)
label                               = 1{ close[T+4m] >= open[T] }
feature_timestamp                   = T+2m
decision_time                       = T+2m
required_latest_closed_minute       = T+1m
row_policy                          = delayed_2m_synthetic_decision_row
```

All requirements below still apply, but every `T+1` feature/decision reference is superseded by `T+2` for the current deployed system.

## Summary

Change the BTC/USDT 5-minute direction system from deciding at the market window start `T` to deciding at `T+1m`, while still predicting the same Polymarket market window `[T, T+5m)`.

The label remains unchanged:

```text
y = 1{ close[T+4m] >= open[T] }
```

The model feature row changes to:

```text
market_t0 / target window start = T
target window                  = [T, T+5m)
label                          = 1{ close[T+4m] >= open[T] }
feature_timestamp              = T+1m
feature visible data           = data closed/landed before T+1m
online order timing            = after T+1m data is available
```

This deliberately allows the model to use the first completed 1-minute candle inside the target market window. The execution engine must therefore order one minute later and must record the delayed-entry convention explicitly.

## Objectives

Primary offline objective:

```text
maximize validation selection_score
subject to validation coverage >= 0.70
```

Final deployment objective:

1. Prove the T+1 delayed-decision model improves validation `selection_score` under `coverage >= 0.70`.
2. Preserve the current label semantics exactly.
3. Deploy an online model trained with all available data through `2026-05-10`.
4. Update `execution_engine` and `aws-poly` deployment so live signals use the T+1 convention.

## Data Requirements

Raw data must be updated through:

```text
2026-05-10
```

This includes all enabled input sources needed by the selected feature set:

```text
1m BTC/USDT OHLCV
1s data / aggTrades if second-level features are enabled
derivatives inputs if enabled in the experiment config
materialized second-level feature stores if used
```

Validation for model selection must use the most recent 1 month ending at `2026-05-10`.

Expected split:

```text
validation_end   = 2026-05-10 23:55:00 UTC, or the latest complete 5m grid before data end
validation_start = validation_end - approximately 30 days
development      = all eligible data before validation_start
```

The exact timestamps must be written into the experiment report:

```text
train_window.start
train_window.end
validation_window.start
validation_window.end
```

## Online Model Training Requirement

After a validation-winning configuration is selected, train a separate online deployment model using all eligible data through `2026-05-10`.

This online model is not the validation acceptance artifact. It is the production artifact trained after selection so it can use the most recent available data.

Required artifacts:

```text
validation_candidate_config
validation_candidate_report
online_full_train_config
online_full_train_report
online_model_artifact
thresholds used by online artifact
```

The online full-train config must preserve the selected feature set, label rule, T+1 alignment, model family/settings, calibration policy, and threshold policy unless a change is explicitly documented.

If thresholds are selected on validation, the online artifact should carry those selected thresholds unless a separate documented threshold policy is approved. Thresholds must not be hardcoded in execution code.

## Time Semantics

The system must distinguish these timestamps:

```text
market_t0          = T, the Polymarket 5m market/window start
market_window_end  = T+5m
label_timestamp    = T
feature_timestamp  = T+1m
decision_time      = T+1m
order_time_policy  = delayed_1m_after_market_open
```

`Signal.t0` must continue to represent `market_t0`, not `feature_timestamp`, because Polymarket slug mapping uses the market start.

The signal/audit context must include:

```text
market_t0
feature_timestamp
decision_time
row_policy
required_latest_closed_minute
minute_latest
market_window_start
market_window_end
```

## Offline Dataset Alignment

Current same-timestamp feature/label merging must not be reused unchanged for this mode.

For each market window start `T`:

```text
label row timestamp    = T
feature row timestamp  = T+1m
target                 = 1{ close[T+4m] >= open[T] }
```

Required offline build flow:

```text
1. Build feature_frame with select_grid_only=false.
2. Build label_frame on 5m grid rows.
3. For each label row with market_t0=T, compute feature_timestamp=T+1m.
4. Join features from feature_frame.timestamp == feature_timestamp.
5. Keep market_t0 and feature_timestamp as explicit columns.
6. Drop rows with incomplete features or missing labels.
7. Exclude raw OHLCV, future columns, target-derived columns, and metadata from feature_columns.
```

Example:

```text
market_t0         = 2026-05-10T12:00:00Z
feature_timestamp = 2026-05-10T12:01:00Z
label             = close[12:04] >= open[12:00]

Feature row 12:01 may use:
  closed 1m candle at 12:00
  second-level / agg data before 12:01

Feature row 12:01 must not use:
  12:01 candle OHLCV
  data at or after 12:01 unless explicitly safe and already landed before decision
  label-derived values
```

## Online Execution Alignment

For live execution:

```text
market_t0     = current_5m_window_start()
decision_time = market_t0 + 1 minute
```

Runtime data requirements:

```text
required_latest_closed_minute = decision_time - 1m = market_t0
required_latest_closed_second = decision_time - 1s
required_latest_agg_trade     = decision_time - 1s - max_agg_trade_lag_seconds
```

Runtime frame policy:

```text
safe_minute = minute rows with timestamp <= market_t0
safe_second = second rows with timestamp < decision_time
safe_agg    = agg trades with timestamp < decision_time
```

Then append a synthetic decision row:

```text
timestamp  = decision_time
OHLCV      = NaN
close_time = NaT
```

Model inference must select:

```text
feature_frame.timestamp == decision_time
```

But market mapping must use:

```text
Polymarket slug/window_start == market_t0
```

Required row policy:

```text
delayed_1m_synthetic_decision_row
```

## Configuration Requirements

All business parameters must live in unified config or execution config, not hardcoded.

Required config shape:

```yaml
objective:
  label: settlement_direction
  optimize_metric: selection_score
  min_coverage: 0.70

decision_alignment:
  enabled: true
  mode: delayed_market_entry
  feature_offset_minutes: 1
  order_delay_seconds_after_feature_time: 5
  row_policy: delayed_1m_synthetic_decision_row

validation:
  mode: chronological_validation
  validation_end: "2026-05-10 23:55:00"
  validation_days: 30

threshold_search:
  hard_constraint: coverage_only
```

If the current config dataclasses do not support these fields, add them conservatively and update tests.

## Evaluation Requirements

Every validation run must report the existing required metrics:

```text
sample_count
coverage
precision_up
precision_down
balanced_precision
all_sample_accuracy
accepted_sample_accuracy
share_up_predictions
share_down_predictions
selected_t_up
selected_t_down
accepted_count
up_prediction_count
down_prediction_count
roc_auc
brier_score
log_loss
utility
downside_risk
selection_score
up_signal_count
down_signal_count
total_signal_count
signal_coverage
overall_signal_accuracy
```

Reports must also include:

```text
decision_alignment_mode
feature_offset_minutes
coverage_constraint_min
coverage_constraint_satisfied
market_t0_start
market_t0_end
feature_timestamp_start
feature_timestamp_end
validation_threshold_tuned = true
validation_result_optimistic = true
```

Acceptance comparison must include:

```text
baseline T decision:
  validation selection_score
  utility
  accepted_sample_accuracy
  coverage
  accepted_count

new T+1 delayed decision:
  validation selection_score
  utility
  accepted_sample_accuracy
  coverage
  accepted_count
```

Pass condition:

```text
validation coverage >= 0.70
validation selection_score > current baseline under the same validation protocol
utility > 0
accepted_sample_accuracy > 0.50
```

If selection_score does not improve under `coverage >= 0.70`, the experiment is not accepted for deployment.

## Leakage Checks

Before claiming improvement, verify:

```text
no future OHLCV columns in feature_columns
no target/future_close/abs_return/signed_return/stage1_target/stage2_target in feature_columns
feature row T+1 does not use T+1 candle OHLCV
feature row T+1 may use T candle only if T candle is closed before decision_time
second-level features sampled at T+1 use only rows before T+1
derivatives joins use backward/asof data available before decision_time
scalers, imputers, calibrators, selectors are fitted only on development during validation
offline and online feature builders remain shared
execution layer does not recompute BTC features
```

## Testing Requirements

Add or update tests for:

```text
1. Label unchanged:
   market_t0=T label remains close[T+4m] >= open[T].

2. Delayed dataset alignment:
   market_t0=T uses feature_timestamp=T+1m.

3. Feature visibility:
   feature row T+1 can use closed T candle,
   but must not use T+1 candle OHLCV.

4. Runtime alignment:
   market_t0=T,
   decision_time=T+1m,
   required_latest_closed_minute=T,
   synthetic row timestamp=T+1m.

5. Polymarket mapping:
   slug/window_start uses market_t0=T, not feature_timestamp.

6. Summary/audit:
   includes market_t0, feature_timestamp, decision_time, row_policy.

7. Threshold search:
   rejects candidates with coverage < 0.70.

8. Train/live parity:
   offline delayed feature row T+1 equals runtime feature row T+1 for the same safe input history.

9. Online full-train artifact:
   can be loaded by execution_engine and carries selected thresholds.
```

## Experiment Reproducibility

Each completed experiment must save:

```text
exact config file
training/evaluation report
feature set
thresholds
split dates
model settings
decision alignment settings
git commit hash
```

Do not overwrite the only copy of an experiment config.

Required final experiment summary:

```text
git_commit
validation_candidate_config_path
validation_candidate_report_path
online_full_train_config_path
online_full_train_report_path
online_artifact_path
primary_metric
validation_selection_score
validation_coverage
coverage_constraint_satisfied
signal_coverage
aws_poly_deployment_status
```

## AWS Deployment Requirements

On `aws-poly`:

```text
1. Pull the deployment commit.
2. Install/update dependencies if needed.
3. Copy or build the online full-train artifact trained through 2026-05-10.
4. Point execution config to the new artifact and settings.
5. Run one dry-run or paper smoke test for a target window.
6. Verify summary fields:
   row_policy == delayed_1m_synthetic_decision_row
   signal.t0 == market.window_start
   feature_timestamp == signal.t0 + 1 minute
   decision_time == signal.t0 + 1 minute
   required_latest_closed_minute == signal.t0
   minute_latest == signal.t0
7. Enable/restart the systemd timer.
8. Verify timer fires after market_t0+1m, not at market_t0.
9. Confirm audit logs and summary JSON are written.
```

Timer policy:

```text
run after T+1m plus a small data-settlement buffer
recommended first attempt: T+1m+5s to T+1m+15s
```

If Binance data is not consistently available by then, increase the buffer in execution config rather than changing model semantics.

## Definition Of Done

The change is complete only when:

```text
1. Raw data is updated through 2026-05-10.
2. Delayed T+1 offline training/evaluation is implemented.
3. Label remains close[T+4m] >= open[T].
4. Validation uses the most recent 1 month ending 2026-05-10.
5. Validation selection_score improves vs the current T-decision baseline.
6. Validation coverage >= 0.70.
7. Online full-train model is trained on all eligible data through 2026-05-10.
8. Thresholds come from config or artifact, not hardcoded.
9. Execution engine uses delayed T+1 feature row.
10. Polymarket market mapping still uses market_t0=T.
11. Tests pass.
12. Leakage checks pass.
13. aws-poly is deployed with the online full-train artifact.
14. aws-poly systemd timer is enabled and verified.
15. Final report states whether selection_score truly improved under coverage >= 0.70.
```
