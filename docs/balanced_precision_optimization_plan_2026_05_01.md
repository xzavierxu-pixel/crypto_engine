# BTC 5m Balanced Precision Optimization Plan

## Context

This document summarizes the first weighted binary selective-direction run and defines a systematic optimization plan for improving:

```text
balanced_precision = (precision_up + precision_down) / 2
subject to coverage >= 0.60
```

Run analyzed:

```text
artifacts/experiments/btc5m_weighted_binary_20260401_run1
```

Data used by that run:

```text
spot source:
  artifacts/tmp/binance_public_phase_bc_light_smoke/normalized/spot/klines/BTCUSDT-1m.parquet

derivatives:
  artifacts/data/derivatives/binance_btcusdt_funding.parquet
  artifacts/data/derivatives/binance_btcusdt_basis.parquet

train:
  2026-01-31 23:55 UTC to 2026-03-02 23:45 UTC
  8129 grid rows

validation:
  2026-03-02 23:55 UTC to 2026-04-01 23:55 UTC
  8162 grid rows

feature count:
  957
```

Important limitation:

```text
The local input only reaches 2026-04-01 23:59 UTC.
For a true current 30d/30d run as of 2026-05-01, data after 2026-04-01 must be downloaded and normalized first.
```

## Current Result

Selected thresholds:

```text
t_up:   0.5000
t_down: 0.4200
```

Validation metrics:

```text
coverage:             0.6011
precision_up:         0.5477
precision_down:       0.6378
balanced_precision:   0.5927
accepted_accuracy:    0.5642
all_sample_accuracy:  0.5474
ROC AUC:              0.5711
Brier score:          0.2461
log loss:             0.6852
accepted_count:       4906 / 8162
UP accepted share:    0.8166
DOWN accepted share:  0.1834
```

Train metrics:

```text
balanced_precision:   0.7732
ROC AUC:              0.7615
coverage:             0.6308
accepted_accuracy:    0.7153
```

Interpretation:

```text
The model has directional signal, but generalization is weak.
The train-valid gap is large.
The validation ROC AUC is only 0.5711.
The selected threshold is barely satisfying the 60% coverage constraint.
```

## Frontier Diagnosis

Best threshold set under the hard requirement:

```text
t_up=0.500
t_down=0.420
coverage=0.6011
precision_up=0.5477
precision_down=0.6378
balanced_precision=0.5927
```

Best lower-coverage regions show stronger precision:

```text
coverage 0.00-0.20:
  t_up=0.585
  t_down=0.400
  coverage=0.1382
  balanced_precision=0.6343

coverage 0.20-0.40:
  t_up=0.565
  t_down=0.400
  coverage=0.2002
  balanced_precision=0.6207

coverage 0.40-0.60:
  t_up=0.525
  t_down=0.400
  coverage=0.4081
  balanced_precision=0.6144

coverage 0.60-0.80:
  t_up=0.500
  t_down=0.420
  coverage=0.6011
  balanced_precision=0.5927

coverage 0.80-1.00:
  t_up=0.500
  t_down=0.470
  coverage=0.8246
  balanced_precision=0.5650
```

Interpretation:

```text
The model ranks some high-confidence samples well, but not enough samples maintain precision at coverage >= 60%.
Improving balanced_precision requires improving score ranking quality, not just threshold search.
```

## Probability Ranking Diagnosis

Validation probability deciles:

```text
lowest decile UP rate: 36.2%
top decile UP rate:    57.5%
```

The top decile is directionally better than random, but not strong enough. For materially better selective precision, the top decile UP rate should move closer to or above 60%.

## Boundary Slice Diagnosis

Validation by realized absolute return:

```text
abs_return < 1bp:
  sample share: roughly 8.1%
  balanced_precision: 0.4869

1bp <= abs_return < 5bp:
  sample share: roughly 31.3%
  balanced_precision: 0.5321

abs_return >= 5bp:
  sample share: roughly 60.5%
  balanced_precision: 0.6350
```

Interpretation:

```text
The model works much better on meaningful moves.
Near-boundary samples remain noisy and drag full validation metrics down.
They must remain in validation, so optimization should reduce their influence in training and improve abstention around them.
```

## Regime Slice Diagnosis

Regime slices show no single catastrophic regime, but balanced precision varies:

```text
volatility low/mid/high:
  roughly 0.586 to 0.597

trend low/mid/high:
  roughly 0.583 to 0.608

volume low/mid/high:
  roughly 0.577 to 0.610

session asia/europe/us:
  roughly 0.587 to 0.598
```

Interpretation:

```text
The current model is not only failing in one narrow slice.
Optimization should focus on global ranking quality plus side-specific improvements.
```

## Feature Importance Diagnosis

Top gain features include:

```text
quote_volume
taker_buy_volume
taker_buy_quote_volume
count
ret_vol_ratio__ret_10__rv_5
htf_ret_15m_1
wick_pressure_1_lag1
rv_10_delta_6
ret_vol_product__ret_10__rv_3
close_location_1_lag3
```

Concern:

```text
Some high-importance fields look like raw Binance kline metadata rather than explicitly shifted feature-pack outputs.
If raw current-row metadata enters the feature matrix, train/online consistency and leakage guarantees become fragile.
```

This must be audited before trusting further model improvements.

## Optimization Principles

1. Fix schema and time alignment before tuning the model.
2. Improve score ranking quality before adding more threshold complexity.
3. Keep validation full-sample; do not drop boundary rows from final metrics.
4. Optimize side precision jointly; do not let one side dominate accepted samples.
5. Prefer repeated chronological windows over a single best validation result.
6. Keep all feature logic in shared feature packs.
7. Execution must consume model signals and shared builder outputs only; it must not recompute separate BTC feature stacks.

## Priority 1: Feature Schema And Time Alignment

Goal:

```text
Ensure every model feature is either:
  1. produced by a registered shared feature pack; or
  2. an explicitly allowed static identifier-free numeric feature.

No raw current-row kline metadata should enter the model accidentally.
```

Actions:

```text
1. Add a strict feature schema gate in dataset building.
2. Exclude raw/non-feature columns from infer_feature_columns().
3. Explicitly exclude at least:
   raw_timestamp
   open_time
   close_time
   quote_volume
   taker_buy_volume
   taker_buy_quote_volume
   count
   number_of_trades
   ignore
   source_file
   source_date
   source_granularity
   source_version
   checksum_status
   ingested_at
4. If a raw field is useful, reintroduce it only through a shared feature pack with shift(1) or stronger no-lookahead handling.
5. Add tests proving current decision timestamp features do not use the current unfinished bar.
```

Expected effect:

```text
Short term metrics may drop if current-row leakage exists.
Long term metrics become trustworthy and production-compatible.
```

Acceptance:

```text
No raw kline metadata appears in artifact_manifest.feature_columns.
Train/live feature parity tests pass.
Feature no-lookahead tests cover all high-risk packs.
```

## Priority 2: Complete Raw Data Coverage

Goal:

```text
Run true recent 30d train / 30d validation using data current through 2026-05-01.
```

Actions:

```text
1. Download and normalize spot BTCUSDT 1m after 2026-04-01.
2. Download and normalize futures funding after 2026-04-10 if available.
3. Download and normalize mark/index/premium basis after 2026-04-10 if available.
4. Add bookTicker data, because current run did not have a usable local bookTicker parquet.
5. Add aggTrades or trades aggregation for real taker flow instead of OHLCV proxy flow.
6. Save the processed train/valid split as reusable parquet artifacts for each experiment.
```

Expected effect:

```text
More recent validation and true book/flow features should improve top-decile ranking quality if short-horizon pressure contains usable signal.
```

Acceptance:

```text
development_frame.parquet and validation_frame.parquet exist for each run.
Validation end timestamp is within expected data availability.
book_pressure features are non-empty when bookTicker is enabled.
flow_pressure uses real trade-side data when available.
```

## Priority 3: Real Microstructure Feature Upgrade

Current issue:

```text
The PRD feature names exist, but true second-level and order-book inputs are only useful when raw data is available.
OHLCV proxy features cannot fully capture the final seconds before a 5-minute settlement.
```

Feature groups to prioritize:

```text
taker flow:
  taker_buy_ratio
  taker_sell_ratio
  taker_imbalance
  rolling taker imbalance mean
  rolling taker imbalance z-score
  taker_imbalance_slope
  signed_dollar_flow

book pressure:
  spread_bps
  mid_price
  microprice
  bid_ask_qty_imbalance
  spread_change
  imbalance_change
  short_horizon_mid_drift

second-level summaries:
  last 5s/10s/30s/60s return
  last 5s/10s/30s/60s realized volatility
  second-level price slope
  second-level taker imbalance
  second-level signed dollar flow
  volume burst
  order-book imbalance change
  microprice drift
  direction flips
  last-second reversal
  late-window acceleration/deceleration
```

Implementation rule:

```text
Do not train directly on raw ticks.
Aggregate raw sub-minute data into fixed decision-time features using only information available at or before t0.
```

Acceptance:

```text
Feature builder can create these features offline and online through the same shared pack.
No execution module reads raw derivatives or raw trade files directly.
```

## Priority 4: Threshold Policy And Side Balance

Current threshold result:

```text
t_up=0.500
t_down=0.420
coverage=0.6011
UP accepted share=81.7%
DOWN accepted share=18.3%
```

Observation:

```text
The best balanced_precision threshold slightly underuses DOWN predictions.
With a 20% side-share guardrail, balanced_precision drops from 0.5927 to about 0.5855, but accepted accuracy improves.
```

Recommended policy:

```text
Research:
  keep enforce_min_side_share=false
  report side shares

Pre-production:
  enforce_min_side_share=true
  min_side_share=0.20
```

Reason:

```text
Without a side guardrail, a narrow side can inflate precision and create unstable production behavior.
```

Acceptance:

```text
Every run reports both unconstrained and side-guarded threshold selections.
Threshold reports include coverage, side shares, precision_up, precision_down, balanced_precision, and accepted accuracy.
```

## Priority 5: Validation Design

Current problem:

```text
One chronological train/validation split is useful for baseline, but not enough to trust a 5m directional signal.
```

Experiment matrix:

```text
train 30d / valid 30d
train 60d / valid 30d
train 90d / valid 30d
```

For each train length:

```text
Run at least 3 rolling validation windows.
Report:
  mean balanced_precision
  worst-window balanced_precision
  mean coverage
  worst-window coverage
  precision_up mean/min
  precision_down mean/min
  threshold stability
```

Acceptance:

```text
Do not promote a config unless it improves mean and does not materially worsen worst-window side precision.
```

## Priority 6: Model Regularization And Feature Selection

Current issue:

```text
Train balanced_precision = 0.7732
Valid balanced_precision = 0.5927
This is a large generalization gap.
```

Initial LightGBM search space:

```yaml
num_leaves: [15, 20, 24, 30]
max_depth: [4, 5, 6]
min_child_samples: [500, 800, 1200, 1500]
subsample: [0.7, 0.8, 0.9, 1.0]
colsample_bytree: [0.6, 0.8, 1.0]
reg_alpha: [0.8, 2.0, 5.0]
reg_lambda: [2.0, 5.0, 10.0, 20.0]
learning_rate: [0.02, 0.03]
n_estimators: [300, 500, 800]
```

Feature selection process:

```text
1. Remove raw metadata and leakage-risk fields first.
2. Train baseline.
3. Compute validation permutation importance.
4. Compare gain importance across rolling windows.
5. Drop high-gain but unstable features.
6. Keep feature packs only when they improve frontier or worst-window side precision.
```

Acceptance:

```text
Reduced model should lower train-valid gap while preserving or improving validation balanced_precision.
```

## Priority 7: Sample Weighting Optimization

Current weighting:

```yaml
min_abs_return: 0.0001
full_weight_abs_return: 0.0005
min_weight: 0.20
max_weight: 1.00
```

Observed boundary behavior:

```text
abs_return < 1bp:
  balanced_precision = 0.4869

1bp to 5bp:
  balanced_precision = 0.5321

>= 5bp:
  balanced_precision = 0.6350
```

Experiment matrix:

```yaml
min_weight: [0.05, 0.10, 0.20, 0.30]
full_weight_abs_return: [0.0003, 0.0005, 0.0008]
```

Additional checks:

```text
1. Monitor weighted class balance after applying sample weights.
2. Report weighted and unweighted training loss.
3. Keep validation metrics unweighted and full-sample.
```

Acceptance:

```text
Chosen weighting improves validation balanced_precision or worst-window side precision without reducing coverage below 60%.
```

## Priority 8: UP-Side Precision Improvement

Current issue:

```text
precision_up = 0.5477
precision_down = 0.6378
```

The main side-specific bottleneck is UP precision.

Diagnostics:

```text
1. Analyze false UP predictions by:
   volatility regime
   trend regime
   volume regime
   session
   spread regime
   basis regime
   funding z-score
2. Compare high-confidence UP true positives vs false positives.
3. Check whether UP false positives are driven by volume-only features.
```

Candidate features:

```text
positive_taker_imbalance x volatility regime
positive_taker_imbalance x trend strength
upward_mid_drift x spread regime
bullish_burst_score x volume_burst
positive_basis_pressure x trend_strength
funding_zscore x high_volatility_flag
last_second_reversal_flag x upward_mid_drift
```

Acceptance:

```text
UP precision improves without materially reducing DOWN precision.
Balanced precision improves under coverage >= 60%.
```

## Priority 9: Reporting Enhancements

Every experiment should write:

```text
artifact_manifest.json
metrics.json
threshold_search.json
threshold_frontier.csv
boundary_slices.csv
regime_slices.csv
feature_importance.csv
probability_deciles.csv
false_up_slices.csv
false_down_slices.csv
```

Additional summary fields:

```text
data_start
data_end
train_start
train_end
validation_start
validation_end
feature_count
raw_metadata_feature_count
book_ticker_available
aggtrade_available
threshold_constraint_satisfied
side_guardrail_constraint_satisfied
```

Acceptance:

```text
A run can be evaluated without opening notebooks.
All key diagnostics are machine-readable.
```

## Recommended Execution Order

### Step 1: Make Metrics Trustworthy

```text
1. Add strict feature schema.
2. Remove raw metadata leakage-risk columns.
3. Add tests for no-lookahead behavior.
4. Rerun current cached split.
```

Expected output:

```text
baseline_after_schema_fix
```

### Step 2: Refresh Data

```text
1. Download data through latest available date.
2. Normalize spot, funding, basis, bookTicker, and aggTrades/trades if available.
3. Save train/valid feature parquet.
4. Rerun 30d/30d baseline.
```

Expected output:

```text
baseline_recent_30d_30d
```

### Step 3: Add Real Microstructure

```text
1. Build true taker-flow features from aggTrades/trades.
2. Build true book-pressure features from bookTicker.
3. Build second-level summary features where data allows.
4. Compare against schema-fixed baseline.
```

Expected output:

```text
microstructure_ablation_report
```

### Step 4: Run Robustness Matrix

```text
1. Train 30d/60d/90d windows.
2. Validate on rolling 30d windows.
3. Report mean and worst-window metrics.
```

Expected output:

```text
rolling_validation_report
```

### Step 5: Tune Model And Weighting

```text
1. Run LightGBM regularization grid.
2. Run sample weighting grid.
3. Select by balanced_precision subject to coverage >= 60%.
4. Use worst-window side precision as a veto metric.
```

Expected output:

```text
selected_baseline_v2
```

## Promotion Criteria

Do not promote a model unless all are true:

```text
coverage >= 0.60
balanced_precision improves over schema-fixed weighted binary baseline
precision_up > validation UP base rate
precision_down > validation DOWN base rate
side shares are stable or side guardrail is enabled
results are not driven by one validation window
feature schema contains no raw metadata leakage-risk fields
execution uses the shared feature builder only
```

## Current Best Interpretation

The current run is useful as a first benchmark, but not yet a production-quality signal.

The most likely path to improve balanced_precision is:

```text
schema cleanup
  -> trustworthy baseline
  -> fresh data
  -> true book/taker microstructure
  -> rolling validation
  -> regularization and sample-weight tuning
```

Threshold tuning alone is unlikely to produce a durable improvement because the current score ranking is only moderately informative.

## Implementation Status

Implemented on 2026-05-01:

```text
strict feature schema and raw metadata exclusion
weighted binary selective training path
asymmetric threshold search with side-guarded reference result
feature importance report
probability decile report
false-UP and false-DOWN slice reports
data availability and threshold constraint fields in artifact_manifest.json
binary rolling validation runner
rolling validation summary by train window
targeted tests for schema, binary training, signal service, artifact loading, and rolling splits
```

Latest schema-fixed cached 30d/30d run:

```text
output_dir: artifacts/experiments/btc5m_weighted_binary_report_20260501
t_up: 0.5300
t_down: 0.5000
coverage: 0.6729
precision_up: 0.5321
precision_down: 0.5068
balanced_precision: 0.5194
roc_auc: 0.5101
raw_metadata_feature_count: 0
book_ticker_features_available: false
threshold_constraint_satisfied: true
side_guardrail_constraint_satisfied: true
```

Rolling validation smoke run:

```text
output_dir: artifacts/experiments/btc5m_weighted_binary_rolling_smoke_20260501
train_days_list: 30
validation_days: 7
fold_count: 2
balanced_precision_mean: 0.5460
balanced_precision_min: 0.5298
coverage_mean: 0.7163
roc_auc_mean: 0.5361
constraint_pass_rate: 1.0
side_guardrail_pass_rate: 1.0
```

The schema-fixed result should replace the earlier 0.5927 benchmark because the earlier run selected raw/current-row metadata-like columns as features. The current trustworthy baseline is weaker, so further work should prioritize fresh data coverage and real bookTicker/aggTrade microstructure before model tuning.
