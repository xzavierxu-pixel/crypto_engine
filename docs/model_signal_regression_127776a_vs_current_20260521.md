# Model Signal Regression Note - Commit 127776a vs Current

## 1. Scope

This note compares model/signal behavior only. It intentionally excludes order price, second-leg optionality, maker/taker mix, and fill quality.

Old profitable run:

```text
commit: 127776abefc5f8fc2470f8f80d81357d96c08983
commit label: profitable version1
live window SGT: 2026-05-15 23:00 to 2026-05-16 09:00
live window UTC: 2026-05-15 15:00 to 2026-05-16 01:00
```

Current comparison window:

```text
UTC: 2026-05-20 15:23:40 to 2026-05-21 00:23:40
```

Input reports:

```text
artifacts/reports/execution_engine/live_profit_analysis_20260515_2300_20260516_0900_sgt.json
artifacts/reports/execution_engine/live_loss_analysis_last_9h.json
artifacts/reports/execution_engine/bot_matched_trades_20260515_2300_20260516_0900_sgt.json
artifacts/reports/execution_engine/bot_matched_trades_last_9h.json
```

## 2. Short Answer

Feature count is not the explanation. The old and current split artifacts both use 1016 features, and the feature column sets are identical.

The important differences are:

```text
1. The offline target changed.
2. The model plugin changed.
3. Calibration changed.
4. The training sample window and row count changed.
5. The selected thresholds changed.
6. The live windows had different first-minute continuation/reversal regimes.
7. The old live window was unusually favorable for a first-minute-following model.
```

The current model is not clearly worse on its offline validation set. Offline validation metrics are almost the same as the old artifact. The large live gap comes from regime shift and threshold/model behavior in a small live window.

## 3. Feature Set Comparison

Old validation artifact:

```text
artifact: artifacts/data_v2/experiments/20260515_t_plus_1_delayed_validation
feature_count: 1016
```

Current accepted split artifact:

```text
artifact: artifacts/data_v2/experiments/20260520_polymarket_resolved_extended_history_baseline
feature_count: 1016
```

Feature diff:

```text
old feature count: 1016
new feature count: 1016
feature set equal: true
feature order equal: true
old-only features: 0
new-only features: 0
```

So this is not a feature-count regression.

## 4. Offline Flow Differences

### Old commit flow

Commit `127776a` still describes the project as a BTC/USDT 5-minute direction prediction project. The old config used:

```yaml
objective:
  label: settlement_direction
  min_coverage: 0.40

horizons:
  "5m":
    label_builder: grid_direction
    label_version: settlement_direction_t0_open_to_t4_close_tie_up_v2

model:
  active_plugin: lightgbm

calibration:
  active_plugin: none
```

The old execution deploy config also pointed to:

```yaml
model_file: lightgbm.binary.pkl
calibrator_file: none.binary.pkl
```

The old artifact itself has:

```text
model_plugin: lightgbm
calibration_plugin: none
t_up: 0.535
t_down: 0.405
base_rate: 0.50898
config_hash: 94524819935a
```

### Current flow

The current accepted baseline is explicitly based on Polymarket resolved labels:

```yaml
label_builder: polymarket_resolved
label_version: polymarket_resolved_gamma_v1
objective.min_coverage: 0.70
```

The current split artifact has:

```text
model_plugin: catboost_lgbm_logit_blend
calibration_plugin: platt_logit
t_up: 0.62
t_down: 0.415
base_rate: 0.51495
config_hash: 344c70c6a3c1
```

This means the old and new offline flows are not the same, even though the feature columns are the same.

## 5. Offline Validation Metrics

Old artifact validation:

```text
sample_count: 7471
coverage: 0.71195
accepted_sample_accuracy: 0.69111
precision_up: 0.68530
precision_down: 0.70014
balanced_precision: 0.69272
selection_score: 0.58027
roc_auc: 0.70378
brier_score: 0.21819
log_loss: 0.62512
t_up: 0.535
t_down: 0.405
```

Current artifact validation:

```text
sample_count: 7468
coverage: 0.70019
accepted_sample_accuracy: 0.69095
precision_up: 0.70657
precision_down: 0.67493
balanced_precision: 0.69075
selection_score: 0.57485
roc_auc: 0.70111
brier_score: 0.21953
log_loss: 0.62777
t_up: 0.62
t_down: 0.415
```

Offline validation difference:

```text
accepted accuracy: old 69.111% vs current 69.095%  -> almost identical
coverage: old 71.195% vs current 70.019%          -> current is lower by 1.18 pp
selection_score: old 0.58027 vs current 0.57485   -> current is lower by 0.00542
```

The current version did not materially improve offline validation. It is slightly worse on the old acceptance metric, but not enough to explain the much larger live gap by itself.

## 6. Live Signal Accuracy and Coverage

Ignoring execution price and using one signal per accepted market window:

### Old profitable live window

```text
resolved cycles: 102
accepted cycles: 76
coverage: 74.51%
signal wins: 61
signal losses: 15
signal accuracy: 80.26%
YES signals: 37
NO signals: 39
actual YES: 40
actual NO: 36
```

### Current loss live window

```text
resolved cycles: 106
accepted cycles: 71
coverage: 66.98%
signal wins: 45
signal losses: 26
signal accuracy: 63.38%
YES signals: 27
NO signals: 44
actual YES: 35
actual NO: 36
```

Live gap:

```text
signal accuracy gap: 80.26% - 63.38% = 16.88 pp
coverage gap: 74.51% - 66.98% = 7.53 pp
```

This gap is much larger than the offline validation gap.

## 7. Why Live Accuracy and Coverage Dropped

### 7.1 Thresholds became more conservative on UP

Old:

```text
t_up: 0.535
t_down: 0.405
```

Current:

```text
t_up: 0.62
t_down: 0.415
```

The current UP threshold is much higher. This naturally reduces accepted UP signals:

```text
old live YES signals: 37
old live NO signals: 39

current live YES signals: 27
current live NO signals: 44
```

The current model became more NO-heavy in this live window. Actual outcomes were nearly balanced:

```text
current actual YES: 35
current actual NO: 36
```

So coverage and side balance both degraded live.

### 7.2 Calibration changed probability geometry

Old:

```text
lightgbm + no calibration
```

Current:

```text
catboost_lgbm_logit_blend + platt_logit
```

The current model was not just a better version of the old model. It changed the probability distribution and the threshold mapping. This matters because the live policy is selective:

```text
UP if p_up >= t_up
DOWN if p_up <= t_down
otherwise abstain
```

Small probability distribution changes can produce large changes in accepted samples, especially near thresholds.

### 7.3 The live regime changed

Both windows show the model is primarily a first-minute follower:

```text
old model side agrees with first-minute side: 98.68%
current model side agrees with first-minute side: 98.59%
```

But continuation quality was different.

Old window:

```text
model agrees with BTC 5m direction: 76.32%
```

Current window:

```text
model agrees with BTC 5m direction: 61.97%
```

The current window had many more first-minute moves that did not continue into the 5-minute direction. Since both versions mostly follow the first minute, this directly cuts live accuracy.

### 7.4 Confidence buckets behaved differently live

Old live summary by confidence bucket:

```text
0.60-0.70: accuracy 87.10%
0.70-0.80: accuracy 76.19%
0.80-0.90: accuracy 81.82%
0.90-1.01: accuracy 100.00%
```

Current live summary:

```text
0.60-0.70: accuracy 61.29%
0.70-0.80: accuracy 47.37%
0.80-0.90: accuracy 81.25%
0.90-1.01: accuracy 100.00%
```

The problematic current bucket is `0.70-0.80`. It was profitable in the old window and bad in the current window. That suggests live calibration/regime instability, not just a static feature issue.

### 7.5 Label target changed from BTC grid direction to Polymarket resolved labels

The old artifact was trained on:

```text
grid_direction
settlement_direction_t0_open_to_t4_close_tie_up_v2
```

The current accepted workflow uses:

```text
Polymarket resolved labels
```

Live evaluation here uses Polymarket resolved outcomes for both windows, but the old model was not trained on that exact target. This can cut both ways:

```text
old live window: BTC first-minute continuation happened to align well with Polymarket resolved outcomes
current live window: that alignment was weaker
```

This is a likely reason the old model looked much better live than its target definition would imply.

## 8. What Is Not The Cause

Not caused by feature count:

```text
old features: 1016
current features: 1016
feature set: identical
feature order: identical
```

Not explained by offline validation alone:

```text
old validation accepted accuracy: 69.111%
current validation accepted accuracy: 69.095%
```

Not only execution:

Even ignoring price/fills, live signal accuracy dropped from:

```text
80.26% -> 63.38%
```

So there is a real signal/regime gap. Execution made the PnL gap worse, but it is not the only issue.

## 9. Interpretation

The most likely explanation is:

```text
1. The old model and current model both mostly follow ret_1 / first-minute direction.
2. The old live window was a favorable continuation regime.
3. The current live window was a worse reversal regime.
4. The current thresholds are stricter on UP and produced lower live coverage and more NO-heavy selection.
5. The current model/calibration changed accepted-sample composition without improving offline validation enough to compensate.
6. The label migration to Polymarket resolved labels is correct for the project, but it changed what offline validation means relative to the old BTC grid-direction model.
```

## 10. Recommended Follow-Up

### 10.1 Replay both artifacts on the same windows

To answer whether the old model is truly better, replay:

```text
old artifact on 2026-05-20/21 window
current artifact on 2026-05-15/16 window
```

Use the same feature frames and the same Polymarket resolved labels. Without this cross replay, the comparison is confounded by market regime.

### 10.2 Compare threshold variants

Run the current model with old thresholds:

```text
t_up: 0.535
t_down: 0.405
```

and old model with current thresholds:

```text
t_up: 0.62
t_down: 0.415
```

This separates model probability quality from threshold policy.

### 10.3 Add continuation/reversal diagnostics to validation

Both models need validation slices:

```text
first_minute_continuation
first_minute_reversal
confidence bucket
side bucket
hour bucket
Polymarket-vs-Binance mismatch bucket
```

### 10.4 Treat current baseline as not proven better

The current artifact is cleaner from a label-alignment standpoint, but it is not clearly better on validation:

```text
old selection_score: 0.58027
current selection_score: 0.57485
```

So the current model should not be assumed superior just because it is newer or uses a blend/calibration.

## 11. Current Conclusion

Feature count and feature columns are the same. The live signal regression is mainly from offline-flow differences plus live regime shift:

```text
old: lightgbm, no calibration, BTC grid-direction target, t_up=0.535, t_down=0.405
current: catboost/lgbm blend, platt calibration, Polymarket resolved target, t_up=0.62, t_down=0.415
```

Offline validation is nearly tied. Live accuracy and coverage are not tied because the old window was a strong continuation regime and the current window was much more reversal-heavy. The correct next step is same-window artifact replay, not another blind feature/model iteration.
