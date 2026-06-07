# Replay Result - Old Artifact on 2026-05-20/21 Window

## 1. Scope

Replay window:

```text
UTC: 2026-05-20T15:23:40 to 2026-05-21T00:23:40
resolved cycles: 107
```

Artifacts:

```text
old artifact:
  source commit: 127776abefc5f8fc2470f8f80d81357d96c08983
  model: lightgbm
  calibration: none
  thresholds: t_up=0.535, t_down=0.405
  feature_count: 1016

current artifact:
  artifact: artifacts/data_v2/experiments/20260520_polymarket_resolved_extended_history_baseline
  model: catboost_lgbm_logit_blend
  calibration: platt_logit
  thresholds: t_up=0.62, t_down=0.415
  feature_count: 1016
```

Report:

```text
artifacts/reports/execution_engine/replay_old_vs_current_20260520_152340_20260521_002340.json
```

The replay rebuilt T0+1m synthetic-row features from Binance 1m/1s/aggTrade data and used Polymarket resolved outcomes as truth.

## 2. 2x2 Replay Matrix

```text
old_model + old_thresholds:
  accepted_count: 82 / 107
  coverage: 76.64%
  wins/losses: 50 / 32
  accepted_accuracy: 60.98%
  up/down: 40 / 42
  selection_score_like: 0.3076

old_model + current_thresholds:
  accepted_count: 70 / 107
  coverage: 65.42%
  wins/losses: 42 / 28
  accepted_accuracy: 60.00%
  up/down: 26 / 44
  selection_score_like: 0.2558

current_model + old_thresholds:
  accepted_count: 84 / 107
  coverage: 78.50%
  wins/losses: 50 / 34
  accepted_accuracy: 59.52%
  up/down: 43 / 41
  selection_score_like: 0.2653

current_model + current_thresholds:
  accepted_count: 72 / 107
  coverage: 67.29%
  wins/losses: 45 / 27
  accepted_accuracy: 62.50%
  up/down: 28 / 44
  selection_score_like: 0.3349
```

The current replay matches the actual live summary:

```text
actual summary:
  accepted_count: 72 / 107
  coverage: 67.29%
  wins/losses: 45 / 27
  accepted_accuracy: 62.50%
```

## 3. What This Separates

### Threshold effect

Old thresholds materially increase coverage:

```text
old model:
  old thresholds coverage: 76.64%
  current thresholds coverage: 65.42%
  coverage delta: +11.21 pp

current model:
  old thresholds coverage: 78.50%
  current thresholds coverage: 67.29%
  coverage delta: +11.21 pp
```

So the coverage drop is mostly threshold policy. The current `t_up=0.62` is much stricter than the old `t_up=0.535`.

### Model effect

At old thresholds:

```text
old model accuracy: 60.98%
current model accuracy: 59.52%
old model advantage: +1.45 pp
```

At current thresholds:

```text
old model accuracy: 60.00%
current model accuracy: 62.50%
current model advantage: +2.50 pp
```

This is not a clear model-quality win for either side. The models mostly agree directionally when both trade. The difference is mainly probability calibration and which rows pass thresholds.

Agreement check:

```text
old_model_old_thresholds vs current_model_current_thresholds:
  both accepted: 66
  same side: 66

old_model_old_thresholds vs current_model_old_thresholds:
  both accepted: 73
  same side: 73

old_model_current_thresholds vs current_model_current_thresholds:
  both accepted: 65
  same side: 65
```

When both models fire, they pick the same side. The issue is not mainly opposite directional views.

### Market regime effect

Old artifact on the bad 5/20 window:

```text
accuracy: 60.98%
coverage: 76.64%
```

Old artifact on the 5/15 profitable window had:

```text
accuracy: 80.26%
coverage: 74.51%
```

The same old artifact does not keep its 80% accuracy on 5/20. That means the 5/15 result was heavily regime-dependent. The 5/20 window was a worse first-minute-continuation regime.

## 4. Interpretation

The replay separates the causes as follows:

```text
1. Threshold explains most of the coverage difference.
2. Model choice does not explain the large 5/15 vs 5/20 accuracy gap.
3. Market regime explains most of the accuracy gap.
4. Current thresholds improve accuracy on this bad window but push coverage below 70%.
5. Old thresholds restore coverage above 70%, but accuracy remains around 60%.
```

## 5. Conclusion

The old artifact would not have saved the 5/20 window from weak signal accuracy. It would have traded more often:

```text
old artifact + old thresholds: 60.98% accuracy, 76.64% coverage
current artifact + current thresholds: 62.50% accuracy, 67.29% coverage
```

So:

```text
coverage regression: mostly threshold policy
accuracy regression vs 5/15: mostly market regime
model regression: not clearly proven by this replay
```

The next useful experiment is not simply reverting the model. It is:

```text
1. keep same-window replay as a standard report,
2. tune thresholds for coverage >= 0.70 on live-like replay windows,
3. add reversal/continuation diagnostics,
4. combine signal thresholds with execution edge controls.
```
