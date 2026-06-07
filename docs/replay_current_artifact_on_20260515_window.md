# Replay Result - Current Artifact on 2026-05-15/16 Window

## 1. Scope

Replay window:

```text
SGT: 2026-05-15 23:00 to 2026-05-16 09:00
UTC: 2026-05-15T15:00:00 to 2026-05-16T01:00:00
resolved cycles: 102
```

Artifacts:

```text
old artifact:
  source commit: 127776abefc5f8fc2470f8f80d81357d96c08983
  model: lightgbm
  calibration: none
  thresholds: t_up=0.535, t_down=0.405

current artifact:
  artifact: artifacts/data_v2/experiments/20260520_polymarket_resolved_extended_history_baseline
  model: catboost_lgbm_logit_blend
  calibration: platt_logit
  thresholds: t_up=0.62, t_down=0.415
```

Report:

```text
artifacts/reports/execution_engine/replay_old_vs_current_20260515_150000_20260516_010000.json
```

## 2. 2x2 Result

```text
old_model + old_thresholds:
  accepted: 76 / 102
  coverage: 74.51%
  wins/losses: 61 / 15
  accepted_accuracy: 80.26%
  up/down: 37 / 39
  selection_score_like: 1.1760

old_model + current_thresholds:
  accepted: 69 / 102
  coverage: 67.65%
  wins/losses: 54 / 15
  accepted_accuracy: 78.26%
  up/down: 26 / 43
  selection_score_like: 0.9971

current_model + old_thresholds:
  accepted: 74 / 102
  coverage: 72.55%
  wins/losses: 61 / 13
  accepted_accuracy: 82.43%
  up/down: 43 / 31
  selection_score_like: 1.3182

current_model + current_thresholds:
  accepted: 59 / 102
  coverage: 57.84%
  wins/losses: 48 / 11
  accepted_accuracy: 81.36%
  up/down: 26 / 33
  selection_score_like: 1.1046
```

## 3. Interpretation

Current artifact on the old profitable window:

```text
with old thresholds:
  accuracy improves from 80.26% to 82.43%
  coverage remains above 70%
  accepted_count is close: 74 vs 76

with current thresholds:
  accuracy remains high at 81.36%
  coverage collapses to 57.84%
```

This means the current artifact is not directionally worse on the old window. The current artifact can perform very well in that favorable regime. The large practical regression comes from current thresholds and the later market regime, not from feature count.

## 4. Combined With 2026-05-20/21 Replay

On 2026-05-20/21:

```text
old_model + old_thresholds:
  accuracy: 60.98%
  coverage: 76.64%

current_model + current_thresholds:
  accuracy: 62.50%
  coverage: 67.29%

current_model + old_thresholds:
  accuracy: 59.52%
  coverage: 78.50%
```

So across both windows:

```text
1. Old thresholds keep coverage above 70%.
2. Current thresholds are too conservative for live coverage.
3. Current model is not worse on the old window.
4. Bad-window accuracy remains around 60% for both models, so the main accuracy gap is market regime.
```

## 5. Conclusion

The answer to "is the new artifact worse?" is:

```text
Not on the old profitable window. With old thresholds, it is actually better:
  old artifact: 80.26% accuracy, 74.51% coverage
  current artifact: 82.43% accuracy, 72.55% coverage
```

The answer to "why is live worse now?" is:

```text
1. Current thresholds reduce coverage too much.
2. 2026-05-20/21 was a worse first-minute-continuation regime.
3. Execution price and second-leg removal then amplified the PnL impact.
```

Next experiment should tune thresholds and reversal filters over multiple live-like windows instead of reverting the model.
