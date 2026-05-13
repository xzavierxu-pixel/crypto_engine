# aws-poly Execution Engine Result Analysis - 2026-05-13

## Scope

Analyzed resolved BTC 5-minute Polymarket execution summaries copied from:

```text
aws-poly:~/opt/crypto_engine/artifacts/logs/execution_engine/summaries
```

The timer was stopped for analysis, but the remote configuration at analysis time had:

```yaml
runtime.mode: live
orders.enabled: true
orders.mode: live
thresholds.t_up: 0.5425
thresholds.t_down: 0.4475
```

So the collected run should be treated as a live execution/shadow-evaluation dataset, not a pure paper-only dataset.

## Goal

For each complete hour:

```text
12 BTC 5-minute markets are released.
Predict at least 6 markets.
Predict correctly for at least 4 of the predicted markets.
```

Evaluation used resolved Polymarket Gamma market outcomes from:

```text
https://gamma-api.polymarket.com/markets?slug=<slug>&closed=true
```

Outcome was inferred from the resolved `outcomePrices` field:

```text
Up/Yes price = 1 -> YES
Down/No price = 1 -> NO
```

## Current Run Result

Using the actual online decisions in the copied summaries:

```text
sample_count: 141
resolved_count: 141
accepted_count: 97
correct_count: 48
coverage: 0.6879432624
accepted_sample_accuracy: 0.4948453608
share_up_predictions: 0.5567010309
share_down_predictions: 0.4432989691
complete_hour_count: 12
passed_hour_count: 9
```

The current run did not meet the hourly goal for every complete hour.

Failing hours:

| Hour UTC | Available summaries | Predicted | Correct | Reason |
|---|---:|---:|---:|---|
| 2026-05-12 15:00 | 11 | 8 | 2 | enough coverage, too few correct |
| 2026-05-12 18:00 | 11 | 5 | 3 | fewer than 6 predictions |
| 2026-05-12 19:00 | 12 | 5 | 3 | fewer than 6 predictions |

Passing hours:

```text
2026-05-12 14:00
2026-05-12 16:00
2026-05-12 17:00
2026-05-12 20:00
2026-05-12 21:00
2026-05-12 22:00
2026-05-12 23:00
2026-05-13 00:00
2026-05-13 01:00
```

## Threshold Replay

I replayed thresholds on the same summary probabilities and resolved outcomes. This does not change model features or labels; it only changes the UP/DOWN/NO-SIGNAL decision rule.

Best replayed candidate for the hourly target:

```text
t_up: 0.51
t_down: 0.50
accepted_count: 139
correct_count: 72
coverage: 0.9858156028
accepted_sample_accuracy: 0.5179856115
complete_hour_count: 12
passed_hour_count: 12
```

This candidate satisfies the hourly goal on the analyzed complete hours:

| Hour UTC | Available | Predicted | Correct |
|---|---:|---:|---:|
| 2026-05-12 14:00 | 10 | 10 | 6 |
| 2026-05-12 15:00 | 11 | 11 | 4 |
| 2026-05-12 16:00 | 12 | 12 | 5 |
| 2026-05-12 17:00 | 12 | 12 | 5 |
| 2026-05-12 18:00 | 11 | 11 | 7 |
| 2026-05-12 19:00 | 12 | 11 | 8 |
| 2026-05-12 20:00 | 10 | 10 | 5 |
| 2026-05-12 21:00 | 12 | 12 | 6 |
| 2026-05-12 22:00 | 12 | 11 | 5 |
| 2026-05-12 23:00 | 12 | 12 | 7 |
| 2026-05-13 00:00 | 10 | 10 | 6 |
| 2026-05-13 01:00 | 11 | 11 | 6 |

## Timeline And Alignment Findings

Current later summaries show the intended online alignment:

```text
signal.t0:          2026-05-13T01:45:00Z
feature_timestamp:  2026-05-13T01:44:00Z
market.slug:        btc-updown-5m-1778636700
market_start:       2026-05-13T01:45:00Z
```

This means the engine predicts the 01:45-01:50 market using the latest closed 1-minute feature row before 01:45, which is 01:44.

There were 5 early summaries where `signal.t0` did not match the actual market slug start. Example:

```text
summary:      2026-05-12T133500Z0000.json
signal.t0:    2026-05-12T13:35:00Z
market_start: 2026-05-12T13:40:00Z
```

Those early summaries also lacked `feature_timestamp`, so they are weaker audit records. Later summaries include `feature_timestamp` and show consistent feature lag of 1 minute.

## Drift Assessment

Observed online drift risks:

1. Threshold drift: remote execution used `t_up=0.5425`, `t_down=0.4475`, while the deployed artifact manifest contains different artifact thresholds. This is a deliberate config override, but it must be tracked as execution behavior drift from artifact defaults.
2. Decision objective drift: current thresholds optimize selective coverage/edge, but the user goal is hourly coverage plus at least 4 correct predictions. The replayed `0.51/0.50` thresholds fit the hourly goal better on this run.
3. Early audit drift: 5 early records have `signal.t0 != market_start` and no `feature_timestamp`, making them hard to trust for offline-online alignment analysis.
4. Source drift: Polymarket resolves BTC 5-minute markets using Chainlink BTC/USD streams, while model features are built from Binance BTCUSDT. This can create unavoidable settlement-source drift near tight boundaries.

No direct evidence of future-feature leakage was found in later online summaries: the feature row is one minute before `signal.t0`, and closed Binance minute bars are used.

## Recommendation

For the next controlled paper experiment:

```yaml
runtime:
  mode: paper
orders:
  enabled: false
  mode: paper
thresholds:
  t_up: 0.51
  t_down: 0.50
```

Run at least one full hour, then evaluate with:

```powershell
python execution_engine/scripts/evaluate_paper_results.py `
  --summary-dir artifacts/logs/execution_engine/summaries `
  --cache-path artifacts/state/execution_engine/outcome_cache.json `
  --threshold-search `
  --output-json artifacts/logs/execution_engine/paper_eval_report.json
```

The current analyzed run is not sufficient to claim the system generally meets the goal, because the passing threshold was selected after seeing this run. It is sufficient evidence that the main current failure mode is threshold/coverage behavior, not missing outcome resolution or feature construction.

## Follow-Up Paper Experiment

After the initial analysis, I ran a controlled paper-only experiment on `aws-poly` with:

```yaml
runtime.mode: paper
orders.enabled: false
orders.mode: paper
thresholds.t_up: 0.51
thresholds.t_down: 0.50
summary_dir: artifacts/logs/execution_engine/paper_summaries_20260513
```

The run used:

```text
execution_engine/scripts/run_paper_experiment.py
execution_engine/config.paper_20260513.yaml
```

Remote evidence paths:

```text
~/opt/crypto_engine/artifacts/logs/execution_engine/paper_summaries_20260513/
~/opt/crypto_engine/artifacts/logs/execution_engine/paper_experiment_20260513.jsonl
~/opt/crypto_engine/artifacts/logs/execution_engine/paper_experiment_20260513_ext.jsonl
~/opt/crypto_engine/artifacts/logs/execution_engine/paper_eval_20260513.json
~/opt/crypto_engine/artifacts/logs/execution_engine/paper_eval_threshold_search_20260513.json
~/opt/crypto_engine/artifacts/state/execution_engine/outcome_cache_paper_20260513.json
```

Paper result:

```text
sample_count: 23
resolved_count: 22
accepted_count: 22
correct_count: 6
coverage: 1.0
accepted_sample_accuracy: 0.2727272727
complete_hour_count: 2
passed_hour_count: 1
```

Hourly result:

| Hour UTC | Available | Resolved | Predicted | Correct | Goal Passed |
|---|---:|---:|---:|---:|---|
| 2026-05-13 09:00 | 11 | 11 | 11 | 5 | true |
| 2026-05-13 10:00 | 12 | 11 | 12 | 1 | false |

The 10:00 hour did not meet the goal. It had enough coverage, but direction accuracy collapsed.

I also compared resolved Polymarket outcomes against the offline Binance label rule:

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

For the 22 resolved paper windows:

```text
Polymarket outcome == Binance offline label: 22/22
```

This means the failure was not caused by Chainlink/Binance settlement-source drift for this sample. It was model/signal direction failure under the current market regime.

The 10:00 hour was almost fully opposite:

```text
resolved windows: 11
original correct: 1
inverse correct: 10
YES predictions: 9
YES outcomes: 1
```

Do not interpret this as permission to invert `p_up` in production. The project signal rule defines `p_up` semantically as UP probability, and changing that would break offline-online consistency unless a separately validated model or policy is introduced.

## Additional Paper Policy Experiments

I added explicit paper-only policy support to `run_paper_experiment.py`:

```text
--policy model
--policy inverse-model
--policy last-window-momentum
```

This is isolated to the paper experiment runner. The main `run_once` execution path still uses the model signal rule.

I also fixed `evaluate_paper_results.py` so unresolved outcome-cache entries are refreshed on later runs. Resolved markets remain stable in cache.

### Last-Window Momentum

Experiment:

```text
config: execution_engine/config.paper_momentum_20260513.yaml
summary_dir: artifacts/logs/execution_engine/paper_momentum_summaries_20260513
eval: artifacts/logs/execution_engine/paper_momentum_eval_20260513.json
```

Result:

```text
sample_count: 22
resolved_count: 21
accepted_count: 21
correct_count: 7
coverage: 1.0
accepted_sample_accuracy: 0.3333333333
```

Complete hour:

| Hour UTC | Available | Resolved | Predicted | Correct | Goal Passed |
|---|---:|---:|---:|---:|---|
| 2026-05-13 12:00 | 12 | 12 | 12 | 3 | false |

### Inverse Model

Experiment:

```text
config: execution_engine/config.paper_inverse_20260513.yaml
summary_dir: artifacts/logs/execution_engine/paper_inverse_summaries_20260513
eval: artifacts/logs/execution_engine/paper_inverse_eval_20260513.json
```

Result:

```text
sample_count: 22
resolved_count: 22
accepted_count: 21
correct_count: 7
coverage: 0.9545454545
accepted_sample_accuracy: 0.3333333333
```

Complete hours:

| Hour UTC | Available | Resolved | Predicted | Correct | Goal Passed |
|---|---:|---:|---:|---:|---|
| 2026-05-13 13:00 | 10 | 10 | 10 | 5 | true |
| 2026-05-13 14:00 | 12 | 12 | 11 | 2 | false |

## Updated Conclusion

The objective is still not met.

Validated paper/live policies so far:

| Policy | Complete target hour tested | Goal result |
|---|---|---|
| model with `0.51/0.50` thresholds | 2026-05-13 10:00 | failed, 1 correct |
| last-window momentum | 2026-05-13 12:00 | failed, 3 correct |
| inverse model | 2026-05-13 14:00 | failed, 2 correct |

The execution timeline and label alignment now look correct, but the model, inverse-model, and last-window-momentum policies were not stable enough to satisfy the hourly requirement.

## Passing Prev3 Momentum Paper Experiment

I then evaluated a stricter short-term momentum policy:

```text
policy: prev3-momentum
definition: predict YES if close[t0 - 1m] >= close[t0 - 3m], else NO
data availability: uses only closed 1m Binance bars before signal.t0
```

Historical replay across the collected complete hours showed:

| Policy | Complete hours checked | Result |
|---|---:|---|
| prev3-momentum | 4 | passed 4/4 |

Per-hour replay:

| Hour UTC | Predicted | Correct | Goal Passed |
|---|---:|---:|---|
| 2026-05-13 10:00 | 11 | 7 | true |
| 2026-05-13 12:00 | 12 | 8 | true |
| 2026-05-13 13:00 | 10 | 8 | true |
| 2026-05-13 14:00 | 12 | 6 | true |

I then ran a fresh aws-poly paper experiment:

```text
config: execution_engine/config.paper_prev3_20260513.yaml
summary_dir: artifacts/logs/execution_engine/paper_prev3_summaries_20260513
eval: artifacts/logs/execution_engine/paper_prev3_eval_20260513.json
```

Result:

```text
sample_count: 22
resolved_count: 22
accepted_count: 22
correct_count: 12
coverage: 1.0
accepted_sample_accuracy: 0.5454545455
complete_hour_count: 1
passed_hour_count: 1
```

Complete target hour:

| Hour UTC | Available | Resolved | Predicted | Correct | Goal Passed |
|---|---:|---:|---:|---:|---|
| 2026-05-13 16:00 | 12 | 12 | 12 | 6 | true |

This satisfies the requested paper-trading hourly criterion for the controlled aws-poly experiment:

```text
predicted >= 6: yes, 12
correct >= 4: yes, 6
```

Important caveat: `prev3-momentum` is a paper experiment policy, not a replacement for the trained model artifact. Promoting it to production would require adding it to unified config and validating it against the project objective over a larger time-based validation window.
