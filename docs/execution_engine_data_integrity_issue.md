# Execution Engine Data Integrity Issue

This document explains the execution-layer data integrity problem observed on `aws-poly` during the last 12 hours of live runs.

## What the issue is

The execution engine sometimes starts a cycle before Binance market data is fully complete for the required pre-signal timestamp. When that happens, the engine raises:

```text
Binance agg trade frame is not complete through the required pre-signal time.
```

This is not a model-training problem. It is an execution-time completeness gate. The engine refuses to finalize a signal if the 1s / agg-trade data does not reach the exact pre-signal boundary required by the runtime alignment rule.

## What the logs show

In the last 12 hours on `aws-poly`, `journalctl -u execution-engine.service` showed many hard failures of the same type:

- Required timestamp example: `2026-05-16T15:55:59+00:00`
- Latest available agg-trade timestamp example: `2026-05-16T15:55:58.763000+00:00`

The gap is usually only a fraction of a second to a few seconds, but the guard is strict, so the cycle exits with a non-zero status instead of producing a summary.

Observed counts for the same 12-hour window:

- `101` successful summary writes
- `43` service start failures
- `18` explicit `selective_binary_abstain` cycles

That means a large part of the wall-clock schedule was lost to data arrival timing rather than to model decisions.

## Where the guard lives in code

The runtime alignment and completeness checks are in [`execution_engine/realtime_data.py`](../execution_engine/realtime_data.py).

Key behavior:

- `decision_timestamp(...)` defines the decision row timestamp.
- `finalize_runtime_frames_for_signal(...)` computes:
  - `required_latest_closed_minute`
  - `required_latest_closed_second`
  - `required_latest_agg_trade`
- The function then checks whether the agg-trade frame reaches that required timestamp.
- If not, it raises the exact runtime error shown in the logs.

The summary writer in [`execution_engine/run_once.py`](../execution_engine/run_once.py) records the alignment fields into the JSON summary, so the problem can be audited after the fact.

The project timeline doc also states the same rule explicitly:

- [`execution_engine/execution_engine_flow_timeline.md`](../execution_engine/execution_engine_flow_timeline.md)

## Why this matters

This failure mode reduces effective coverage. Even when the model is fine, the engine cannot produce a usable signal if the data is late by a small amount.

Current mitigation: the Binance agg-trade completeness gate now allows a small lag window by default (`max_agg_trade_lag_seconds = 2.0`), so sub-second or near-2-second delays no longer force a hard failure.

Concrete effects seen in the 12-hour window:

- Some cycles never produced a summary at all.
- Hourly coverage became uneven because failed cycles were dropped.
- Accuracy and PnL interpretation become biased if you only look at successful cycles, because failed cycles are missing from the denominator.

The important distinction is:

- model issue: the signal exists but is wrong
- data integrity issue: the signal cannot be finalized because the runtime input set is incomplete

This document is about the second case.

## What is not the main problem

The logs do not suggest a feature-leakage problem here.

The summaries show a consistent one-minute offset between `signal.t0` and `feature_timestamp`, which matches the current delayed-row policy.

So the main problem is not that the engine is using future data. The main problem is that the engine is waiting for exact completeness and sometimes misses the boundary.

## Practical interpretation

If the goal is to run every 5-minute cycle reliably, this issue should be treated as an availability / completeness problem:

- the Binance agg-trade stream arrives too close to the decision boundary
- the pre-signal guard is strict
- the service fails instead of degrading gracefully

That makes the system look more unstable than the model alone would suggest.

## Suggested next checks

1. Measure how often the latest agg-trade timestamp trails the required pre-signal timestamp by less than 1 second.
2. Compare failures against `systemd` trigger time to see whether the schedule is too aggressive for current data latency.
3. Inspect whether the prewarm window or the agg-trade lookback buffer is too small for live conditions.
4. Separate "hard data completeness failures" from "model abstentions" in reporting so coverage loss is not misattributed to the model.
