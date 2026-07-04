# Execution Engine

The execution engine is the runtime adapter between the accepted BTC 5-minute Polymarket signal artifact and Polymarket order placement. It does not retrain models or reimplement feature formulas; runtime inference goes through the shared `src/` feature, model, and signal-policy code.
Runtime source of truth:

```text
execution_engine/config.example.yaml
execution_engine/deploy/baseline/artifact_manifest.json
execution_engine/deploy/price_estimator_expected_return_h14/artifact_manifest.json
config/settings.yaml
```

## Current Workflow

One `run_once.py` cycle does the following:

1. Load `execution_engine/config.yaml`, the baseline artifact, the optional price-estimator artifact, and `config/settings.yaml`.
2. Wait until `schedule.trigger_delay_seconds` for the current 5-minute window when no explicit target window is passed.
3. Pull Binance 1m, 1s, and aggTrades data, then validate that the runtime frame is complete through the decision cutoff.
4. Append the delayed synthetic decision row at `T+1m` for market window `[T,T+5m)`.
5. Build second-level and 1m features through the shared feature builder.
6. Run the baseline classifier and shared selective binary policy.
7. If the signal trades, evaluate the active H14 expected-return price estimator and attach its outputs to the signal context.
8. Map `signal.t0` to the Polymarket BTC 5-minute slug, read the target token order book, build BUY limit orders, and submit only when `mode == live` and `orders.enabled == true`.

## Current Artifacts

Baseline classifier:

```yaml
artifact_dir: execution_engine/deploy/baseline
experiment_id: 20260611_catboost_calendar_coordinate_search
training_mode: calendar_coordinate_offline_train
model_plugin: catboost
calibration_plugin: none
feature_count: 569
t_up: 0.5792857142857143
t_down: 0.43142857142857144
validation_selection_score: 0.6413740846017104
validation_coverage: 0.7000535618639528
validation_accepted_sample_accuracy: 0.70734506503443
```

The baseline deploy directory currently contains:

```text
artifact_manifest.json
catboost.binary.pkl
none.binary.pkl
metrics.json
report.json
```

Keep `baseline.model_file` and `baseline.calibrator_file` as `null` unless intentionally overriding manifest resolution.
Price estimator:

```yaml
enabled: true
active_artifact: expected_return_h14
artifact_dir: execution_engine/deploy/price_estimator_expected_return_h14
model_file: expected_return_hazard.npz
prediction_column: expected_return_bid
feature_count: 576
order_price_policy: min(best_ask - 0.01, expected_return_optimal_bid)
```

The H14 estimator is loaded and logged for traded signals. The current first order leg still uses `reference_multiplier_offset_and_cap`; H14 directly controls price only if a leg is configured with `price_mode: expected_return_optimal_bid`.

## Current Runtime Defaults

Tracked template defaults:

```yaml
runtime:
  mode: live
schedule:
  trigger_delay_seconds: 68
  max_data_wait_seconds: 20
price_estimator:
  enabled: true
  active_artifact: expected_return_h14
orders:
  enabled: true
  mode: live
  first:
    enabled: true
    price_mode: reference_multiplier_offset_and_cap
    price_cap: 0.75
    offset: 0.01
    reference_multiplier: 1.0
    size: 5.0
  second:
    enabled: false
guards:
  require_market_accepting_orders: true
  require_best_bid: true
  enforce_idempotency: true
execution_edge:
  enabled: false
```

Model alignment and feature state from `config/settings.yaml`:

```yaml
decision_alignment:
  enabled: true
  mode: delayed_feature_offset
  feature_offset_minutes: 1
  row_policy: delayed_1m_synthetic_decision_row
second_level:
  enabled: true
  feature_profile: expanded_v2
  require_agg_trade_through_last_second: true
  max_agg_trade_lag_seconds: 2.0
```

The current deploy artifact includes `sl_` features, so 1s and aggTrades data are part of the live model input path.

## Order Pricing

For the default first leg, when `best_bid` is available:

```text
raw_price = min(best_bid * reference_multiplier + offset, price_cap)
price = floor_to_tick(max(raw_price, min_price), tick_size)
```

With the current defaults this is approximately:

```text
price = floor_to_tick(max(min(best_bid + 0.01, 0.75), 0.10), tick_size)
```

The second leg is disabled by default. If enabled, it uses its own cap, multiplier, rounding, and size. `execution_edge.enabled` remains false; enabling it adds edge, spread, price, and notional skip guards and should be treated as a separate policy change.

## Timing

For market window `[T,T+5m)`:

```text
prewarm timer: T+00:23
execution timer: T+01:08
decision_time: T+01:00
feature_timestamp: T+01:00
latest 1m candle: T
latest required 1s: T+00:59
latest required agg: T+00:57 by default
market slug: btc-updown-5m-<T epoch>
```

`run_once.py` also honors `schedule.trigger_delay_seconds` internally when no target window is passed. The systemd timer should still fire at the intended delay so the service does not start late in the market window.

## Local Smoke Test

Create a local runtime config:

```bash
cp execution_engine/config.example.yaml execution_engine/config.yaml
```

For a non-submitting smoke test, set:

```yaml
runtime:
  mode: paper
orders:
  enabled: false
```

Prewarm and run one cycle:

```bash
. .venv/bin/activate
python execution_engine/prewarm.py \
  --config execution_engine/config.yaml \
  --cache-output artifacts/state/execution_engine/prewarm \
  --print-json
python execution_engine/run_once.py \
  --config execution_engine/config.yaml \
  --mode paper \
  --print-json
```

Outputs are written to:

```text
artifacts/logs/execution_engine/live.jsonl
artifacts/logs/execution_engine/summaries/
```

## Linux Deploy

Canonical deployment path:

```text
/home/ubuntu/opt/crypto_engine
```

Use the tracked config template as the source of truth. For production updates, back up the server config and replace it from the tracked template before applying environment-only secrets in `execution_engine/secrets.env`.
Systemd examples live under `execution_engine/scheduler/`:

```text
execution-engine-prewarm.timer.example T+00:23
execution-engine.timer.example T+01:08
execution-engine-prewarm.service.example
execution-engine.service.example
```

Live credentials must stay out of git:

```bash
cat > execution_engine/secrets.env <<'EOF'
POLYMARKET_PRIVATE_KEY=...
CLOB_API_KEY=...
CLOB_SECRET=...
CLOB_PASS_PHRASE=...
POLYMARKET_SIGNATURE_TYPE=3
POLYMARKET_FUNDER=0x...
EOF
chmod 600 execution_engine/secrets.env
```

## Verification

Verify the loaded baseline:

```bash
python - <<'PY'
from execution_engine.artifacts import load_baseline_artifact
from execution_engine.config import load_execution_config

cfg = load_execution_config("execution_engine/config.yaml")
artifact = load_baseline_artifact(cfg.baseline)
print(artifact.manifest["experiment_id"])
print(artifact.manifest["training_mode"])
print(artifact.t_up, artifact.t_down)
print(len(artifact.feature_columns))
PY
```

Expected:

```text
20260611_catboost_calendar_coordinate_search
calendar_coordinate_offline_train
0.5792857142857143 0.43142857142857144
569
```

In live or paper summaries, check at least:

```text
signal.t0 == market.window_start
signal.feature_offset_minutes == 1
signal.row_policy == delayed_1m_synthetic_decision_row
signal.feature_timestamp == signal.t0 + 1m
signal.minute_latest == signal.t0
signal.second_latest < signal.t0 + 1m
signal.agg_trade_latest < signal.t0 + 1m
signal.t_up / signal.t_down
signal.price_estimator_expected_return_bid
orders
submitted
```

## Common Failures

`market_not_found`: Check clock sync, Gamma API availability, and the BTC 5-minute slug.
`missing_best_bid`: The current guard requires a target-token best bid before order planning.
`Runtime feature frame is missing ... baseline features`: Check Binance 1m, 1s, aggTrades coverage and `second_level.enabled`.
`expected_return_policy_rejected`: A leg using `expected_return_optimal_bid` did not receive an eligible positive H14 bid or lacked `best_ask`.
`idempotency_key_already_seen`: The same window, token, side, and leg already submitted or attempted an order.
