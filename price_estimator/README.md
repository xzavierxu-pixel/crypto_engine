# Price Estimator

Independent Polymarket BTC 5m winner-token price estimator.

The active baseline is now `safe_lowest_price_gap`: one low-latency model
predicts a raw safe-price location `f(X)` in probability space, then a calibrated
global margin plus bucket abstain policy turns it into `p_pred <= p_side`.
For execution deployment, use the deploy-feature experiment
`price_estimator/safe_lowest_price_gap/experiments/20260617_deploy_baseline_features/config.yaml`
and the tracked artifact in
`execution_engine/deploy/price_estimator_safe_lowest_price_gap`.

Primary objective:

```text
minimize covered_gap_norm.mean
subject to coverage_feasible >= 0.90
```

Where:

```text
target_raw = lowest_trade_price_next4
y          = target_safe = target_raw + 0.01
p          = p_side
s          = p - y
feasible   = y <= p
covered    = p_pred >= y and p_pred <= p
gap_norm   = (p_pred - y) / s, only for covered feasible rows
```

The baseline trains on all rows, including infeasible rows where `y > p_side`.
`target_safe`, `s`, and `s_eff` are label-derived quantities used only by the
training loss and offline evaluation; they must not be model features.

The old `upper_bound_mlp` and CatBoost Q70/Q80/Q90 quantile scripts and
artifacts are retained as historical/deprecated baseline material.

Target build command:

```powershell
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/fetch_btc5m_sell_taker_trades.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/build_price_target.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
```

Safe lowest-price normalized-gap baseline:

```powershell
rtk python price_estimator/safe_lowest_price_gap/train_safe_lowest_price_gap.py --config price_estimator/safe_lowest_price_gap/config.yaml
```

This script:

- splits the train dataset chronologically, using the last 31 days as
  calibration;
- trains a single MLP with normalized asymmetric loss for `alpha in {2,4,8}`;
- selects `delta` and bucket abstain threshold on calibration only;
- evaluates the selected configuration once on validation;
- writes `summary_metrics.json`, `calibration_frontier.csv`, predictions, and a
  model checkpoint under `price_estimator/safe_lowest_price_gap/reports` and
  `price_estimator/safe_lowest_price_gap/models`.

Deprecated upper-bound MLP command:

```powershell
rtk python price_estimator/upper_bound_mlp/train_upper_bound_mlp.py --config price_estimator/upper_bound_mlp/configs/upper_bound_mlp_aug_lagrangian.yaml
```

`time_to_lowest_trade_sec` is saved for evaluation strata only and is not used
as a model input.
