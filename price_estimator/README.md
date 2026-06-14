# Price Estimator

Independent Polymarket BTC 5m winner-token price estimator.

The active modeling flow is now `upper_bound_mlp`: one PyTorch MLP predicts a
raw logit and `p_upper_bound = sigmoid(z_raw)` as a tight upper bound over
`target_raw`.

The old CatBoost Q70/Q80/Q90 quantile scripts and artifacts are retained only
as historical/deprecated baseline material.

Target build command:

```powershell
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/fetch_btc5m_sell_taker_trades.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/build_price_target.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
```

Upper-bound MLP command:

```powershell
rtk python price_estimator/upper_bound_mlp/train_upper_bound_mlp.py --config price_estimator/upper_bound_mlp/configs/upper_bound_mlp_aug_lagrangian.yaml
```

`time_to_lowest_trade_sec` is saved for evaluation strata only and is not used
as a model input.
