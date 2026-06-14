# Price Estimator

Independent Polymarket BTC 5m winner-token price estimator baseline.

This module reads existing direction-model feature frames, uses the deploy
artifact predictions from `execution_engine/deploy/baseline`, builds
`lowest_trade_price_next4` from Polymarket SELL/takerOnly trades, and trains
CatBoost quantile regressors for Q70/Q80/Q90.

Baseline commands:

```powershell
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/fetch_btc5m_sell_taker_trades.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/build_price_target.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
rtk proxy powershell -NoProfile -Command "python price_estimator/scripts/train_catboost_quantile.py --config price_estimator/configs/catboost_quantile_baseline.yaml"
```

`time_to_lowest_trade_sec` is saved for evaluation strata only and is not used
as a model input.
