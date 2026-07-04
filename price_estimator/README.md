# Price Estimator

Price estimation for the Polymarket BTC 5m execution engine.

The current execution workflow uses the expected-return hazard artifact:

```text
active_artifact: expected_return_h14
artifact_dir: execution_engine/deploy/price_estimator_expected_return_h14
model_file: expected_return_hazard.npz
prediction_column: expected_return_bid
```

This artifact is `20260619_expected_return_h14_h2_gc_gt_0p75`. It predicts the
bid for the classifier-selected side (`selected_side` = `UP` or `DOWN`) and the
runtime prices the order as:

```text
min(best_ask - 0.01, expected_return_optimal_bid)
```

The deployed policy skips rows without a valid price-estimator output. The
manifest reports 576 features, validation coverage `0.7000535618639528`, order
coverage `0.49674827850038256`, and validation `sum_pnl = 27.44`.

`safe_lowest_price_gap` is retained as a historical normalized-gap baseline and
backup artifact, but it is no longer the active execution price estimator. The
old `upper_bound_mlp` and CatBoost quantile material is deprecated.
