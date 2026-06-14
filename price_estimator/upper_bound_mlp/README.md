# Upper-Bound MLP Price Estimator

This is the active replacement for the deprecated CatBoost quantile flow.

It trains one PyTorch MLP that outputs raw logits `z_raw`; probabilities are
computed as `p_upper_bound = sigmoid(z_raw)`.

Training optimizes a tight upper bound over `target_raw` with the constraint:

```text
p_upper_bound >= target_raw + epsilon
```

The augmented Lagrangian constraint is applied in logit space with configurable
price-aware constraint weights. Model selection requires validation coverage
and the configured max-violation cap first, then minimizes validation mean gap,
without final-bias repair or per-bin calibration.

Run:

```powershell
rtk python price_estimator/upper_bound_mlp/train_upper_bound_mlp.py --config price_estimator/upper_bound_mlp/configs/upper_bound_mlp_aug_lagrangian.yaml
```

Primary outputs:

- `price_estimator/upper_bound_mlp/models/upper_bound_mlp.pt`
- `price_estimator/upper_bound_mlp/reports/epoch_metrics.csv`
- `price_estimator/upper_bound_mlp/reports/predictions_train.parquet`
- `price_estimator/upper_bound_mlp/reports/predictions_validation.parquet`
- `price_estimator/upper_bound_mlp/reports/upper_bound_mlp_metrics.json`

The summary report includes `validation_diagnostics.by_p_bin`,
`validation_diagnostics.by_p_side_bin`, and
`validation_diagnostics.by_price_bin` with sample count, coverage, violation
rate, mean gap, and max violation.
