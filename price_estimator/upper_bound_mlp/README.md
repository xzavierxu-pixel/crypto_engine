# Upper-Bound MLP Price Estimator

This is the active replacement for the deprecated CatBoost quantile flow.

It trains one PyTorch MLP that outputs raw logits `z_raw`; probabilities are
computed as `p_upper_bound = sigmoid(z_raw)`.

Training optimizes a tight upper bound over `target_raw` with the constraint:

```text
p_upper_bound >= target_raw + epsilon
```

The augmented Lagrangian constraint is applied in logit space. A final scalar
bias repair can be enabled to shift the MLP's last-layer bias by the minimum
amount needed to make the training set feasible. This remains a single MLP
checkpoint and does not use per-bin correction or a calibration table.

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
