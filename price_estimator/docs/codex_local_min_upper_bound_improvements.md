## Codex Task: Two Improvements for Local Min Upper Bound

Please implement two changes for `price_estimator/upper_bound_mlp`.

### 1. Exclude high `p_side` samples

Add a hard acceptance filter:

```python
eligible = p_side < 0.80
```

Config:

```yaml
acceptance_filter:
  enabled: true
  max_p_side: 0.80
```

Apply before sample acceptance/ranking. Report:

- removed sample count/rate
- accepted_count
- accepted_rate
- accepted_coverage
- covered_mean_gap
- covered_q90_gap
- side_violation_rate

Add assertion:

```python
assert accepted_df["p_side"].max() < 0.80
```

Reason: latest diagnostics show `p_side >= 0.80` has very high gap, especially `0.80-0.90` with mean gap around `0.347`, so these samples are toxic for minimizing gap.

---

### 2. Add CatBoost LogCosh base model

Add alternative base model:

```yaml
base_model:
  type: catboost_logcosh
  loss_function: LogCosh
  iterations: 2000
  learning_rate: 0.03
  depth: 6
  l2_leaf_reg: 10
  random_seed: 42
  early_stopping_rounds: 100
  eval_metric: MAE
```

Use:

```python
target_column = "target_raw"
mu = np.clip(mu, 0.001, 0.999)
```

Then reuse existing `non_normalized_local_conformal`.

Compare three variants:

```text
1. current MLP + non_normalized_local_conformal
2. CatBoost LogCosh + non_normalized_local_conformal
3. CatBoost LogCosh + p_side<0.80 filter + non_normalized_local_conformal
```

Main success metric:

```text
validation covered_mean_gap < current 0.144
accepted_coverage >= 0.70
side_violation_rate == 0
```

Also report validation base MAE, covered_q90_gap, accepted_rate, and train-validation gap.
