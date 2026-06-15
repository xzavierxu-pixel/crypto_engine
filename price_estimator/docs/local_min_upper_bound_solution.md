# Local Minimum Upper Bound Prediction Scheme

## 1. Objective

For each sample:

```text
covered_i = 1[p_pred_i >= y_i + epsilon]
```

Main optimization objective:

```text
minimize covered_gap
subject to accepted_coverage >= 0.70
```

Where:

```text
accepted_coverage = mean(p_pred_i >= y_i + epsilon | accepted_i = 1)
covered_gap = mean(p_pred_i - y_i | accepted_i = 1 and covered_i = 1)
```

Side constraint:

```text
p_side_i - 0.5 <= p_pred_i <= p_side_i
```

Important principle:

```text
Samples that cannot be covered with a small margin should be rejected, not forced to be covered.
```

---

## 2. Step 1: Train Base Fitting Model

Train a center prediction model to fit the target `y` as accurately as possible.

Recommended loss:

```text
log_cosh_loss = log(cosh(pred - y))
```

Purpose:

```text
mu_i = base_model(x_i) ≈ y_i
```

This model should not directly optimize coverage. It should only learn a good local estimate of `y`.

Recommended base model candidates:

```text
LogCosh
Huber
MSE
MAE
```

Primary evaluation for base model:

```text
MAE
RMSE
median_abs_error
residual_q70
residual_q90
```

---

## 3. Step 2: Train Local Scale Model

After base model prediction:

```text
mu_i = base_model(x_i)
residual_i = y_i - mu_i
abs_residual_i = abs(y_i - mu_i)
```

Train a scale model:

```text
sigma_i = scale_model(x_i)
```

Target:

```text
abs_residual_i
```

Purpose:

```text
sigma_i ≈ local prediction uncertainty
```

Apply a floor to avoid unstable normalized scores:

```text
sigma_i = max(sigma_i, sigma_floor)
```

Recommended initial value:

```text
sigma_floor = 0.01
```

---

## 4. Step 3: Normalized Conformal Calibration

On the calibration set, compute signed normalized scores:

```text
score_i = (y_i - mu_i) / sigma_i
```

Take the target coverage quantile:

```text
q = Quantile_70%(score_i)
```

For each prediction sample:

```text
required_margin_i = q * sigma_i
p_raw_i = mu_i + required_margin_i
```

Interpretation:

```text
p_raw_i = center prediction + local minimum required buffer
```

---

## 5. Step 4: Selective Gate

Reject samples that require too much margin or violate the side upper bound.

Recommended logic:

```text
if p_raw_i > p_side_i:
    accepted_i = 0

elif required_margin_i > margin_threshold:
    accepted_i = 0

else:
    p_pred_i = max(p_raw_i, p_side_i - 0.5)
    accepted_i = 1
```

Important rule:

```text
Do not clip p_raw_i down to p_side_i when p_raw_i > p_side_i.
```

Reason:

```text
If p_raw_i > p_side_i, the sample needs a price above the side-implied fair value to satisfy coverage. Clipping it down would break coverage and create hidden under-prediction.
```

---

## 6. Step 5: Evaluation Metrics

Evaluate only on accepted samples.

Required metrics:

```text
accepted_rate = mean(accepted_i)
accepted_coverage = mean(p_pred_i >= y_i + epsilon | accepted_i = 1)
covered_mean_gap = mean(p_pred_i - y_i | accepted_i = 1 and p_pred_i >= y_i + epsilon)
covered_median_gap
covered_q90_gap
covered_q90_q10_gap
side_violation_rate = mean(p_pred_i > p_side_i | accepted_i = 1)
```

Valid model condition:

```text
accepted_coverage >= 0.70
side_violation_rate = 0
```

Optional minimum trade constraint:

```text
accepted_rate >= min_accepted_rate
```

---

## 7. Model Selection Rule

Among valid models, select the one with the smallest covered gap.

Recommended score:

```text
score =
    covered_mean_gap
    + 0.5 * covered_q90_gap
    + 2.0 * max(0, 0.70 - accepted_coverage)
    + 0.2 * max(0, min_accepted_rate - accepted_rate)
```

If no minimum trade rate is required, remove the last term.

Final selection priority:

```text
1. accepted_coverage >= 0.70
2. side_violation_rate = 0
3. minimize covered_mean_gap
4. minimize covered_q90_gap
5. maintain acceptable accepted_rate
```

---

## 8. Final Prediction Formula

Final production logic:

```text
mu = base_model(x)
sigma = max(scale_model(x), sigma_floor)
required_margin = q * sigma
p_raw = mu + required_margin

if p_raw > p_side:
    reject
elif required_margin > margin_threshold:
    reject
else:
    p_pred = max(p_raw, p_side - 0.5)
    accept
```

The final model is therefore:

```text
LogCosh base fitting model
+ local scale model
+ normalized conformal residual calibration
+ selective margin gate
+ p_side feasibility constraint
```

---

## 9. Key Design Principle

Do not train a single asymmetric loss to solve both fitting and coverage.

Use a two-stage structure:

```text
Stage 1: fit y well
Stage 2: calibrate the minimum local residual buffer required for coverage
Stage 3: reject samples that require excessive margin
```

This directly targets:

```text
small covered gap
stable accepted coverage
no forced coverage of bad samples
```
