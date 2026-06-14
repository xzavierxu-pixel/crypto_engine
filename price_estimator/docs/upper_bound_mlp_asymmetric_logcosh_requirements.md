# Route A: Single-Model Upper-Bound MLP with Asymmetric Log-Cosh Loss

## 1. Goal

Train a single PyTorch MLP model to predict an upper-bound probability `p_pred` directly.

The practical objective is no longer hard feasibility on every sample. The objective is:

```text
minimize validation mean_gap
subject to validation coverage >= target_coverage
```

Where:

```text
mean_gap = mean(p_pred - y_raw)
coverage = mean(p_pred >= y_raw + epsilon)
```

This replaces the previous hard Augmented Lagrangian objective, because hard constraints were too sensitive to outliers and produced excessive mean_gap.

---

## 2. Target and Output Definition

Use raw target price as the training target:

```text
y_raw = target_raw / lowest_trade_price_next4
```

Do not use `target_logit` as `y`.

Model output:

```text
z_raw = MLP(features)
p_pred = sigmoid(z_raw)
```

Target upper bound:

```text
target = clip(y_raw + epsilon, 1e-6, 1 - 1e-6)
```

All loss terms are calculated in probability space, not logit space.

---

## 3. Model Choice

Use one standalone PyTorch MLP model.

No CatBoost model, no per-bin correction, no post-calibration table.

Recommended first structure:

```text
Input
→ Linear(256) + SiLU + LayerNorm + Dropout(0.10)
→ Linear(128) + SiLU + LayerNorm + Dropout(0.05)
→ Linear(64) + SiLU
→ Linear(1)
→ raw logit z_raw
→ sigmoid(z_raw) = p_pred
```

The final layer must output raw logits. Do not put sigmoid inside the model module.

---

## 4. Asymmetric Log-Cosh Loss

Define:

```python
target = torch.clamp(y_raw + epsilon, 1e-6, 1 - 1e-6)
p_pred = torch.sigmoid(z_raw)

under = torch.relu(target - p_pred)  # violation / under-cover
over = torch.relu(p_pred - target)   # excess upper-bound gap
```

Use asymmetric log-cosh:

```python
def log_cosh(x, scale=0.05):
    u = x / scale
    return scale * scale * torch.log(torch.cosh(u))

loss = (
    w_under * log_cosh(under, scale)
    + w_over * log_cosh(over, scale)
).mean()
```

Recommended default:

```text
w_over = 1.0
w_under = 5.0 or 10.0
scale = 0.03 or 0.05
```

Rationale:

```text
under-cover should be punished more than over-cover,
but large outlier violations should not dominate the whole model.
```

---

## 5. Alternative Preferred Variant: Mean-Gap + Soft Violation Penalty

Because the main KPI is low `mean_gap`, also support this variant:

```python
gap = p_pred - y_raw
violation = torch.relu(target - p_pred)

loss = gap.mean() + C * log_cosh(violation, scale).mean()
```

Recommended grid:

```text
C = 2, 5, 10, 20, 50
scale = 0.02, 0.03, 0.05, 0.08
```

This variant directly optimizes the business objective:

```text
reduce mean_gap while charging a soft cost for violations
```

This should be treated as the primary experiment if `mean_gap` is more important than strict coverage.

---

## 6. Hyperparameter Grid

Run both variants if possible.

### Variant A: Asymmetric Log-Cosh

```text
w_under = 3, 5, 10, 20
w_over = 1
scale = 0.02, 0.03, 0.05, 0.08
```

### Variant B: Mean-Gap + Soft Log-Cosh Violation

```text
C = 2, 5, 10, 20, 50
scale = 0.02, 0.03, 0.05, 0.08
```

Do not use per-sample `alpha_i` or Augmented Lagrangian multiplier updates in this version.

---

## 7. Model Selection Criteria

Primary model selection:

```text
1. validation coverage >= target_coverage
2. validation mean_gap is smallest
3. validation p95_violation and p99_violation are acceptable
4. validation max_violation is monitored but should not dominate selection
```

Recommended target coverage levels to test:

```text
target_coverage = 0.85, 0.90
```

Do not require:

```text
train violation_rate == 0
train min_gap >= epsilon
```

Those hard feasibility criteria caused excessive mean_gap and are no longer the main goal.

---

## 8. Required Evaluation Metrics

For train and validation, calculate:

```python
gap = p_pred - y_raw
violation = torch.relu(y_raw + epsilon - p_pred)
```

Required summary metrics:

```text
coverage
violation_rate
mean_gap
median_gap
min_gap
max_violation
p90_violation
p95_violation
p99_violation
sample_count
```

Also report diagnostics by:

```text
price_bin based on y_raw
p_bin based on p_side
selected_side
```

---

## 9. Diagnostic Fix: p_bin

Do not put `p_side < 0.5` into `missing`.

Use complete p-side bins:

```text
0.00-0.10
0.10-0.20
0.20-0.30
0.30-0.40
0.40-0.50
0.50-0.55
0.55-0.60
0.60-0.65
0.65-0.70
0.70-0.80
0.80-0.90
0.90-1.00
missing only for true null values
```

This is for diagnostics and optional feature encoding only.

---

## 10. Price-Bin Diagnostics

The previous ALM result showed:

```text
low price bins had excessive mean_gap
high price bins had weak coverage
```

Therefore, always report by `target_raw` price bin:

```text
0.00-0.10
0.10-0.20
0.20-0.30
0.30-0.40
0.40-0.50
0.50-0.60
0.60-0.70
0.70-0.80
0.80-0.90
0.90-1.00
```

The desired improvement is:

```text
lower overall mean_gap
no massive low-price over-cover waste
acceptable high-price coverage
```

---

## 11. Early Stopping

Do not early stop on training loss only.

Use validation-based selection:

```text
eligible checkpoint: validation coverage >= target_coverage
best checkpoint: lowest validation mean_gap among eligible checkpoints
tie-breaker: lower p99_violation
```

If no checkpoint reaches target coverage, select the checkpoint with the best coverage / mean_gap trade-off and mark the run as not passing.

---

## 12. Expected Output Artifacts

The experiment should produce:

```text
1. trained PyTorch model checkpoint
2. config YAML used for training
3. epoch_metrics.csv
4. predictions_train.parquet
5. predictions_validation.parquet
6. diagnostics_by_price_bin.csv
7. diagnostics_by_p_bin.csv
8. summary metrics JSON
```

Prediction files should include:

```text
sample_id
y_raw
target
z_raw
p_pred
gap
violation
selected_side
p_side
p_bin
price_bin
```

---

## 13. Success Criteria

Minimum pass condition:

```text
validation coverage >= 0.85
validation mean_gap materially below ALM baseline
p99_violation does not explode
```

Preferred pass condition:

```text
validation coverage >= 0.90
validation mean_gap materially below 0.20
p99_violation acceptable
price-bin diagnostics show less low-price gap waste
```

Main target:

```text
A single MLP model learns a low-gap soft upper bound over y_raw, without being dominated by rare outlier samples.
```
