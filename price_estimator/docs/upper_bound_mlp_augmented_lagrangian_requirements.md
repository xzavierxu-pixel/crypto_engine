# Route A: Single-Model Upper-Bound MLP with Augmented Lagrangian

## 1. Goal

Train a single PyTorch MLP model to predict calibrated probability `p_i` directly.
训练时用gpu，infer时用cpu
The optimization target is:

```text
minimize sum(p_i - y_i)
subject to p_i >= y_i + epsilon
```

This means the model should learn the tightest possible upper bound over the realized target `y_i`, with minimal excess gap.

---

## 2. Model Choice

Use one standalone PyTorch MLP model.

No CatBoost model, no per-bin correction, no post-calibration table.

Model input:

```text
all selected numerical / categorical encoded features
```

Model output:

```text
z_i = raw logit
p_i = sigmoid(z_i)
```

The model must output raw logits, not probabilities directly.

---

## 3. Model Structure

Recommended first version:

```text
Input
→ Linear(256) + SiLU + LayerNorm + Dropout(0.10)
→ Linear(128) + SiLU + LayerNorm + Dropout(0.05)
→ Linear(64) + SiLU
→ Linear(1)
→ raw logit z_i
```

Final probability:

```python
p_i = sigmoid(z_i)
```

---

## 4. Constraint Definition

For each sample:

```text
target_p_i = y_i + epsilon
```

Clip before logit transform:

```python
target_p_i = clip(y_i + epsilon, 1e-6, 1 - 1e-6)
target_z_i = logit(target_p_i)
```

The hard constraint becomes:

```text
z_i >= target_z_i
```

Define violation:

```text
g_i = target_z_i - z_i
```

If `g_i > 0`, the sample violates the constraint.

If `g_i <= 0`, the sample satisfies the constraint.

---

## 5. Augmented Lagrangian Training Objective

Each training sample has its own multiplier:

```text
alpha_i >= 0
```

Use the canonical inequality augmented Lagrangian penalty:

```python
shifted = g_i + alpha_i / rho
aug_penalty_i = 0.5 * rho * (relu(shifted) ** 2 - (alpha_i / rho) ** 2)
```

Total loss:

```python
loss = mean(p_i - y_i) + mean(aug_penalty_i)
```

Where:

```python
p_i = sigmoid(z_i)
g_i = target_z_i - z_i
```

---

## 6. Multiplier Update

After each optimizer step, recompute violation and update alpha:

```python
alpha_i = max(0, alpha_i + rho * g_i)
```

Recommended to clip alpha:

```python
alpha_i = clip(alpha_i, 0, alpha_max)
```

Initial settings:

```text
rho = 10
alpha_i = 0 for all training samples
alpha_max = 1000
```

If violation does not decrease, gradually increase:

```text
rho: 10 → 30 → 100 → 300
```

---

## 7. Training Requirements

The dataloader must return sample indices:

```python
x_batch, y_batch, idx_batch
```

This is required because `alpha_i` is stored per training sample and updated by `idx_batch`.

The model should be trained on the training set only.

Validation set is used only for evaluation and model selection.

---

## 8. Evaluation Metrics

For train and validation, calculate:

```python
gap = p_i - y_i
violation = y_i + epsilon - p_i
```

Required metrics:

```text
mean_gap = mean(p_i - y_i)
min_gap = min(p_i - y_i)
violation_rate = mean(p_i < y_i + epsilon)
max_violation = max(y_i + epsilon - p_i)
coverage = mean(p_i >= y_i + epsilon)
```

Model selection priority:

```text
1. train violation_rate == 0
2. train min_gap >= epsilon
3. smallest train mean_gap
4. validation violation_rate as low as possible
5. validation mean_gap not excessively large
```

---

## 9. Early Stopping Logic

Do not select model by ordinary validation loss only.

Primary selection should follow feasibility first:

```text
first satisfy constraint, then minimize gap
```

Save checkpoints where:

```text
train violation_rate == 0
```

Among feasible checkpoints, choose the one with the lowest train `mean_gap`, while checking validation stability.

---

## 10. Key Implementation Notes

- Use logit-domain constraint, not probability-domain constraint.
- Always clip `y_i + epsilon` before applying logit.
- The final layer should not use sigmoid inside the model.
- Sigmoid should only be applied during loss and evaluation.
- Monitor top alpha samples to detect outliers.
- If a few samples dominate alpha, inspect whether the target data has noise.
- This method enforces constraints on the training set; it does not mathematically guarantee unseen validation samples satisfy the constraint.

---

## 11. Expected Output Artifacts

The experiment should produce:

```text
1. trained PyTorch model checkpoint
2. training metrics by epoch
3. validation metrics by epoch
4. final prediction file with columns:
   - sample_id
   - y
   - z_raw
   - p_pred
   - gap
   - violation
   - alpha
5. summary report comparing:
   - baseline q99 model
   - MLP augmented Lagrangian model
```

---

## 12. Success Criteria

Minimum success criteria:

```text
train violation_rate == 0
train min_gap >= epsilon
train mean_gap is materially lower than conservative q99 baseline
validation violation_rate is acceptable and does not explode
```

Main target:

```text
A single MLP model learns p_i directly as the tightest feasible upper bound over y_i.
```
