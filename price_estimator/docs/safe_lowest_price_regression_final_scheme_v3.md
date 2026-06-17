# Safe Lowest Price Regression 最终方案

## 目标

训练一个回归模型，在 inference 时只已知 `p_side`，输出最终价格 `p_pred`。

训练 / validation 时已知：

```text
p_side
lowest_price
```

inference 时已知：

```text
p_side
```

目标：

```text
p_pred <= p_side
p_pred 尽量接近 lowest_price + 0.01
尽量满足 p_pred >= lowest_price + 0.01
最终 p_pred 输出两位小数
```

其中：

```python
buffer = 0.01
```

---

## 1. 核心定义

先定义安全最低价格：

```python
lowest_safe = lowest_price + buffer
```

定义最大安全下调比例：

```python
eps = 1e-8

target_r = np.maximum(p_side - lowest_safe, 0) / np.maximum(p_side, eps)
```

含义：

```text
target_r = 在不低于 lowest_price + 0.01 的前提下，
           p_side 最多可以向下偏离的比例
```

如果：

```text
lowest_price + 0.01 >= p_side
```

则：

```text
target_r = 0
```

此时没有安全下调空间，最优输出就是：

```text
p_pred = p_side
```

---

## 2. 模型输出

模型预测一个非负下调比例：

```python
r_hat >= 0
```

最终价格由：

```python
p_pred_raw = p_side * (1 - r_used)
```

得到。

安全条件是：

```python
r_used <= target_r
```

如果：

```text
r_used = target_r
```

则：

```text
p_pred_raw = lowest_price + 0.01
```

刚好最优。

如果：

```text
r_used < target_r
```

则预测更保守，`p_pred` 更高，安全但 gap 更大。

如果：

```text
r_used > target_r
```

则：

```text
p_pred < lowest_price + 0.01
```

违反安全目标。

---

## 3. 训练目标不是普通 MSE

`target_r` 不是普通 regression target，而是每个样本的最大安全上界。

因此训练目标不是简单拟合：

```text
r_hat ≈ target_r
```

而是：

```text
r_hat 尽量接近 target_r，但不要超过 target_r
```

等价于：

```text
maximize r_hat
subject to r_hat <= target_r
```

---

## 4. 推荐 Loss：Normalized Safe Interval Loss

定义：

```python
eps = 1e-8

positive = target_r > eps
u = r_hat / np.maximum(target_r, eps)
```

其中：

```text
u < 1  安全，但预测偏保守
u = 1  最优
u > 1  预测过头，会低于 lowest_price + 0.01
```

loss：

```python
under = np.maximum(1 - u, 0)
over = np.maximum(u - 1, 0)

loss_pos = beta * np.log1p(under) + alpha * over**2
```

对于没有安全下调空间的样本：

```python
loss_zero = alpha_zero * r_hat**2
```

完整形式：

```python
loss = np.where(
    positive,
    beta * np.log1p(np.maximum(1 - u, 0))
    + alpha * np.maximum(u - 1, 0)**2,
    alpha_zero * r_hat**2
)
```

推荐初始参数：

```python
alpha = 10.0
beta = 1.0
alpha_zero = 10.0
```

含义：

```text
预测太保守：轻罚
预测低于 lowest_price + 0.01：重罚
没有安全空间却下调：重罚
```

---

## 5. 可选 Target Transform

如果模型框架更适合学习 transformed target，可以使用：

```python
scale_r = 0.01

z_target = np.log1p(target_r / scale_r)
```

其中：

```python
np.log1p(x) = log(1 + x)
```

当：

```text
target_r = 0
```

有：

```text
z_target = 0
```

不会出现 `log(0)` 问题。

但最终推荐的核心仍然是：

```text
在 loss 中惩罚 r_hat > target_r
```

而不是普通拟合 `z_target` 的均值。

---

## 6. Inference：使用单一 r_limit 控制 coverage

最终只使用一个全局控制参数：

```python
r_limit
```

含义：

```text
最终价格最多允许相对 p_side 向下偏离 r_limit
```

应用：

```python
r_used = np.minimum(r_hat, r_limit)
```

然后：

```python
p_pred_raw = p_side * (1 - r_used)
```

`r_limit` 越小：

```text
p_pred 越接近 p_side
coverage 越高
mean_gap 越大
```

`r_limit` 越大：

```text
p_pred 越接近 lowest_price + 0.01
coverage 可能下降
mean_gap 可能下降
```

---

## 7. 两位小数输出

最终价格只做一次向上取两位小数：

```python
p_pred = np.ceil(p_pred_raw * 100 - 1e-12).astype(int) / 100
```

不要再额外使用：

```python
np.round(...)
np.minimum(p_pred, p_side)
```

前提是：

```text
p_side 本身已经是两位小数 tick price
```

---

## 8. Validation 选择 r_limit

在 validation 上扫描 `r_limit`：

```python
target_coverage = 0.70

best = None

for r_limit in np.arange(0.00, 0.401, 0.0025):

    r_used = np.minimum(r_hat_valid, r_limit)

    p_pred_raw = p_side_valid * (1 - r_used)

    p_pred = np.ceil(p_pred_raw * 100 - 1e-12).astype(int) / 100

    covered = p_pred >= lowest_price_valid + 0.01

    coverage = covered.mean()
    mean_gap = np.mean(p_pred - lowest_price_valid)
    covered_gap = np.mean(p_pred[covered] - lowest_price_valid[covered])

    if coverage >= target_coverage:
        score = mean_gap

        if best is None or score < best["score"]:
            best = {
                "r_limit": r_limit,
                "coverage": coverage,
                "mean_gap": mean_gap,
                "covered_gap": covered_gap,
                "score": score,
            }
```

选择规则：

```text
在 coverage >= target_coverage 的候选 r_limit 中，
选择 mean_gap 最小的 r_limit。
```

---

## 9. 最终 Inference 公式

```python
# model predicts r_hat
r_hat = model.predict(X)
r_hat = np.maximum(r_hat, 0)

# apply global coverage control
r_used = np.minimum(r_hat, best_r_limit)

# price
p_pred_raw = p_side * (1 - r_used)

# two-decimal safe output
p_pred = np.ceil(p_pred_raw * 100 - 1e-12).astype(int) / 100
```

---

## 10. 最终结论

最终方案是：

```text
1. 用 p_side 和 lowest_price + 0.01 构造最大安全下调比例 target_r
2. 模型预测 r_hat，即相对 p_side 的下调比例
3. 训练时不做普通 MSE，而是使用 Normalized Safe Interval Loss
4. loss 轻罚 r_hat < target_r，重罚 r_hat > target_r
5. inference 时用单一 r_limit 控制最大下调比例和 coverage
6. 最终 p_pred 向上取两位小数
```

核心公式：

```python
target_r = max(p_side - (lowest_price + 0.01), 0) / p_side

u = r_hat / target_r

loss = beta * log1p(max(1 - u, 0)) + alpha * max(u - 1, 0)^2

r_used = min(r_hat, r_limit)

p_pred = ceil_to_2dp(p_side * (1 - r_used))
```
