# Local Min Upper Bound 优化方案

## 1. 当前结果摘要

本次实验：`20260615_local_min_upper_bound`

目标：

```text
minimize covered_mean_gap
subject to accepted_coverage >= 0.70
and side_violation_rate = 0
```

当前 validation 结果：

| 指标 | 数值 | 解释 |
|---|---:|---|
| sample_count | 7,432 | validation 总样本数 |
| accepted_count | 751 | 被 gate 接受的样本数 |
| accepted_rate | 10.10% | 只接受约 10% 样本 |
| accepted_coverage | 71.50% | 勉强满足 70% coverage 约束 |
| covered_mean_gap | 0.2167 | 主要优化目标，偏大 |
| covered_median_gap | 0.1945 | 中位数 gap 也偏大 |
| covered_q90_gap | 0.4133 | tail gap 很大 |
| side_violation_rate | 0.00% | side 约束满足 |

关键诊断：

| 模块 | Fit | Calibration | Validation |
|---|---:|---:|---:|
| base MAE | 0.0194 | 0.1500 | 0.1546 |
| base RMSE | 0.0275 | 0.1895 | 0.1928 |
| residual q70 | 0.0240 | 0.1925 | 0.2023 |
| residual q90 | 0.0408 | 0.3100 | 0.3191 |

结论：**base model 的 in-sample fit 很好，但 out-of-sample 泛化明显退化。最终 mean gap 大，不是因为训练集拟合不够，而是因为 validation residual 本身已经很大，再叠加 conformal safety margin。**

---

## 2. 核心问题拆解

### 2.1 base fit MAE 低是假象

当前 `fit MAE = 0.0194`，但 calibration / validation MAE 分别是 `0.1500 / 0.1546`。

所以不能用 fit MAE 判断模型质量。后续所有模型选择必须只看：

```text
calibration MAE
validation MAE
validation residual q70
validation residual q90
final covered_mean_gap
final covered_q90_gap
```

`fit MAE` 只能作为过拟合诊断指标，不能作为优化目标。

---

### 2.2 conformal margin 形成了结构性 gap floor

当前 calibration 参数：

```text
coverage_quantile = 0.9
q = 14.8629
sigma_floor = 0.01
```

所以最小 margin 为：

```text
q * sigma_floor = 14.8629 * 0.01 = 0.1486
```

validation 中：

```text
mean_required_margin = 0.1488
margin_threshold = 0.1520
```

这说明被接受的样本几乎都被加了约 `0.149` 的 margin。也就是说，即使 base prediction 完美等于 y，最终 gap 也很难低于 `0.149`。

当前 validation covered_mean_gap：

```text
0.2167 ≈ 0.1488 margin floor + 0.0679 extra validation slippage
```

因此最终 mean gap 大的主因是：

```text
1. base out-of-sample residual 大；
2. scale model 低估 sigma；
3. sigma_floor 被 q 放大，形成硬 margin floor；
4. gate 只是在 margin floor 附近选样本，没有真正选到 low-gap 样本。
```

---

### 2.3 scale model 没有学到真实不确定性

如果真实 residual 大约是 `0.15`，但 sigma 被压到 `0.01` 附近，则 normalized residual 会变成：

```text
0.15 / 0.01 = 15
```

这正好对应当前 `q = 14.86`。

这说明 scale model 没有给出合理的 uncertainty estimate，而是大量样本落在 `sigma_floor` 附近。结果是 conformal 只能通过一个巨大的 q 来补偿。

---

### 2.4 local 机制可能没有真正生效

validation accepted 样本中，大量样本落入 `<NA>` / `nan`：

```text
by_price_bin <NA>: 663 / 751
by_p_bin nan: 663 / 751
by_selected_side nan: 663 / 751
```

这说明当前 local calibration / local gating 可能没有真正使用 price bin、p bin、selected side 等条件信息。

如果 local 信息缺失，模型会退化成一个近似全局 conformal upper bound，无法针对不同 price 区间学习不同 margin。

这是最高优先级 bug。

---

## 3. 优化优先级

## P0：先修数据与指标口径

### P0.1 修复 p_bin / price_bin / selected_side 的 NA 问题

必须检查：

```text
1. p_bin 是否在 train / calibration / validation 全部生成；
2. selected_side 是否在 prediction artifact 中正确保留；
3. groupby diagnostics 是否因为 dtype / category / parquet serialization 导致 NA；
4. p_side / p_up 是否存在缺失；
5. local calibration 是否 fallback 到 global；
6. validation accepted 样本是否真的有可用的 local features。
```

验收标准：

```text
validation accepted 样本中：
price_bin / p_bin / selected_side 的 NA 占比 < 5%
```

如果这个不满足，不继续调参。

---

### P0.2 明确所有 gap 是否在同一空间计算

当前 target 曾使用 logit 形式训练，但最终约束是 probability price bound。必须确认以下变量的空间一致：

```text
y_raw: 原始价格 / 概率空间
z = logit(y_raw): logit 空间
base_pred_raw: probability 空间预测
base_pred_logit: logit 空间预测
p_final: 最终 probability 空间 upper bound
covered_gap = p_final - y_raw
```

禁止混用：

```text
p_final_raw - y_logit
base_pred_logit + margin_raw
clip 后再用 logit gap 做 coverage
```

验收标准：

```text
所有最终业务指标都在 probability 空间计算：
coverage = mean(p_final_raw >= y_raw + epsilon)
covered_gap = mean(p_final_raw - y_raw | covered)
side constraint = p_final_raw <= p_side
```

---

## P1：降低 base model 的 validation residual

当前最根本的问题是 base 泛化误差太大：

```text
fit MAE = 0.0194
validation MAE = 0.1546
```

这表示模型在训练集上学得太细，但 out-of-sample 不能稳定预测 lowest price。

### P1.1 模型选择只看 calibration / validation

训练过程中必须记录：

```text
train MAE
calibration MAE
validation MAE
calibration residual q70 / q90
validation residual q70 / q90
```

early stopping 不要看 train loss，而要看 calibration loss 或 rolling validation loss。

推荐选择指标：

```text
primary: validation residual q70
secondary: validation MAE
third: validation residual q90
```

原因：你的最终目标是 coverage 70%，所以 residual q70 比 MAE 更贴近业务目标。

---

### P1.2 降低 MLP 过拟合

建议先尝试以下正则化配置：

```yaml
base_model:
  hidden_dims: [256, 128]
  dropout: 0.10
  weight_decay: 0.001
  batch_norm: true
  early_stopping_metric: calibration_mae
  early_stopping_patience: 10
  max_epochs: 80
```

如果当前特征数量超过 1,000，MLP 很容易在 15k 样本上过拟合。可以优先减少模型容量，而不是继续增加 epoch。

---

### P1.3 增加更稳健的 base loss 对照组

建议跑三个 base 版本：

```text
A. symmetric log-cosh
B. asymmetric log-cosh
C. Huber / pseudo-Huber
```

如果目标是先学一个稳定的 conditional center / conditional lower movement，再由 calibration 做 upper bound，base loss 不应该过度追求极端 upper bound。

推荐主线：

```text
base model: symmetric log-cosh 或 pseudo-Huber
calibration: 负责 coverage
gate: 负责 low-gap selection
```

不建议让 base model 同时承担：

```text
fit y + satisfy upper bound + minimize covered gap + obey side cap
```

这几个目标会互相冲突。

---

## P2：重做 scale / conformal margin

### P2.1 先加入 non-normalized conformal baseline

当前 normalized conformal 出现 q 巨大问题，所以需要一个简单 baseline：

```text
residual_upper = max(y_raw - base_pred_raw, 0)
margin_global = quantile(residual_upper on calibration, target_q)
p_final = base_pred_raw + margin_global
```

然后再做 local 版本：

```text
margin_local = quantile(residual_upper | price_bin, p_side_bin, selected_side)
p_final = base_pred_raw + margin_local
```

这可以判断：问题到底来自 base model，还是来自 sigma model。

验收标准：

```text
non-normalized conformal 的 covered_mean_gap 必须作为 baseline 写入 report
normalized conformal 只有在 beat 这个 baseline 时才保留
```

---

### P2.2 scale model 改成直接预测 residual size

当前 scale model 很可能学出了过小 sigma。建议目标改为：

```text
scale_target = abs(y_raw - base_pred_raw)
或
scale_target = max(y_raw - base_pred_raw, 0)
```

输出层使用：

```text
sigma = softplus(raw_sigma) + sigma_min
```

loss 使用：

```text
log_cosh(log(sigma) - log(scale_target + eps))
```

或简单版本：

```text
Huber(sigma, scale_target)
```

诊断指标必须增加：

```text
mean_sigma
median_sigma
sigma_q10 / q50 / q90
fraction_sigma_at_floor
corr(sigma, abs_residual)
mean(abs_residual / sigma)
q90(abs_residual / sigma)
```

验收标准：

```text
fraction_sigma_at_floor < 20%
q 不应长期 > 5
q * median_sigma 接近 calibration residual target quantile
```

---

### P2.3 不要让 sigma_floor 决定最终 gap

当前实际最小 gap 接近：

```text
q * sigma_floor ≈ 0.149
```

后续报告必须明确输出：

```text
margin_floor = q * sigma_floor
margin_floor / covered_mean_gap
```

验收标准：

```text
margin_floor / covered_mean_gap < 40%
```

如果 margin floor 占比太高，说明模型是在靠 floor cover，而不是靠真实 uncertainty ranking。

---

## P3：重做 gate，让它真正优化 low-gap selection

当前 gate 的 accepted rate 只有 10.1%，但 covered_mean_gap 仍然 0.2167，说明 gate 没有很好地区分低 gap 样本。

### P3.1 gate 不要只按 required_margin 小筛选

当前 selected_margin_threshold 约为：

```text
0.1520
```

而 accepted 样本 mean_required_margin 为：

```text
0.1488
```

这说明 gate 主要只是筛掉 margin 高的样本。但如果所有样本 margin 都被 floor 固定在 0.149 附近，gate 就失去分辨率。

新 gate 应该学习两个量：

```text
1. cover_prob = P(p_final >= y_raw)
2. expected_gap = E[p_final - y_raw | covered]
```

最终 score：

```text
score = expected_gap + lambda * max(0, target_coverage - cover_prob)
```

选择 score 最低的一批样本，并在 calibration / validation 上用 threshold search 保证：

```text
accepted_coverage >= 0.70
side_violation_rate = 0
```

---

### P3.2 gate 增加 side-cap awareness

因为业务约束中有：

```text
p_final <= p_side
```

所以 gate 必须显式考虑 headroom：

```text
headroom = p_side - base_pred_raw
headroom_after_margin = p_side - p_final
```

如果：

```text
headroom_after_margin < 0
```

则该样本不能接受，或者 p_final 被 clip 后 coverage 会变差。

建议 gate 特征加入：

```text
p_side
base_pred_raw
p_side - base_pred_raw
required_margin
required_margin / max(p_side - base_pred_raw, eps)
selected_side
price_bin
p_side_bin
```

---

## P4：实验矩阵

建议按照以下顺序跑，不要一次性全改。

### Experiment 0：Sanity Fix

目的：确认数据、bin、side、transform 没问题。

改动：

```text
1. 修复 p_bin / price_bin / selected_side NA；
2. 确认所有 final metrics 在 raw probability 空间计算；
3. 增加 margin_floor diagnostics；
4. 增加 sigma diagnostics。
```

成功标准：

```text
NA rate < 5%
side_violation_rate = 0
report 中能拆出 base gap / margin gap / final gap
```

---

### Experiment 1：Base Regularization

目的：降低 validation residual。

对照组：

```text
A. current MLP
B. smaller MLP + dropout + weight_decay
C. CatBoost regression baseline
D. pseudo-Huber / log-cosh base loss
```

选择标准：

```text
primary: validation residual q70 lowest
secondary: validation MAE lowest
third: validation residual q90 lowest
```

目标：

```text
validation MAE 从 0.155 降到 <= 0.12
validation residual q70 从 0.202 降到 <= 0.16
```

---

### Experiment 2：Non-normalized Local Conformal

目的：建立简单可靠 baseline。

实现：

```text
margin = quantile(max(y_raw - base_pred_raw, 0), q | local group)
p_final = base_pred_raw + margin
```

local group 优先级：

```text
1. selected_side + p_side_bin
2. selected_side + price_bin
3. p_side_bin only
4. global fallback
```

小样本 fallback：

```text
if group_count < 100:
    use parent group
```

目标：

```text
accepted_coverage >= 0.70
covered_mean_gap < current 0.2167
covered_q90_gap < current 0.4133
```

---

### Experiment 3：Improved Normalized Conformal

目的：只有在 scale model 有效时，才恢复 normalized conformal。

要求：

```text
fraction_sigma_at_floor < 20%
q < 5
corr(sigma, abs_residual) > 0.20
```

如果不满足，normalized conformal 不上线。

---

### Experiment 4：Coverage-Gap Gate

目的：让 accepted set 真正以 low-gap 为目标。

训练 label：

```text
covered_i = 1[p_final_i >= y_i]
gap_i = p_final_i - y_i if covered_i else large_penalty
```

训练两个 head：

```text
cover_prob_head
gap_head
```

selection：

```text
sort by expected_gap + lambda * fail_risk
choose threshold with accepted_coverage >= 0.70
```

目标：

```text
accepted_rate >= 8%
accepted_coverage >= 70%
covered_mean_gap <= 0.17
covered_q90_gap <= 0.30
side_violation_rate = 0
```

---

## 4. 报告必须新增的 diagnostics

每次实验 report 必须输出：

```text
base_fit_mae
base_calibration_mae
base_validation_mae
base_validation_residual_q70
base_validation_residual_q90

q
sigma_floor
margin_floor = q * sigma_floor
mean_required_margin
margin_floor / covered_mean_gap

mean_sigma
median_sigma
sigma_q10
sigma_q90
fraction_sigma_at_floor
corr_sigma_abs_residual

accepted_count
accepted_rate
accepted_coverage
covered_mean_gap
covered_median_gap
covered_q90_gap
side_violation_rate

NA rate for price_bin / p_bin / selected_side
by selected_side diagnostics
by p_side_bin diagnostics
by price_bin diagnostics
```

---

## 5. 最终验收标准

### Minimum acceptable improvement

```text
accepted_coverage >= 0.70
side_violation_rate = 0
covered_mean_gap < 0.20
covered_q90_gap < 0.38
```

### Good result

```text
accepted_coverage >= 0.70
side_violation_rate = 0
accepted_rate >= 0.08
covered_mean_gap <= 0.17
covered_q90_gap <= 0.30
```

### Excellent result

```text
accepted_coverage >= 0.70
side_violation_rate = 0
accepted_rate >= 0.10
covered_mean_gap <= 0.15
covered_q90_gap <= 0.25
```

注意：如果 conformal margin floor 本身仍然接近 `0.149`，则 covered_mean_gap 很难低于 `0.15`。所以 excellent result 的前提是修复 scale model 或改用更有效的 local margin。

---

## 6. 推荐下一步执行顺序

```text
Step 1: 修复 p_bin / price_bin / selected_side NA 问题
Step 2: 确认 probability / logit 空间没有混用
Step 3: 增加 diagnostics，拆分 base error / margin / final gap
Step 4: 降低 base validation residual，优先看 residual q70
Step 5: 跑 non-normalized local conformal baseline
Step 6: 修 scale model，避免 sigma 全部贴 floor
Step 7: 重做 gate，用 expected_gap + coverage risk 排序
```

最重要的判断：

```text
如果 base validation residual q70 仍然在 0.20 附近，
不要继续调 conformal，mean gap 很难明显下降。

如果 q * sigma_floor 仍然在 0.15 附近，
不要期待 covered_mean_gap 低于 0.15。

如果 local bin 大量 NA，
不要相信 local calibration / local gate 的结果。
```
