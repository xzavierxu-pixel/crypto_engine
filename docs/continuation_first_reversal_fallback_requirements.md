# Continuation-First Reversal Fallback Overlay 需求说明

## 1. 实验目的

构建一个更保守的双专家组合策略：

- **顺势专家 Continuation Expert** 作为主模型，拥有最高优先级。
- **反转专家 Reversal Expert / reversal_weight_8** 只在顺势专家选择 `ABSTAIN` 时作为补充判断。
- 禁止反转专家覆盖或推翻顺势专家已经接受的交易。

目标是在不破坏顺势专家高胜率边界的前提下，验证反转专家是否能在顺势模型放弃的残差样本中提供正增量收益。

---

## 2. 使用模型

### 2.1 顺势专家

来源：

```text
artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_continuation_side_coordinate_search
```

使用该实验中 `continuation_accepted_accuracy` 最高的最优配置。

### 2.2 反转专家

来源：

```text
artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_reversal_sample_weight_search/reversal_weight_8
```

使用 `reversal_weight_8`，因为该版本拥有最高的 `reversal_accepted_accuracy`。

---

## 3. 核心路由逻辑

最终决策必须遵循以下优先级：

```python
if continuation_expert_accepts:
    final_signal = continuation_signal
    final_source = "continuation"

elif reversal_expert_passes_threshold:
    final_signal = reversal_signal
    final_source = "reversal_fallback"

else:
    final_signal = "ABSTAIN"
    final_source = "abstain"
```

关键规则：

1. 顺势专家 accept 的样本，直接采用顺势信号。
2. 反转专家只允许在顺势专家 abstain 的样本上运行。
3. 反转专家不得 override 顺势专家。
4. 不再训练 hard router / gate model。
5. 第一版只通过阈值 sweep 验证反转 fallback 的增量价值。

---

## 4. 阈值搜索要求

不要直接沿用 `reversal_weight_8` 在全样本上的最优阈值。

需要在以下样本空间重新搜索阈值：

```text
continuation_accept == 0
```

即只在顺势专家 abstain 的 residual samples 上评估反转模型。

建议阈值网格：

```python
tau_rev_grid = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]
```

如使用 confidence / margin：

```python
rev_margin_grid = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20]
```

---

## 5. 需要输出的指标

必须分别输出以下指标：

| 指标 | 说明 |
|---|---|
| `continuation_coverage` | 顺势专家单独覆盖率 |
| `continuation_accepted_accuracy` | 顺势专家单独 accepted accuracy |
| `reversal_fallback_coverage_on_residual` | 反转专家在 residual 样本中的覆盖率 |
| `reversal_fallback_accepted_accuracy` | 反转 fallback 样本准确率 |
| `reversal_fallback_utility` | 反转 fallback 样本单独 utility |
| `combined_coverage` | 顺势 + 反转 fallback 后总覆盖率 |
| `combined_accepted_accuracy` | 合并后的 accepted accuracy |
| `combined_utility` | 合并后的 total utility |
| `incremental_utility_from_reversal` | 反转 fallback 相对 continuation-only 的增量 utility |

---

## 6. 成功标准

实验只有在同时满足以下条件时才视为有效改进：

```text
combined_coverage >= 0.70
combined_utility > continuation_only_utility
incremental_utility_from_reversal > 0
reversal_fallback_accepted_accuracy > 0.50
```

更理想的反转 fallback 要求：

```text
reversal_fallback_accepted_accuracy >= 0.55
```

如果反转 fallback 在 residual 样本中的 utility 为负，则应禁用反转补仓逻辑。

---

## 7. 建议实验名称

```text
20260612_continuation_first_reversal_weight8_abstain_overlay
```

或：

```text
20260612_continuation_primary_reversal_weight8_fallback
```
