# 20260625 expected_return 实验总结

本文档总结 `price_estimator/expected_return/experiments` 下所有 `20260625*` 实验的探索方向，并重点解释 `validation.sum_pnl` 最好的实验。

## 1. 范围和数据来源

本次梳理覆盖 20 个实验目录：

- `20260625_expected_return_catboost_deeper_lowbid_group_no_leak`
- `20260625_expected_return_catboost_lgbm_blend_no_leak`
- `20260625_expected_return_catboost_lowbid_group_no_leak`
- `20260625_expected_return_lgbm_ev_isotonic_no_leak`
- `20260625_expected_return_lgbm_lowbid_offset_group_no_leak`
- `20260625_expected_return_xgb_catboost_blend_fine_groups_no_leak`
- `20260625_expected_return_xgb_catboost_blend_fine_lowbid_group_no_leak`
- `20260625_expected_return_xgb_catboost_blend_finebid_no_leak`
- `20260625_expected_return_xgb_catboost_blend_highbid_no_leak`
- `20260625_expected_return_xgb_catboost_blend_lowbid_group_no_leak`
- `20260625_expected_return_xgb_catboost_blend_qgate_no_leak`
- `20260625_expected_return_xgb_ev_isotonic_no_leak`
- `20260625_expected_return_xgb_lowbid_group_cal14_no_leak`
- `20260625_expected_return_xgb_lowbid_group_cal3_no_leak`
- `20260625_expected_return_xgb_lowbid_group_gciso_no_leak`
- `20260625_expected_return_xgb_lowbid_group_min_ev_no_leak`
- `20260625_expected_return_xgb_lowbid_group_noiso_no_leak`
- `20260625_expected_return_xgb_lowbid_group_qiso_no_leak`
- `20260625_expected_return_xgb_lowbid_isotonic_no_leak`
- `20260625_expected_return_xgb_lowbid_offset_group_no_leak`

使用的原始文件：

- 每个实验的 `config.yaml`
- 每个实验的 `reports/config_used.yaml`
- 每个实验的 `reports/summary_metrics.json`
- 带分组策略的实验使用 `reports/group_min_ev_policy.json`
- 最佳实验额外检查了 `reports/predictions_validation.parquet`

## 2. 共同实验协议

这些实验都是 no-leak validation diagnostic 类型实验，不是部署 artifact 重训。

共同设置：

- `objective.min_coverage = 0.70`
- `objective.target_validation_sum_pnl = 1000.0`
- `deploy_training_mode = no_leak_validation_diagnostic`
- `offline_validation_metric_source = execution_engine/deploy/baseline/artifact_manifest.json`
- `split.calibration_tail_days = 7`
- `timestamp_column = timestamp`
- `feature_count = 1826`
- train dataset: `price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet`
- validation dataset: `price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet`

窗口：

| window | rows | start | end |
| --- | ---: | --- | --- |
| train | 15696 | 2026-02-12 00:35:00+00:00 | 2026-04-10 23:40:00+00:00 |
| calibration tail | 1242 | 2026-04-03 23:45:00+00:00 | 2026-04-10 23:25:00+00:00 |
| validation | 7468 | 2026-04-11 00:15:00+00:00 | 2026-05-10 23:50:00+00:00 |

泄漏控制记录在 `summary_metrics.json` 中：

- 模型和 policy selection 只使用 fit/calibration labels。
- validation labels 只用于最终评价。
- 特征排除 pattern 包含 `target|label|winner|correct|chosen_low|future|closed|endDate|condition|market_id|question|slug|outcome|fetched|source|time_to|trade_time|timestamp|date|pnl`。

## 3. `validation.sum_pnl` 是什么

`validation.sum_pnl` 来自验证集预测文件 `predictions_validation.parquet` 中的 `realized_pnl` 汇总，不是分类指标。

最佳实验的 validation parquet 字段包括：

- 信号字段: `selected_side`, `p_up`, `p_side`, `selected_t_up`, `selected_t_down`
- 真实结果字段: `target`, `correct`
- 价格和成交字段: `chosen_low`, `q_used`, `bid`, `expected_ev`, `model_fill_prob`, `filled`, `printed_filled`
- PnL 字段: `realized_pnl`

核心口径：

- `accepted_count`: 被方向阈值接受的样本数。
- `order_count`: policy 决定提交订单的样本数。
- `trade_count`: 模拟成交的订单数，也就是 `filled=True` 的数量。
- `fill_rate`: `trade_count / accepted_count`。
- `order_coverage`: `order_count / accepted_count`。
- `sum_pnl`: `sum(realized_pnl)`。
- `win_pnl_sum + loss_pnl_sum = sum_pnl`。
- 未成交样本的 `realized_pnl` 为 0，因此 `sum_pnl` 主要由实际成交订单贡献。

因此，`selection_score` 衡量方向模型在覆盖率约束下的分类选择质量，而 `validation.sum_pnl` 衡量在 bid、EV gate、fill model 和分组 policy 之后的交易收益。20260625 这批实验里，很多实验的方向指标完全相同，但 `sum_pnl` 不同，原因是 bid/policy 改变了提交订单和成交分布。

## 4. validation sum_pnl 排名

所有实验都满足方向信号覆盖率约束：`validation.coverage = 0.7000535619 >= 0.70`。其中大多数实验方向指标相同：

- `sample_count = 7468`
- `accepted_count = 5228`
- `accepted_sample_accuracy = 0.7073450650`
- `selection_score = 0.6413740846`
- `utility = 0.2903053026`
- `downside_risk = 0.4526302350`
- `up_prediction_count = 2856`
- `down_prediction_count = 2372`

排名按 `validation_metrics.sum_pnl` 降序：

| rank | experiment | validation sum_pnl | trade_count | order_count | fill_rate | mean_pnl_filled | mean_bid | policy / group |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `xgb_catboost_blend_fine_lowbid_group_no_leak` | 961.84 | 3036 | 3725 | 0.580719 | 0.316812 | 0.459621 | blend, `p_side_bin` |
| 1 | `xgb_catboost_blend_highbid_no_leak` | 961.84 | 3036 | 3725 | 0.580719 | 0.316812 | 0.459621 | blend, `p_side_bin` |
| 1 | `xgb_catboost_blend_qgate_no_leak` | 961.84 | 3036 | 3725 | 0.580719 | 0.316812 | 0.459621 | blend, `p_side_bin` |
| 4 | `xgb_catboost_blend_fine_groups_no_leak` | 961.68 | 3052 | 3745 | 0.583780 | 0.315098 | 0.459660 | blend, `p_side_bin` |
| 5 | `xgb_catboost_blend_lowbid_group_no_leak` | 960.90 | 3050 | 3751 | 0.583397 | 0.315049 | 0.460178 | blend, `selected_side,p_side_bin` |
| 6 | `xgb_catboost_blend_finebid_no_leak` | 950.53 | 2992 | 3768 | 0.572303 | 0.317691 | 0.455385 | blend, `hour` |
| 7 | `catboost_lgbm_blend_no_leak` | 949.67 | 3053 | 3784 | 0.583971 | 0.311061 | 0.459332 | blend, `hour` |
| 8 | `catboost_lowbid_group_no_leak` | 948.75 | 3050 | 3738 | 0.583397 | 0.311066 | 0.458785 | catboost, `selected_side,p_side_bin` |
| 9 | `xgb_lowbid_group_min_ev_no_leak` | 946.91 | 2974 | 3704 | 0.568860 | 0.318396 | 0.453632 | xgboost, `hour` |
| 9 | `xgb_lowbid_offset_group_no_leak` | 946.91 | 2974 | 3704 | 0.568860 | 0.318396 | 0.453632 | xgboost, `hour` |
| 11 | `xgb_lowbid_isotonic_no_leak` | 945.02 | 2958 | 3692 | 0.565800 | 0.319479 | 0.455350 | xgboost, global |
| 12 | `xgb_ev_isotonic_no_leak` | 943.70 | 3026 | 3741 | 0.578806 | 0.311864 | 0.462902 | xgboost, global |
| 13 | `xgb_lowbid_group_qiso_no_leak` | 940.31 | 3279 | 3761 | 0.627200 | 0.286767 | 0.479028 | xgboost, `selected_side,p_side_bin` |
| 14 | `catboost_deeper_lowbid_group_no_leak` | 938.88 | 2902 | 3689 | 0.555088 | 0.323529 | 0.451117 | catboost, `p_side_bin` |
| 15 | `xgb_lowbid_group_noiso_no_leak` | 936.91 | 3264 | 3844 | 0.624331 | 0.287044 | 0.472492 | xgboost, `hour` |
| 16 | `lgbm_ev_isotonic_no_leak` | 936.55 | 2884 | 3673 | 0.551645 | 0.324740 | 0.452401 | lightgbm/global |
| 17 | `xgb_lowbid_group_gciso_no_leak` | 930.72 | 2918 | 3777 | 0.558148 | 0.318958 | 0.442955 | xgboost, `hour` |
| 18 | `xgb_lowbid_group_cal14_no_leak` | 929.23 | 2862 | 3716 | 0.547437 | 0.324679 | 0.447917 | xgboost, `selected_side,hour` |
| 19 | `xgb_lowbid_group_cal3_no_leak` | 928.07 | 2868 | 3692 | 0.548585 | 0.323595 | 0.447919 | xgboost, `selected_side` |
| 20 | `lgbm_lowbid_offset_group_no_leak` | 903.33 | 2822 | 3712 | 0.539786 | 0.320103 | 0.442563 | lightgbm, `hour` |

## 5. 做了哪些探索

### 5.1 模型族探索

实验覆盖了单模型和 blend：

- XGBoost: `xgb_*`
- LightGBM: `lgbm_*`
- CatBoost: `catboost_*`
- XGBoost + CatBoost blend: `xgb_catboost_blend_*`
- CatBoost + LightGBM blend: `catboost_lgbm_blend_no_leak`

结论：本批结果里 XGBoost + CatBoost blend 明显最好。最高 `sum_pnl=961.84` 来自 XGBoost/CatBoost blend，而单模型最好的 CatBoost 是 `948.75`，单 XGBoost 基线型结果是 `946.91`，LightGBM 结果较弱，最低为 `903.33`。

### 5.2 blend 权重探索

最佳实验的 policy 明确选择：

```text
prediction_candidate = blend_xgboost_0.450_catboost_0.550
blend_weight_first = 0.45
blend_weight_second = 0.55
blend_members = [xgboost, catboost]
```

也就是说，最佳组合不是 50/50，而是 CatBoost 权重略高。`xgb_catboost_blend_lowbid_group_no_leak` 使用 50/50，`sum_pnl=960.90`，只比最佳低 0.94。说明 blend 权重有影响，但本批实验里主要收益来自 blend 本身和 p_side 分桶 policy。

### 5.3 低 bid / bid 上限 / bid offset 探索

policy 搜索覆盖：

- `bid_min` 通常为 0.01。
- `bid_max` 多数为 0.91，`highbid` 放宽到 0.99。
- `bid_step = 0.05`。
- `bid_offset_steps_grid = [-4, -3, -2, -1, 0, 1]`。

最佳三组中：

- `fine_lowbid_group`: `bid_max=0.91`
- `highbid`: `bid_max=0.99`
- `qgate`: `bid_max=0.91` 且额外搜索 `min_q_grid`

三者最终都选到了同一套有效 policy：

```text
bid_offset_steps = 0
fallback_min_ev = 0.015
selected_group_columns = [p_side_bin]
selected_group_count = 7
```

因此 `highbid` 放宽 bid 上限没有带来额外验证收益，`qgate` 的质量门槛也最终退化为 `min_q=0.0`，没有实际过滤更多样本。

### 5.4 全局 min_ev 与分组 min_ev 探索

这些实验的核心不是改变方向阈值，而是在接受信号之后搜索是否下单：

```text
expected_ev = q_used * payoff_estimate - bid / fill-adjusted logic
submit order if expected_ev >= min_ev policy
```

本批实验主要比较：

- 全局 `min_ev`
- 按小时 `hour` 分组的 `min_ev`
- 按方向 `selected_side` 分组的 `min_ev`
- 按置信度分桶 `p_side_bin` 分组的 `min_ev`
- 组合分组，如 `selected_side,p_side_bin`、`selected_side,hour`

最佳实验使用 `p_side_bin` 分组，而不是 `hour` 或 `selected_side`：

```json
{
  "(0.55, 0.6]": -0.005,
  "(0.6, 0.65]": 0.08,
  "(0.65, 0.7]": 0.13,
  "(0.7, 0.75]": 0.115,
  "(0.75, 0.8]": 0.075,
  "(0.8, 0.85]": 0.015,
  "(0.85, 1.01]": -0.005
}
```

这说明 policy 最终认为不同置信度区间的 EV 门槛应该不同。中等偏高置信度区间需要更严格的 EV 门槛，而最高置信度区间可以接受更低门槛。

### 5.5 分组粒度探索

几个相近实验展示了分组粒度的效果：

- `fine_lowbid_group`: `p_side_bin` 7 组，`sum_pnl=961.84`
- `fine_groups`: `p_side_bin` 8 组，加入 `(0.499, 0.55]`，`sum_pnl=961.68`
- `lowbid_group`: `selected_side,p_side_bin` 13 组，`sum_pnl=960.90`
- `finebid`: `hour` 24 组，`sum_pnl=950.53`

结论：本批验证集上，按 `p_side_bin` 单独分组最优。更细的方向乘置信度分组、小时分组没有改善，可能因为 calibration tail 只有 1242 行，过细分组更容易让分组阈值变得不稳定。

### 5.6 isotonic / no-isotonic 探索

XGBoost 相关实验对 `isotonic_q` 和 `isotonic_gc` 做了拆分：

- `xgb_lowbid_isotonic_no_leak`: isotonic 开启，`sum_pnl=945.02`
- `xgb_ev_isotonic_no_leak`: EV 版本 isotonic，`sum_pnl=943.70`
- `xgb_lowbid_group_qiso_no_leak`: `isotonic_q=true, isotonic_gc=false`，`sum_pnl=940.31`
- `xgb_lowbid_group_gciso_no_leak`: `isotonic_q=false, isotonic_gc=true`，`sum_pnl=930.72`
- `xgb_lowbid_group_noiso_no_leak`: 两者关闭，`sum_pnl=936.91`

结论：单看 XGBoost，isotonic 开关没有稳定地提高验证 PnL。`qiso` 增加了交易数和填充率，但 `mean_pnl_filled` 下滑；`gciso` 在本批验证上更差。最佳 blend 实验的最终 policy 报告为 `isotonic_q=false`、`isotonic_gc=false`，说明最终提交订单逻辑没有依赖额外 isotonic gate。

## 6. validation sum_pnl 最好的实验

并列第一的三个实验是：

- `20260625_expected_return_xgb_catboost_blend_fine_lowbid_group_no_leak`
- `20260625_expected_return_xgb_catboost_blend_highbid_no_leak`
- `20260625_expected_return_xgb_catboost_blend_qgate_no_leak`

它们的最终 validation 指标完全相同：

| metric | value |
| --- | ---: |
| `validation.sum_pnl` | 961.84 |
| `win_pnl_sum` | 998.52 |
| `loss_pnl_sum` | -36.68 |
| `sample_count` | 7468 |
| `accepted_count` | 5228 |
| `coverage` | 0.7000535619 |
| `accepted_sample_accuracy` | 0.7073450650 |
| `selection_score` | 0.6413740846 |
| `utility` | 0.2903053026 |
| `downside_risk` | 0.4526302350 |
| `up_prediction_count` | 2856 |
| `down_prediction_count` | 2372 |
| `order_count` | 3725 |
| `order_coverage` | 0.7125095639 |
| `trade_count` | 3036 |
| `fill_rate` | 0.5807192043 |
| `mean_pnl_filled` | 0.3168115942 |
| `mean_accepted_pnl` | 0.1839785769 |
| `mean_bid` | 0.4596212701 |
| `median_bid` | 0.61 |
| `mean_expected_ev` | 0.2022549621 |
| `mean_model_fill_prob` | 0.8762976812 |
| `realized_correct_fill_rate_submitted` | 0.8097736057 |
| `submitted_fill_calibration_gap` | 0.0665240754 |
| `negative_expected_ev_share` | 0.2792654935 |

最佳 policy：

```text
model_family = blend
prediction_candidate = blend_xgboost_0.450_catboost_0.550
selected_group_columns = [p_side_bin]
selected_group_count = 7
bid_offset_steps = 0
fallback_min_ev = 0.015
min_group_order_count = 20
selection_source = calibration
```

为什么三个实验并列：

- `fine_lowbid_group`、`highbid`、`qgate` 的搜索空间不同。
- 但 calibration selection 最终都选到了同一个 prediction candidate、同一个 `p_side_bin` 分组、同一个 `fallback_min_ev=0.015`、同一个 `bid_offset_steps=0`。
- `highbid` 虽然允许 `bid_max=0.99`，最终没有产生不同成交路径。
- `qgate` 虽然搜索了 `min_q_grid`，最终选择 `min_q=0.0`，等价于不加 q gate。

## 7. 与本批内部 baseline 的比较

多个 config 把 `20260625_expected_return_xgb_lowbid_group_min_ev_no_leak` 作为内部 baseline：

```text
baseline validation_sum_pnl = 946.91
```

最佳实验：

```text
validation_sum_pnl = 961.84
```

差异：

```text
absolute improvement = +14.93
relative improvement = +1.58%
```

覆盖率：

```text
baseline coverage = 0.7000535619
best coverage = 0.7000535619
coverage constraint satisfied = yes
```

目标：

```text
target_validation_sum_pnl = 1000.0
best validation_sum_pnl = 961.84
target satisfied = no
```

也就是说，最佳实验相对本批内部 baseline 有小幅提升，但仍未达到 1000 的目标 PnL。

## 8. 结论

本批 20260625 实验的主要发现：

1. 方向模型层面，XGBoost + CatBoost blend 是本批最强选择。
2. 最佳 blend 权重为 `xgboost=0.45, catboost=0.55`。
3. 对验证 PnL 最有效的 policy 是按 `p_side_bin` 做 group min EV，而不是按小时或方向做更细分组。
4. `highbid` 和 `qgate` 扩展了搜索空间，但最终没有改变最优 policy。
5. 单模型 XGBoost、CatBoost、LightGBM 都没有超过最佳 blend。
6. isotonic/q/gc 校准在本批交易 PnL 上没有给出稳定收益。
7. 最佳验证 `sum_pnl=961.84`，满足 `coverage >= 0.70`，但没有达到 `target_validation_sum_pnl=1000.0`。

本批最值得作为后续起点的实验是：

```text
price_estimator/expected_return/experiments/20260625_expected_return_xgb_catboost_blend_fine_lowbid_group_no_leak
```

使用它作为后续起点的原因是：它和另外两个实验并列第一，但配置语义最直接，没有 `highbid` 或 `qgate` 这些最终未生效的额外搜索维度。

## 9. 最佳方案的详细解释

推荐优先分析和复用的最佳方案是：

```text
20260625_expected_return_xgb_catboost_blend_fine_lowbid_group_no_leak
```

它和 `highbid`、`qgate` 并列第一，但这个目录的 config 最干净，搜索空间和最终 policy 的关系最直接。

### 9.1 这套方案实际做了什么

这套方案可以拆成四层：

1. 方向模型先判断 5m Polymarket BTC UP/DOWN 的方向概率。
2. 方向阈值只保留约 70% 的高置信度样本。
3. expected-return policy 决定哪些已接受信号值得挂单。
4. fill/PnL 模拟根据 bid、是否成交和真实结果计算 validation `realized_pnl`。

关键点是：方向模型负责“选方向”，expected-return policy 负责“是否值得以这个 bid 出价”。所以它不是单纯追求更高方向准确率，而是在已有方向信号之上优化交易 PnL。

### 9.2 方向信号层

验证集方向指标：

| metric | value |
| --- | ---: |
| `sample_count` | 7468 |
| `accepted_count` | 5228 |
| `coverage` | 0.7000535619 |
| `selected_t_up` | 0.5791249044 |
| `selected_t_down` | 0.4314498852 |
| `accepted_sample_accuracy` | 0.7073450650 |
| `precision_up` | 0.7107843137 |
| `precision_down` | 0.7032040472 |
| `balanced_precision` | 0.7069941805 |
| `share_up_predictions` | 0.5462892119 |
| `share_down_predictions` | 0.4537107881 |
| `roc_auc` | 0.7436403330 |
| `brier_score` | 0.2029322450 |
| `log_loss` | 0.5928327457 |
| `utility` | 0.2903053026 |
| `downside_risk` | 0.4526302350 |
| `selection_score` | 0.6413740846 |

信号规则仍是标准二分类选择：

```text
UP signal   if p_up >= selected_t_up
DOWN signal if p_up <= selected_t_down
NO-SIGNAL   otherwise
```

本方案的方向层没有牺牲覆盖率换 PnL，仍刚好满足 `coverage >= 0.70`。这也解释了为什么本批很多实验的 `selection_score` 完全一样：它们用的是同一批 accepted signals，差异主要在交易 policy。

### 9.3 模型层为什么用 XGBoost + CatBoost blend

最终选择的 prediction candidate 是：

```text
blend_xgboost_0.450_catboost_0.550
```

含义：

- 第一个成员是 XGBoost，权重 0.45。
- 第二个成员是 CatBoost，权重 0.55。
- CatBoost 权重略高，说明 calibration tail 上 CatBoost 对 expected-return policy 更有帮助。
- 50/50 blend 也很接近，但最终低 0.94 PnL。

这不是大幅复杂化模型，而是在已有树模型族内做低风险集成。它的收益主要体现在交易 policy 之后的 PnL，而不是让方向层指标大幅变化。

### 9.4 下单 policy 的结构

最佳 policy：

```text
type = q_plus_bid_gc_ev
mode = group_min_ev
model_family = blend
selection_source = calibration
bid_min = 0.01
bid_max = 0.91
bid_step = 0.05
bid_offset_steps = 0
fallback_min_ev = 0.015
selected_group_columns = [p_side_bin]
selected_group_count = 7
min_group_order_count = 20
isotonic_q = false
isotonic_gc = false
```

policy 的决策逻辑可以理解为：

```text
1. 已经有 UP/DOWN 方向信号。
2. 根据该方向的 p_side 进入一个置信度分桶。
3. 在该分桶中读取对应的 min_ev 门槛。
4. 如果 expected_ev >= 该分桶 min_ev，则提交订单。
5. bid 使用搜索得到的正常 bid，不做 offset，bid_offset_steps = 0。
```

最终分桶门槛：

| p_side_bin | min_ev |
| --- | ---: |
| `(0.55, 0.6]` | -0.005 |
| `(0.6, 0.65]` | 0.080 |
| `(0.65, 0.7]` | 0.130 |
| `(0.7, 0.75]` | 0.115 |
| `(0.75, 0.8]` | 0.075 |
| `(0.8, 0.85]` | 0.015 |
| `(0.85, 1.01]` | -0.005 |

这个形状很重要：它不是简单地“概率越高，EV 门槛越低或越高”。中间置信度段 `(0.6, 0.75]` 需要更高 EV，最高置信度段反而允许更低 EV。原因是最高置信度段虽然胜率高，但 bid 往往也更贵，policy 需要在胜率、价格和成交概率之间折中。

### 9.5 PnL 是怎样形成的

最佳方案 validation PnL：

| metric | value |
| --- | ---: |
| `sum_pnl` | 961.84 |
| `win_pnl_sum` | 998.52 |
| `loss_pnl_sum` | -36.68 |
| `order_count` | 3725 |
| `trade_count` | 3036 |
| `order_coverage` | 0.7125095639 |
| `fill_rate` | 0.5807192043 |
| `mean_pnl_filled` | 0.3168115942 |
| `mean_accepted_pnl` | 0.1839785769 |
| `mean_bid` | 0.4596212701 |
| `median_bid` | 0.61 |
| `mean_expected_ev` | 0.2022549621 |
| `mean_model_fill_prob` | 0.8762976812 |
| `realized_correct_fill_rate_submitted` | 0.8097736057 |
| `submitted_fill_calibration_gap` | 0.0665240754 |

解释：

- 5228 个 accepted signals 中，有 3725 个被 policy 选择提交订单。
- 3036 个订单在验证模拟中成交。
- 成交订单平均 PnL 为 0.3168。
- 所有 accepted signals 平均 PnL 为 0.1840，因为未成交样本记 0。
- 总盈利 998.52，总亏损 -36.68，净值 961.84。
- `submitted_fill_calibration_gap=0.0665` 表示提交订单后的实际正确成交率和模型 fill 估计仍存在偏差，后续仍可改进。

这套方案赚钱的关键不是交易更多，而是维持较高成交后收益。比如 `xgb_lowbid_group_noiso_no_leak` 的 `trade_count=3264` 更多，但 `mean_pnl_filled=0.2870` 明显更低，最终 `sum_pnl=936.91`，低于最佳方案。

### 9.6 UP/DOWN 两侧表现

按方向拆分 validation parquet：

| selected_side | rows | filled | sum_pnl | mean_pnl | mean_bid | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DOWN | 2372 | 1368 | 444.22 | 0.187277 | 0.447159 | 0.703204 |
| UP | 2856 | 1668 | 517.62 | 0.181239 | 0.469972 | 0.710784 |

观察：

- UP 和 DOWN 都贡献正 PnL，不是单边策略。
- UP 的样本数和总 PnL 更高。
- DOWN 的平均 PnL 略高，平均 bid 更低。
- 两侧 accuracy 接近，说明方向模型没有明显只靠一侧赚钱。

### 9.7 p_side 分桶表现

按 `p_side_bin` 拆分 validation parquet：

| p_side_bin | rows | filled | sum_pnl | mean_pnl | mean_bid | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(0.499, 0.55]` | 58 | 32 | 14.38 | 0.247931 | 0.330517 | 0.689655 |
| `(0.55, 0.6]` | 808 | 468 | 179.97 | 0.222735 | 0.367611 | 0.658416 |
| `(0.6, 0.65]` | 1269 | 707 | 256.63 | 0.202230 | 0.389425 | 0.649330 |
| `(0.65, 0.7]` | 1079 | 583 | 198.97 | 0.184402 | 0.419870 | 0.671918 |
| `(0.7, 0.75]` | 859 | 508 | 145.17 | 0.168999 | 0.494051 | 0.727590 |
| `(0.75, 0.8]` | 649 | 407 | 106.33 | 0.163837 | 0.543852 | 0.767334 |
| `(0.8, 0.85]` | 354 | 234 | 47.26 | 0.133503 | 0.685113 | 0.884181 |
| `(0.85, 1.01]` | 152 | 97 | 13.13 | 0.086382 | 0.786842 | 0.927632 |

这张表解释了为什么按 `p_side_bin` 分组有效：

- 最高置信度区间准确率最高，但 bid 也最高，平均 PnL 反而最低。
- 最大 PnL 来自 `(0.6, 0.65]` 和 `(0.65, 0.7]`，这些区间样本多、价格仍可接受。
- 低置信度但低 bid 的区间也能贡献正收益，因此不能简单用高 `p_side` gate 过滤掉。
- 分桶 min EV 可以让每个置信度区间使用不同门槛，比全局门槛更贴合价格/胜率结构。

### 9.8 为什么 `qgate` 没有进一步提升

`qgate` 实验额外搜索：

```text
min_q_grid = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
```

但最终选择：

```text
min_q = 0.0
```

这说明 calibration tail 上，额外的 q 门槛没有提高 PnL。原因从 p_side 分桶表现也能看出来：低/中置信度区间仍有大量正 PnL，如果硬加 q gate，可能会过滤掉价格便宜且仍有正期望的订单。

### 9.9 为什么 `highbid` 没有进一步提升

`highbid` 把 `bid_max` 从 0.91 放宽到 0.99，但最终 validation 路径完全相同：

```text
sum_pnl = 961.84
trade_count = 3036
order_count = 3725
mean_bid = 0.4596212701
```

这说明最优 policy 并不需要更高 bid 上限。更高 bid 上限理论上可能提高成交概率，但也会压低 payout edge；本次 calibration selection 没有选择那条路径。

### 9.10 风险和注意事项

使用这个方案作为后续起点时，要注意：

1. `validation.sum_pnl` 是在 validation 上最终评价出来的交易模拟 PnL，不等价于线上真实收益。
2. `target_validation_sum_pnl=1000.0` 没有达到，当前最佳只是本批内部最优。
3. 最优 policy 由 7 天 calibration tail 选择，分组样本不大，存在过拟合 calibration tail 的风险。
4. `submitted_fill_calibration_gap=0.0665` 说明 fill 概率估计还有系统偏差。
5. 最高置信度区间平均 bid 很高，不能只看 accuracy，需要继续看价格和成交后收益。
6. 后续如果改变 bid step、fill model、label window、特征集或方向阈值，这个 policy 的 PnL 不能直接外推。

### 9.11 后续最合理的改进方向

基于这批结果，下一步优先级建议：

1. 保留 `xgboost=0.45, catboost=0.55` blend 和 `p_side_bin` group min EV，先做时间分段稳定性检查。
2. 对 `submitted_fill_calibration_gap` 做诊断，检查哪些 bid / p_side / hour 区间 fill 估计偏差最大。
3. 在不降低 `coverage >= 0.70` 的前提下，尝试更稳健的分组门槛，例如对 group min EV 加 shrinkage 或限制相邻分桶跳变。
4. 不优先加更复杂模型，因为本批收益主要来自 policy 层，而不是方向分类指标大幅变化。
5. 如果继续追 `validation_sum_pnl >= 1000`，优先改 fill/price policy，而不是单纯提高方向模型 accuracy。
