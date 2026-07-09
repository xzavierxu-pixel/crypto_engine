# Arbor Direct Policy V2 — Loss and Price-Aware Decision

日期：2026-07-08
用途：在 `.arbor/sessions` 继续开下一轮 direct policy 实验，重点验证 loss 改造和 price-aware 决策能否减少错边亏损。

## 0. 上一轮结论

参考 session：

```text
.arbor/sessions/20260708_direct_policy_market_order
```

B_test 结果：

```text
DP1_static_ev_baseline      sum_pnl = 50.1197
DP2_direct_policy_regret_ce sum_pnl = 137.0812
DP2 oracle_pnl              sum_pnl = 3224.2443
```

结论：

- `DP2 > DP1`，说明 direct policy 方案已经比当前 static EV 估计更有信号。
- `DP2 capture_ratio = 4.25%`，距离 oracle 上限仍很远。
- `DP2 trade_rate = 96.32%`，问题不是下单太少。
- 主要漏损来自 cross-side error：买 YES 但 NO 赢、买 NO 但 YES 赢。

关键 B_test 拆解：

```text
trade_count    = 7193
trade_accuracy = 63.27%
win_pnl_sum    = +1616.21
loss_pnl_sum   = -1479.12
net_pnl        = +137.08

wrong BUY_NO when oracle BUY_YES gap  ≈ 1541.30
wrong BUY_YES when oracle BUY_NO gap  ≈ 1447.28
NO_TRADE missed oracle gap            ≈   94.18
```

因此 V2 的目标不是继续提高 trade_rate，而是降低错边交易。

## 1. 核心假设

V1 的 hard-label multiclass CE 有两个缺陷：

```text
1. 把 BUY_WRONG_SIDE 和 NO_TRADE 都当成普通分类错误，不能充分表达错边亏损更大。
2. 推理层 action = argmax(pi_yes, pi_no, pi_none)，没有显式比较 m_yes/m_no 下的经济效用。
```

V2 要验证：

```text
H1: reward soft-target CE 能减少 cross-side error。
H2: cross-side regret weighting 能进一步压低买反边亏损。
H3: price-aware utility decision 能比直接 argmax(pi) 更稳定地提升 PnL。
H4: XGBoost custom PnL objective 能比加权分类更直接优化 validation sum_pnl。
```

## 2. 统一约束

- 市价仍使用 M 系列 1:08 口径：`market_t0 + 68s` 前最后一笔 YES/NO trade price。
- 不使用方向分类器的 accepted threshold / `t_up` / `t_down` 做样本筛选。
- 所有有合法 1:08 市价的行都进入评估。
- `B_test` 每个 named 节点记录一次，不能用于调参或反复筛候选。
- 不改 `2mins`、deploy artifact、live config。
- 主模型特征默认仍保持 current deploy feature manifest；`m_yes/m_no` 可以进入 reward、label、utility decision，不默认进入 feature columns。
- 若某个节点把 `m_yes/m_no` 加入模型 feature，必须标记为 research-only price-context ablation，不能和 deploy-manifest-only 节点混成同一结论。
- 若使用 `p_base`，训练集必须是 OOF / walk-forward prediction；B_test 只能使用由 B_test 之前数据训练出的 base model prediction，禁止用全量模型回填。

建议新开 session：

```text
.arbor/sessions/20260708_direct_policy_loss_price_aware
```

## 3. 共同数据帧

复用 V1 frame 逻辑，必需字段：

```text
sample_id
decision_time
condition_id
y
m_yes
m_no
has_yes
has_no
m_yes_trade_time
m_no_trade_time
fold_id or split
<current_deploy_feature_columns>
```

奖励定义：

```text
reward_yes  = y       - m_yes - c
reward_no   = (1 - y) - m_no  - c
reward_none = 0
reward_vec  = [reward_yes, reward_no, reward_none]
oracle_action = argmax(reward_vec)
oracle_pnl    = max([y - m_yes, (1-y) - m_no, 0])
```

不可交易一侧在 reward 中置为 `-inf`，并在推理前 mask。

## 4. 实验节点

### DP0_frame_qa_v2

目的：确认 V2 与 V1 使用同一 1:08 market-order universe。

检查：

```text
row_count
m_yes coverage
m_no coverage
any-side coverage
late_join_count == 0
feature_columns == current_deploy_feature_manifest
forbidden_feature_intersection == []
```

### DP1_replay_v1_reference

目的：在 V2 session 中复现 V1 的 DP2 决策，作为同代码环境 reference。

要求：

```text
loss = hard-label multiclass CE
label = oracle_action
weight = clip(log1p(oracle_margin / 0.02), 0.25, 5.0)
decision = argmax([pi_yes, pi_no, pi_none])
c in {0.00, 0.01, 0.02}
```

预期：B_test 应接近 V1 `137.08`。如果偏差很大，先停止并定位 frame 或随机种子差异。

### DP2_soft_reward_ce

目的：验证 soft target 能否减少 hard label 过度交易和错边。

训练 target：

```text
target_i = softmax(reward_i / temperature)
```

网格只做小范围：

```text
c in {0.00, 0.01, 0.02}
temperature in {0.03, 0.05, 0.08, 0.10}
```

实现建议：

- 如果 XGBoost wrapper 不方便直接吃 soft label，可先用 custom objective 或转成 3 个 one-vs-rest value/probability heads。
- 若实现成本过高，先用 `sample_weight` 近似：对非 oracle class 的 cross-side regret 加大权重，但报告必须标注为 approximation。

推理第一版仍用：

```text
action = argmax([pi_yes, pi_no, pi_none])
```

这样可以单独观察 loss 改造的影响。

### DP3_cross_side_regret_weighted_ce

目的：保持 hard label，但显式提高错边样本权重。

定义：

```text
best_reward = max(reward_vec)
wrong_side_reward =
  reward_no  if oracle_action == BUY_YES
  reward_yes if oracle_action == BUY_NO
  max(reward_yes, reward_no) if oracle_action == NO_TRADE

cross_side_regret = best_reward - wrong_side_reward
weight = clip(log1p(cross_side_regret / regret_scale), 0.5, 10.0)
```

小网格：

```text
c in {0.00, 0.01, 0.02}
regret_scale in {0.03, 0.05, 0.08}
```

推理仍用 direct argmax，先隔离 weight 改造效果。

### DP4_price_aware_utility_layer

目的：验证 price-aware 决策层是否比直接 `argmax(pi)` 更适合市价单 PnL。

输入模型可以先复用 DP2 或 DP3 的 `pi_yes/pi_no/pi_none`，但最终 action 改为 utility argmax。

候选 utility：

```text
utility_yes  = s_yes - m_yes - tau
utility_no   = s_no  - m_no  - tau
utility_none = 0
```

其中 `s_yes/s_no` 先用最简单变换：

```text
s_yes = pi_yes / (pi_yes + pi_no)
s_no  = pi_no  / (pi_yes + pi_no)
```

小网格：

```text
tau in {0.00, 0.01, 0.02, 0.03}
source_model in {DP2_soft_reward_ce_best, DP3_cross_side_regret_weighted_ce_best, DP5_xgb_direct_pnl_objective_best}
```

注意：这里的 `tau` 是 utility edge，不是方向分类器概率阈值，也不是 accepted threshold。

### DP5_xgb_direct_pnl_objective

目的：补做 End-to-End XGBoost Direct Policy PnL Objective。这个方案 V1 没做过；V1 的 DP2 是 hard-label `multi:softprob` 分类，不是 custom expected-reward objective。

动作：

```text
class 0 = BUY_YES
class 1 = BUY_NO
class 2 = NO_TRADE
```

价格口径分两种，但主结论优先看 actual-price：

```text
actual-price: m_yes = YES 1:08 executable price, m_no = NO 1:08 executable price
proxy-price:  m = m_yes, m_no = 1 - m_yes
```

`proxy-price` 只用于复刻用户原始公式；如果 actual NO price 可用，实际研究结论以 `actual-price` 为准。

奖励：

```text
actual-price:
  reward_yes  = y       - m_yes - c
  reward_no   = (1 - y) - m_no  - c
  reward_none = 0

proxy-price:
  reward_yes  = y - m - c
  reward_no   = m - y - c
  reward_none = 0
```

新增 research-only 特征：

```text
m_yes
m_no
m_yes_minus_m_no
abs(m_yes - 0.5)
abs(m_no - 0.5)
logit_m_yes = log(m_yes / (1 - m_yes))
logit_m_no  = log(m_no  / (1 - m_no))
p_base
p_base - m_yes
(1 - p_base) - m_no
abs(p_base - m_yes)
abs((1 - p_base) - m_no)
abs(p_base - 0.5)
```

如果跑 `proxy-price`，可额外输出与原始方案一致的列：

```text
m
1 - m
p_base - m
abs(p_base - m)
```

特征标记：

```text
research_only_price_and_base_features = true
p_base_source = OOF / walk-forward only
```

模型输出 raw scores：

```text
z_yes, z_no, z_none
pi = softmax(z / temperature)
```

Direct policy objective：

```text
expected_reward = pi_yes * reward_yes + pi_no * reward_no + pi_none * reward_none
loss = -expected_reward
```

custom objective 实现要求：

```text
grad_j = pi_j * (expected_reward - reward_j) / temperature
```

XGBoost 需要非负 Hessian；由于该 objective 的真实 Hessian 可能不是 PSD，第一版使用稳定的 diagonal surrogate：

```text
hess_j = max(pi_j * (1 - pi_j) * reward_scale / temperature^2, hess_floor)
reward_scale = max(abs(reward_yes), abs(reward_no), 0.05)
hess_floor = 1e-4
```

报告必须写明使用 surrogate Hessian，不要把它描述成精确二阶优化。

Validation hard action：

```text
action = argmax([z_yes, z_no, z_none])
```

真实 PnL：

```text
BUY_YES  pnl = y       - m_yes
BUY_NO   pnl = (1 - y) - m_no
NO_TRADE pnl = 0
```

小网格：

```text
c in {0.00, 0.01, 0.02, 0.03}
temperature in {0.05, 0.08, 0.10, 0.15}
max_depth in {2, 3}
eta in {0.03, 0.05}
subsample = 0.9
colsample_bytree = 0.8
num_boost_round <= 800
early_stopping_rounds = 50
```

选择标准仍然只用 w1-w4：

```text
primary = sum_pnl(w1-w4)
tie_breaker = lower wrong_side_loss_abs, then lower drawdown
```

必报额外诊断：

```text
objective_train_curve
validation_sum_pnl_by_round
temperature
price_mode = actual-price / proxy-price
p_base_leakage_check
custom_hessian_mode = surrogate_diagonal
```

### DP6_optional_price_context_model

暂不作为主结论。只有 DP2/DP3/DP5 loss 改造有效但 DP4 utility 仍明显过度交易时再跑。

目的：检查 `m_yes/m_no` 作为 research-only feature 是否能帮助模型学习价格状态。

新增 feature 只允许：

```text
m_yes
m_no
m_yes_minus_m_no
min(m_yes, m_no)
max(m_yes, m_no)
```

报告必须单独标注：`research_only_price_context=true`。

### DP7_stable_universe_diagnostic

目的：排除 w1-w6 与 B_test 的 1:08 价格覆盖率差异造成的误判。

V1 中 B_test 接近满覆盖，但早期 rolling fold 价格覆盖率更低；这可能让 tune 阶段低估满覆盖环境下的亏损暴露。

诊断方式：

```text
universe_a = has_yes or has_no
universe_b = has_yes and has_no
```

对 DP1/DP2/DP3/DP4/DP5 的 winner 配置分别在两个 universe 上重算 w1-w6 与 B_test 指标。这个节点不重新选参，只用于解释结果稳定性。

必报：

```text
coverage_by_fold
sum_pnl_by_fold_by_universe
trade_rate_by_fold_by_universe
wrong_side_loss_by_fold_by_universe
```

## 5. 选择协议

沿用 rolling discipline：

```text
w1-w4: tune / rough comparison
w5-w6: holdout diagnostic
B_test: frozen final month, every named experiment records once
```

选择规则：

```text
tune_score = sum_pnl(w1-w4)
tie_breaker = lower wrong_side_loss_abs, then lower drawdown, then lower trade_rate
holdout_passed = holdout_sum > 0 and both holdout weeks > 0
```

如果一个候选 `tune_score` 更高但 `wrong_side_loss_abs` 明显更高，报告必须指出这是高暴露策略，不可直接 promotion。

## 6. 必报指标

除 V1 指标外，V2 必须新增错边指标：

```text
sum_pnl
sample_count
trade_count
trade_rate
trade_accuracy
mean_pnl_per_trade
YES_pnl
NO_pnl
win_pnl_sum
loss_pnl_sum
wrong_side_loss
wrong_side_count
wrong_buy_yes_count
wrong_buy_no_count
wrong_buy_yes_loss
wrong_buy_no_loss
no_trade_missed_oracle_pnl
oracle_pnl
capture_ratio
cross_side_gap_to_oracle
worst_week_pnl
selection_drawdown
```

B_test 报告必须包含 action x oracle_action confusion table：

```text
rows = action
cols = oracle_action
values = count, realized_pnl, oracle_gap
```

## 7. 成功标准

V2 算正信号，需要同时满足：

```text
holdout_passed == true
B_test sum_pnl > V1 DP2 B_test 137.08
B_test wrong_side_loss_abs < V1 DP2 1479.12
B_test capture_ratio > V1 DP2 4.25%
trade_rate not higher than V1 by more than 2pct unless sum_pnl improves materially
leakage_check passed
```

强信号标准：

```text
B_test sum_pnl >= 300
wrong_side_loss_abs reduced by at least 15%
no single side contributes more than 70% of total pnl
```

如果只通过 B_test、但 w5-w6 不稳，仍记录结果，但结论写为 diagnostic only。

## 8. 产物要求

每个节点输出到：

```text
.arbor/sessions/20260708_direct_policy_loss_price_aware/experiments/<node_name>/
```

最低产物：

```text
REPORT.md
config_used.yaml
feature_manifest.json
leakage_check.json
metrics_bdev.json
metrics_btest.json
predictions_btest.parquet
confusion_btest.csv
wrong_side_decomposition_btest.csv
```

`predictions_btest.parquet` 最少列：

```text
sample_id
decision_time
y
m_yes
m_no
pi_yes
pi_no
pi_none
utility_yes
utility_no
utility_none
action
realized_pnl
oracle_action
oracle_pnl
gap_to_oracle
```

## 9. 本轮不做

```text
大规模 XGBoost 调参
新 research feature pack
order-book depth sizing
手续费/滑点上线化
promotion / live config change
用 B_test 反复筛阈值
```

本轮目标很窄：验证 loss 改造和 price-aware utility 是否能减少错边亏损，并确认 direct policy 相对 static EV 的优势是否可扩大。