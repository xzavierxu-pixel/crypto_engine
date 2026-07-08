# Arbor Direct Policy Brief — Market-Order PnL

## 执行结果（2026-07-08）

已按本文实现独立 session：

```text
.arbor/sessions/20260708_direct_policy_market_order
```

最终每组实验 B_test：

```text
DP0_frame_qa                QA only
DP1_static_ev_baseline      sum_pnl = 50.1197097360   holdout_passed = true
DP2_direct_policy_regret_ce sum_pnl = 137.0812075043  holdout_passed = true
```

关键结论：

- `DP2` 明显优于 `DP1`，相对 `DP1` 增加 `+86.9614977683`。
- `DP2` 的 `trade_count=7193`、`trade_rate=0.9632`、`worst_week_pnl=-5.3170`、`selection_drawdown=26.3519`。
- `DP0` 确认 B_test `m_yes coverage=0.9888`、`m_no coverage=0.9867`、`any-side coverage=0.9896`、`late_join_count=0`。
- 详细产物见 session 下 `REPORT.md`、`direct_policy_btest_ledger.csv` 和各节点 `metrics_btest.json`。

日期：2026-07-08
用途：给 `crypto_engine/.arbor/sessions` 继续开 Arbor 实验用的精简执行说明。

## 0. 结论先行

本轮只做一个粗比较实验：在 M1 市价单语义下，用当前 deploy feature manifest 训练一个三动作 direct policy：

```text
BUY_YES / BUY_NO / NO_TRADE
```

不做大规模调参，不新增 research-only 特征，不做 promotion。目标是快速判断 direct policy 是否值得进入下一轮完整优化。

比较对象：

```text
no_trade baseline
static market EV baseline: score - m_yes > tau / (1 - score) - m_no > tau, if score exists
DP1 direct policy: regret-weighted multiclass logloss
oracle upper bound
```

## 1. 起点与约束

参考 session：`.arbor/sessions/20260706_gc_market_stop_all_clean`。

已知事实：

- M1 市价单 `raw_tree_blend / tau=0.0`，B_test `sum_pnl=122.63`。
- M2 同 5164 行 legal-m universe：市价 `122.63`，限价 anchor `45.41`。
- 用户已验证：方向 ranking 的 AUC 衰减很少，主要问题是 calibration level 漂移。

本实验约束：

- 特征列保持和当前 deploy 模型一致。
- `m_yes/m_no/y` 只用于 reward、oracle label、baseline、PnL 计算，默认不进入模型特征。
- 若需要 `score` 做 static EV baseline，只能用 deploy 同口径 OOF / walk-forward score。
- 不使用已有方向分类器的 accepted threshold / `t_up` / `t_down` 做样本筛选；所有有合法 1:08 市价的行都进入粗比较。
- 每个 named 节点都必须记录 B_test 结果；B_test 只用于最终记录和粗比较，不能用于调参、改配置或反复筛候选。
- 不改 `2mins`、deploy artifact、live config。

## 2. 市价单语义

每行是一次交易机会，不拆 YES/NO 两行。

必需字段：

```text
sample_id
decision_time
condition_id
y                    # YES resolve => 1, NO resolve => 0
m_yes                # YES price from trade data at 1:08 cutoff
m_no                 # NO price from trade data at 1:08 cutoff, or proxy
fold_id or split
<current_deploy_feature_columns>
```

市价 `m_yes/m_no` 使用 Arbor 之前 M 系列相同口径：从 `price_estimator/data/sell_taker_trades_daily/date=*.parquet` 读取 trade 数据，按 `condition_id + outcome` 分组，在 `market_t0 + 68s` 之前取最后一笔成交价。

```text
cutoff = market_t0 + 68s
m_yes = last price where condition_id matches, outcome == YES, trade_time <= cutoff
m_no  = last price where condition_id matches, outcome == NO,  trade_time <= cutoff
```

没有 cutoff 前 trade 的一侧视为该侧不可交易；该动作不能被提交。`decision_time` 统一记为 `cutoff`，并报告 `m_yes/m_no` 覆盖率与 late join 数。

第一版若没有 NO 可执行价，可用 research proxy：

```text
m_no = 1 - m_yes
```

但报告必须标注该 proxy 不是上线口径。

动作收益用于训练 label：

```text
reward_yes  = y       - m_yes - c
reward_no   = (1 - y) - m_no  - c
reward_none = 0
```

回测 PnL 不扣 `c`，只算真实经济收益：

```text
BUY_YES  pnl = y       - m_yes
BUY_NO   pnl = (1 - y) - m_no
NO_TRADE pnl = 0
```

`c` 是 required edge / safety margin，不是手续费。真实手续费和滑点若有数据，应单独扣在 PnL 中。

## 3. Arbor 节点

建议新开 session：

```text
.arbor/sessions/20260708_direct_policy_market_order
```

### DP0_frame_qa

目的：构造 direct policy frame，并确认价格、标签、特征 manifest 可用。

检查：

```text
row_count
m_yes coverage
m_no coverage or proxy flag
late_join_count == 0
m_yes_trade_time <= market_t0 + 68s for all legal YES prices
m_no_trade_time <= market_t0 + 68s for all legal NO prices, unless proxy
feature_columns == current_deploy_feature_manifest
forbidden_feature_intersection == []
```

产物：`experiments/DP0_frame_qa/REPORT.md`、`metrics_bdev.json`、`leakage_check.json`、`feature_manifest.json`。

### DP1_static_ev_baseline

目的：给 direct policy 一个同 universe 的粗基线。

若有 OOF / walk-forward deploy score：

```text
BUY_YES if score - m_yes > tau
BUY_NO  if (1 - score) - m_no > tau
else NO_TRADE
```

这里的 `tau` 是市价 EV edge 门槛，不是方向分类器的概率阈值。不要再套用 deploy 方向模型的 accepted threshold；否则会把“方向筛选”和“市价 EV 决策”混在一起，无法公平判断 direct policy。

只扫小网格：

```text
tau in {0.00, 0.01, 0.02}
```

若没有 score，则跳过 static EV，只报告 `no_trade` 和 `oracle`。

### DP2_direct_policy_regret_ce

目的：主实验。用三分类 XGBoost 直接学习动作。

训练 label：

```text
oracle_action = argmax([reward_yes, reward_no, reward_none])
```

样本权重：

```text
oracle_margin = max(reward) - second_largest(reward)
weight = clip(log1p(oracle_margin / 0.02), 0.25, 5.0)
```

模型固定配置，先不调参：

```text
objective = multi:softprob
num_class = 3
max_depth = 3
eta = 0.05
subsample = 0.9
colsample_bytree = 0.8
min_child_weight = 50
lambda = 5
num_boost_round = 500
early_stopping_rounds = 50
```

只做极小 `c` 比较，不做分类概率二次阈值筛选：

```text
c in {0.00, 0.01, 0.02}
```

决策：

```text
action = argmax([pi_yes, pi_no, pi_none])
```

若某一侧没有合法 1:08 价格，则该动作在 argmax 前 mask 掉；不要用分类模型概率阈值替代价格可交易性检查。

### DP3_optional_soft_target

暂不实现。只有 DP2 出现明显 hard-label 过度交易，但粗比较仍有正信号时，再开下一轮。

## 4. 选择协议

沿用 Arbor rolling discipline：

```text
w1-w4: tune / rough comparison
w5-w6: holdout diagnostic
B_test: frozen final month, every named experiment records once
```

选择规则保持简单：

```text
tune_score = sum_pnl(w1-w4)
tie_breaker = lower drawdown, then lower trade_rate
holdout_passed = holdout_sum > 0 and both holdout weeks > 0
```

DP0、DP1、DP2 都要写 `metrics_btest.json` 并追加 session ledger。若 holdout 失败，仍记录 B_test，但报告必须标注 `holdout_passed=false`，结论不得写成可 promotion 或可上线，只能作为诊断证据。

## 5. 粗比较指标

主指标：

```text
sum_pnl
```

必报辅助指标：

```text
sample_count
trade_count
trade_rate
mean_pnl_per_trade
YES_pnl
NO_pnl
wrong_side_loss
no_trade_count
oracle_pnl
capture_ratio = sum_pnl / oracle_pnl
worst_week_pnl
```

暂不做复杂 regret 分解、月度风控、完整调参表。报告只需要判断：

```text
DP2 是否粗略优于 no_trade / static EV / M1 same-universe baseline
DP2 的收益是否来自单边 YES/NO 偶然暴露
DP2 是否出现明显过度交易
```

## 6. 产物要求

每个节点输出到：

```text
.arbor/sessions/20260708_direct_policy_market_order/experiments/<node_name>/
```

最低产物：

```text
REPORT.md
config_used.yaml
feature_manifest.json
leakage_check.json
metrics_bdev.json
metrics_btest.json          # 每个 named 节点都必须记录
predictions_btest.parquet   # DP0 可为空 schema；DP1/DP2 写实际 B_test predictions
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
action
realized_pnl
oracle_action
oracle_pnl
```

## 7. 防泄漏

禁止进入 feature columns：

```text
y
target
future_*
abs_return
signed_return
stage1_target
stage2_target
stage1_sample_weight
chosen_low
chosen_low_trade_time
time_to_chosen_low_sec
correct
winner
pnl
post-decision trade path fields
```

若使用 `market_t0 + 68s` 价格，必须验证：

```text
m_yes_trade_time <= market_t0 + 68s
m_no_trade_time <= market_t0 + 68s, unless m_no uses proxy
```

## 8. 成功标准

研究成功，不等于 promotion。

DP2 算正信号需同时满足：

```text
holdout_passed == true
B_test sum_pnl > static EV baseline on same universe, if baseline exists
B_test sum_pnl > no_trade
trade_rate not obviously degenerate
leakage_check passed
```

若只在 w1-w4 好、w5-w6 失败，仍记录 B_test，但结论写为：direct policy failed holdout; B_test is diagnostic only, not promotion evidence.

## 9. 下一轮再做

本轮暂不做：

```text
大规模 XGBoost 调参
custom expected-reward objective
soft-target CE
price/logit/rolling residual 新特征
真实 YES/NO order-book depth sizing
promotion / live config change
```

只有 DP2 粗比较通过后，再开第二份文档细化这些内容。
