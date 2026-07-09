# Arbor Direct Policy V3 — Trade/L2 特征 + 下单价格 Cap

日期：2026-07-09
用途：在 `.arbor/sessions` 开下一轮 direct policy（1:08 市价单口径）实验。两步走：
1. **先**把 trade / L2 / market-path 新数据特征（含「1 分钟前最后一笔 trade 价格」）加入三方向分类模型训练，验证能否提高方向 edge 从而提高 B_test PnL。
2. **再**在最优特征模型上，给下单价格加 cap（依然市价单，但只在 chosen 侧市价 ≤ cap 时成交），扫描 0.4 / 0.5 等阈值，验证「降低平均成交价格」能否提高 PnL。

前置分析文档：`crypto_engine/docs/direct_policy_pnl_gap_analysis_20260709.md`（务必先读）。

---

## 0. 上一轮结论与本轮动机

### 0.1 支配性恒等式（V2 分析已确证）

市价单口径下：

```text
单笔盈亏 = y_chosen − m_chosen        （买对 y_chosen=1，买错 y_chosen=0）
E[pnl/trade] = P(chosen 侧赢) − E[m_chosen] = accuracy − avg_entry_price
```

- V1 DP2（best，B_test=137.08）：`0.6327 − 0.6136 = +0.0191`
- Oracle：`1.0 − 0.563 = +0.437`，PnL=3224
- 3224 → 137 的缺口几乎 100% 来自方向命中率不足。市场价 `m` 本身已是高度有效的胜率估计，模型相对市场只有约 **+2 pp** 的净 edge，且只在 `price ≥ 0.35` 区间为正。

因此只有两个杠杆：
```text
杠杆 1：提高相对市场价的方向 edge（accuracy − price 变大）—— 支配性
杠杆 2：降低平均买入价（avg_entry_price 变小）—— 但会连带降低 accuracy，净效果待验证
```

本轮 V3 正好各打一个：**Phase 1 打杠杆 1（trade/L2 特征），Phase 2 打杠杆 2（价格 cap）。**

### 0.2 为什么选 trade/L2 特征做杠杆 1

前几轮实验里，trade/L2 / market-path 是少数真正加入 forward market state、有信息增量的方向：

| 来源 | 机制 | 结果 |
|---|---|---|
| `20260703_trade_btc_feature_research` node 1 | decision-time trade + BTC feature bank | 六周 dev PnL `62.5 / 60.8 / 63.1 / 80.0 / 58.7 / 69.8` |
| `20260702_l2_A2_direction_l2` | classifier 加 L2 特征 | calib accepted acc `0.667`，ROC-AUC `0.725`（train AUC `0.843`）|
| `20260704_btest500_full_pipeline` node 2.1 | path-signature + liquidity-shape | Brier 改善 |

但注意：**这些以前都是在 limit-bid / Gc（expected-return）框架里测的**，B_test anchor 是 `42.43`，而 `20260705_trade_l2_all` 里所有 challenger 都没能在 B_test 超过 42.43。原因是 limit-bid 框架里方向信号被 fill 选择性稀释了。

**本轮的关键不同**：direct-policy 市价单框架下，方向 edge 直接线性驱动 PnL（见恒等式），没有 fill 稀释。所以「trade/L2 有信号但在 limit 框架没转化成 B_test PnL」不代表在市价单框架也不行——这正是值得重测的点。

### 0.3 关于价格 cap 的重要前置证据（必须正视）

用户直觉「降低平均成交价格 → 降低错单亏损 → 提高 PnL」在**单笔、固定 accuracy 下成立**，但 accuracy 会随价格下降而下降。两份硬证据：

1. **DP2 逐笔（B_test）**：模型实际胜率 ≈ chosen 侧市场价（每个价格桶都成立）。在便宜侧（price < 0.35）edge 为**负**（模型买冷门时比市场还差）。把 DP2 限制在 `chosen_m ≤ 0.45` 反而把 PnL 从 137 降到 122。
2. **线上真实成交**（`docs/aws_poly_20260607_to_20260702_runtime_analysis.md`，6986 周期）：利润集中在**高价段**，便宜段亏钱：

   | 下单价桶 | 胜率 | PnL |
   |---|---:|---:|
   | 0.3–0.4 | 35.4% | −13.5 |
   | 0.5–0.6 | 53.5% | **−68.3**（最大拖累） |
   | 0.6–0.7 | 65.7% | **+88.2** |
   | 0.7–0.8 | 75.7% | +49.9 |

**所以「把 cap 压到 0.4/0.5」在当前特征下大概率会砍掉赚钱的高价段、留下亏钱的便宜段，净效果为负。** 但这不是拒绝实验的理由，而是本轮要严格验证的假设：cap 只有在 **Phase 1 的新特征真的修好了便宜侧 accuracy** 之后才可能翻正。因此顺序必须是「先特征、后 cap」，且 cap 实验必须输出 accuracy-vs-price 的完整分解，用来解释而不是只看总 PnL。

---

## 1. 核心假设

```text
H1（杠杆1）：trade/L2/market-path 特征能把「分价格桶的 edge = 实测胜率 − 市场价」
            从当前 ~+0.02 稳定抬高，从而把 B_test sum_pnl 抬到 > 137。
H2（关键特征）：1 分钟前最后一笔 trade 价格 + 由它派生的短窗动量，是独立、强的
            方向特征，单独加入即可提升 edge。
H3（杠杆2）：在固定特征模型上加下单价格 cap 会降低 avg_entry_price；净 PnL 变化 =
            −Δaccuracy 的损失 + Δprice 的收益。当前特征下预期为负（便宜段亏钱），
            需要用逐桶分解验证真实方向。
H4（交互）：只有当 Phase 1 特征把便宜侧 edge 修到 ≥ 0 时，低 cap 才可能提高 PnL；
            否则最优「价格门」是中高价 band 而非低 cap。
```

---

## 2. 统一约束

- 口径仍是 1:08 市价单：`market_t0 + 68s` 前 chosen 侧最后一笔成交价，和 V1/V2 完全一致。
- 全量 universe：**不使用方向分类器的 accepted threshold / `t_up` / `t_down` 做样本筛选**，所有有合法 1:08 市价的行都进入评估。
- 基础特征 = 当前 deploy feature manifest（`execution_engine/deploy/baseline/artifact_manifest.json`，569 列）。trade/L2 新特征作为 **research-only 追加列**，必须单独标注，不改 deploy manifest 本身。
- 价格 cap 是**决策层**改造，不进入 feature columns（不能把 cap/`m_chosen` 作为泄漏特征回灌训练标签）。
- `B_test` 每个 named 节点记录一次，禁止用 B_test 反复筛参/筛阈值。
- 不改 `2mins`、deploy artifact、live config、用户分支。
- 泄漏纪律：禁止任何 `decision_time` 之后的信息进入特征；`trade_time` 只能用于构造 `< decision_time` 的聚合，不能作为输入列。
- 若使用 base model 预测 `p_up / p_base`，训练集必须是 OOF / walk-forward；B_test 只能用 B_test 之前数据训练出的 base 预测。
- **产物强制落盘**（修正 V2 教训：V2 各节点只落了 REPORT+config，缺 metrics/predictions，无法逐笔归因）。每节点必须写出 §9 全部产物。

建议新开 session：

```text
.arbor/sessions/20260709_direct_policy_trade_l2_price_cap
```

---

## 3. 数据源（全部只用 decision_time 之前的信息）

```text
方向基础帧（特征+target+market_t0+condition_id）：
  artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features/development_frame.parquet
  artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features/validation_frame.parquet

Polymarket 成交（构造 trade-path 与 1:08 / 1min-ago 价格）：
  price_estimator/data/sell_taker_trades_daily/date=*.parquet
    列：condition_id, outcome(YES/NO→UP/DOWN), price, trade_time

L2 first-minute 特征（已产出）：
  artifacts/data_v2/polymarket_l2/first_minute_features/         # 每市场首分钟盘口特征
  artifacts/data_v2/polymarket_l2/l2_frozen_splits.parquet       # L2 可用市场与 split
  构建代码：src/data/polymarket_l2.py (build_first_minute_features / build_trade_products)
           scripts/data/step4_features/build_polymarket_l2_features.py

结算标签：
  artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet

滚动 folds（调参/holdout，沿用）：
  .arbor/sessions/20260703_prefinal_rolling/folds/w{1..6}
  日期：w1 02-27..03-05, w2 03-06..03-12, w3 03-13..03-19, w4 03-20..03-26,
        w5 03-27..04-02, w6 04-03..04-10；B_test = 04-11..05-10
```

**L2 覆盖率警告**：L2 数据只覆盖 `l2_frozen_splits` 里的 eligible markets，覆盖率 < trade 覆盖率。所有含 L2 特征的节点必须报告 L2 coverage，并同时在「L2-covered」和「full」两个 universe 上评估，缺 L2 的行用 NaN + missing-flag，不得丢弃。

---

## 4. 特征定义

### 4.A Trade-path 特征（`price_estimator/data/sell_taker_trades_daily`）

对 chosen/opposite 两侧，窗口 = `decision_time` 之前（分 10s/30s/60s/120s 子窗）：

```text
last、mean、VWAP、min、max、range、std、slope(最小二乘斜率)
count、volume、notional、buy/sell imbalance（若方向可得）
最新成交距 decision_time 的秒数（recency）
selected_last − opposite_last、ratio、complement_deviation = |selected_last + opposite_last − 1|
pre-decision 卖压：短窗下行斜率、下破次数、低价成交占比
side-switching：selected/opposite 价格交替领先次数
path signature：一阶/二阶增量、回撤、反弹
```

### 4.B L2 / order-book 特征（`first_minute_features`，families 见 A5–A8 ablation）

```text
book_depth：best_bid/ask、mid、spread、bid/ask depth_1/3/5/10
order_flow：quote update rate、cancel/replace、near-bid 挂单厚度 depth(bid, bid+0.03)
trade_dynamics：首分钟成交强度、买卖冲击、mirror trade 占比
cross_side：selected vs opposite 的 depth/imbalance 相对强弱
imbalance = (bid_depth − ask_depth)/(bid_depth + ask_depth)
price_impact：买/卖固定 notional 需穿透的档数
market_mid：首分钟中价及其变化（与 deploy L2 direction 实验一致）
```

### 4.C ★ 1 分钟前最后一笔 trade 价格（用户重点，单列一个节点）

对 chosen 与 opposite 两侧定义**滞后参考价**与派生动量：

```text
锚点（需与用户确认，默认取 A）：
  A: lag_cutoff = decision_time − 60s            （= market_t0 + 8s，首分钟起点）
  B: lag_cutoff = market_t0                       （市场开盘参考）

对每一侧 s ∈ {YES, NO}：
  m_s_lag1m   = 最后一笔 (trade_time ≤ lag_cutoff) 的成交价
  m_s_now     = 现有 1:08 价（最后一笔 ≤ decision_time）
派生：
  d_s_1m      = m_s_now − m_s_lag1m               （首分钟价格变化 = 短窗动量）
  slope_s_1m  = d_s_1m / 60
  m_yes_lag1m − m_no_lag1m、以及 (d_yes_1m − d_no_1m)（双侧动量差）
  has_lag1m 覆盖标记；缺失用 NaN + flag，不回退成 now 价（避免零信息伪装）
```

理由：1:08 是一个静态快照，缺少「往哪个方向在动」。首分钟价格变化是 BTC 5 分钟 up/down 市场里最直接的短期方向信号，且与市场静态价 `m` 正交，最可能提高 `edge = acc − price`。

### 4.D 严禁作为输入列

```text
target, future_*, abs_return, signed_return, stage1_target, stage2_target,
chosen_low*, correct, winner, pnl, y, trade_time, endDate, condition_id,
market_id, slug, outcome, accepted, threshold_accepted, predicted_side,
m_yes, m_no, m_chosen（价格进 reward/decision，不进 feature，避免与标签同源泄漏）
```

---

## 5. 实验节点

### Phase 1 — Trade/L2 特征（打杠杆 1）

#### F0_frame_qa
确认 universe 与 V1/V2 一致 + 新数据覆盖率。
```text
row_count、m_yes/m_no coverage、any-side coverage、late_join_count==0
trade_path_coverage、l2_coverage、lag1m_coverage
feature_columns == deploy_manifest ∪ research_additions（列出追加列清单）
forbidden_feature_intersection == []
```

#### F1_baseline_replay
在 V3 代码环境里复现 V1 DP2（regret-CE，B_test≈137）作为 anchor。
```text
loss = hard-label multiclass CE, label = oracle_action
weight = clip(log1p(oracle_margin/0.02), 0.25, 5.0)
decision = argmax([pi_yes, pi_no, pi_none]);  c ∈ {0.00, 0.01, 0.02}
预期 B_test ≈ 137；偏差大先停下定位 frame/seed。
（注意：V2 的 DP1_replay 实际跑成了 static-EV=50，本轮务必用 regret-CE 对齐 137。）
```

#### F2_trade_path
基础特征 + 4.A trade-path。其余同 F1。

#### F3_l2
基础特征 + 4.B L2。报告 L2-covered 与 full 两个 universe。

#### F4_lag1m_price ★
基础特征 + 4.C（1 分钟前价格 + 动量派生）。单列以隔离该特征贡献。

#### F5_trade_l2_combined
基础特征 + 4.A + 4.B + 4.C 全量。选参网格保持最小（只调 `c`），避免过拟合。

**Phase 1 主判据不是只看 B_test，而是看 `edge = 实测胜率 − 市场价（分价格桶）` 是否稳定抬升**（edge 是 PnL 的领先指标，见分析文档）。每个 F 节点必须输出逐桶 edge 表并与 F1 对比。

### Phase 2 — 下单价格 Cap（打杠杆 2）

在 Phase 1 的 winner（记为 `M*`，通常是 F5 或 edge 最优者）上，**只改决策层，不重训**（除非 C3）。

#### 价格 cap 机制（主口径，匹配「依然市价单」）
```text
模型选出 side 与 m_chosen（1:08 chosen 侧市价）：
  if m_chosen ≤ cap:  下市价单，realized_pnl = y_chosen − m_chosen
  else:               NO_TRADE（放弃该单），realized_pnl = 0
```

#### C1_price_cap_grid
```text
cap ∈ {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00(=无 cap)}
对每个 cap 报告：sum_pnl、trade_rate、accuracy、avg_entry_price、
                edge=acc−price、wrong_side_loss、win/loss 笔数与金额
```

#### C2_price_gate_band（诊断，回答「最优价格区间在哪」）
把「cap」推广成「价格门」，同时扫上界和下界，找真实最优区间：
```text
trade iff  floor ≤ m_chosen ≤ cap
floor ∈ {0.00, 0.35, 0.45, 0.55};  cap ∈ {0.50, 0.60, 0.70, 0.80, 1.00}
```
目的：直接检验「利润在中高价段、便宜段亏钱」是否在 direct-policy B_test 上重现；如重现，则用户的低 cap 会被数据否定，最优是中高价 band。

#### C3_cap_aware_retrain（可选，只有 C1/C2 出现正信号才做）
把 cap 作为**已知约束**在训练时告诉模型（例如只对 `m_chosen ≤ cap` 的样本算 oracle/reward，或在 reward 里对超过 cap 的一侧置 −inf），让模型学会在 cap 约束下选侧，而不是事后砍单。

#### C4_limit_at_cap（低优先，对照）
对照「静止限价单挂在 cap」而非市价放弃：
```text
if m_chosen ≤ cap: 市价成交 y_chosen − m_chosen
else:              限价挂 cap，仅当窗口内 chosen 侧最低价 ≤ cap 才成交
```
预期更差（20260706 已证 resting limit 输给 market order，adverse selection）；仅作对照，不作主结论。

### 诊断节点

#### D1_stable_universe
Phase 1 winner 与 Phase 2 winner 在 `has_yes or has_no`、`has_yes and has_no`、`l2_covered` 三个 universe 上重算，不重新选参，只解释稳定性。

---

## 6. 选择协议（沿用滚动纪律）

```text
w1–w4：tune / 选参（选 c、cap、band）
w5–w6：holdout，未参与选参
B_test：冻结月 04-11..05-10，每个 named 节点记一次

tune_score      = sum_pnl(w1–w4)
tie_breaker     = higher edge(acc−price) → lower wrong_side_loss_abs → lower drawdown
holdout_passed  = holdout_sum > 0 且两周都 > 0
```

Phase 2 的 cap / band **只能在 w1–w4 上选**，禁止用 B_test 挑 cap。

---

## 7. 必报指标

每个节点（B_dev / rolling / B_test 都要）：

```text
sum_pnl、sample_count、trade_count、trade_rate
accuracy(=P(realized>0 | trade))、avg_entry_price、edge = accuracy − avg_entry_price
mean_pnl_per_trade、YES_pnl、NO_pnl、win_pnl_sum、loss_pnl_sum
wrong_side_loss(_abs)、wrong_buy_yes/no_count/loss、no_trade_count
oracle_pnl、capture_ratio、worst_week_pnl、selection_drawdown
恒等式自检：accuracy − avg_entry_price ≈ mean_pnl_per_trade
```

逐桶 edge 表（Phase 1 核心，Phase 2 也要）：

```text
按 m_chosen 分桶 [0.2,0.3,...,0.9,1.0]：n、mean_price、win_rate、edge、bucket_pnl
```

Phase 2 每个 cap/band 额外报：

```text
cap、trade_rate、avg_entry_price、accuracy、edge、sum_pnl、
Δsum_pnl_vs_nocap、Δavg_entry_vs_nocap、Δaccuracy_vs_nocap
（用来把总 PnL 变化拆成「价格下降的收益」和「accuracy 下降的损失」两项）
```

L2 节点额外报：`l2_coverage`、L2-covered vs full 两套上表。

---

## 8. 成功标准

Phase 1 正信号：
```text
holdout_passed == true
B_test sum_pnl > 137（V1 DP2）
中高价桶 edge 相对 F1 稳定抬升（不是单桶噪声）
leakage_check 通过；无 L2 覆盖导致的选择性偏差
强信号：B_test sum_pnl ≥ 300（≈ accuracy 0.66–0.67），且 edge 抬升在 w5–w6 稳定
```

Phase 2 正信号：
```text
某个 cap/band 的 w1–w4 tune_score 高于无 cap，且 holdout 两周为正，
B_test sum_pnl 高于同模型无 cap 版本，
且分解显示「价格下降收益 > accuracy 下降损失」（不是靠 B_test 过拟合挑出来的）。
若 C2 显示最优在中高价 band、低 cap 全负，则如实结论：用户的低 cap 假设被 direct-policy 数据否定，
真正的价格杠杆是「避开负 edge 的便宜段」，方向与「降低平均价」相反。
```

---

## 9. 产物要求（强制，逐节点）

输出到 `.arbor/sessions/20260709_direct_policy_trade_l2_price_cap/experiments/<node>/`：

```text
REPORT.md
config_used.yaml
feature_manifest.json          # 含 research_additions 列清单 + deploy manifest 交集校验
leakage_check.json
metrics_bdev.json
metrics_btest.json
predictions_btest.parquet
edge_by_price_bucket_btest.csv
confusion_btest.csv
wrong_side_decomposition_btest.csv
（Phase 2 额外）cap_sweep_btest.csv
```

`predictions_btest.parquet` 最少列：

```text
sample_id, decision_time, y, m_yes, m_no, m_chosen,
m_yes_lag1m, m_no_lag1m, d_yes_1m, d_no_1m,
pi_yes, pi_no, pi_none, action, cap_applied,
realized_pnl, oracle_action, oracle_pnl, gap_to_oracle,
has_yes, has_no, l2_covered
```

并维护 session 级账本 `v3_btest_ledger.csv`（每节点一行：experiment_id、feature_family、B_dev、w1–w6、B_test、Δvs137、accuracy、avg_entry、edge、leakage_passed、notes）。

---

## 10. 本轮不做

```text
改 deploy manifest / live config / 2mins / 用户分支
promotion / 上线
用 B_test 反复筛 cap 或 band
大规模 XGBoost 超参搜索（每节点只调最小网格）
回到 limit-bid / Gc 框架做 bid 调参（本轮是 direct-policy 市价单）
把某个价格/置信弃单区间写死成永久规则（必须每实验用自身训练分布无泄漏重学）
```

---

## 11. 一页纸执行顺序

```text
1. F0 QA → F1 复现 137 anchor（不对齐先别继续）
2. F2 / F3 / F4 分别隔离 trade / L2 / 1min-ago 三类特征，看逐桶 edge
3. F5 合并全部特征，得到 winner M*
4. 在 M* 上跑 C1 cap 扫描 + C2 价格 band → 得到价格杠杆的真实方向
5.（可选）C3 cap-aware 重训；C4 limit 对照
6. D1 稳定性诊断 → 汇总 ledger → 结论（不 promotion）
核心记住：edge = accuracy − price 是领先指标；先用特征把 edge 抬上去，
cap 只有在便宜侧 edge 被修正后才可能帮上忙。
```
