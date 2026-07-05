# Trade/L2 主线优化计划：每次实验强制记录 B_test

日期：2026-07-05

目标：把 trade/L2 / market path 特征作为下一轮主探索方向，系统验证它是否能超过当前冻结 B_test 最好结果 `42.43`，并建立“每次实验都记录 B_test”的实验账本。

## 0. 背景结论

前几轮 Arbor 实验里，真正有新信息增量的方向是 trade/L2 / market path，而不是继续扩 `q-alpha / Gc floor / min_ev` 这种后处理网格。

已有关键结果：

| 来源 | 机制 | 开发/滚动结果 | B_test 状态 |
|---|---|---:|---|
| `20260703_trade_btc_feature_research` node 1 | decision-time trade + BTC feature bank | 六周 PnL `62.48, 60.81, 63.07, 79.98, 58.74, 69.75` | 本组未测 B_test |
| `20260703_trade_btc_feature_research` node 1.1 | trade-conditioned monotone Gc calibrator | w1-w4 `290.3`, w5-w6 `138.58` | 本组未测 B_test |
| `20260703_trade_btc_feature_research` node 5 | 28-day p-bin empirical Gc | tune `272.94`, holdout `148.22` | 本组未测 B_test |
| `20260704_btest500_full_pipeline` node 2.1 | path-signature + liquidity-shape features | Brier 改善 | B_test `28.82` |
| `20260704_btest500_full_pipeline` node 2.2 | path features + q gate + exposure cap | holdout `139.86` | B_test `29.27` |

解释：trade/L2 确实能提供新信号，但简单把特征塞进 q/Gc 并不能保证最终月 PnL。下一轮需要围绕 submitted action 的盈利 fill 事件来建模，并且每次都记录 B_test。

### 0.1 线上真实运行结果（重要证据）

参考 `docs/aws_poly_20260607_to_20260702_runtime_analysis.md`（2026-06-07 至 07-02，6,986 个周期，真实成交结算）。它是真金白银的结果，比回测更硬，且直接支持“某些价格/置信区间干脆不做”的想法。

关键分桶（真实成交）：

| 维度 | 区间 | 胜率 | PnL | ROI | 判断 |
|---|---|---:|---:|---:|---|
| 下单限价 | 0.5–0.6 | 53.53% | **-68.33** | -1.83% | 最大拖累段 |
| 下单限价 | 0.3–0.4 | 35.37% | -13.46 | -3.51% | 亏损段 |
| 下单限价 | 0.6–0.7 | 65.70% | +88.24 | +2.16% | 主要正收益 |
| 下单限价 | 0.7–0.8 | 75.66% | +49.93 | +2.99% | 正收益 |
| 模型置信 | 0.5–0.6 | 46.85% | **-56.85** | -3.19% | 胜率低于 50%，最明确风险段 |
| 模型置信 | 0.6–0.7 | 55.17% | +85.18 | +1.91% | 正收益主力 |
| 模型置信 | 0.7–0.8 | 63.40% | +10.23 | +0.28% | 安全边际很薄 |

从真实结果能得到三个直接结论：

1. **低置信段结构性亏钱。** 模型置信 0.5–0.6 桶胜率只有 46.85%，低于 50%，在 forced-wrong-fill 经济下必然亏；这不是 bid 调不好，而是方向本身在这个置信带不可靠。
2. **中价段是最大现金拖累。** 下单价 0.5–0.6 桶亏 68.33，成交均价 0.544、胜率 53.5%，基本卡在盈亏平衡线上，噪声和 adverse selection 一叠加就转负。
3. **利润集中在中高置信、中高价段。** 0.6–0.8 价格段和 0.6–0.7 置信段贡献主要正收益。

所以“对很差的价格/置信区间直接不做”是被真实数据支持的方向，但必须满足两点：不能把这一段窗口的分桶结果写死成永久规则；必须在每个实验里用该实验自己的训练/开发分布，按时间顺序、无泄漏地重新学习弃单区间。这条正是本计划新增的主线 D。

### 0.2 回测与真实收益的差距

回测 anchor 是 `sum_pnl=42.43`，但线上真实区间 ROI 只有 +0.78%，且当前 U11 状态真实 PnL 为 -2.36、ROI -0.03%。这说明：

- 回测 `sum_pnl` 是理论口径，不等于真实成交后的收益；真实成交有部分成交、撤单、adverse selection 等摩擦。
- 部署策略当前基本是盈亏平衡，很小的分布漂移就能把它推成负。
- 因此不要只看回测 `sum_pnl` 的绝对值，要同时看它是否稳定为正、loss_pnl 是否可控、被选中订单是否真的能成交为赢家。

## 1. 新实验协议：B_test 必测、必记录

用户要求：B_test 数据集只有约七八千行，evaluation 很快，所以每一次实验都必须记录 B_test 结果。

本计划采用这个要求，规则如下：

1. 每个实验必须输出 `B_dev / rolling / B_test` 三类结果；如果某个 split 暂时不可运行，报告必须写明原因。
2. 每个实验必须保存 `btest_metrics.json`，不得只在终端里打印。
3. 每个实验必须保存 `btest_predictions.parquet` 或等价预测明细，至少包含 sample id、decision time、selected side、q、bid、expected EV、fill probability、filled、correct、realized PnL。
4. 每个实验必须追加一行到 `trade_l2_btest_ledger.csv`。
5. 不允许只报告 B_dev 高分；B_test 是每个实验的强制列。
6. 如果 B_test 结果变差，也必须记录，不能因为失败而省略。
7. 实验报告必须明确写出当前 best anchor：`42.43`，以及本实验相对它的 delta。
8. 任何 promotion 仍需要单独用户确认；记录 B_test 不等于自动上线。

### B_test 账本字段

建议每个 session 下维护：

```text
trade_l2_btest_ledger.csv
```

字段：

```text
experiment_id
run_time_utc
code_ref_or_git_status
data_manifest_path
feature_manifest_path
config_path
report_path
prediction_path
train_window
calibration_window
bdev_window
btest_window
feature_family
model_family
policy_family
bdev_sum_pnl
w1_sum_pnl
w2_sum_pnl
w3_sum_pnl
w4_sum_pnl
w5_sum_pnl
w6_sum_pnl
btest_sum_pnl
btest_delta_vs_42p43
btest_order_count
btest_order_coverage
btest_trade_count
btest_fill_rate
btest_mean_bid
btest_win_pnl_sum
btest_loss_pnl_sum
btest_correct_fill_rate
btest_wrong_submitted_count
btest_wrong_submitted_rate
btest_avg_loser_bid
btest_submitted_fill_calibration_gap
btest_q_brier
btest_gc_brier
abstained_segments
leakage_check_passed
notes
```

## 2. 主假设

trade/L2 方向值得继续，不是因为它已经证明能在 B_test 赢，而是因为它是少数真正加入 forward market state 的方向。它可能解释两个关键现象：

1. **winner fill 选择性。** 正确方向订单能不能以较低 bid 成交，取决于 pre-decision 市场路径、流动性、卖压、盘口厚度和短期冲击。
2. **loser adverse selection。** 很便宜、很容易成交的订单，可能正是市场在卖错方向 token；如果模型不能识别这一点，bid 会变成 loser forced-fill 损失。

因此下一轮主目标不是单独提高 q，也不是单独提高 Gc，而是直接提高 submitted bid 的 realized PnL。

## 3. 不变的交易语义

所有实验必须保持同一 fill / PnL 口径：

- 如果 `correct = 1`，订单只有在 `winner_low <= bid` 时成交，收益约为 `1 - bid`。
- 如果 `correct = 0` 且 `bid > 0`，订单按 forced wrong-fill 处理，损失约为 `bid`。
- 不允许把 wrong side 的 printed fill 当成主 PnL 口径；printed fill 只能作为诊断。
- 不允许把未下单样本从 accepted universe 中删除后再报告“条件收益”；必须同时报告 accepted-level PnL 和 order-level PnL。

## 4. 特征方向

### 4.1 Polymarket trade path 特征

每个样本只允许使用 `decision_time` 之前的数据。

建议特征：

- selected side 最近成交价：last、mean、VWAP、min、max、range、std、slope。
- opposite side 最近成交价：同上。
- selected/opposite 相对价差：`selected_last - opposite_last`、ratio、complement deviation。
- 成交量与成交次数：count、volume、notional、buy/sell imbalance，如果方向可得。
- 时间衰减特征：last 10s、30s、60s、120s 的 path stats。
- 最新成交距离 decision_time 的秒数。
- pre-decision sell pressure：短窗内价格下行斜率、下破次数、低价成交占比。
- side-switching：selected 与 opposite 价格交替领先的次数。
- path signature / trajectory shape：短窗路径的增量、二阶变化、回撤、反弹。

### 4.2 L2 / order book 特征

如果 L2 snapshot 或增量数据可用，优先做以下特征：

- best bid / best ask / mid / spread。
- top-k depth：bid_depth_1/3/5/10、ask_depth_1/3/5/10。
- imbalance：`(bid_depth - ask_depth) / (bid_depth + ask_depth)`。
- price impact：买/卖固定 notional 需要穿透多少档。
- selected side depth 与 opposite side depth 的相对强弱。
- spread widening / narrowing over last 10s/30s/60s。
- quote update rate、cancel/replace rate，如果可得。
- bid 附近的挂单厚度：`depth_between(bid, bid+0.03)`、`depth_near_bid`。

### 4.3 BTC second-level regime 特征

这些不是主线，但要保留作为辅助解释变量：

- BTC 1s/5s/15s/30s/60s return。
- realized volatility、range、micro-trend slope。
- impulse / reversal flags。
- volume burst、trade count burst。
- 与 Polymarket path 的交互：BTC 上冲但 selected side 被卖、BTC 稳定但 selected side 深跌等。

### 4.4 严禁使用的列

任何模型特征都必须排除：

```text
target
future_*
abs_return
signed_return
stage1_target
stage2_target
chosen_low
correct
winner
pnl
trade_time
endDate
condition_id
market_id
slug
outcome
```

注意：`trade_time` 可以用于构造 decision_time 之前的聚合特征，但不能作为模型输入列直接使用。

## 5. 建模主线

### 主线 A：trade/L2 joint profitable-fill model

这是第一优先级。

目标事件：

```text
r(b, X) = P(correct = 1 and winner_low <= b | X, b)
```

这个目标把方向正确和 winner fill 合成一个事件，避免 `q * Gc` 独立相乘的误差放大。

EV 形式：

```text
EV(b, X) = r(b, X) * (1 - b) - P(wrong | X) * b
```

其中 `P(wrong | X)` 可以先用 `1 - q(X)`，后续再试 joint loss model。

实验重点：

- 使用 trade/L2 path 特征作为 `X`。
- bid 作为显式输入特征。
- 约束 `r(b, X)` 随 bid 单调不减。
- 输出 bid-level predicted profitable-fill probability。
- 对 submitted action 做 calibration，而不是只看全体样本 Brier。

### 主线 B：trade/L2 Gc debias + hazard ranking

这是第二优先级。

已有证据说明：完全 empirical Gc 会丢掉 row-level ranking，但低方差 empirical Gc 可以修正 H2/hazard 过度自信。

做法：

- 保留 hazard/H2 的 row-level ranking。
- 用 trade/L2 buckets 做 shrinkage correction。
- 校准对象不是全局 Gc，而是 submitted bid 的 fill probability。

候选 correction：

- `Gc_corrected = (1 - lambda) * Gc_hazard + lambda * Gc_empirical_bucket`
- `Gc_corrected = Gc_hazard ^ gamma(bucket)`
- `Gc_corrected = min(Gc_hazard, upper_confidence_empirical_bucket)`
- `Gc_corrected = isotonic(Gc_hazard, trade_l2_bucket)`

### 主线 C：anchor-safe trade/L2 challenger

这是上线最安全的策略层。

默认 anchor：当前 best B_test `42.43`。

思路：

- trade/L2 模型给出 challenger bid / no-order。
- anchor policy 仍作为默认。
- 只有当 trade/L2 challenger 相对 anchor 的历史 posterior lower bound 为正，才允许替代 anchor。
- 这样可以避免 trade/L2 模型在某个月份过度自信时扩大 loser exposure。

主线 C 不替代 A/B，而是 promotion 层；A/B 负责产生 challenger，C 负责决定是否相信 challenger。

### 主线 D：数据驱动的分区间弃单（segment abstention）

这是本次评审新增的主线，直接来自线上真实结果，也直接采纳“某些价格/置信区间很差就不做”的想法。

它和 EV 策略是互补关系，不是替代：

- 理论上 EV 策略应该在负 EV 区间自动弃单。
- 但 q/Gc 一旦过度自信，EV 会在真实为负的区间给出正 EV，于是照样下单。
- segment abstention 是一层稳健护栏：即使模型说这里能赚，只要该区间在历史训练/开发分布上稳定亏，就直接不做。

核心原则：弃单区间必须由每个实验自己的数据分布决定，不能写死。

方法：

- 仅在训练/开发窗口（时间上早于评估窗口）上，按维度分桶统计 realized PnL / ROI。
- 维度至少包括：模型置信 `p_side`、下单价 `bid`、`p_side × bid` 联合网格；可扩展到 side、UTC session、liquidity bucket、trade-pressure bucket。
- 弃单判据用 PnL 的置信下界，而不是点估计：只有当某桶的 mean PnL 的 lower confidence bound < 0 且样本量足够时，才标记为弃单区间。
- 冻结这张弃单表，再在 w5-w6 和 B_test 上评估。
- 记录被弃单区间在 B_test 的真实分布：这些区间在 test 上是不是也亏、弃单救回多少 loss_pnl、误杀多少 win_pnl。

必须防的坑：

- **过拟合单一窗口。** 一个月很差的桶，换一个月可能不差。所以要用多个 rolling folds 检查该桶是否稳定为负，而不是只看一个 split。
- **样本太少的桶。** 线上 0.1–0.2 价格桶只有 10 单却 +69% ROI，这种桶不能作为“该做/不该做”的证据；小样本桶默认走 anchor/EV，不单独下弃单结论。
- **和 EV 双重否定。** 弃单表是 OR 逻辑：EV 说不做就不做；弃单表说不做也不做。两者任一否决即弃单。
- **低置信直接弃单可能是最便宜的大杠杆。** 线上 `p_side` 0.5–0.6 胜率 46.85%，仅提高该段弃单阈值就可能显著减亏，成本远低于任何复杂模型。所以主线 D 的最小版本应尽早测。

主线 D 的最小版本不需要新模型，只需要在现有 anchor / EV policy 上加一层区间闸门，因此它是性价比最高、最该先测的方向之一。

每个实验都必须跑 B_test 并记录账本。

### T0：复现 trade/L2 基线

目的：先复现已有高分方向，建立可比较 baseline。

做法：

- 复现 `20260703_trade_btc_feature_research` node 1 / 1.1 的 feature set 和 policy。
- 不加新机制。
- 输出 B_dev、w1-w6、B_test。

必须回答：

- 原 rolling 高分在当前代码和数据下能否复现？
- B_test 是否超过 `42.43`？
- 如果 B_test 很低，是 q drift、Gc drift、还是 loss exposure？

### T1：trade/L2 submitted-action calibration audit

目的：不改策略，只诊断已有 trade/L2 candidate 的失败点。

做法：

- 对每个 submitted order 记录 `model_fill_prob`、真实 fill、bid、q、expected EV。
- 分桶看 predicted fill vs realized fill。
- 分别看 correct submitted 和 wrong submitted。

必须输出：

- submitted fill calibration gap。
- by bid bucket calibration。
- by q bucket calibration。
- by selected/opposite liquidity bucket calibration。
- B_test PnL decomposition。

### T2：joint profitable-fill baseline

目的：直接建模 `P(correct and winner_low <= bid | X,bid)`。

做法：

- 使用 trade path + L2 + BTC 特征。
- bid grid 先用 5-cent 或已有 legal grid，不追求 1-cent 细化。
- 模型可以先用 LightGBM / CatBoost / monotone GBDT。
- 后处理保证 probability 随 bid 单调。

必须输出：

- B_test `sum_pnl`。
- joint-event Brier。
- submitted joint-event calibration。
- win/loss PnL decomposition。

### T3：joint profitable-fill + submitted calibration

目的：修正 T2 在 submitted subset 上的过度自信。

做法：

- 在 calibration window 上只用 policy 会提交的 action 拟合 calibration map。
- 可试 isotonic、temperature scaling、bucket shrinkage。
- calibration 不允许使用 B_test。

必须比较：

- T2 raw vs T3 calibrated 的 B_test。
- 是否降低 loss_pnl_sum。
- 是否过度降低 order_count。

### T4：trade/L2 empirical Gc shrink

目的：保留 hazard ranking，同时用 trade/L2 buckets 降低 Gc overconfidence。

做法：

- buckets 包含 `p_side`、bid、selected/opposite liquidity、short-window sell pressure。
- 对每个 bucket 估 empirical fill CDF。
- 与 hazard Gc 做 shrink blend。

必须比较：

- 完全 empirical。
- hazard only。
- shrink blend。
- upper-cap。

### T5：side-specific UP/DOWN path features

目的：不要只看 selected side，要显式利用 opposite side 状态。

做法：

- 为 UP 和 DOWN 各自构造 trade/L2 path features。
- 构造相对特征：UP_pressure - DOWN_pressure、UP_depth / DOWN_depth、UP_last + DOWN_last deviation。
- 仍可先保持 legacy direction，不急着 full-universe 改方向。

必须回答：

- side-relative features 是否提高 B_test PnL？
- 是改善 q，还是改善 fill selection？

### T6：loss-exposure constrained bid policy

目的：trade/L2 模型可能会提高 order_count，但也可能增加 loser exposure；T6 强制控制亏损侧。

做法：

- 在 T2/T3/T4 的基础上加入 avg loser bid proxy、max bid cap、q floor、liquidity disagreement gate。
- gate 的目标不是提高 Brier，而是降低 `loss_pnl_sum`。

必须输出：

- loss_pnl_sum。
- win_pnl_sum。
- wrong submitted rate。
- avg loser bid。
- B_test delta vs unconstrained policy。

### T7：anchor-safe challenger

目的：用 trade/L2 challenger 只替换 anchor 明显不好的样本。

做法：

- anchor = `42.43` policy。
- challenger = T2/T3/T4 最好候选。
- 在 calibration / rolling 历史上估计 challenger 相对 anchor 的 reward delta posterior。
- 只有 lower credible bound > 0 时替换。

必须输出：

- anchor B_test。
- challenger B_test。
- safe-gated B_test。
- 替换样本数。
- 替换样本的 win/loss PnL。

### T8：two-minute trade/L2 auxiliary horizon

目的：two-minute 方向不能直接替代 first-minute，但可作为辅助信号。

做法：

- 不把 two-minute policy 单独上线。
- 将 two-minute trade/L2 profitable-fill score 作为 first-minute policy 的辅助 feature 或 no-order gate。

必须回答：

- two-minute score 是否能识别 first-minute 的 loser exposure？
- 是否降低 B_test loss_pnl_sum？

### T9：数据驱动的分区间弃单（对应主线 D，建议尽早做）

目的：用每个实验自己的训练/开发分布，学出“很差就不做”的价格/置信区间，验证它能否减亏并提高 B_test `sum_pnl`。这一步不需要新模型，是性价比最高的护栏，建议在 T0/T1 之后立刻做。

做法：

- 基线可以是当前 anchor policy 或 T2/T3 的 EV policy。
- 只在训练/开发窗口按 `p_side`、`bid`、`p_side × bid` 分桶统计 realized PnL、ROI、样本量。
- 用 mean PnL 的 lower confidence bound 判据标记弃单桶，并设最小样本量门槛。
- 冻结弃单表，在 w5-w6 和 B_test 上评估。
- 至少测三档激进度：只弃 `p_side` 0.5–0.6 低置信段、只弃亏损价格段、两者联合。

必须输出：

- 弃单前后 B_test `sum_pnl`。
- 弃单救回的 loss_pnl 与误杀的 win_pnl。
- 被弃单区间在 B_test 上的真实 PnL 分布（验证弃单是否正确）。
- 弃单桶在多个 rolling folds 上的稳定性（是否只在一个窗口为负）。
- order_count / order_coverage 下降幅度。

必须回答：

- 弃单是提高了 `sum_pnl`，还是只是缩量降波动？
- 被弃的区间在 test 上是否也确实亏，还是被误杀？
- 低置信段弃单这一个便宜杠杆，单独能贡献多少减亏？

## 7. 每个实验报告模板

每个实验目录必须有：

```text
REPORT.md
config_used.yaml
feature_manifest.json
leakage_check.json
metrics_bdev.json
metrics_btest.json
predictions_btest.parquet
trade_l2_btest_ledger.csv 或追加到 session 级 ledger
```

`REPORT.md` 模板：

```markdown
# Experiment T? - short name

## Hypothesis

## What Changed

## What Did Not Change

- Direction artifact:
- Calibration method:
- Fill semantics:
- Accepted universe:

## Data Windows

- Train:
- Calibration:
- B_dev:
- B_test:

## Leakage Check

- Forbidden columns intersection:
- Feature cutoff check:
- Fit-on-validation check:

## Metrics

| Split | sum_pnl | delta_vs_anchor | order_count | order_coverage | win_pnl | loss_pnl | fill_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_dev | | | | | | | |
| B_test | | | | | | | |

## B_test Required Result

- btest_sum_pnl:
- btest_delta_vs_42p43:
- btest_wrong_submitted_rate:
- btest_avg_loser_bid:
- btest_submitted_fill_calibration_gap:

## Diagnosis

## Decision

- Continue / modify / stop:
- Reason:
```

## 8. B_test 指标要求

每次 B_test 至少报告：

```text
accepted_count
order_count
order_coverage
trade_count
fill_rate
sum_pnl
mean_accepted_pnl
mean_pnl_filled
mean_bid
win_pnl_sum
loss_pnl_sum
correct_fill_rate
wrong_fill_forced
wrong_fill_printed
wrong_submitted_count
wrong_submitted_rate
avg_loser_bid
submitted_fill_calibration_gap
q_brier
gc_brier
joint_event_brier, if applicable
```

还必须按以下维度拆 B_test：

- week / UTC day。
- bid bucket（价格分桶，直接对照线上 0.5–0.6 亏损段）。
- q bucket / p_side bucket（置信分桶，直接对照线上 0.5–0.6 亏损段）。
- `p_side × bid` 联合桶（用于主线 D 弃单表）。
- selected/opposite liquidity bucket。
- trade pressure bucket。

## 9. 成功标准

### 最低继续标准

满足任一条即可继续迭代：

- B_test `sum_pnl > 42.43`。
- B_test 未超过 `42.43`，但显著降低 `loss_pnl_sum` 且保留大部分 `win_pnl_sum`。
- B_test 未超过 `42.43`，但 submitted calibration 明显改善，并且下一步有明确可测修正。

### 强候选标准

强候选需要同时满足：

- B_test `sum_pnl >= 42.43 + 10`。
- B_test `loss_pnl_sum` 不比 anchor 更差，或 deterioration 有清晰收益补偿。
- submitted fill calibration gap 明显低于 anchor。
- B_dev / rolling 与 B_test 方向一致，不是 dev 高、test 崩。
- 无泄漏列、无 label proxy、无 timestamp 违规。

### Promotion 标准

即使每次都记录 B_test，也不自动 promotion。Promotion 需要：

- 至少一个完整 experiment report。
- B_test 明确超过 anchor。
- 用户确认接受。
- 只推广最小必要改动，不把实验脚本直接塞进 live execution。

## 10. 推荐执行顺序

1. **T0 复现 trade/L2 baseline。** 先知道高 rolling 分数对应的 B_test 到底是多少。
2. **T1 做 submitted-action audit。** 找到 trade/L2 候选在 B_test 失败或成功的具体位置。
3. **T9 分区间弃单（护栏）。** 最便宜的大杠杆，先用数据驱动的弃单表处理线上已知的低置信/中价亏损段，建立减亏基线。
4. **T2/T3 joint profitable-fill。** 主攻 `P(correct and winner_low <= bid | X,bid)`。
5. **T4 empirical Gc shrink。** 并行修正 hazard overconfidence。
6. **T5 side-specific path。** 把 UP/DOWN 相对市场状态补齐。
7. **T6 loss-exposure constraints。** 控制 forced wrong-fill。
8. **T7 anchor-safe challenger。** 用 trade/L2 候选替代 anchor 的部分订单，而不是全量替换。
9. **T8 two-minute auxiliary。** 只作为辅助，不作为主策略替换。

T9 和 T2/T3 可以并行：T9 是不依赖新模型的护栏，能立刻减亏；T2/T3 是提升上限的主模型。两者最终在主线 C 的 anchor-safe 层合并。

## 11. 不建议继续的分支

以下方向不是主线，除非作为 ablation：

- 继续扩大 `q-alpha / Gc floor / min_ev` 网格。
- 纯 q calibration，包括 isotonic、fixed clipping、p_side-bin residual q。
- 完全 empirical CDF 替代 hazard ranking。
- 低维 Bayesian optimization 只调 q-to-bid 曲线。
- standalone direct reward / contextual bandit 单独决定 bid。
- two-minute 单独替代 first-minute。

## 12. 关键判断

trade/L2 仍应作为主 **建模** 方向，因为它是少数真正加入 forward market state 的路径，是提升收益上限的主要来源。但仅有 trade/L2 不是当前最优的完整方向：线上真实数据显示，最便宜、最确定的减亏杠杆是“对结构性亏损的价格/置信区间直接不做”。

所以当前最优方向是两条腿并行：

- **建模腿（提升上限）：** trade/L2 joint profitable-fill + submitted calibration。
- **护栏腿（守住下限）：** 数据驱动的分区间弃单，尤其是低置信 `p_side` 0.5–0.6 段。

下一轮的正确做法是：

1. 每个实验都记录 B_test。
2. 先用 T9 分区间弃单守住线上已知亏损段，建立减亏基线。
3. 把 trade/L2 特征接到 joint profitable-fill 和 submitted calibration，提升收益上限。
4. 用 loss-exposure 和 anchor-safe gate 防止 B_test 上 forced wrong-fill 扩大。
5. 以 `42.43` 为明确 anchor，所有实验都报告 delta；弃单区间必须由每个实验自己的数据分布决定，不写死。

补充说明用户的想法：对很差的价格区间直接不做是对的，而且被真实成交数据支持；关键是把它做成“每个实验按自身分布、用置信下界、跨多折稳定性判定”的动态弃单表，而不是一条写死的静态规则。
