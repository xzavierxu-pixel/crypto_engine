# Predicted-Side Soft-PnL Bid Policy 设计方案

日期：2026-06-24  
目标：在已有方向预测模型基础上，只学习下单价格 `bid`，最大化 validation PnL。  
硬约束：允许不下单，但 accepted 样本上的 `order_coverage >= 0.70`。  
相关背景：[20260624_r9_full_chain_experiment_documentation.md](20260624_r9_full_chain_experiment_documentation.md)，[20260624_safe_lowest_price_gap_results_analysis.md](20260624_safe_lowest_price_gap_results_analysis.md)

## 0. 一句话结论

不要把 bid 拆成 action grid，也不要继续只预测 winner token low。建议训练一个 **predicted-side single-output bid policy**：

```text
model(X) -> bid_raw in [0, p_side]
calibration threshold -> bid = 0 or tick-rounded bid_raw
```

训练时使用 predicted-side accepted 样本，loss 直接平滑近似真实 PnL：

```text
correct=True:  soft_pnl = soft_fill(bid, chosen_low) * (1 - bid)
correct=False: soft_pnl = -lambda_wrong * bid
loss = -mean(soft_pnl)
```

这样保留 R9 的“单模型直接出价”思想，但解决原 R9 的样本不一致问题：模型会同时见到约 70% correct 样本和约 30% wrong 样本；correct 样本教它“bid 要接住赢家回撤但不能太贵”，wrong 样本教它“方向错时 bid 越高亏越多”。

最终不在 validation 上调参。fit 训练模型，calibration 选择 `tau/lambda_wrong/order-threshold` 并强制 `order_coverage >= 0.70`，validation 只报告一次。

## 1. 为什么不用 action expansion

把每个样本展开为所有候选 tick：

```text
b in {0, 0.01, 0.02, ..., floor(p_side/0.01)*0.01}
```

理论上最直接，但这里不推荐作为主线：

1. 样本量膨胀，训练、校准和分桶诊断都会变重。
2. 单个 `(X,b)` 的 PnL label 是阶跃函数，局部噪声大。
3. 模型容易学到不可泛化的 bid 细节。
4. 你真正需要的是每个样本一个 bid，不是完整收益面。

因此本文方案改成单输出：每条样本只产生一次模型输出，直接训练 `bid=f(X)`。

## 2. 问题定义

已有方向模型给出 `p_up` 和部署阈值：

```text
accepted = (p_up >= t_up) or (p_up <= t_down)
selected_side = UP   if p_up >= t_up
              = DOWN if p_up <= t_down
p_side = p_up       if selected_side == UP
       = 1 - p_up   if selected_side == DOWN
```

对每个 accepted 样本，唯一决策是 limit buy 价格 `bid`：

```text
0 <= bid <= p_side
bid = 0 表示不下单
bid > 0 表示提交订单
```

真实 PnL 口径沿用 expected-return 分支的 forced-wrong-fill 规则：

```text
correct=True:
  if chosen_low <= bid: pnl = 1 - bid
  else:                 pnl = 0

correct=False:
  if bid > 0:           pnl = -bid
  else:                 pnl = 0
```

其中：

- `correct` 表示方向模型选中的 side 是否最终盈利。
- `chosen_low` 是 **预测侧 token** 在订单窗口中的最低成交价，不是赢家 token low。
- wrong 样本的 `chosen_low` 可以作为诊断字段保存，但 loss 不应把它当成“需要 cover 的目标”。

主指标：

```text
mean_accepted_pnl = sum(pnl over accepted validation rows) / accepted_count
```

硬约束：

```text
order_coverage = mean(bid > 0 over accepted rows) >= 0.70
```

注意：abstain 样本 PnL 记 0，但仍在 accepted 分母内；这能防止策略靠只下极少数样本刷高成交单均值。

## 3. 数据重建

原 R9 的问题是：`selected_side=resolved_outcome`，训练的是 winner token 未来低点。新方案必须使用 predicted-side 数据。

### 3.1 样本筛选

只保留方向模型 accepted 样本：

```text
accepted = (p_up >= t_up) or (p_up <= t_down)
```

这对应实际会考虑下单的 universe。此前 validation 上 accepted 约 `5228 / 7468 = 0.70005`，方向准确率约 `0.70735`。

### 3.2 预测侧和正确性

构造：

```text
predicted_side = UP/DOWN from threshold policy
predicted_outcome = up/down
p_side = predicted side probability
correct = predicted_side == final_winner_side
```

`final_winner_side` 只用于训练 label 和离线评估；推理时不可用。

### 3.3 predicted-side low

对每个 accepted 样本，用 `(condition_id, predicted_outcome, decision_time)` join trades，取订单窗口内预测侧 token 的最低成交价：

```text
decision_time < trade_time <= order_window_end
chosen_low = min(price of predicted_outcome token)
```

订单窗口建议先沿用 expected-return 分支的 `until_settlement` 口径，因为 wrong direction 在该口径下 `bid>0` 保守视为强制成交。也可以另开实验评估 `next4m` 窗口，但主方案先和 PnL 回放口径一致。

### 3.4 数据字段

训练 frame 至少包含：

| 字段 | 用途 |
|---|---|
| `timestamp`, `decision_time`, `condition_id`, `polymarket_slug` | 对齐、诊断、时间切分 |
| 原方向模型特征 | 模型输入 |
| `p_up`, `p_side`, `selected_side`, `direction_confidence`, `p_bin` | 模型输入和约束 |
| `correct` | loss 分支与评估 label |
| `chosen_low` | correct 样本 soft fill 的 label |
| `target` / resolved outcome | 只用于构造 `correct`，禁止作为模型输入 |

禁止作为特征：

```text
correct, chosen_low, final outcome, target, realized_pnl, future trade fields
```

## 4. 模型形式

模型仍然是单输出 MLP，可以复用 R9 的特征预处理和网络结构。

推荐输出参数化：

```text
u = sigmoid(MLP(X))
bid_raw = p_side * u
```

这样天然满足：

```text
0 <= bid_raw <= p_side
```

可选增强：在输入里加入 R9 的输出作为辅助特征：

```text
r9_f_model
r9_p_pred
p_side - r9_p_pred
r9_action / r9_conf_ok
```

但第一版可以先不依赖 R9 输出，避免链路复杂。第二版再比较：

```text
S0: soft-PnL policy from scratch
S1: R9-initialized / R9-feature policy
```

## 5. Soft-PnL loss

真实 correct fill 是硬阶跃：

```text
filled_correct = chosen_low <= bid
```

硬阶跃不可导，所以用 sigmoid 平滑：

```text
soft_fill = sigmoid((bid_raw - chosen_low) / tau)
```

单样本 soft PnL：

```text
if correct:
  soft_pnl = soft_fill * (1 - bid_raw)
else:
  soft_pnl = -lambda_wrong * bid_raw
```

合并写法：

$$
\text{soft\_pnl}_i = c_i \cdot \sigma\left(\frac{b_i - \ell_i}{\tau}\right)(1-b_i)
 - \lambda_\text{wrong}(1-c_i)b_i
$$

其中：

```text
c_i = 1 if correct else 0
b_i = bid_raw
ell_i = chosen_low
tau = soft fill temperature
lambda_wrong = wrong loss weight
```

训练 loss：

$$
L = -\frac{1}{N}\sum_i \text{soft\_pnl}_i
$$

### 5.1 为什么这个 loss 对应真实机制

correct 样本：

```text
bid 远低于 chosen_low: soft_fill≈0, pnl≈0，模型有动力提高 bid 接近 low。
bid 接近 chosen_low: soft_fill 快速上升，是主要学习区域。
bid 高于 chosen_low 很多: soft_fill≈1, pnl≈1-bid，继续抬 bid 只会降低收益。
```

因此 correct 样本的最优输出会靠近“刚好能成交”的低点附近，而不是无脑报高价。

wrong 样本：

```text
bid>0 就亏 bid
```

所以 loss 直接惩罚 `bid_raw`，让模型在看起来像 wrong 的区域压低 bid。

### 5.2 参数影响

| 参数 | 作用 | 建议网格 | 影响 |
|---|---|---|---|
| `tau` | 平滑 correct fill 阶跃 | `[0.005, 0.01, 0.02, 0.05]` | 小：更接近真实阶跃但梯度集中；大：更平滑但可能鼓励过宽区域抬 bid |
| `lambda_wrong` | wrong 样本亏损权重 | `[0.5, 1.0, 1.5, 2.0]` | 大：更保守，wrong bid 降低，但 correct fill 可能下降；小：更激进，correct fill 高但 wrong loss 增大 |
| `bid_floor_train` | 训练时最小正 bid，可选 | 默认不加 | 不加更利于模型学会接近 0；若业务必须挂单再加 |
| `bid_threshold` | calibration 中把低 bid 置 0 的阈值 | calibration 选 | 直接控制 order coverage 和质量 |

### 5.3 数值例子

设 `tau=0.02`。

Correct 样本：

```text
chosen_low=0.50
bid=0.45 -> soft_fill=sigmoid(-2.5)=0.076, soft_pnl≈0.076*(0.55)=0.042
bid=0.50 -> soft_fill=sigmoid(0)=0.500, soft_pnl=0.500*(0.50)=0.250
bid=0.55 -> soft_fill=sigmoid(2.5)=0.924, soft_pnl≈0.924*(0.45)=0.416
bid=0.70 -> soft_fill≈1.000, soft_pnl≈0.300
```

这里最优不会无限抬高，通常在 low 上方一点点。因为 `bid` 越高，`1-bid` 越低。

Wrong 样本，`lambda_wrong=1.5`：

```text
bid=0.10 -> soft_pnl=-0.15
bid=0.30 -> soft_pnl=-0.45
bid=0.50 -> soft_pnl=-0.75
```

wrong 样本会强烈推动模型压低 bid。

## 6. 允许不下单但 coverage >= 0.70

训练输出 `bid_raw` 后，最终下单还要经过 calibration gate：

```text
if bid_raw >= theta:
    bid = tick_round_or_floor(bid_raw)
else:
    bid = 0
```

其中 `theta` 只在 calibration 上选择，validation 不参与调参。

### 6.1 order coverage 定义

```text
order_coverage = mean(bid > 0 over accepted rows)
```

硬约束：

```text
order_coverage >= 0.70
```

因为 accepted universe 本身已经是方向模型阈值过滤后的样本，所以这里的 0.70 是 accepted 内部下单覆盖率，不是全市场样本覆盖率。

如果希望全市场最终下单覆盖率也接近 0.70，需要另行定义：

```text
overall_order_coverage = direction_acceptance_rate * order_coverage
```

当前本文按用户要求使用 accepted 内 `order_coverage >= 0.70`。

### 6.2 theta 怎么选

在 calibration 上枚举 theta 分位或绝对阈值：

```text
theta candidates:
  absolute: [0.00, 0.01, 0.02, 0.03, 0.05]
  quantile-derived: choose thresholds that produce order_coverage in [0.70, 0.75, 0.80, 0.90, 1.00]
```

对每个 theta，用真实 hard PnL 回放打分：

```text
correct=True:
  pnl = 1 - bid if chosen_low <= bid else 0
correct=False:
  pnl = -bid if bid > 0 else 0
```

选择规则：

```text
valid if order_coverage >= 0.70
primary: maximize calibration mean_accepted_pnl
tie 1: higher order_coverage, if PnL within one-SE / tolerance
tie 2: lower wrong_mean_bid
tie 3: lower mean_bid
```

为什么 tie 偏向更高 coverage：用户明确要求覆盖不能低于 0.7，且 calibration PnL 很薄时，过度贴着 0.70 的门槛容易在 validation 掉到 0.7 以下。

### 6.3 calibration buffer

建议 calibration 约束比 validation 目标略高：

```text
calibration_order_coverage >= 0.72 或 0.73
validation target >= 0.70
```

这和 R9 的 `calibration_coverage_buffer=0.03` 思路一致，用来防止时间漂移导致 validation 覆盖跌破线。

## 7. Tick 和下单价格后处理

建议后处理：

```text
bid_raw = clip(bid_raw, 0, p_side)
if bid_raw < theta:
    bid = 0
else:
    bid = floor_to_tick(bid_raw, tick=0.01)
    bid = max(bid, 0.01)
    bid = min(bid, floor_to_tick(p_side))
```

为什么倾向 `floor_to_tick` 而不是 ceil：

- R9 gap 目标中用 ceil 是为了确保 cover low。
- PnL 目标中 bid 高一 tick 会直接增加 wrong loss。
- correct fill 的少量 miss 可以由 soft-PnL 和 calibration 学到；不应无条件上取整。

需要在报告里同时输出 ceil/floor 对照，确认一 tick 差异对 fill 和 wrong loss 的影响。

## 8. 训练/校准/验证切分

建议仍用时间切分：

```text
fit:         训练 MLP 参数
calibration: 选择 tau/lambda_wrong/theta/early stop
validation: 只评估最终策略一次
```

不要在 validation 上选 `theta` 或 `lambda_wrong`。此前 O0/O1/O2 的 `min_ev=0.02` 是 validation 上选出来的，结果只有 `+0.00231`，属于乐观值；新方案要避免这个问题。

建议第一版 split：

```text
fit: train accepted 中除最后 7~14 天外的数据
calibration: train accepted 最后 7~14 天
validation: 原 validation accepted
```

如果 calibration 太薄，优先扩大到 14 天；因为 order coverage 和 PnL 都是薄边际，7 天可能太抖。

## 9. 指标和报告

主报告必须包含：

| 指标 | 说明 |
|---|---|
| `mean_accepted_pnl` | 主指标，accepted 全样本均值，abstain 计 0 |
| `sum_pnl` | 总 PnL |
| `order_coverage` | `bid>0` 占 accepted 的比例，必须 >= 0.70 |
| `mean_bid` | 所有 submitted orders 的平均 bid |
| `correct_fill_rate` | correct accepted 样本中最终成交比例 |
| `wrong_order_rate` | wrong accepted 样本中下单比例 |
| `wrong_mean_bid` | wrong submitted orders 的平均 bid |
| `win_pnl_sum` | correct filled 的总收益 |
| `loss_pnl_sum` | wrong submitted 的总亏损 |
| `filled_accuracy` | filled orders 中 correct 比例 |
| `pnl_by_p_side_bucket` | 分桶看收益来源 |
| `coverage_by_p_side_bucket` | 分桶看是否靠某些桶 abstain |

尤其要和 R9 / fixed baselines 对比：

| baseline | 作用 |
|---|---|
| R9 safe-gap bid | 当前 gap 最强但 PnL 为负的锚点 |
| fixed `0.5 * p_side` | 简单低价 baseline |
| fixed `0.75 * p_side` | 更高成交 baseline |
| pay `p_side` | 全价兜底下界 |
| bid=0 for all | PnL=0、coverage=0，不满足约束但作为 sanity check |

## 10. 实验矩阵

第一批实验不要太多，优先验证这个方向是否能在 `order_coverage >= 0.70` 下打过 R9 和 fixed baseline。

| 实验 | 变更 | 关键问题 |
|---|---|---|
| S0 | predicted-side soft-PnL MLP from scratch | 单输出 PnL policy 是否有效？ |
| S1 | S0 + `tau` 网格 `[0.005,0.01,0.02,0.05]` | correct fill 阶跃平滑度怎么选？ |
| S2 | S1 + `lambda_wrong` 网格 `[0.5,1,1.5,2]` | wrong loss 权重能否提高净 PnL？ |
| S3 | S2 + R9 features | R9 winner-low 信号是否还能增益？ |
| S4 | S2 + R9 initialized MLP | 从 R9 权重 fine-tune 是否更稳？ |
| S5 | S2 + calibration coverage buffer `{0.70,0.72,0.75}` | 防止 validation coverage 掉线 |

验收门槛：

```text
validation order_coverage >= 0.70
validation mean_accepted_pnl > R9 safe-gap PnL
validation mean_accepted_pnl > fixed_0.5*p_side baseline
wrong_mean_bid 显著低于 R9
correct_fill_rate 不出现塌方
```

R9 PnL 锚点来自 safe-gap 回放：

```text
R9 total_pnl_per_one_share = -15.98
R9 mean_pnl_per_order      = -0.00306
R9 order_count             = 5228
```

这里 R9 的 order_count 是全 accepted 都有 `p_pred>0`，而新方案允许 bid=0，但要求 order_coverage 至少 0.70。

## 11. 风险和缓解

### 11.1 Soft loss 和 hard PnL 有偏差

`soft_fill` 是平滑近似，可能优化出 calibration 上 hard PnL 不佳的 bid。缓解：

- 每个 epoch 都在 calibration 上用 hard PnL 评估。
- 早停和最终选择只看 hard PnL + order_coverage，不看 soft loss。

### 11.2 Wrong 样本只有约 30%，但亏损决定成败

方向准确率约 70%，wrong 样本少但亏损硬。缓解：

- `lambda_wrong` 网格必须包含 `>1`。
- 报告 `wrong_order_rate` 和 `wrong_mean_bid`。
- 按 `p_side_bucket` 看 wrong loss 是否集中在高价桶。

### 11.3 Coverage 约束会迫使边界样本下单

`order_coverage >= 0.70` 意味着不能只挑最稳的 20% 样本。缓解：

- calibration 用 `0.72~0.73` buffer。
- theta tie-break 偏向更高 coverage。
- 按 bucket 检查是否某些桶被系统性放弃。

### 11.4 模型可能把 bid 普遍压太低

如果 `lambda_wrong` 太高或 `tau` 太小，correct fill 会塌。缓解：

- 报告 `correct_fill_rate`。
- 加 correct fill 下限作为 soft/hard 约束，例如 calibration `correct_fill_rate >= baseline * 0.8`。
- 或在 loss 中给 correct 项权重 `lambda_correct`，但第一版先不用。

## 12. 与 R9 的关系

原 R9 学的是：

```text
winner-side future low -> safe bid
```

新方案学的是：

```text
predicted-side features -> bid that maximizes forced-wrong-fill PnL
```

它不是否定 R9，而是把 R9 的单输出思想迁移到真实部署分布：

- correct 样本上，它仍会学“接住 winner 回撤”。
- wrong 样本上，它会学“少亏或不下单”。
- calibration 层明确保证 `order_coverage >= 0.70`。

因此这条路线比 EV 组件分解更少依赖 `q/Gc/Gw` 的显式估计，也比 action expansion 更轻。

## 13. 最终建议

下一步实现 `predicted_side_soft_pnl_bid_policy` 作为 R9 后续分支：

1. 先重建 predicted-side accepted train/validation 数据。
2. 训练单输出 MLP：`bid_raw = p_side * sigmoid(MLP(X))`。
3. 用 soft-PnL loss 训练，用 hard PnL 在 calibration 上早停/选参。
4. 在 calibration 上选择 `theta`，要求 `order_coverage >= 0.72~0.73`。
5. validation 只报告一次，硬约束 `order_coverage >= 0.70`。

如果 S0/S1 在 `order_coverage >= 0.70` 下无法超过 fixed baseline，就说明“只靠出价模型”确实很难突破，需要回到方向模型/accepted universe 选择；如果能超过，再把 R9 features 或 R9 initialization 加进来做第二轮。

---

## 14. Optimization（问题分析与修订建议）

本节是对上面 GPT 方案的逐项 review。结论分三档优先级：**P0 必须修正**（不改会得到误导性结论或直接跑偏）、**P1 应当修正**（影响能否真正突破历史瓶颈）、**P2 实现细节**。最后给出修订后的实验矩阵与验收门槛。

### 14.0 总体判断

方案本身**自洽、写得清楚**，single-output soft-PnL policy 是 action-expansion 与 q/Gc/Gw 显式分解之外一个**更轻、合理的第三条路**。把 q 隐式折进出价策略，还顺带绕开了历史上失败的脆弱 q 重标定（H4 isotonic 把 cal `+55.74` 翻成 val `-16.08`）。

但它有一个根本性问题：**它只是把交易目标重参数化，没有引入任何新的信号**。winner-low 分辨率（`sd(f-y)=0.157`）、q 分辨率（AUC `0.79`）、以及最关键的**结构性 zero-EV 墙**都原封不动。历史上 O 系列 EV、H 系列 hazard 全部收敛在 `mean_accepted_pnl ≈ +0.002 ~ +0.003` 这条极薄的线上（见 [20260624_safe_lowest_price_gap_results_analysis.md](20260624_safe_lowest_price_gap_results_analysis.md) 第 5 节）。soft-PnL 大概率也会落在同一条线上。因此本节的核心不是否定方案，而是**校正它的预期、基线和验收口径**，并加上几个能防止“假胜利”的护栏。

### 14.1 P0 必须修正

#### P0-1 基线和验收门槛选错了（最严重）

§10 的验收写的是“打过 R9 safe-gap PnL 和 fixed `0.5*p_side`”。但这两个都是**已知很弱的锚点**：R9 PnL 是负的（`-15.98` / mean `-0.00306`），fixed baselines 也全负。真正的非泄漏最强者是 **H2 hazard/survival**：

```text
H2 validation sum_pnl   = +17.75
H2 validation mean/样本  = +0.00340
win/loss 计数            ≈ 1.04（几乎打平，全靠 70/30 方向占比撑住）
```

H2 比本方案早（2026-06-19），方案却完全没提它。**只打赢 R9/fixed 没有意义**——它们本来就被 H2 甩开。请把 §10 验收改成：

```text
主门槛: validation mean_accepted_pnl > H2 hazard (+0.00340) 且 sum_pnl > +17.75
        且 order_coverage >= 0.70（非泄漏，calibration 选参）
辅助门槛: win_pnl_sum / |loss_pnl_sum| 必须显著 > 1.04（不能靠计数占比硬撑）
```

参考：H2 数字见 [20260624_safe_lowest_price_gap_results_analysis.md](20260624_safe_lowest_price_gap_results_analysis.md) 第 5 节与 `expected_return/experiments/20260619_expected_return_h1_hazard_survival/`。

#### P0-2 没有触及 zero-EV 结构墙，需要明确写进“预期上限”

binary token limit-buy 机制下存在一堵已被反复确认的墙：

```text
赢家要成交 -> bid 必须 ~0.6 -> 利润仅 1-0.6 = 0.4
输家只要 bid>0 -> 强制成交 -> 亏整个 bid（~0.5）
=> 同一个抬高的 bid 既提高赢家成交、也放大输家亏损
=> 仅靠 70/30 的方向计数差，净值勉强为正
```

H1 实证：1060 笔赢家成交均值 `+0.394`（合 `+418`），746 笔输家成交均值 `-0.541`（合 `-403`），净 `+14.75`，**输家只要再多亏 3.5% 就归零**。

soft-PnL 的 correct 项（接住回撤）受限于 `sd(f-y)=0.157`，wrong 项（压低/不下单）受限于 q(X) 分辨率，两者方案都没加新信号。**因此必须在文档开头（§0 或 §10）明确写：本方案预期上限就是逼近 H2 的 `~+0.003` 量级，不要期待数量级改善；若做不到，结论就是“bid 侧已耗尽”。** 否则容易把一个噪声级的 `+0.0005` 波动误读成胜利。

#### P0-3 soft surrogate 的逐样本最优系统性**偏高约 3τ**，方向正好踩反

这是 loss 设计里一个隐蔽但要命的偏置。对 correct 样本，`g(b)=σ((b-ℓ)/τ)·(1-b)`，令 `g'(b)=0`：

$$
(1-\sigma)\,(1-b^\*) = \tau,\qquad \sigma=\sigma\!\left(\frac{b^\*-\ell}{\tau}\right)
$$

硬机制下 correct 的最优可行 bid 是 `b=ℓ`（恰好成交、利润 `1-ℓ` 最大）。但 soft 最优 `b*` 满足上式，对小 τ 会逼 `σ→1`，即 `b* ≫ ℓ`。数值上 τ=0.02、ℓ=0.50：

```text
hard 最优:  b = 0.50  -> pnl = 0.50
soft 最优:  b*≈ 0.56  -> 比 ℓ 高 ~0.06 ≈ 3τ
```

§5.3 自己的数表也印证了这点：soft_pnl 从 bid 0.50 的 `0.25` 一路升到 bid 0.55 的 `0.416`——**surrogate 在主动鼓励抬价**。而 [results_analysis 第 6.3 节](20260624_safe_lowest_price_gap_results_analysis.md) 已证明 **realized PnL 随 bid 单调变差**。也就是说 soft 平滑引入的偏置方向，恰好是历史上最赔钱的方向。τ 网格上端 `0.05` 对应过冲 `~0.15`，在这种薄 margin 机制下是灾难性的。

建议：

```text
1. τ 网格去掉 0.05，主用 [0.005, 0.01, 0.02]，并默认偏小。
2. 在 soft_fill 里显式减一个 margin：σ((b - ℓ - m)/τ)，m≈2~3τ，抵消过冲。
   或对 correct 项目标改成 σ(...)·(1-b) - β·b 的小负偏置，让最优回落到 ℓ 附近。
3. 训练全程严格用 hard PnL 早停/选 epoch（§11.1 已提，但要强调它是抵消 3τ 偏置的主要手段）。
4. 报告里加 mean(bid_raw_train_optimum) vs mean(ℓ|correct) 的差，监控过冲。
```

#### P0-4 数据重建漏了 trades_coverage 过滤，且订单窗口口径没锁死

§3.3 重建 predicted-side `chosen_low` 时没提 `trades_coverage_start` 过滤。历史上这正是 train 巨量 missing-low 的根因：未加该过滤时 train missing_low 高达 `8595`，加上 `trades_coverage_start=2026-02-12` 后降到 `42`（见 repo 记录与上游 `build_price_target.py`）。现有 R9 train 也是 `24249 -> 缺 8592 -> 15657`，窗口从 `2026-02-12` 起。**predicted-side 重建必须复用同一过滤**，否则 correct 样本的 `chosen_low` 会在 2 月前严重缺失、分布有偏，soft loss 直接学偏。

另外两点必须锁死并写清：

1. **订单窗口口径**：§3.3 一会儿说沿用 `until_settlement`，但现有 winner-low label 是 `next4`（4 分钟）。两者对 correct_fill 定义不同（`until_settlement` 的低点 ≤ `next4` 的低点 → correct_fill 偏高）。**训练用的 `chosen_low` 窗口必须与 §10 PnL 回放窗口完全一致**，否则 train 与评估错位。建议主线先统一到 `until_settlement`（与 G_w≡1 一致），并在文档显式声明。
2. **predicted-side 的 wrong 样本 `chosen_low ≈ 0`**（loser token 价格趋于 0），所以 wrong 几乎必然 `chosen_low <= bid` → 强制成交，这与 §2 的 forced-wrong-fill 一致；但要在 §3.3 说明：wrong 的 `chosen_low` 不能进 correct 的 fill 监督，只能进 wrong 的 `-λ·b` 分支（方案已分支，补一句口径说明即可）。

### 14.2 P1 应当修正

#### P1-1 优化者诅咒：直接最大化 PnL 会在噪声 `chosen_low` 上过拟合

这是 hazard 已经踩过的坑（BLOCKER#2）：EV/PnL argmax 会**挑中那些恰好 `chosen_low` 偏低的噪声样本**，高估赢家成交。实测 hazard 在 EV 提交的 correct 子集上 `model_fill = 0.679` vs `realized = 0.509`，**+0.17 的乐观偏差**。soft-PnL 直接对 `chosen_low` 做策略优化，会有同样的 selection bias：模型学到的“能贴住低点”在 validation 上塌掉。

建议：

```text
1. cross-fit / 样本外评估 chosen_low：用 holdout 折估计每个样本的 fill，避免 in-sample 乐观。
2. 对 soft_fill 做收缩（temperature 调大一点、或对 bid_raw 加正则把它往保守拉）。
3. 报告 submitted-correct 子集的 model_fill vs realized_fill 差距，目标 < 0.05（与 hazard 同口径对标）。
```

#### P1-2 7~14 天 calibration tail 已被证明极脆（cal→val 14x 崩塌）

§8/§6.3 把 τ/λ_wrong/θ 全放在 7~14 天 calibration 上选。但 hazard 在同样的 7 天尾巴上出现 **cal `+0.040` → val `+0.0028`，14 倍 edge 蒸发**，frontier 非单调（`+14.75/+18.73/-1.84/+1.82/+3.76`）= 纯噪声。本方案要同时在这条薄尾巴上选 3 个超参 + θ 操作点，过拟合风险更高。

建议：

```text
1. 不要单窗：用 rolling / nested CV，多个 calibration 窗口取稳健点（中位数或最差窗口）。
2. 必报 cal->val 退化幅度；若某操作点 cal 很高但相邻窗口抖动大，判为噪声、不选。
3. θ 不要贴着 0.70 选（§6.3 buffer 思路对，但要扩展到 τ/λ：选 cal 上 PnL 在邻域稳定的点，而非单点最大）。
4. λ_wrong 和 τ 优先用更宽的 fit 内部时间折交叉验证，θ 才放 calibration。
```

#### P1-3 压低 wrong bid 的能力 = q(X) 分辨率，方案没加新 q 信号

wrong 项 `-λ·b` 要起作用，前提是 X 能把 wrong 样本和 correct 样本分开——这本质就是 `q(X)=P(correct|X)` 的分辨率。历史 q（=p_side 经标定）AUC 只有 `0.79`，且在高 p_side 仍混着 29% 输家。soft-PnL 没引入新的 q 信号，所以它压 wrong bid 的上限被钉死。

建议：把 q 相关信号显式喂进去做对照（注意 H4 朴素 isotonic q 重标定**反向**了，不要直接照搬）：

```text
S-q0: 仅原方向特征（方案现状）
S-q1: + 校准后的 q（用数据充足的折，不用 7 天尾巴）作为输入特征
对比两者在 wrong_order_rate / wrong_mean_bid / 净 PnL 上的差异，验证 q 信号是否还有空间。
```

#### P1-4 correct 分支本质≈一个 winner-low 分位头，预期增益只来自 wrong 融合

把 correct 分支拆开看：训练时 `ℓ` 是 label，模型只见 X，要让 `bid=f(X)` 贴着各样本实现的低点——这**数学上几乎就是对 `chosen_low` 做 pinball/分位回归**，团队已经探索过，天花板是 `sd(f-y)=0.157`。所以 soft-PnL **不会**在“接住回撤”上赢过 R9/hazard 的 winner-low 精度。

含义（应写进文档预期）：**本方案相对 hazard 的潜在增益，只可能来自“correct+wrong 联合训练 + 隐式 q 折叠”，而不是更好的低点分辨率**。如果联合训练带不来增益，方案就退化成“一个更难调的 winner-low 分位头”，没有理由替换 hazard。

#### P1-5 train/eval 失配：训练用连续 `bid_raw`，部署经 θ gate + floor_to_tick

模型在训练时从没见过 θ 门控和 `floor_to_tick`，但部署时一个想报 `0.015` 的样本会被 floor 到 `0.01` 或被 θ 清零。calibration 用 hard PnL 选 θ 部分缓解，但**模型本身没有对齐到最终离散动作**。

建议：

```text
1. 训练 loss 里就纳入一个可微的 soft gate（如 σ((bid_raw-θ)/τθ) 作为下单概率），或用 straight-through 估计离散 floor，让模型感知门控。
2. 至少在 calibration 报告 “连续 bid_raw 的 PnL” vs “θ+floor 后的 PnL” 差距，量化失配成本。
```

### 14.3 P2 实现细节

| 项 | 问题 | 建议 |
|---|---|---|
| `λ_wrong` 语义 | 训练用 `λ_wrong`（可>1）与评估 hard PnL 的真实 `λ=1` 不一致 | 文档写清：这是**风险厌恶正则**，不是无偏 PnL 估计；同时报告 `λ=1` 口径的 hard PnL，避免把“调保守”误当“更赚钱” |
| 输出参数化 | `bid_raw = p_side·sigmoid(MLP)` 把上界钉在 `p_side`，但 EV 最优常在**很低**的 bid；sigmoid 在小 `u` 区梯度小，模型**难学到极低 bid** | 改用对 logit 加负偏置 / softplus 变体，或显式给“0 下单”一个独立 head，避免 coverage 因参数化被人为撑高、bid 黏在中段 |
| coverage>=0.70 的代价 | `max_possible_coverage` 才 `0.717`，且强制 `bid>0` 于 70% 必然踩中一批 wrong → 钉死亏损下界（任一 `bid=0.01` 的 wrong 强制成交亏 `0.01`）。覆盖率是历史**第一大杠杆**（90%→70% 可省 ~0.19 gap） | 报告**完整 PnL–coverage frontier**（放开 0.70 看曲线），把 0.70 这条硬约束的“成本”量化出来给决策者；若 0.70 处 PnL 显著为负而 0.55 处为正，应回去和业务谈这条约束 |
| 全局 τ | 单个 τ 对所有样本，但 room/p_side 异质（中段 room 0.31、两端 0.07~0.13） | τ 按 room 或 p_side_bucket 缩放，或对 `bid_raw` 在归一化 room 空间里算 soft_fill |
| `chosen_low` 缺失处理 | predicted-side 重建后仍可能有 no-trade 行 | 明确：correct 缺 low 行如何处理（右删失 or 丢弃），并报告 train/val 的 missing 率对齐（避免 train 34.7% vs val 0.5% 这类历史偏差重演） |

### 14.4 修订后的实验矩阵（替换/补充 §10）

把 H2 设为主基线，并把上面的护栏做成消融维度：

| 实验 | 变更 | 验证的问题 |
|---|---|---|
| S0 | predicted-side soft-PnL MLP from scratch，**严格 hard-PnL 早停** | 能否逼近 H2（不是只打 R9）？ |
| S1 | S0 + τ ∈ `[0.005,0.01,0.02]` + **margin 抵消 3τ 过冲** | 过冲偏置修掉后是否更稳/更高？ |
| S2 | S1 + `λ_wrong ∈ [1.0,1.5,2.0]`，并报 `λ=1` hard PnL | wrong 加权能否真提净 PnL（非仅调保守）？ |
| S3 | S2 + **cross-fit / OOF chosen_low**，报 submitted-correct `model_fill - realized_fill` | 优化者诅咒是否被压到 <0.05？ |
| S4 | S2 + **多窗 rolling calibration**，报 cal→val 退化 | 操作点在薄尾巴上是否稳健？ |
| S5 | S2 + q 校准特征（数据充足折，非 7 天尾） | q 融合还有没有空间（P1-3）？ |
| S6 | S2 + 完整 PnL–coverage frontier（放开 0.70） | 量化 0.70 硬约束的成本 |

### 14.5 修订后的验收门槛

```text
硬约束:
  validation order_coverage >= 0.70（calibration 选参，validation 只报一次）

主门槛（必须同时满足）:
  validation mean_accepted_pnl > +0.00340  (H2)
  validation sum_pnl          > +17.75     (H2)
  submitted-correct: |model_fill - realized_fill| < 0.05
  win_pnl_sum / |loss_pnl_sum| 显著 > 1.04

稳健性门槛:
  cal->val 的 mean_accepted_pnl 退化不得是“正翻负”或数量级蒸发
  bid_raw 训练最优相对 ℓ 的过冲受控（监控 mean(bid)-mean(ℓ|correct)）

若 S0~S2 在以上口径下打不过 H2:
  结论 = bid 侧已耗尽（与 results_analysis 第 6.3 节一致）
  下一步 = 回到 selection（mean_accepted_pnl vs coverage frontier，保持低 bid）
           或 upstream 方向/acceptance 阈值，而不是继续调出价模型。
```

### 14.6 一句话总结

方案是一条**合理但增益有限**的旁路：它优雅地把 q 折进策略、避开脆弱 q 重标定，但**没有新增 winner-low 或 q 分辨率**，且 soft surrogate 自带 `~3τ` 的抬价偏置，方向与历史 PnL 规律相反。**先做最小 S0 严格对标 H2**；过了，再叠 R9 features / 初始化；过不了，就坐实“出价侧已到顶”，把精力转向 selection 与方向模型。