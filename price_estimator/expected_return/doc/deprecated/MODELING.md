# Expected Return 建模方案：从「最低成交价」到「期望收益」

> 日期：2026-06-18
> 目录：`price_estimator/expected_return/`（所有新实验、脚本、数据都在此）
> 取代目标：旧的 `safe_lowest_price_gap` 以 covered gap 为目标的建模
> 关联：[safe_lowest_price_gap_optimization_directions_v3.md](../docs/safe_lowest_price_gap_optimization_directions_v3.md)

---

## 0. 一句话结论

```text
旧目标（最小化 covered gap）只对“预测正确”的样本是对的。
真实世界有不可消除的不对称：
  - 预测正确：挂低于市价的限价买单 → 不一定成交（价格未必跌到你的出价）；成交则赚 (1 - b)。
  - 预测错误：持有的 token 必然归 0 → 价格一路下穿你的出价 → 必然成交；成交即亏 b。
所以正确目标是**期望收益 E[PnL]**，不是 gap。
出价 b 越低：对赢家“少成交但更便宜”，对输家“少亏”。两端都偏好更低的 b，
但太低会错过赢家的成交 → 存在一个由“正确概率 q(X)”驱动的最优 b*(X)。
“放弃下单”不再是单独机制——当 q(X) 低到 E[PnL]≤0 时，最优 b* 自动趋于 0 / 不挂单。

本版简化（按决定）：输家必成交 G_w≡1；不考虑手续费 fee=0；
q(X) 直接用现有方向分类模型的 p_side，不再单独训练/校准。
→ 唯一要训练的新组件只剩“赢家低点 CDF”G_c。
```

---

## 1. 交易机制（你的澄清，建模的地基）

在 `decision_time`（开盘 + 60s）对**模型下注那一侧**的 token 挂一个**低于市价的限价买单**，出价 `b`，在 4 分钟窗口内有效。结果二选一：

| 情形 | 持有 token 走向 | 是否成交 | 成交后盈亏 |
|---|---|---|---|
| **预测正确**（下注侧获胜） | → 1 | **不一定**：仅当窗口内最低价 `ℓ ≤ b` | `+ (1 − b)` |
| **预测错误**（下注侧落败） | → 0 | **几乎必然**：价格下穿任何 `b` | `− b` |

**统一写法**：成交条件永远是 `ℓ ≤ b`（窗口最低价跌到出价）。
不对称**不在成交规则里，而在 `ℓ` 的分布与盈亏里**：

- 正确：`ℓ` 是赢家 token 的低点（较高，常 `> b` → 可能不成交），成交赚 `1 − b`；
- 错误：`ℓ` 是输家 token 的低点（趋近 0，`≤ b` 恒成立 → 必成交），成交亏 `b`。

> 这把“正确不一定成交 / 错误必成交”收敛成**同一个成交规则 `ℓ ≤ b`**，
> 差异完全由 `ℓ` 的条件分布承担。建模因此非常干净。

---

## 2. 目标函数：单笔订单的期望收益

记号（决策时刻可得 / 事后可得）：

```text
X        决策时刻特征（开盘+60s 快照）
s(X)     下注侧 = UP if p_up≥t_up ; DOWN if p_up≤t_down ; 否则不接受（不交易）
q(X)     = P(下注侧获胜 | X) = 现有方向模型的 p_side（不另训，直接用）
ℓ        下注侧 token 在订单存活窗口内的最低成交价（事后可观测）
b        限价买单出价（决策变量，tick 网格 0.01..p_side）
```

按本版决定：**G_w ≡ 1**（输家必成交）、**fee = 0**。成交条件 `ℓ ≤ b`。
单笔期望收益化简为：

$$
\mathbb{E}[\text{PnL}\mid b, X] \;=\;
\underbrace{q(X)\,G_c(b\mid X)\,(1-b)}_{\text{赢家：可能成交，赚 }1-b}
\;-\;
\underbrace{\big(1-q(X)\big)\,b}_{\text{输家：必成交，亏 }b}
$$

其中（只剩一个需要建模的量 G_c）

```text
q(X)     = p_side                直接取自现有方向分类模型（不另训）
G_c(b|X) = P(ℓ ≤ b | 正确, X)    赢家 token 低点的 CDF（成交概率），随 b 增大而增大
G_w(b|X) ≡ 1                     输家必成交（固定，不建模）
```

**最优出价**：对每个样本在 tick 网格上取

$$
b^*(X) = \arg\max_{0 \le b \le p_\text{side}} \Big[\, q\,G_c(b\mid X)\,(1-b) - (1-q)\,b \,\Big]
$$

---

## 3. 这个目标自动产生的两个行为（为什么它对）

对 `b` 求偏导（已代入 G_w≡1, fee=0），两项方向相反：

```text
赢家项 ∂/∂b:  q·[ G_c'(b)(1-b) − G_c(b) ]   有内点最优（提高成交率 vs 牺牲利润）
输家项 ∂/∂b:  −(1−q)                        恒为负常数，一味把 b 往下压
```

于是：

1. **高 q(X)（很可能对）** → 输家惩罚 (1−q) 小 → `b*` 抬高去“追成交”，退化为旧的「尽量低但要能成交」= **min-gap**。
2. **低 q(X)（可能错）** → 输家惩罚大 → `b*` 压低，甚至压到 `ℓ` 之下 → **不成交 = 自动放弃**。
3. **中间 q(X)** → 在“赚赢家 / 躲输家”之间**连续插值**。

> 由于 G_w≡1，输家项就是干净的线性惩罚 `−(1−q)·b`，b* 的解析直觉更清楚：
> 仅当存在 b 使 `q·G_c(b)·(1−b) > (1−q)·b` 才值得挂单，否则 b*→0（放弃）。

**关键收益**：旧设计里强行加的 abstain / coverage 约束 / 0.6·p_side 地板**全部消失**——
“该不该下单、出多低”都由 `E[PnL]` 内生决定。你说的“最好别下单但当时不知道”被 `q(X)` 显式接管。

---

## 4. 要建什么模型（本版只需训一个）

由于 q(X)=p_side 直接复用、G_w≡1 固定、fee=0，**唯一需要训练的是 G_c**。

### 4.1 q(X) = P(正确 | X)：直接用现有方向模型（不训）

```text
q(X) := p_side = (p_up if 下注 UP else 1 - p_up)。
按决定：不另训、不重校准，直接取部署方向模型输出。
注：p_side 是原始模型概率（阈值过滤后仍 ~33% 出错），其校准度直接影响 PnL；
   本版不动它，后续若要提升可单独加校准层（不在本范围）。
```

### 4.2 G_c(b|X)：赢家 token 低点的 CDF（成交概率）——本版唯一要训的模型

```text
这正是现有 R9 低价模型在做的事，只是口径改为“下注侧且正确”的样本，
并把点预测升级为分布/CDF。
做法（二选一）:
  (a) 复用 R9：点预测 f + 残差分布 → G_c(b)=Φ((b−f)/σ)；σ 用全局或分桶残差尺度。
  (b) 直接多分位/CDF 头：输出若干分位 → 线性插值成 CDF。
训练集: 仅 correct（下注侧获胜）样本，标签 = 下注侧 token 窗口低点 ℓ。
R9 配方延用: alpha 0.01 / s_floor 0.05 / room-weight / 正则。
```

### 4.3 G_w ≡ 1（输家必成交，不建模）

```text
按决定直接固定 G_w≡1（输家 token → 0，任何 b 都会被下穿）。
不需验证、不需训练。
```

**组装推理**：对每个 accepted 样本，用 §2 公式 `q·G_c(b)·(1−b) − (1−q)·b` 在 tick 网格上取 `b*`。
若 `max_b E[PnL] ≤ min_ev`（默认 0，可加风险偏好）→ **不挂单**。

> 备选 Design B（直接策略网络）：训练网络直接输出 `b*(X)`，用可微的 PnL 代理损失端到端优化。
> 更难校准/调试，列为后续；先用组件式打底、可解释、可复用。

---

## 5. 数据重建（根因修复：消除 look-ahead）

**当前 bug**：[build_price_target.py](../scripts/build_price_target.py#L165) 用了

```python
selected_side = "UP" if target==1 else "DOWN"   # ← 用了“最终赢家”（未来信息）
```

→ 数据集只含“持有赢家 token”的轨迹；实盘里 ~33% 持有的是输家 token，从未进入训练/评估。

**新构建脚本** `build_expected_return_target.py`，关键改动：

```text
1) 下注侧 = 部署阈值决定（与实盘一致）:
     chosen_side = UP   if p_up >= t_up
                   DOWN if p_up <= t_down
                   else  DROP            # 不接受 → 不进数据集
   t_up/t_down 取自 execution_engine/deploy/baseline/artifact_manifest.json
   （t_up=0.5792857, t_down=0.4314286）。
2) chosen_outcome = up/down（按 chosen_side）。
3) correct = (target==1 & chosen UP) | (target==0 & chosen DOWN)。
4) 取“下注侧 token”的窗口低点（不是赢家！）:
     trades.merge(on chosen_outcome)；窗口 [decision_time, decision_time+window]；
     chosen_low_next4 = min(price)；time_to_low。
     —— 对正确样本= 赢家低点；对错误样本= 输家低点(≈0)。
5) 列: 特征…, p_up, p_side(=下注侧概率), chosen_side, correct,
        chosen_low_next4, time_to_low, accepted=1。
6) 只保留 accepted 行（与部署口径一致）。
```

数据可行性：现有 `read_trades()` 保留 `outcome` 列、按 `outcome` merge，
**两侧 token 的成交都在**，所以把 merge 键从“赢家 outcome”换成“下注侧 outcome”即可，无需新数据源。

> 订单存活窗口 `window_minutes`：若订单挂到**结算**，则输家必成交 `G_w≡1`、赢家低点取整段；
> 若只挂 4min，则 `G_w` 需从数据估。两种都做成 config 开关，先按你的实盘真实挂单时长设。

---

## 6. 推理 / 决策流水线

```python
def decide_bid(X, p_up, p_side, Gc_model, cfg):
    if not (p_up >= cfg.t_up or p_up <= cfg.t_down):
        return None                      # 不接受 → 不交易
    q = p_side                           # §4.1：直接用现有模型，不另训
    # 在 tick 网格上最大化期望收益（G_w≡1, fee=0）
    best_b, best_ev = 0.0, 0.0
    for b in tick_grid(0.0, p_side, cfg.tick):
        Gc = Gc_model.cdf(b, X)          # P(ℓ≤b | 正确)
        ev = q*Gc*(1-b) - (1-q)*b
        if ev > best_ev:
            best_b, best_ev = b, ev
    if best_ev <= cfg.min_ev:            # 默认 0，可加风险阈值
        return None                      # 自动放弃
    return best_b
```

```text
- 一次前向(q) + 一次前向(Gc) + 廉价网格 → 满足低延迟。
- 输出每条: bid b*（或不挂单），及预期 E[PnL]、q、成交概率，便于审计。
```

---

## 7. 评估指标（成败口径换成 PnL）

```text
主指标（回测，按真实成交规则 ℓ ≤ b*）:
  realized_PnL_per_sample =
     if ℓ ≤ b*:  (correct ? (1 - b*) : (-b*))
     else:       0
  → 报 sum 与 mean（accepted 全样本，含赢家盈利 − 输家亏损）。fee=0、输家必成交。

诊断:
  - 成交率 overall / 分 correct / 分 wrong
  - 赢家且成交的平均入场价（= 旧 gap，降级为子 KPI）
  - 输家且成交的平均亏损、占比
  - 放弃率（b*=0 / 不挂单）
  - q(X) 的校准曲线（可靠性图）

基线对照:
  (a) 旧 min-gap 策略（bid = R9 的 p_pred）
  (b) 恒报 p_side（最贵兜底）
  (c) 固定比例 bid = k·p_side
  → 新策略应在 mean PnL 上显著优于全部基线。
```

---

## 8. 目录结构与文件

```text
price_estimator/expected_return/
├── MODELING.md                       # 本文档
├── config.yaml                       # 阈值/窗口/网格/模型超参
├── build_expected_return_target.py   # §5 数据重建（下注侧 + accepted-only）
├── train_low_cdf.py                  # §4.2 赢家低点 CDF（唯一要训的模型，复用 R9 骨架）
├── decide_and_backtest.py            # §6 决策（q=p_side, G_w≡1, fee=0）+ §7 PnL 回测/指标
├── data/                             # 复制过来的数据（已放 extracted_labels.csv）
├── models/
├── reports/
└── experiments/                      # 每次实验一个子目录（config+reports）
```

**数据复制**（在有数据的机器执行；本机仅有 extracted_labels.csv，已复制）：

```bash
# 在 Windows/Databricks 上把真实数据复制进来
cp price_estimator/data/price_estimator_train.parquet      price_estimator/expected_return/data/
cp price_estimator/data/price_estimator_valid.parquet      price_estimator/expected_return/data/
# trades 目录（两侧 outcome 都需在内）与 deploy 预测（p_up）按 config 路径引用即可
```

> 脚本可基于现有 `train_safe_lowest_price_gap.py` / `build_price_target.py` 改写——
> 本机无训练数据无法验证运行，建议在数据就位的机器上落地与重训。

---

## 9. 实验计划

| 实验 | 内容 | 关键问题 |
|---|---|---|
| **E0** | 重建 target（§5） | accepted 占比/correct 率与部署一致？输家低点是否趋 0（佐证 G_w≡1 合理）？ |
| **E2** | 训 G_c（§4.2，复用 R9 骨架，correct-only） | 赢家低点 CDF 是否准（覆盖/分位校准）？ |
| **E3** | 组装决策（q=p_side, G_w≡1, fee=0）+ PnL 回测 vs 三基线 | mean PnL 是否显著优于 min-gap / 恒报 p_side？ |
| **E4** | 风险阈值 min_ev、窗口时长敏感性 | 放弃率与 PnL 的折中 |
| **E5** | 端到端策略网络（Design B，可选） | 直接学 b*(X) 是否超过组件式？ |

> 注：原 E1（训 q-head）已删除 —— q 直接用 p_side；原「G_w 验证」也删 —— 按决定 G_w≡1。

---

## 10. 假设与风险（需在 E0/E1 验证）

```text
1) 本版按决定 G_w≡1（输家必成交）、fee=0。若后续发现输家在 4min 窗口未必跌透，
   再考虑放开为从数据估 G_w（E0 可顺带看一眼输家低点分布，但不阻塞主线）。
2) q(X)=p_side 直接复用现有模型（按决定）。其校准度直接决定输家亏损能否避免；
   本版不动，作为已知局限记录（高 p_side 仍 33% 出错）。
3) 数据重建必须无泄漏：chosen_side 只能用决策时刻可得的 p_up 与阈值，
   correct / chosen_low 是事后标签，禁作特征。
4) 训练样本量：accepted-only 会减少样本(~70%)，注意与 calibration tail、正则配合。
```

---

## 11. 与旧结论的衔接

```text
保留并复用:
  - R9 低价模型骨架 → 作为 G_c（赢家低点 CDF）。alpha 0.01 / s_floor 0.05 / 正则等配方延用。
  - 部署阈值过滤(accepted) → 现在是数据构建的**前置条件**，而非可选项。
作废:
  - covered gap 作为主目标（仅降级为“赢家成交紧度”子 KPI）。
  - abstain / coverage 硬约束 / 0.6·p_side 地板 → 被 E[PnL] 内生取代。
新增（本版简化）:
  - 按样本 b* 的 PnL 决策与回测。q(X)=p_side 直接复用（不训）；G_w≡1、fee=0。
  - 唯一新训模型 = G_c（赢家低点 CDF）。
```
