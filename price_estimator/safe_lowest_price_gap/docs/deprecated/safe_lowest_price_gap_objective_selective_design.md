# Safe Lowest Price：以 normalized gap 为目标的单模型选择性预测设计（v2）

本文从**已确定的业务语义**出发，重新设计建模方案，重点解决三件事：

1. **单模型、低延迟**（不训练多分位、不做 sigma/conformal 两阶段）；
2. **loss 直接对齐 normalized gap**（修正旧 log-cosh / 旧 soft-violation 的错位）；
3. **infeasible 样本（约占 30%）如何训练与推理**（旧设计「只训 feasible」在 infer 时无解）。

记号（与现有代码一致）：

```text
target_raw  = lowest_trade_price_next4               # 概率空间 0~1
buffer      = 0.01
y           = target_safe = target_raw + buffer      # 真正要“贴近但不低于”的下界（label，未来量）
p           = p_side                                 # 报价上界，已是两位小数 tick，inference 已知
s           = p - y                                  # “可行空间/房间”：>0 feasible，<=0 infeasible（label 派生）
covered     = (p_pred >= y) & (p_pred <= p)
gap_norm    = (p_pred - y) / s                        # 仅 covered & s>0
feasible    = (s > 0)
```

> 关键事实：`y`、`s` 都是**未来标签派生量**，inference 时不可见、也禁止作为特征。
> 它们只在**训练 loss 与离线评估**里出现。

---

## 0. 一句话结论

```text
模型：单个回归器，一次前向输出报价定位 m(X)。
      首选 CatBoost 自定义目标（与现有栈一致、推理快）；
      若要顺带学不确定性，用单个双头 MLP（location + scale）。
loss：非对称，但在 normalized 空间度量——
      欠预测(miss) 用线性强惩罚（控覆盖，并把 30% infeasible 样本的预测往上推）；
      过预测(covered gap) 用 (gap/s)^2 / Huber 惩罚（直接压 normalized gap，普通 quantile 没有这一项）。
训练：用全部样本（含 infeasible）；归一化只出现在“过预测”分支并对 s 设下限，避免 1/s 爆炸。
覆盖：loss 只学“形状”，覆盖率由校准集上的全局 margin δ 精确卡到 0.9（feasible 口径）。
推理：全样本同一流水线 m(X)+δ → 截到 p_side → 向上取 tick；
      低置信样本（廉价 bucket 查表 / 可选 σ 头）直接返回 p_side。
输出：每条带 action（active / abstain_low_conf / clamp_over_pside / infeasible[评估期]）。
```

---

## 1. 业务语义与优化问题（先把问题写对）

### 1.1 语义

对每条样本输出报价 `p_pred ∈ (0, p]`：

- **不能低于** `y`，否则 miss（区间外**只有**这一种坏情形）；
- covered 前提下**越接近 `y` 越好**（`gap_norm` 越小）；
- 若算出的报价超过 `p`，**截断到 `p`**（gap_norm=1，但仍 covered）；
- 若对该样本**没把握**，允许**主动放弃**直接报 `p`（安全但 gap=1）。

`p_pred = p` 永远是「最安全最贵」的兜底（feasible 时必 covered）。

### 1.2 优化问题（注意是**边际**覆盖约束）

$$
\min_{p_\text{pred}(\cdot)}\ \mathbb{E}\big[\text{gap\_norm}\mid \text{covered}\big]
\quad\text{s.t.}\quad \Pr(\text{covered})\ge 0.9,\ \ p_\text{pred}\le p.
$$

两个决定方案形态的点：

1. **目标是 normalized gap，不是原始价差。** 同样 1 分价差，在 `s=0.05` 是 20% gap，
   在 `s=0.4` 只有 2.5%。**loss 必须在 `/s` 归一化后度量**才与指标一致。
2. **约束是整体 coverage ≥ 0.9，不是逐条件。** 这给了「好样本压低、差样本兜底」的重分配自由。
   固定 alpha 的单分位对每条都做同一条件分位，**放弃了这个自由**，所以天然次优。

### 1.3 infeasible（≈30%）的定位（直接回答你的疑问）

```text
infeasible := y > p（未来最低价已逼近/超过 p_side）。这类样本：
- 理论上永远 miss（p_pred<=p<y），是不可避免损失；
- 因此 coverage_overall 的天花板 ≈ 1 - 30% = 70%，
  → 0.9 的约束只能落在 coverage_feasible（feasible 子集内的覆盖）。
- inference 时无法判定某条是否 infeasible（y 未知），
  但**不需要判定**：它们会被同一条流水线推到高位 → 截断为 p_side（见 §3.4、§7）。
```

> 结论先行：**不要把 infeasible 从训练里剔除**。它们的标签 `y` 是定义良好的（只是很高），
> 把它们喂进去能让模型学到「高价区」，并在推理时自然把这些样本推向 p_side。
> 旧设计「只训 feasible」恰恰制造了你指出的「infer 没见过」的问题——本设计纠正它。

---

## 2. 为什么三类旧方案都不够（逐条回应）

| 方案 | 失败机理 | 本设计的修正 |
|---|---|---|
| **普通 quantile** | pinball 斜率固定、只管分位、**不管 gap 大小**；且对每条做同一条件分位，放弃边际重分配；多分位还要多次/多输出预测、**infer 慢**。 | 过预测分支用 `(gap/s)^2`／Huber，**显式压 gap 量级**；单模型单输出，**infer 快**；abstain 拿回边际自由（§5）。 |
| **旧 Asymmetric Log-Cosh MLP** | 在 `z=log1p(r/scale)` 空间做非对称惩罚，与价格空间的 coverage/gap **不对应**；无覆盖保证，靠预测越过 p_side 刷覆盖（above_p_side_share 0.68，非法）。 | 残差直接在**价格/归一化空间**度量；硬截断 p_side；覆盖交给**校准 δ**，不靠 loss（§3、§5）。 |
| **旧 Mean-gap + Soft Violation** | 直接压 mean_gap、违约软惩罚太弱 → coverage 崩到 0.41~0.58。 | 同一思想保留，但 (a) gap 放到**归一化空间**，(b) 覆盖由**校准 δ 硬卡 0.9**、并在选点时作为**硬约束**，不让它崩（§5、§7）。 |
| **center+sigma+conformal** | 三段误差叠加、sigma 系统性低估、conformal 乘子爆掉 → gap 0.75 vs 0.38。 | **单模型**，不显式拆 μ/σ；σ（若用）只作 abstain 的**排序信号**，阈值在校准集定，不参与乘性 margin（§6）。 |

> 你说「log-cosh 与 soft-violation 还可考虑、需调整」——本设计正是它们的**修正合体**：
> 非对称（log-cosh 的欠/过两支）+ 软违约（欠预测支）+ **归一化（新增）** + **覆盖用校准卡死（新增）**。

---

## 3. 核心：单模型 + normalized 非对称损失

### 3.1 残差与归一化

```text
f = f(X)              # 模型原始输出（概率空间的一个“定位”）
r = f - y             # 残差：r<0 欠预测(会 miss)，r>=0 过预测(覆盖, 产生 gap)
s_eff = clip(s, s_floor, +inf)    # 只在过预测分支使用；s_floor 例如 0.02
```

### 3.2 损失公式（分支）

$$
L(r)=
\begin{cases}
\alpha\big(\sqrt{r^2+c^2}-c\big), & r<0 \quad(\text{miss：平滑-}|r|，近线性、稳健，控覆盖)\\[6pt]
H_\kappa\!\Big(\dfrac{r}{s_\text{eff}}\Big), & r\ge 0 \quad(\text{covered：归一化 gap 的 Huber 惩罚})
\end{cases}
$$

其中过预测分支用归一化 Huber（`t=r/s_eff\ge0`）：

$$
H_\kappa(t)=
\begin{cases}
\tfrac12 t^2, & t\le\kappa \quad(\text{小 gap：二次，越大罚越狠 → 贴紧 }y)\\
\kappa\,(t-\tfrac12\kappa), & t>\kappa \quad(\text{大 gap：转线性，抗离群、稳梯度})
\end{cases}
$$

直觉：

```text
- 欠预测：每多一条 miss，付固定斜率 α 的“上推力” → α 越大覆盖越高；对 infeasible 也成立（把 f 往上推）。
- 过预测：付 (gap/s)^2 → 房间 s 越小，过冲一点点也被重罚 → 模型对“窄房间”样本贴得更紧；
         这正是普通 quantile 缺的“在乎 gap 量级”。κ 之外转线性，避免个别样本主导。
```

### 3.3 梯度 / Hessian（可直接落到 CatBoost / 自动微分）

对 `f` 求导（`dr/df=1`，`dt/df=1/s_eff`）：

```text
r<0 :  dL/df = α · r / sqrt(r^2 + c^2)            # ∈(-α,0)
       d2L/df2 = α · c^2 / (r^2 + c^2)^{3/2}      # >0
r>=0:  dL/df = min(t, κ) / s_eff                  # t=r/s_eff
       d2L/df2 = (1/s_eff^2) if t<=κ else 0       # >=0
```

两支 Hessian 均非负，GBDT/Newton 友好。`c`（如 0.01）平滑欠预测尖点。

### 3.4 infeasible 样本如何被训练与推理（重点）

```text
训练：infeasible 的 y>p 很高，模型输出 f 一般落在可行价区 → r=f-y<0 → 走“欠预测支”。
     该支不含 1/s、不会爆，只是以斜率 α 把 f 往上推（学到“这类样本该报很高”）。
     → 所以可以、且应当把 infeasible 一起训练（解决“infer 没见过”的问题）。
推理：对所有样本一视同仁算 p_pred = clip(ceil(f+δ), 0, p)。
     infeasible 的 f+δ 偏高 → 多半 >= p → 截断为 p_side（action=clamp_over_pside）。
     它们仍会 miss（y>p 不可避免），但拿到的是“最安全的 p_side 报价”，符合业务。
     无需在 inference 判定 feasibility（也无法判定）。
```

### 3.5 数值稳健化（务必）

```python
s_floor = 0.02            # 过预测分支 1/s 的下限，防爆
c       = 0.01            # 欠预测平滑尺度
kappa   = 1.0             # 归一化 gap 的 Huber 拐点（=1 个“满房间”）
alpha   = tune in {2,4,8} # 欠/过不对称比 → 粗调覆盖（细调交给 δ）
# 训练用全部样本；对 r>=0 且 s<=0 的极少数情形用 s_eff=s_floor（强罚把 f 拉回）。
```

---

## 4. 模型选择（单模型、低延迟）

### 4.1 首选：单个 CatBoost 回归 + 自定义目标（栈一致、推理最快）

```python
class NormalizedAsymObjective:
    def __init__(self, s_eff, alpha=4.0, c=0.01, kappa=1.0):
        self.s_eff = s_eff; self.alpha=alpha; self.c=c; self.kappa=kappa
    def calc_ders_range(self, approxes, targets, weights):
        out = []
        for i, f in enumerate(approxes):
            r = f - targets[i]                    # targets 传 y=target_safe
            if r < 0.0:
                d  = self.alpha * r / (r*r + self.c*self.c) ** 0.5
                h  = self.alpha * self.c*self.c / (r*r + self.c*self.c) ** 1.5
            else:
                t  = r / self.s_eff[i]
                tc = t if t <= self.kappa else self.kappa
                d  = tc / self.s_eff[i]
                h  = (1.0 / (self.s_eff[i]**2)) if t <= self.kappa else 1e-6
            w = weights[i] if weights is not None else 1.0
            out.append((-w * d, -w * h))          # CatBoost 约定：返回 (-dL, -d2L)
        return out
```

```text
- 单模型、单标量输出 → 一次 predict，infer 与现有 quantile 同量级、远快于多分位/两阶段。
- s_eff 需按样本传入（训练期可知，因 y 已知）；推理期不需要 s。
- 训练标签传 y=target_safe；与现有 train_catboost_quantile.py 同栈，改 loss 即可。
```

### 4.2 备选：单个双头 MLP（location + scale，一次前向同时给报价与不确定性）

```python
# 修复你已有的 log-cosh MLP：两处关键改动
# (1) 残差归一化到 /s 再进 Huber；(2) 增加 scale 头供 abstain（只做排序信号）
def loss_fn(f, log_sigma, y, s, alpha=4.0, c=1e-2, kappa=1.0, s_floor=2e-2, lam=0.1):
    r = f - y
    s_eff = s.clamp_min(s_floor)
    t = (r / s_eff).clamp_min(0.0)
    over = torch.where(t <= kappa, 0.5*t*t, kappa*(t - 0.5*kappa))
    under = alpha * (torch.sqrt(r*r + c*c) - c)
    loc = torch.where(r < 0, under, over).mean()
    # scale 头：用 NLL 让 sigma 单调反映条件不确定性（仅用于 abstain 排序，不做乘性 margin）
    sigma = log_sigma.exp().clamp_min(1e-3)
    nll = (0.5*(r.detach()/sigma)**2 + log_sigma).mean()
    return loc + lam * nll
```

```text
- 仍是“单模型一次前向”，infer 快；顺带得到 σ(X) 供 abstain。
- σ 只当“排序信号”，阈值在校准集定（§6）——规避旧 sigma 方案“被迫乘性放大”的坑。
- 若不想要学习型不确定性，用 §6 的 bucket 查表即可，连第二个头都不需要。
```

### 4.3 明确不再做

```text
- 多分位（MultiQuantile / 多个 Quantile 模型）：infer 慢、且仍不直接管 gap 量级。
- center+sigma+conformal 两/三阶段：误差叠加、已被实验否决。
```

---

## 5. 覆盖率如何精确卡到 0.9（与 loss 解耦）

旧 soft-violation 的崩盘根因是「让 loss 自己保覆盖」。本设计**把覆盖从 loss 里拿出来**，
用一个**全局 margin δ** 在校准集上精确卡：

```text
1. 训练得到 f(X)（loss 只决定“形状/排序”）。
2. 在 calibration 的 feasible 子集上，取 δ = quantile_{0.10}(y - f)
   → 使 P(f + δ >= y | feasible) = 0.90（恰好 coverage_feasible=0.9）。
3. 推理：p_pred = clip(ceil((f+δ)*100 - 1e-9)/100, 0, p)。
```

```text
为何稳：覆盖是“f+δ 是否 >= y”的边际频率，由 δ 直接、单调控制，不受 loss 调参漂移影响。
为何不牺牲 gap：δ 是“最省”的统一上移；真正压低 gap 的是 §3 的归一化过预测惩罚 +
              §6 的选择性 abstain（把不得不的大覆盖成本转嫁到“本来就该兜底”的样本）。
```

---

## 6. 选择性决策与置信度（abstain，把边际自由用起来）

### 6.1 为什么 abstain 有用

```text
abstain 把一条样本直接报 p_side（gap=1，feasible 必 covered）。
它只买覆盖、不买 gap。最优用法：用一个低 δ 让“有把握的主体”报得紧（小 gap），
再把“没把握的尾巴”abstain 去补覆盖——只要尾巴本就 miss 或 gap 接近 1，
这样换来的 covered 主体更紧，E[gap_norm|covered] 反而更低。
```

### 6.2 置信度信号（二选一）

```text
A. 廉价 bucket 查表（首选，零额外模型、近零延迟）
   桶键：p_side_bucket × time_to_close 桶 ×（可选）近端波动桶(sl_rv_30s 或价格派生 vol)
   在 calibration 上对每桶统计：在选定 δ 下的 miss_rate 或 残差 (y-f) 的离散度。
   → miss_rate 高的桶（不可信）整桶 abstain。
B. σ(X) 头（§4.2，需双头 MLP）
   按 σ 的分位阈 abstain；σ 仅排序、阈值校准集定，不做乘性 margin。
```

### 6.3 联合选点

```text
旋钮：alpha(训练，小网格) + δ(覆盖) + abstain 阈值 a。
在 calibration 上联合选，使 coverage_feasible>=0.9 且 covered_gap_norm.mean 最小（§8）。
```

---

## 7. 推理流水线（含 action 分类）

```python
def infer_one(f, p_side, conf_ok, delta, tol=1e-9):
    # f: 模型输出; conf_ok: 该样本是否“可信”(由 §6 给出); delta: 校准好的全局 margin
    if not conf_ok:
        return p_side, "abstain_low_conf"          # 没把握 → 兜底
    raw = f + delta
    if raw >= p_side:
        return p_side, "clamp_over_pside"          # 越过上界 → 截断（含多数 infeasible）
    p_pred = math.ceil(raw * 100 - tol) / 100.0    # 向上取 tick
    return min(p_pred, p_side), "active"           # 有把握 → 贴近 y 的报价
```

```text
- infeasible 不在此处显式出现：它们多半 f+δ>=p → clamp_over_pside（拿到 p_side）。
- 一次模型前向 + 一次查表 + 常数运算 → 满足低延迟。
```

---

## 8. 训练 / 切分 / 选点协议

```text
1. 切分：按时间 train → calibration → validation（时序不可打乱）。
2. 训练：在 train（全部样本，含 infeasible）拟合 §3 损失，标签 y=target_safe，传 s_eff。
3. 校准：在 calibration 上定 δ（§5）与 abstain 阈值 a / 桶集合（§6），小网格扫 alpha。
   目标：minimize covered_gap_norm.mean；硬约束：coverage_feasible>=0.9。
   tie-break：covered_gap_norm.median，再 abstain_rate 低者。
4. 报告：用 calibration 选出的唯一配置，在 validation 出最终指标与分类报告（§9），不再调参。
5. 概率空间：所有指标在概率价格空间；禁 logit。
6. 无泄漏：禁用 forbidden_columns；y、s 为标签派生量，禁作特征；样本经 apply_sample_filter。
```

---

## 9. 评估与分类报告（满足“说明每条为何兜底”）

### 9.1 覆盖口径（必须区分）

```python
feasible = (y <= p)                                    # = s>0
max_possible_coverage = feasible.mean()                # 上限（≈0.70，因 30% infeasible）
covered  = (p_pred >= y) & (p_pred <= p)
coverage_overall  = covered.mean()
coverage_feasible = covered[feasible].mean()           # ← 0.9 约束用这个口径
```

### 9.2 covered 上的 gap（q25 / median / mean / q75）

```python
gap_norm = (p_pred - y) / (p - y)                      # covered & s>0
```

### 9.3 动作分类（核心交付物）

| action | 含义 | 期望 gap | 是否 covered |
|---|---|---|---|
| `active` | 有把握，报价压低(f+δ<p) | 小（目标） | 多数 covered，少量 miss（吃 10% 预算） |
| `abstain_low_conf` | 没把握，主动报 p_side | =1 | feasible 必 covered |
| `clamp_over_pside` | f+δ>=p，截断到 p_side | =1 | feasible 必 covered（含多数 infeasible→仍 miss） |
| `infeasible`（仅评估期可标） | y>p，理论不可 covered | — | 必 miss |

汇总需报告：

```text
- 各 action 占比；
- active 内部：coverage、gap_norm 的 q25/median/mean/q75、miss_rate；
- abstain_low_conf / clamp_over_pside 占比（贡献覆盖但 gap=1）；
- infeasible 占比与 max_possible_coverage；
- 整体：coverage_feasible、全样本 gap_norm.mean（abstain/clamp 计 1）。
→ “因超过 p_side / 因没信心 / 本就不可行” 三类原因一目了然。
```

---

## 10. 约束清单（硬性）

```text
[输出]
- p_pred <= p_side（硬 cap，禁非法报价）；向上取 tick：ceil(raw*100 - 1e-9)/100；
- abstain / clamp 一律输出 p_side。
[数据]
- 无泄漏特征（forbidden_columns）；y、s 为未来标签派生，禁作特征；
- accepted_signal 过滤（apply_sample_filter）；时间三段切分，旋钮只在 calibration 选。
[模型]
- 单模型、单前向；概率空间评估，禁 logit；
- 过预测分支 1/s 必须 s_floor 下限；欠预测分支平滑(c)；alpha 小网格。
[覆盖]
- 用全局 δ 把 coverage_feasible 精确卡到 0.9；不让 loss 承担保覆盖职责。
[选点]
- 目标 covered_gap_norm.mean 最小；约束 coverage_feasible>=0.9；
- 无候选满足则出 coverage–gap 前沿，不强行选点。
```

---

## 11. 落地步骤（最小改动）

```text
1. build：在 target 表加 y=target_raw+0.01、s=p_side-y、s_eff=clip(s,s_floor)。
2. train：复用 train_catboost_quantile.py 骨架，改三处：
   - 标签列 → y；
   - loss → §4.1 自定义目标（传 s_eff 数组）；
   - 输出 OOF f 到 predictions_{train,validation}。
   （或用 §4.2 双头 MLP，得到 f 与 σ。）
3. calibrate：在 calibrate_capped_quantile.py 的扫描骨架上：
   - make_pred 升级为 §7 的 infer_one（含 δ、abstain、clamp、ceil）；
   - 扫 alpha / δ / abstain 阈值，按 §8 选点。
4. report：在现有 metrics 基础上加 §9.3 的 action 分类汇总。
5. 产物：每条落 p_pred 与 action；汇总落 coverage_feasible / gap 分位 / 各类占比。
```

---

## 12. 与已否决方案的关系

```text
保留：单模型回归作为骨架（被验证为前沿最优的形态），infer 快。
修正并启用：log-cosh 的非对称 + soft-violation 的软违约
           → 统一为“欠预测线性 + 过预测归一化 Huber”，并把覆盖交给校准 δ。
新增：normalized(/s) 度量（对齐指标）+ 选择性 abstain（拿回边际自由）+ 全样本训练(解决 infeasible)。
坚决不做：多分位(慢)、center+sigma+conformal(误差叠加被否)、z=log1p 空间非对称(与指标错位)。
```

> 相对 `safe_lowest_price_target_loss_constraints.md` 的**核心新增主张**：
> 用**单模型 + 归一化非对称损失**直接压 normalized gap，
> 用**校准 δ** 保覆盖、用**廉价 abstain**换更紧的 covered 主体，
> 并把 **infeasible 一起训练**——这三点正是普通 quantile 与旧 log-cosh/soft 方案拿不到的部分。
