# Expected Return 优化方案 v2：用「离散生存 / Hazard 模型」替换 Gc

> 日期：2026-06-19
> 基线实验：[`experiments/20260619_expected_return_trade_coverage_start`](../experiments/20260619_expected_return_trade_coverage_start)
> 关联：[MODELING.md](MODELING.md) · [OPTIMIZATION.md](OPTIMIZATION.md)
> 下一实验 id（建议）：`20260619_expected_return_hazard_survival`
> 适用对象：只升级 §4.2 的 `Gc`，EV 决策规则、q、Gw≡1、forced-wrong-fill 回测口径全部不变。

---

## 0. 一句话结论

```text
基线 trade_coverage_start 已修好「训练侧缺低点」问题（missing_low 42，曾 8595），
但 Gc 仍是「点预测 f + 全局残差 CDF」——同方差、薄、系统性高估成交率：
  validation 上 model_fill_prob 0.242 vs 实测 correct_fill 0.03–0.18。
EV 门被这个高估污染，min_ev 在 validation 上搜出来的 sum_pnl 在 -7.92~+7.47 之间变号 = 噪声。

新方法：把 Gc 换成「离散时间生存 / hazard 模型」。
  在出价网格上预测每档 hazard，survival = 累乘 → CDF 天然单调、逐档自带概率校准、形状随 X 变。
  成交本身就是一次 first-passage 事件，hazard 有干净物理意义（价格再往下穿一个 tick 的条件概率）。

这一步根治 Gc 的「不诚实 + 同方差」，让 EV 门可信。
但它本身不破「accepted universe ≈ 零 EV」这堵墙——把 PnL 拱正仍要靠 q 校准 + 选择（L2/L3）。
```

---

## 1. 基线快照（`trade_coverage_start`，validation，forced wrong fill）

数据口径（[target_build_summary.json](../experiments/20260619_expected_return_trade_coverage_start/reports/target_build_summary.json)）：

| 项 | train | validation |
|---|---:|---:|
| source rows（coverage filter 前） | 24249 | 7468 |
| coverage filter 丢弃（< 2026-02-12 无成交） | 8553 | 0 |
| predicted-side rows（join 前） | 15696 | 7468 |
| all-side correct | 10328 | 4855 |
| threshold-accepted rows | 11385 | 5228 |
| **missing_low** | **42** | **35** |

Gc 点模型（[summary_metrics.json](../experiments/20260619_expected_return_trade_coverage_start/reports/summary_metrics.json) `point_model_metrics`）：

| 集合 | n | bias(pred−y) | mae | rmse |
|---|---:|---:|---:|---:|
| fit_correct | **9119** | +0.0099 | 0.121 | 0.153 |
| calibration_correct | 1185 | −0.0075 | 0.140 | 0.174 |
| validation_correct | 4829 | −0.0071 | 0.151 | 0.185 |

全局残差 CDF：来自 1185 个 calibration_correct，std 0.174，q05 −0.332 / q50 +0.042 / q95 +0.231。

validation `min_ev` 搜索（**当前在 validation 上选，泄漏**）：

| min_ev | mean_accepted_pnl | sum_pnl | order_coverage | trade_count | correct_fill_rate | model_fill_prob |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | −0.00151 | −7.92 | 0.774 | 1915 | 0.181 | 0.242 |
| 0.01 | −0.00022 | −1.13 | 0.308 | 996 | 0.147 | 0.242 |
| 0.02 | +0.00094 | +4.89 | 0.211 | 723 | 0.111 | 0.242 |
| 0.03 | −0.00013 | −0.67 | 0.146 | 500 | 0.077 | 0.242 |
| 0.05 | +0.00143 | +7.47 | 0.061 | 210 | 0.034 | 0.242 |

固定基线（同口径）：fixed_0.50 −0.0041 / fixed_0.75 −0.0066 / pay_pside −0.0072。**出价越高 PnL 越差，单调。**

---

## 2. 为什么换 Gc 的参数化（三个病根）

1. **同方差 + 系统性高估**：现有 Gc = `empirical_cdf(residuals, grid − f)`（[train_low_cdf_and_backtest.py](../train_low_cdf_and_backtest.py#L160)），**所有样本共用一条残差形状**，只靠 `f(X)` 平移。结果 `model_fill_prob 0.242` 远高于实测成交（0.03–0.18）。EV 用的就是这个成交概率，被吹高后 `min_ev` 门形同虚设。

2. **薄**：残差 CDF 只有 1185 个 calibration_correct 点；分位粗，尾部（深出价）尤其不可靠，而 EV 最优出价恰恰落在深出价/低 bid 区。

3. **单调性其实已有，但方式脆**：`empirical_cdf` 对 b 已是非降，所以 b-单调没问题；问题是它**不条件化、不校准**。若改成「直接学多分位头」又会引入分位交叉。hazard 模型一举解决：**条件化 + 校准 + 单调**全保证。

> 注意一个常见误判：**这不是「数据太少」一个数能修的**。correct-only 总量 ≈ 10328（9119 fit + 1185 calib），且 `fit` 已经包含 threshold-rejected 的 correct 行（`fit = fit_all[correct]`，`fit_all` 含 rejected）。要再加行只有两条路：把 `trades_coverage_start` 往前推（找回被丢的 8553 行，属数据工程）或缩 calibration tail（已 7 天，空间很小）。**hazard 模型的增益来自「每行监督多个 tick」+ 校准，不是更多行。**

---

## 3. 新方法：离散时间生存 / Hazard 的 Gc

### 3.1 目标量

`Gc(b|X) = P(ℓ ≤ b | 正确, X)`，`ℓ = chosen_low`（赢家 token 窗口低点），随 b 增大而增大。

### 3.2 网格（关键：用 EV 的出价网格，别用 0.30 封顶的 drawdown）

在**绝对价位**网格上离散，与 [`price_grid`](../expected_return_common.py) 用的 bid 网格对齐：

```text
b_1 < b_2 < ... < b_K,   b_k = k · tick,  tick = 0.01,  k = 1..K
K = floor(max_price / tick),  max_price 取 config（默认 0.85，覆盖到 p_side 上限）
```

> 你原稿用 drawdown `δ_k = m − b_k`（m≈p_side）做下标，数学等价；但 `δ` 封顶 0.30 会截断深出价的 Gc，而 EV 最优出价（min_ev=0.05 时 mean_bid≈0.035 ⇒ drawdown 可达 0.5+）正好在那之外。直接在绝对价位网格上做，Gc 格点与 argmax 的 bid 格点天然对齐，省插值、不截断。

### 3.3 Hazard → Survival → CDF（单调由构造保证）

每个样本输出 K 个「填充 hazard」（从低价往高价方向，给定还没在更低处成交）：

$$\lambda_k(x) = P(\ell \le b_k \mid \ell > b_{k-1},\ x) = \sigma\big(z_k(x)\big) \in (0,1)$$

则「尚未成交」的生存概率与 CDF 为：

$$S_k(x)=P(\ell > b_k\mid x)=\prod_{j=1}^{k}\big(1-\lambda_j(x)\big),\qquad
G_c(b_k\mid x)=1-S_k(x)$$

因为每个 $0\le 1-\lambda_j\le 1$，所以 $S_k$ 非增、$G_c(b_k)$ **非降**——

$$G_c(b_1)\le G_c(b_2)\le\dots\le G_c(b_K)\quad\text{（单调由构造保证，无需排序/约束）}$$

> 这正是你写的 hazard 版本 $G_k=\prod_{j<k}(1-h_j)$，只是把"drawdown 生存 $P(D\ge\delta)$ 递减"改写成"价位 CDF $P(\ell\le b)$ 递增"，两者一一对应。

### 3.4 离散生存似然（每行监督多个 tick）

对一个 correct 样本，观测低点 ℓ 落在桶 $k^*$（$b_{k^*-1} < \ell \le b_{k^*}$）：

- $j < k^*$：$\ell > b_j$ → 没成交 → 标签 0 → 因子 $(1-\lambda_j)$
- $j = k^*$：$\ell \le b_{k^*}$ → 事件 → 标签 1 → 因子 $\lambda_{k^*}$
- $j > k^*$：无关（已在更低处成交）

负对数似然（即逐 tick 的 Bernoulli CE，监督到 $k^*$ 为止）：

$$\mathcal{L}(x,\ell)=-\sum_{j<k^*}\log\big(1-\lambda_j(x)\big)-\log\lambda_{k^*}(x)$$

**右截尾**：若只知道 $\ell > b_c$（无更低成交/窗口数据不全），只取存活项 $-\sum_{j\le c}\log(1-\lambda_j)$、无事件项。离散生存天然支持，正好对上历史 trade-coverage 断层。

### 3.5 为什么它适配这个问题（而非 quantile / direct-surface）

- 成交 = **首次穿越（first-passage）**事件；hazard「再往下穿一个 tick 的条件概率」物理意义干净。
- 逐 tick Bernoulli **自带概率校准**，而 EV 要的就是成交概率本身（不是分位点）。
- 单调由构造保证，无需事后排序（优于多分位/pinball 头）。
- 仍是组件式 `q·Gc·(1-b) − (1-q)·b`，可解释、可复用 q、可单独验收 Gc（优于一步到位的 direct EV surface，后者误差不可分解）。

---

## 4. 接进 EV 决策（决策规则一行不改）

对每个 accepted 样本，先用 hazard 模型在价位网格上算出 $G_c(\cdot|x)$ 整条曲线，再在其 bid 网格上最大化 EV：

```python
# 替换 choose_expected_return_bids 内部这一行：
#   gc = empirical_cdf(residuals, grid - f)        # 旧：点 + 全局残差
# 改为按 survival 模型取该样本在 grid 上的 CDF：
#   gc = survival_cdf[i, idx_of(grid)]             # 新：hazard 累乘 CDF（已单调）
ev = q * gc * (1.0 - grid) - (1.0 - q) * grid       # ← 完全不变
j  = int(np.argmax(ev))
bid = grid[j] if ev[j] > min_ev else 0.0            # ← 完全不变（弃单仍从 EV 内生）
```

`realized_pnl`（forced wrong fill：correct 成交 iff `chosen_low≤bid` 赚 `1-bid`；wrong 且 `bid>0` 强制成交亏 `bid`）保持不变。

---

## 5. 训练与数据规格

| 项 | 规格 |
|---|---|
| 训练集 | `fit = fit_all[correct & chosen_low.notna()]`（≈9119，已含 rejected-correct） |
| 早停/选择 | `calibration_correct`（≈1185）上的离散生存 NLL（替换原 calibration_mae） |
| 标签 | `chosen_low` 落桶 $k^*$；右截尾样本按存活项计入 |
| 网格 | 价位 0.01 tick，K = floor(max_price/tick)，max_price∈config |
| 模型 | `HazardMLP`：共享 trunk（256/128/64，dropout 0.10/0.05/0.0，沿用 R9 正则）→ 线性到 K logits → sigmoid |
| 损失 | 逐 tick masked Bernoulli CE（mask 到 $k^*$）+ 可选相邻档平滑罚 $\beta\sum_k(\lambda_{k+1}-\lambda_k)^2$ |
| 优化 | AdamW lr 1e-3 / wd 1e-4 / clip 5 / batch 512 / epochs 45 / early stop 10（沿用现配置） |
| q(X) | = p_side，不重训（本步不动；见 §9 / L2） |
| Gw | ≡ 1，不建模 |

> 不要把 wrong 样本喂进 Gc——Gc 条件在 correct 上；wrong 由 Gw≡1 接管。

---

## 6. 三个必须改对（否则不 work）

1. **网格覆盖到低 bid / 深 drawdown**：用绝对价位网格、`max_price` 覆盖 p_side 上限；**别用 0.30 封顶的 drawdown**，否则 EV argmax 选错。
2. **右截尾**：无更低成交按截尾，不当作"没成交=失败"，否则系统性低估深档成交、和现有"高估"反向但同样有偏。
3. **correct-only**：训练/标签只用 correct 行；wrong 不进 Gc。

---

## 7. 决策 / 回测口径修正（L0，必须随这步一起做）

当前 [main()](../train_low_cdf_and_backtest.py#L315) 在 **validation** 上选 `min_ev`（`min_ev_selection_source: "validation"`，并自带 `validation_optimism_note`）。**改为只在 calibration 上选，validation 只报一次：**

```python
# 旧：在 validation_accepted 上 argmax mean_accepted_pnl 选 min_ev（泄漏）
# 新：在 calibration_accepted 上选，附加 turnover 约束，validation 只评估一次
selected_min_ev = select_min_ev_on_calibration(
    calibration_accepted, f_or_survival_cal,
    grid=min_ev_grid,
    objective="mean_accepted_pnl",          # forced wrong fill 口径
    min_order_count=MIN_ORDERS,             # 防止退化成极少数样本
    tie_break="higher_min_ev",              # one-SE 同分取更高门槛
)
report["min_ev_selection_source"] = "calibration"   # 不再是 "validation"
```

---

## 8. 校准与验收标准

报告必须新增：

1. **逐 tick reliability**：分档 `λ_k` / `Gc(b_k)` 预测 vs 实测成交率（按 b_k 分桶），看是否贴对角线。
2. **submitted 子集成交校准**：validation 下单子集上 `mean(model_fill_prob)` 对 `realized correct_fill_rate`，**两者背离要小**（目标 |gap| ≲ 0.05；当前 0.242 vs 0.11 不可接受）。
3. **frontier**：`mean_accepted_pnl vs order_coverage`（随 calibration 选定的策略在 validation 上画一条），替代只看单点。
4. **forced 口径**：`wrong_fill_forced = 1.0` 主报；`wrong_fill_printed` 仅诊断。
5. **单调性自检**：断言每个样本 `Gc` 沿 grid 非降（构造应恒真）。

**验收门槛**：

```text
A. min_ev 在 calibration 选；validation 只报一次。
B. submitted 子集 predicted_fill ≈ realized_fill（背离显著收窄，远好于 0.242 vs 0.11）。
C. validation mean_accepted_pnl ≥ 0 且 ≥ 最优 fixed 基线；report order_coverage + frontier。
D. 若 B 达成但 C 仍 ≈0 → 结论升级：accepted universe 在此机制下无可交易 edge，
   必须上移到 q 校准 / 方向模型 / 接受阈值（出价侧已到顶），见 §9。
```

---

## 9. 诚实的天花板（这步能修什么、不能修什么）

**能修**：Gc 的同方差/高估/薄——把 0.242 vs 0.11 这个病根除掉，让 EV 门和弃单（min_ev）变得诚实，frontier 可信，可能在高 EV 小子集上把 PnL 从噪声里拣出一点真正正值。

**不能修**：`win_pnl ≈ loss_pnl` 的「零 EV 墙」。逆向选择是结构性的（赢家要跌到出价才成交 ~18%，输家归零必成交 100%）。把 PnL「大幅」拱正仍要：

- **L2 q 校准**：对 p_side 做 isotonic（EV 输家项 `(1-q)·b` 对 q 极敏感，Brier 0.20）。
- **L3 选择**：EV 门挑「高 q + 赢家会深跌」子集，**出价保持低**；用 frontier 调 coverage。
- **L4 上移**：若 L1–L3 后仍 ≈0，改方向模型 / 接受阈值 / 换标的。

> hazard Gc 是 L1 的最佳落地，是"让选择诚实"的地基，不是"一上就大正"的银弹。

---

## 10. 代码改动清单（[train_low_cdf_and_backtest.py](../train_low_cdf_and_backtest.py) + [config.yaml](../config.yaml)）

| # | 位置 | 改动 |
|---|---|---|
| 1 | 新增 `HazardMLP(nn.Module)` | 共享 trunk → K logits；`forward` 返回 logits（K 维） |
| 2 | 新增 `build_tick_grid(tick, max_price)` | 返回价位网格 `b_1..b_K` |
| 3 | 新增 `survival_cdf(logits) -> Gc[N,K]` | `λ=sigmoid(logits)`；`S=cumprod(1-λ)`；`Gc=1-S` |
| 4 | 新增 `train_hazard_model(...)` | 替换 `train_point_model`：masked Bernoulli CE + 可选平滑罚；early-stop 用 calibration NLL |
| 5 | 改 `choose_expected_return_bids(...)` | 入参从 `(f_pred, residuals)` 改为 `(gc_matrix, grid)`；内部 `gc = gc_matrix[i]`，EV/argmax/弃单不变 |
| 6 | 改 `main()` 切分后 | 用 hazard 模型对 train/cal/val 算 `Gc[N,K]`；删 `residuals = y_cal_correct - f_cal_correct` |
| 7 | 改 `main()` min_ev 选择 | 从 `validation_accepted` 改 `calibration_accepted`；`report["min_ev_selection_source"]="calibration"` |
| 8 | 改 checkpoint | 存 `tick_grid`、`max_price`、hazard `state_dict`；去掉 `residual_cdf` |
| 9 | 改 `write_predictions` / `point_metrics` | `gc_point_pred` → 记录 `model_fill_prob`（取 argmax bid 处 Gc）+ 逐 tick 校准所需列；新增 reliability 导出 |
| 10 | 新增 reliability 报告 | per-tick 与 submitted 子集 predicted vs realized 成交校准表 |

> `UpperBoundMLP` / `train_point_model` / `empirical_cdf` 可保留为 fallback（`model.family` 开关），默认走 hazard。

---

## 11. config 片段（建议）

```yaml
model:
  family: hazard_survival_cdf      # 取代 r9_point_plus_global_residual_cdf
  hidden_dims: [256, 128, 64]
  dropout: [0.10, 0.05, 0.00]
  hazard:
    tick_size: 0.01
    max_price: 0.85                # 价位网格上限，覆盖 p_side 上限
    smoothness_penalty: 0.0        # 相邻档平滑罚 β，先 0，过拟合再开

target:
  # tick_size / min_bid / order_window / min_ev_grid 保持不变
  min_ev_selection_source: calibration   # 新增：禁止在 validation 上选

split:
  calibration_tail_days: 7         # 不变
```

---

## 12. 实验计划

| run | 配置 | 目的 |
|---|---|---|
| H0 | 复跑基线（global residual Gc） | 锚定 0.242 vs 0.11 与噪声 frontier |
| **H1** | hazard Gc + min_ev 在 **calibration** 选 | 主实验：成交校准是否收窄、frontier 是否诚实 |
| H2 | H1 + 相邻档平滑罚 β>0 | 抑制稀疏高/低价档的 hazard 抖动 |
| H3（可选） | H1 + 把 `trades_coverage_start` 前移找回 8553 行 | 验证「更多行」是否还有边际增益（数据工程） |
| H4（接 L2） | H1 + q=p_side isotonic 校准 | 把"诚实的 Gc"接到"诚实的 q"，看 PnL 是否过零 |

> 训练数据 parquet 不在本地（路径为 Windows `C:\Users\ROG\...`）。按既定流程：**先把代码改动同步到 Databricks/Windows，再在那边重建/重训**，回传 `summary_metrics.json` + reliability 表 + predictions parquet 做分桶诊断。

---

## 13. 验收 checklist

```text
[ ] Gc 沿 grid 单调（构造自检通过）
[ ] min_ev 在 calibration 选；report.min_ev_selection_source == "calibration"
[ ] validation 只评估一次，不参与任何选择
[ ] submitted 子集 predicted_fill ≈ realized_fill（远好于 0.242 vs 0.11）
[ ] 报告 forced wrong_fill = 1.0；printed 仅诊断
[ ] 报告 order_coverage + mean_accepted_pnl vs order_coverage frontier
[ ] 与 fixed/pay 基线、H0 global-residual Gc 对比
[ ] 若校准达成但 PnL 仍 ≈0 → 结论写明上移 q/方向/接受阈值
```
