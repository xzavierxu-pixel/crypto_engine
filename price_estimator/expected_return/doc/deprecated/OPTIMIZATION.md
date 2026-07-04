# Expected Return 优化方案：从「准确率高但 PnL≈0」到「大幅提高 validation mean accepted EV」

> 日期：2026-06-19
> 关联：[MODELING.md](MODELING.md)
> 实验对象：`experiments/20260618_expected_return_r9_residual_cdf`
> 目标：把 validation 的 **mean accepted PnL（每个 accepted 样本的真实 realized EV）** 抬到稳定为正。当前报告里的 −0.0024 是 **printed-fill 乐观口径**；真实口径必须对预测错误样本强制 100% 成交，实际起点会更低。

---

## 0. 一句话结论

```text
主指标必须明确为 mean accepted PnL：
  mean_accepted_pnl = sum(realized_pnl over accepted samples) / accepted_sample_count
当前报告里的 mean_pnl 分母已经是 accepted_sample_count=5201，应该重命名为 mean_accepted_pnl。

更重要：当前回测的 filled 口径不符合你的真实机制。
  预测正确：只有 chosen_low <= bid 才成交；成交赚 1-bid，否则 pnl=0。
  预测错误：必须强制 100% 成交（G_w≡1）；只要提交了 bid>0 的订单，就亏 bid。

所以 current report 的 wrong_fill=0.661 只是「历史成交打印 chosen_low<=bid」下的乐观诊断，
不是实盘真实机制。真实机制下 wrong_fill 应固定为 1.000。

实验另一个缺口是 MODELING.md §3/§6 的「放弃下单（abstain）」没有落地。
代码里 allow_negative_ev_bid=true + min_bid=0.01 → 策略对每个被接受样本都强制挂 ≥0.01 的单，
永远不弃单。由于错误样本强制成交，这些 1 美分 floor 单仍会稳定亏损。

而「准确率 70.7% 却不赚钱」的根因是成交的逆向选择（与方向准确率无关）：
  赢家几乎不成交（17.6%），输家真实口径下强制 100% 成交。
  → 当前 printed-fill 表里 1654 笔成交只有 39% 是赢家；真实 forced-wrong-fill 口径下赢家占比会更低。
  → 所有基线（fixed_0.5 / 0.75 / pay_pside）的 printed-fill mean accepted PnL 也都≈0 或为负；forced-wrong-fill 后会更差。

在当前 printed-fill report 中，EV 被高估 0.02 主要来自 Gc（赢家低点 CDF）高估赢家成交率
（模型 0.249 vs 实测 0.176），来源是时间漂移：fit 窗口截至 3/10，validation 是 4/11–5/10，
point model 在 val 上 bias −0.017。改成 forced-wrong-fill 后，还会额外补上所有未打印成交的错误样本亏损。

要「大幅」提高，必须先修 **回测口径（错误强制成交）**，再改 **选择（只在高 EV 子集下注）**
和 **校准（让 q、Gc 诚实）**，而不只是重调出价。

数据切分改为硬规则：calibration 只用最近 7 天；7 天之前全部可定义 predicted side 的样本放入 fit。
不要在数据构建阶段先丢弃未过 threshold 的样本。正确做法是：
  predicted_side = UP if p_up>=0.5 else DOWN
  p_side = max(p_up, 1-p_up)
  threshold_accepted = (p_up>=t_up or p_up<=t_down)
训练 Gc / direct EV 时可以用未过 threshold 的样本；最终回测/下单时再按 threshold_accepted 丢弃。

训练口径也要拆清楚：最终 direct EV/PNL 训练应使用全部 predicted-side rows（correct + wrong，包括 rejected）；
只有当模型明确估计 `Gc=P(赢家低点<=b | correct,X)` 这个条件 CDF 时，低点头才可以 correct-only。
如果要把预测正确和预测错误样本一起训练低点/EV，模型目标必须改成 direct EV/PNL surface，不能再叫 Gc。
```

---

## 1. 实验结果快照（来自 `reports/summary_metrics.json`）

### 1.1 三段窗口与样本

当前报告仍是旧的 31 天 calibration 切分，并且数据构建在低点 join 之前先过滤成 accepted-only。下一轮固定改为：

```text
calibration_all_side = train 窗口最后 7 天 predicted-side rows（accepted + rejected 都保留）
fit_all_side         = train 窗口中 calibration 之前的全部 predicted-side rows（accepted + rejected 都保留）
threshold_accepted   = 最终回测/下单过滤标记，而不是训练前丢弃条件
```

| 窗口 | 时间 | 行数 | accepted accuracy |
|---|---|---|---|
| fit_correct（旧报告，仅 Gc 低点头） | 2026-02-12 → 03-10 | 3806 | — |
| calibration | 2026-03-10 → 04-10 | 5928 | 0.705 |
| validation | 2026-04-11 → 05-10 | 5201 | 0.707 |

> 这里的 `fit_correct=3806` 不是下一轮推荐的主训练样本数，它只是当前 accepted-only 数据里 component Gc 低点头的 correct-only 样本数。下一轮如果训练 direct EV/PNL surface，fit 应使用 `fit_all_side` 的全部 predicted-side 行，包括 threshold rejected、accepted、correct 和 wrong。

### 1.2 当前报告表（validation，printed-fill 乐观口径）

| 策略 | mean_accepted_pnl* | mean_expected_ev | fill_rate | correct_fill | wrong_fill_printed** | mean_bid | trade_count |
|---|---|---|---|---|---|---|---|
| **expected_return（本模型）** | **−0.00244** | +0.01779 | 0.318 | **0.176** | **0.661** | 0.205 | 1654 |
| fixed 0.50·p_side | +0.00008 | — | 0.418 | 0.194 | 0.961 | 0.336 | 2175 |
| fixed 0.75·p_side | −0.00444 | — | 0.567 | 0.392 | 0.989 | 0.507 | 2950 |
| pay p_side | −0.00717 | — | 0.882 | 0.833 | 1.000 | 0.678 | 4588 |

\* 当前 JSON 里列名叫 `mean_pnl`，但分母是 5201 个 accepted 样本，所以应理解为 `mean_accepted_pnl`。

\** `wrong_fill_printed` 只表示历史成交打印里 `chosen_low <= bid` 的比例。按真实机制，预测错误时下注侧 token 归 0，提交的正 bid 订单必须视为成交：`wrong_fill_forced = 1.000`。

> **关键观察**：这张表不是最终实盘 EV 表，而是当前代码的 printed-fill 乐观诊断表。本模型 printed-fill `mean_accepted_pnl = −0.0024` 已经为负；如果把预测错误样本强制 100% 成交，真实 `mean_accepted_pnl` 会更低。精确 forced 数字需要重新跑回测或读取逐样本 bid；仅凭 summary 至少可知：本模型有约 516 笔错误样本未在 printed-fill 下成交，因 `min_bid=0.01`，真实亏损至少再多 `516×0.01≈5.16`，所以 forced 口径下 `mean_accepted_pnl <= (-12.67-5.16)/5201 ≈ -0.00343`，实际可能更差。

### 1.3 train / calibration / validation 漂移

| 指标 | train | calibration | validation |
|---|---|---|---|
| mean_accepted_pnl（printed-fill） | **+0.0301** | **+0.0225** | **−0.0024** |
| mean_expected_ev | +0.0195 | +0.0199 | +0.0178 |
| correct_fill_rate | 0.261 | 0.237 | 0.176 |
| wrong_fill_printed | 0.641 | 0.645 | 0.661 |

> 训练/校准上为正、验证上转负 → 典型的**泛化/时间漂移**。但这些仍是 printed-fill 数字；真实 forced-wrong-fill 口径要把所有错误样本的正 bid 全部记为成交亏损。

### 1.4 point model（赢家低点回归）与残差 CDF

| 集合 | bias(pred−y) | mae | rmse | pred_mean | target_mean | n |
|---|---|---|---|---|---|---|
| fit_correct | +0.0105 | 0.120 | 0.153 | 0.517 | 0.507 | 3806 |
| calibration_correct | −0.0059 | 0.144 | 0.180 | 0.505 | 0.511 | 4180 |
| validation_correct | **−0.0170** | 0.157 | **0.191** | 0.505 | **0.522** | 3678 |

残差 CDF（来自 calibration_correct，全局）：mean 0.006，std **0.180**，q05 −0.357，q50 +0.047，q95 +0.233。

---

## 2. 是否忠实实现了 MODELING.md？——两个关键缺口：forced wrong fill + abstain

| MODELING.md 条目 | 是否实现 | 证据 / 备注 |
|---|---|---|
| §5 数据重建：下注侧=阈值决定、accepted-only、chosen_low=下注侧 token 窗口低点、无 look-ahead | ✅ 忠实 | `build_expected_return_target.py` `choose_side()` 用 `p_up`+阈值；`correct` 为事后标签；merge 用 `selected_outcome` 非赢家 |
| §4.1 q(X)=p_side 直接复用、不重训 | ✅ | `choose_expected_return_bids` 用 `df["p_side"]` |
| §4.2 Gc=point model f + 全局残差 CDF（方案 a） | ✅ 忠实 | `empirical_cdf(residuals, grid − f)`，residuals 取自 calibration_correct |
| §4.3 G_w≡1（输家必成交） | ❌ 回测未实现 | **策略 EV** 用 `−(1−q)·b`（隐含 G_w=1），但 **realized PnL 回测** 对 wrong 样本仍用 `chosen_low ≤ bid` 判断成交。真实机制下 wrong 且 bid>0 必须强制成交，`wrong_fill=1.000` |
| §7 PnL 回测 + 三基线 | ⚠️ printed-fill 口径 | realized_pnl 对 correct 样本是对的；对 wrong 样本偏乐观。基线 fixed_0.5/0.75/pay_pside 也必须同步改成 forced-wrong-fill 后再比较 |
| **§3/§6 abstain：当 max_b E[PnL] ≤ min_ev 时不挂单** | ❌ **未实现** | `choose_expected_return_bids` 永远 `argmax` 一个 ≥ `min_bid` 的 bid；`allow_negative_ev_bid` 只写进 report、**不参与决策**；没有 `min_ev` 门限 |

**结论**：当前实现有两个必须先修的口径问题。第一，**wrong 样本必须强制成交**，不能用历史成交打印判断是否成交；否则 `mean_accepted_pnl` 被系统性高估。第二，**MODELING.md 的灵魂机制——「弃单从 EV 内生涌现」——没有落地**。当前代码把 b* 钳在 `[min_bid, p_side]`，最低也要挂 0.01；在 forced-wrong-fill 口径下，这些 floor 单只要预测错就必亏。

> 代码定位：forced wrong fill 缺失在 [train_low_cdf_and_backtest.py](../train_low_cdf_and_backtest.py) 的 `realized_pnl`；弃单缺失在同文件 `choose_expected_return_bids`；floor 在 [config.yaml](../config.yaml) `target.min_bid: 0.01` 与 `allow_negative_ev_bid: true`。

---

## 3. 为什么「准确率高、mean accepted PnL 低」——根因拆解

### 3.1 根因 A：成交的逆向选择（与方向准确率脱钩）

机制（MODELING.md §1）：限价**低于市价**买入下注侧。
- 赢家 token → 1：价格留在高位，**很少**跌到你的低出价 → 极少成交。
- 输家 token → 0：价格穿过任何正出价 → **必须按 100% 成交**。

把 validation 数字代进去：

```text
被接受样本 5201：correct 3678 (70.7%)，wrong 1523 (29.3%)
当前 printed-fill report：
  赢家成交 = 3678 × 0.176 ≈ 647
  错误样本中有历史打印穿 bid 的数量 = 1523 × 0.661 ≈ 1007
  printed-fill 合计 1654 笔；其中赢家仅 647/1654 = 39%

真实 forced-wrong-fill 口径：
  赢家成交仍 ≈647（正确样本只有 chosen_low <= bid 才成交）
  错误样本成交 = 1523（wrong_fill = 1.000）
  forced 合计 ≈647+1523=2170 笔；其中赢家仅 647/2170 = 29.8%
→ 尽管方向准确率 70.7%，真实成交单里赢家占比只有约 30%。
```

当前 printed-fill realized 分解：
```text
win_pnl_sum  = +260.70（647 笔赢家成交，平均 +0.40/笔，赚 1−b）
loss_pnl_sum = −273.37（1007 笔输家成交，平均 −0.27/笔，亏 b）
net = −12.67 / 5201 = −0.00244
```

真实 forced-wrong-fill 分解应为：
```text
win_pnl_sum_forced  = +sum(1-bid_i for correct_i and chosen_low_i <= bid_i)
loss_pnl_sum_forced = -sum(bid_i for wrong_i and bid_i > 0)
mean_accepted_pnl   = (win_pnl_sum_forced + loss_pnl_sum_forced) / accepted_count
```

当前 summary 没有保存「printed-fill 下未成交的 wrong 样本 bid 总和」，所以无法仅凭 JSON 精确还原 forced PnL；但它一定比 −0.00244 更差。

> **核心**：赢家收益（+0.40/笔）大但稀疏；输家亏损只要预测错就必然发生。方向准确率被「赢家不一定成交 / 输家强制成交」的不对称吃掉了。**准确率≠mean accepted PnL**。

### 3.2 根因 B：策略试图用「floor 出价」躲输家，但躲不掉

median_bid = **0.01**（≥一半样本挂在 floor）。但在 0.01：
- 赢家：低点 ≤ 0.01 几乎不可能 → 赢家成交率→0。
- 当前成交打印：`wrong_low_lte_001_share = 0.529`，说明历史打印里已有 **52.9%** 的输家穿到 0.01。
- 真实机制：错误样本强制成交，floor bid=0.01 不是躲掉亏损，而是 **100% 亏 0.01**。

所以 floor 不是「弃单」，而是「用最小正出价仍提交了一张错误时必亏的订单」。叠加 `min_bid=0.01` 强制下注 → 持续小额失血。**真正的弃单（bid=0 / 不挂单）才能止血**。

### 3.3 根因 C：printed-fill 口径下的 EV 高估主要来自 Gc 高估

- 模型预测的赢家成交率（`mean_model_fill_prob`）= **0.249**；实测 `correct_fill_rate` = **0.176**。高估 +0.073（相对 +41%）。
- 反推：若赢家成交率真为 0.249，realized 赢家收益 ≈ 260.7 × (0.249/0.176) ≈ 368；净 = (368 − 273) / 5201 ≈ **+0.018** ≈ 预测 EV。
- **⇒ 在当前 printed-fill report 里，EV 高估主要来自 Gc 对赢家成交率的高估。**

为什么 Gc 在 validation 上乐观？
1. **时间漂移**：fit 截至 3/10，validation 4/11–5/10，相差约 1 个月；point model 在 val 上 `bias = −0.017`（f 低估赢家低点 → `Gc(b)=P(r≤b−f)` 中 f 偏小 → b−f 偏大 → CDF 偏大 → 成交率高估）。
2. **残差 CDF 用 3 月的 calibration 窗口**，std 0.180，未反映 val 期更宽的误差（val rmse 0.191 > cal 0.180）。
3. q=p_side 的平均校准其实略**保守**（mean p_side≈0.68 < 准确率 0.707），所以乐观不来自 q，而来自 Gc。

改成 forced-wrong-fill 后，评估缺口会新增一部分：当前 printed-fill 下没成交的 wrong 样本要补记为 `-bid`。因此 O0 必须先重跑 forced 口径，再重新判断 q 与 Gc 各自贡献。

### 3.4 根因 D（数据完整性风险）：训练侧 35% accepted 行因「无成交」被丢弃

`target_build_summary.json`：
- train：`missing_low_rows = 6021 / 17371 accepted = 34.7%` 被 drop（下注侧 token 决策后无成交）。
- validation：`missing_low_rows = 27 / 5228 = 0.5%`。

train 与 val 的 trades 覆盖度差异巨大（35% vs 0.5%）。这意味着：
- fit/calibration 的残差只建立在「有成交」的子集上 → **选择偏差**，Gc 可能系统性偏移；
- 也可能放大了 train(+0.030) 与 val(−0.0024) 的落差。

这不是 threshold 直接造成的：`missing_low_rows` 是在 **threshold accepted 之后**，用 `(condition_id, selected_outcome, decision_time)` 去 trades 里找 `decision_time < trade_time <= endDate` 的 selected-side 成交时产生的。train 和 validation 用的是同一个 `utc_day_session_coordinate` threshold policy，accepted coverage 也接近（train 71.6%、validation 70.0%），但 missing_low 差异是 34.7% vs 0.5%，所以更像 **trade 覆盖/成交打印/时间段数据完整性问题**，不是阈值用得不同。

下一轮诊断必须在 drop 之前输出：

```text
missing_low_rate by date / week
missing_low_rate by selected_side / predicted_side
missing_low_rate by threshold_accepted
trade row count and condition_id coverage by date
for missing rows: trades 是否完全没有该 condition_id，还是有 condition_id 但没有 selected_outcome / 时间窗内成交
```

若 rejected 样本也加入 Gc/direct EV 训练，这个诊断要同时覆盖 accepted 和 rejected；否则 rejected 里的无低点样本仍会被隐式丢掉，训练集继续有选择偏差。

### 3.5 根因 E（硬口径错误）：预测错误必须强制 100% 成交

当前 `realized_pnl()` 对所有样本统一使用：

```python
filled = chosen_low <= bid
```

这对 correct 样本是对的，但对 wrong 样本不对。真实机制应写成：

```python
if correct:
  filled = chosen_low <= bid
  pnl = 1 - bid if filled else 0
else:
  filled = bid > 0      # 只要提交订单就强制成交
  pnl = -bid if filled else 0
```

因此 `wrong_fill` 在主报告里不应由数据打印决定，而应固定为：

```text
wrong_fill_forced = 1.000  # 在 bid>0 的 submitted orders 上
```

当前 `wrong_fill=0.661` 只能作为「历史成交打印覆盖 / 市场最低打印是否穿 bid」的诊断列，不能作为主 PnL 口径。

---

## 4. 优化杠杆（按预期收益排序）

> 主指标：**validation mean accepted PnL（forced-wrong-fill 口径，每个 accepted 样本，弃单计 0）**。
> 选 `min_ev` / 阈值时，用 calibration 的 forced `mean_accepted_pnl` 选，validation 只报结果。

### L0 —— 先修回测口径：mean accepted PnL + forced wrong fill（必须先做）

**做法**：
1. 把报告列名从 `mean_pnl` 改为 `mean_accepted_pnl`，明确分母是 accepted 样本数，不是成交笔数。
2. `realized_pnl()` 改为分支逻辑：correct 样本用 `chosen_low <= bid`；wrong 样本只要 `bid > 0` 就 `filled=True, pnl=-bid`。
3. 报告两个 fill 指标：
  - `wrong_fill_forced`：主口径，submitted wrong orders 上固定为 1.000；
  - `wrong_fill_printed`：诊断口径，保留原 `chosen_low <= bid`，用于观察成交打印覆盖。
4. 所有基线也同步用 forced-wrong-fill 回测，否则策略比较不公平。

**预期**：重算后当前模型的 validation `mean_accepted_pnl` 会低于 printed-fill 的 −0.00244。这不是坏消息，而是把真实损益口径拉直；后续优化必须以这个数为起点。

### L1 —— 落地真正的弃单 + EV 风险边际 `min_ev`（最高优先，直接修 §3/§6）

**假设**：当前对每个样本强制挂 ≥0.01，在「无利可图」样本上失血。引入弃单后，被接受域里只对 `max_b EV > min_ev` 的子集下注，其余 realized PnL=0（而非负）。

**做法**：
1. 在 `choose_expected_return_bids` 增加弃单：若 `best_ev ≤ min_ev` → `bid = 0`（不提交订单，correct/wrong 都 `filled=False`、`pnl=0`）。
2. 让 `allow_negative_ev_bid` 真正生效：为 `false` 时启用弃单分支。
3. 新增 config `target.min_ev`，在网格 `{0.00, 0.01, 0.02, 0.03, 0.05}` 上按 **calibration forced mean_accepted_pnl** 选取，validation 只报。
4. 因为预测 EV 在 printed-fill 下已乐观约 0.02，`min_ev ≈ 0.02–0.03` 才可能把「预测微正、真实为负」的边际单挡掉。

**预期**：把当前 `negative_expected_ev_share=0.158` + 一批「预测微正、实测为负」的边际单清零；forced `mean_accepted_pnl` 从真实起点抬向 0 以上；`mean_pnl_filled` 转正。**这是把曲线翻正的第一杠杆。**

### L2 —— 固定 7 天 calibration，前面全部进 fit_all_side；threshold 最后过滤

**原则**：calibration 只负责最近窗口的阈值/偏差/EV 门限选择，不应该吃掉半个训练集。当前 `calibration_tail_days=31` 让 fit 只剩很少、且离 validation 太远；下一轮固定改为 **7 天**。同时，threshold 不能在数据构建阶段过早 drop；它应该作为 `threshold_accepted` 标记保留到最后回测/下单阶段。

**做法**：
1. `split.calibration_tail_days = 7`。
2. 在 target build 里新增 `predicted_side = argmax(p_up)`、`p_side=max(p_up,1-p_up)`、`threshold_accepted`，不要先 `df = df.loc[df["accepted"]]`。
3. `fit_all_side = train_all_side[timestamp < train_end - 7d]`，包含全部 predicted-side rows：threshold rejected + accepted、correct + wrong。
4. `calibration_all_side = train_all_side[timestamp >= train_end - 7d]`，也保留全部 predicted-side rows；其中 min_ev/最终策略选择只在 `threshold_accepted=True` 子集上评估。
5. 不再用 31 天 calibration。7 天足够做校准，也能把更多、更早的数据放回 fit，并让 fit 结束时间更接近 validation。

**预期**：减少时间漂移，增加 fit 样本量，避免 calibration 过大导致模型主体训练不足。

### L3 —— 训练口径：主线改为 all-side direct EV/PNL surface；Gc correct-only 只作为组件备选

你的判断对主目标是对的：如果目标是提高 **mean accepted PnL / expected EV**，训练阶段不应该只看 threshold accepted 行。应该先看全部分类模型能定义 predicted side 的行，包括 threshold rejected 和 accepted，也包括 correct 和 wrong。原因是 rejected 行仍提供“某种 X 下赢家低点/输家亏损结构”的样本信息；最终是否下单再由 threshold_accepted 和 min_ev 决定。

但要避免一个坑：**不能把 wrong 行的低点混进 `Gc` 训练**。因为 `Gc` 的定义是：

```text
Gc(b|X) = P(赢家低点 <= b | correct, X)
```

它是条件在 `correct=True` 上的赢家低点 CDF。如果把 wrong 行也放进去，wrong token 的低点接近 0，会把 `Gc` 人为抬高；再代入 `q*Gc*(1-b)` 时，就等于把输家低点当成赢家成交概率，正项会被污染。

因此下一轮推荐两条路线：

**主线 A：direct EV/PNL surface（推荐，使用全部 predicted-side 行）**

对每个样本、每个 bid 网格直接构造 forced-wrong-fill 真实收益标签：

```text
if correct:
  pnl_label(b) = 1-b  if chosen_low <= b else 0
else:
  pnl_label(b) = -b   if b > 0 else 0
```

训练 `EV_hat(X,b)` 或直接训练每个 bid 的 ranking/utility，使用 `fit_all_side` 全部 predicted-side rows。推理/回测时先算 EV，再在最后应用：

```text
b* = argmax_b EV_hat(X,b)
if threshold_accepted is False: no order
else if max_b EV_hat(X,b) <= min_ev: no order
else submit bid=b*
```

这条路线最贴近目标函数，天然包含 rejected/accepted、correct/wrong 四类样本，也直接优化最终 threshold-accepted 子集上的 mean accepted PnL。

**组件 B：保留 q + Gc（可解释备选）**

```text
q 校准：用全部 calibration_all_side predicted-side rows，标签 correct；最终 min_ev/threshold 表现只在 threshold_accepted=True 子集上选择和报告。
Gc 低点头：可用全部 predicted-side rows 中 `correct=True` 的行，标签为 predicted-side winner/chosen_low；不应只限 threshold accepted。
策略/门限/回测：训练/校准可看全部 predicted-side rows；最终指标必须在 threshold_accepted=True 子集上，forced-wrong-fill 口径。
```

组件 B 的 correct-only 只限于 `Gc` 低点头，不代表整个实验只训练正确样本，也不代表未过 threshold 的样本应该在训练前丢弃。当前文档和报告要把这点写清楚。

### L4 —— 校准 Gc（赢家低点 CDF）+ 修时间漂移（组件 B 专用）

**目标**：让预测 EV ≈ 实测 EV（消除 0.02 乐观），从而 L1 的弃单门限才可信。

**做法**（可叠加）：
1. **point model 偏差校正**：用 7 天 calibration correct rows 估 `bias_recent = mean(f − y)`，推理时 `f' = f − bias_recent`（或直接对残差中心化）。val bias −0.017，校正后 Gc 不再系统高估。
2. **残差尺度时间加权 / 滚动**：残差 CDF 用最近 7 天 calibration correct rows，必要时对残差 std 乘一个 >1 的保守系数（如 ×1.1）做悲观 CDF。
3. **（可选）异方差**：按 `p_side_bucket` 或 `direction_confidence` 分桶估残差尺度，替代单一全局 CDF（repo 既往结论：全局 room proxy 已够，分桶尺度增益有限，列为低优先）。

**预期**：`mean_model_fill_prob` 从 0.249 收敛到 ≈实测 correct_fill 0.176；预测 EV 与 forced-wrong-fill realized PnL 对齐；配合 L1 让弃单门限真实有效。

### L5 —— 校准 q = p_side（修 EV 的方向项，per-sample）

**假设**：mean p_side≈0.68 vs 准确率 0.707，平均略保守，但**分布上**高 p_side 区间过自信（repo 记录：高 p_side 仍 33% 出错）。EV 对 q 线性敏感，per-sample 误校准会让弃单/抬价判断错位。

**做法**：在 calibration 窗口对 `p_side→correct` 做 **isotonic / Platt** 校准，得到 `q_cal(X)`，EV 用 `q_cal` 而非原始 p_side。MODELING.md §10 已把这列为已知局限，本优化将其纳入。

**预期**：高 p_side 区间的过自信被压低 → 这些样本的 EV 下修 → 在真实低 EV 处更早弃单；低 p_side 区间若被低估则可能恢复少量正 EV 单。整体让 L1 的门限切在正确位置。

### L6 —— EV 门限驱动的「强选择」：只在高 q + 有跌幅的赢家上下注（抬 EV 天花板）

**结构事实**：正 EV 需要 `q·Gc(b)·(1−b) > (1−q)·b`。对高 q（如 q≥0.85）且赢家在窗口内有**实质跌幅**（Gc(b) 在某个仍有利可图的 b 处够大）才成立。当前阈值（t_up≈0.58 / t_down≈0.43）放进来大量 q≈0.6–0.7 的边际样本，注定 floor + 失血。

**做法**：
1. 不新增硬阈值，**用 L1 的 `min_ev` 让弃单自然把选择收紧到高 EV 子集**（与 MODELING.md 「选择从 EV 涌现」一致）。
2. 诊断：按 `q_cal` 分桶报告 forced mean accepted PnL 与最优 `min_ev`，确认正 EV 主要集中在高 q + 大 room（`p_side − f` 大）子集。
3. 若弃单后 trade_count 过低，接受「低频高质」——主指标是 forced mean accepted PnL，不是成交量。

**预期**：交易数从 1654 大幅下降，但每笔与总体 forced mean accepted PnL 显著转正；这是「大幅提高」的主要来源（把零 EV 的大众样本剔除，保留少数真正有 edge 的样本）。

### L7 —— 固化 G_w≡1：策略 EV、回测、报告三处口径一致

**做法**：
1. 策略 EV 已经使用 `−(1−q)·b`，保持不变：这就是 `G_w≡1`。
2. 回测 realized PnL 必须同步使用 forced wrong fill：wrong 且 bid>0 → `pnl=-bid`。
3. 报告里不要再把 `wrong_fill_printed=0.661` 当成策略表现；主口径展示 `wrong_fill_forced=1.000`。
4. 如果未来另做「只挂 4min 且错误未必成交」的短窗版本，那是另一个模型假设，需要单独建 `G_w(b|X)`；本实验目标按你的机制固定 `G_w≡1`。

**预期**：消除「策略假设 G_w=1、回测却用 printed-fill」的偏差，让 mean accepted PnL 与真实交易机制一致。

### L8 —— 风险感知目标与门限选择（稳健化）

**做法**：
1. 选 `min_ev` 时优化「validation forced mean accepted PnL − λ·下行波动（或 CVaR）」，而非纯均值，降低单边大亏。
2. 报告 bootstrap 置信区间，确认正 EV 非噪声（当前 sum_pnl −12.67、单位小，需统计显著性）。
3. 时间衰减 retrain（缩短 fit→deploy 间隔），缓解漂移（与 L2 协同）。

---

## 5. 实验计划（建议在 Databricks/Windows 上数据就位后跑）

| 实验 | 改动 | 主要验证 |
|---|---|---|
| **O0 回测口径修正** | L0/L7：mean_accepted_pnl + forced wrong fill；保留 wrong_fill_printed 诊断 | 得到真实起点；wrong_fill_forced=1.000；当前 −0.0024 被修正为更保守值 |
| **O1 七天切分 + 数据体检** | L2：calibration 固定 7 天；前面全部 predicted-side rows 进 fit_all_side；排查 `missing_low_rows=35%` | fit_all_side 样本恢复；Gc/EV 标签是否仍有交易覆盖偏差 |
| **O2 弃单+min_ev** | L1：`choose_expected_return_bids` 加弃单；网格选 `min_ev` | validation forced mean_accepted_pnl 是否转正；mean_pnl_filled 转正 |
| **O3 direct EV/PNL surface** | L3 主线 A：用 fit_all_side 全部 predicted-side rows 训练 `EV_hat(X,b)` | all-side 训练是否显著超过 q+Gc 组件式 |
| **O4 Gc 组件备选校准** | L4 组件 B：7 天 calibration correct rows 做 bias/残差 CDF | `mean_model_fill_prob`→≈correct_fill；预测 EV≈forced realized |
| **O5 q 校准** | L5：isotonic/Platt(p_side)，用全部 calibration_all_side predicted-side rows | 高 p_side 桶 EV 下修；accepted 子集弃单切点更准 |
| **O6 强选择曲线** | L6：扫 `min_ev`，画 PnL–trade_count 前沿 | 高 q+大 room 子集 forced mean_accepted_pnl 显著正 |
| **O7 稳健化** | L8：CVaR 目标 + bootstrap CI | 正 EV 统计显著、抗漂移 |

**落地顺序**：O0（先修真实口径）→ O1（7 天切分 + 保留 rejected）→ O2（先有弃单）→ O3（all-side direct EV 主线）→ O5（q 校准）→ O6（选择曲线）→ O7。O4 是组件式 q+Gc 的备选校准线，可并行但不作为主线阻塞。

---

## 6. 成功标准（验收口径）

```text
主指标：validation forced mean accepted PnL（每 accepted 样本，弃单计 0）
  目标：先用 O0 得到 forced 真实起点，再提升到 ≥ +0.01（每样本），且
       预测 mean_expected_ev 与 forced realized 的缺口 ≤ 0.005。
辅指标：
  - mean_pnl_filled > 0（当前 −0.0077）；
  - 预测 winner fill ≈ 实测 winner fill（缺口从 0.073 → ≤0.02）；
  - wrong_fill_forced = 1.000（submitted wrong orders），wrong_fill_printed 仅作为诊断；
  - 在所有基线（fixed_0.5/0.75/pay_pside）之上显著为正；
  - bootstrap 95% CI 下界 > 0。
诚实性：所有阈值（min_ev、calib_tail）用 calibration 选、validation 只报；
       不得用 validation 调参。
```

---

## 7. 风险与注意

```text
1) 数据偏差（根因 D）：train 期 35% 行无成交被丢，Gc 可能系统偏移；O1 必须做。
2) 口径硬约束（根因 E）：预测错误必须强制 100% 成交；任何 printed-fill PnL 都只能做诊断，不能做主指标。
3) 选择过紧：L1/L6 弃单后 trade_count 可能很低；主指标是 mean accepted PnL，但要确认样本量足够做统计推断（O7 bootstrap）。
4) q 校准外溢：p_side 来自部署方向模型，重校准只在本模块用于 EV，不回写部署模型。
5) 漂移：calibration 改 7 天后 fit 结束时间更接近 validation；direct EV 主线仍需滚动重训/时间衰减防漂移。
6) 「大幅提高」的本质是「少做、做对」：forced-wrong-fill 后被接受域会比当前 printed-fill 更差，
   提升来自把零 EV 大众样本弃单、只保留高 q+大 room 的少数样本——会牺牲成交量。
```

---

## 8. 与 MODELING.md 的衔接

```text
修正：§4.3 G_w≡1 必须同时用于策略 EV 和 realized PnL；wrong 样本强制成交（L0/L7）。
补齐：§3/§6 的 abstain（本优化 L1）——MODELING.md 的核心机制，代码漏实现。
更新：calibration 固定 7 天，7 天之前全部 predicted-side rows 进入 fit_all_side；threshold_accepted 最后过滤（L2）。
新增：主线改为 direct EV/PNL surface，用全部 predicted-side rows 训练（L3）；q+Gc 仅为组件备选。
强化：§4.1 q 现在做校准（L5，MODELING.md §10 已列为已知局限）。
强化：§4.2 Gc 加偏差校正+时间加权（L4，组件 B 专用），修 §10 假设(2)的乐观。
不变：§5 数据重建口径正确（无 look-ahead），保留；但需补 O1 数据完整性体检。
```
