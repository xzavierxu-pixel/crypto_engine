# 研究计划：高 Gc 成交门槛 / 市价单 / 止损止盈三方向优化

日期：2026-07-06
面向执行者：Arbor
主指标：冻结 `B_test sum_pnl`。当前 anchor = `42.43`。

## 0. 背景与 baseline

当前最好的、可验证的冻结 B_test 结果来自 `20260703_prefinal_rolling`：

- 策略：`q = raw_tree_blend`、`Gc floor = 0.85`、`min_ev = 0.02`、`q shrink = 0.0`。
- 冻结 B_test `sum_pnl = 42.43`（相对旧 anchor `27.44`）。
- 代码入口：`.arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py`
  - `prepare(train_path, dev_path, hazard_path, seed)` 返回 `(dev, accepted_dev, q_dict, gc, grid, calibration, n_features)`。
  - `evaluate(prepared, q_name, shrink, floor, min_ev)` 返回 backtest metrics。
  - 底层：`choose_survival_expected_return_bids(q, gc, grid, min_bid=0.01, min_ev, min_fill_probability=floor)`、`backtest_with_bid(accepted, bid, ev, fill)`、`backtest_metrics(accepted, result, n_dev)`。
- anchor 复现表达式：`evaluate(prepared, "raw_tree_blend", 0.0, 0.85, 0.02)`，参考 `.arbor/sessions/20260703_prefinal_rolling/diagnose_tested_btest_policy.py`。

本计划以该 anchor 为唯一 baseline。三个研究方向分别改的是：成交门槛（方向一）、订单类型（方向二）、成交后的持仓管理（方向三）。

## 1. 固定评估协议（所有实验通用）

### 1.1 数据与切分

- 开发/滚动：`.arbor/sessions/20260703_prefinal_rolling/folds/w1..w6`，每个含 `data/train.parquet`、`data/dev.parquet`、`models/hazard_survival_cdf.pt`。
  - `w1-w4` = tune，用于搜索和选择候选。
  - `w5-w6` = untouched holdout gate，候选冻结后才看，不参与选择。
- 冻结 B_test（最终月 2026-04-11..2026-05-10）：
  - train: `price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet`
  - B_test: `price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet`
  - hazard: `price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt`
- trade 路径数据：`price_estimator/data/sell_taker_trades_daily/date=*.parquet`，schema：`condition_id`、`outcome`（`up`/`down`）、`price`、`trade_time`（UTC）。方向二、方向三都依赖它。

### 1.2 成交与 PnL 语义（限价单基线，不可静默修改）

$$
EV(b \mid X) = q(X)\cdot G_c(b\mid X)\cdot(1-b) - (1-q(X))\cdot b
$$

- 方向正确的限价单：只有 `winner_low = chosen_low <= b` 才成交，成交收益 `1-b`。
- 方向错误的已提交限价单：`wrong_fill_forced = 1.0`，一律成交，PnL `= -b`。
- 未成交行 PnL = 0。
- `chosen_low`、`chosen_low_trade_time`、`time_to_chosen_low_sec` 已在 target frame 中，语义见 `price_estimator/expected_return/build_expected_return_target.py`（窗口 `(decision_time, endDate]` 内 selected-outcome token 的最低成交价）。

方向二、方向三会显式引入新的成交/持仓语义，必须作为独立 track 报告，且必须同时给出 anchor 在同一 universe 下的对照值（见各方向的“公平对照”条款），不能用改语义后的数字直接和 `42.43` 混排名。

### 1.3 B_test 账本（每个实验强制）

沿用 `docs/trade l2 optimization-0705.md` 的规则：

1. 每个 named 实验必须输出 `B_dev`、`w1..w6`、`B_test` 三类结果。
2. 每个实验保存 `metrics_bdev.json`、`metrics_btest.json`、`predictions_btest.parquet`（至少含 sample id、decision_time、selected_side、q、bid 或 market_price、expected_ev、fill_probability 或 fill_flag、filled、correct、realized_pnl、以及本方向特有字段）。
3. 追加一行到本 session 的 `gc_market_stop_btest_ledger.csv`。
4. B_test 每个候选只跑一次；不得围绕 B_test 反复调参。若候选在 `w1-w4` 或 `w5-w6` 已失败，不跑 B_test，并写明原因。
5. 必须报告相对 anchor `42.43` 的 delta；记录 B_test 不等于 promotion。

### 1.4 必报指标

`accepted_count`、`order_count`、`order_coverage`、`trade_count`、`fill_rate`、`mean_accepted_pnl`、`mean_pnl_filled`、`mean_bid`（或 `mean_market_price`）、`win_pnl_sum`、`loss_pnl_sum`、`correct_fill_rate`、`realized_correct_fill_rate_submitted`、`wrong_submitted_count`、`avg_loser_cost`、`submitted_fill_calibration_gap`、`q_brier`、`gc_brier`。

### 1.5 防泄漏

- 决策时特征禁止包含：`target, future_*, abs_return, signed_return, stage1_target, stage2_target, stage1_sample_weight, chosen_low, chosen_low_trade_time, time_to_chosen_low_sec, correct, winner, pnl, endDate, condition_id, market_id, slug, outcome`，以及任何 `decision_time` 之后（方向二为 `market_t0 + 68s` 之后）的 trade 信息。
- `chosen_low` / 路径信息只能作为回测标签或成交后执行规则输入，不能作为下单决策特征。
- 每个实验输出 `leakage_check.json`，记录特征列与禁列的交集必须为空。
- 模型、校准表、阈值、止损参数只能在评估窗口之前的数据上拟合/选择；B_test 只读一次。

---

## 2. 方向一（G 系列）：只在 `Gc >= 0.90` 的价格里找最大 EV 的 bid

### 2.1 动机

用户要求下单有 90% 以上概率成交。机制上就是把 bid 搜索限制在 `Gc(b|X) >= 0.90` 的合法价格集合里，再在其中取 EV 最大的 bid。`choose_survival_expected_return_bids` 已有 `min_fill_probability` 参数（anchor 用 0.85），因此该门槛可直接实现。

关键风险：`Gc >= 0.90` 是“模型自认的成交概率”，不是实际成交率。历史记录显示 submitted fill calibration gap 常在 `0.17-0.25`，即模型 0.90 的 fill 在最终月可能只实现约 0.65-0.72。所以本方向必须同时验证“实际 submitted correct_fill_rate 是否真的接近 0.90”，否则 90% 门槛只是名义值。此外，抬高 fill 门槛会把 bid 推高，放大错误方向的 forced-fill 损失，收益并不必然改善——必须做 win/loss 分解。

### 2.2 节点

**G0 — anchor 复现（前置校验，不跑 B_test 排名）**
- 机制：`evaluate(prepared, "raw_tree_blend", 0.0, 0.85, 0.02)`，确认能复现 B_test `42.43`。
- 通过标准：复现值与 `42.43` 一致（允许浮点误差）。用于确认环境与数据一致，然后才做 G1+。

**G1 — 高 fill 门槛扫描**
- 机制：`q = raw_tree_blend`，`min_fill_probability(floor) ∈ {0.90, 0.925, 0.95}`，`min_ev ∈ {0.0, 0.01, 0.02, 0.03, 0.05}`。
- 选择：`w1-w4` robust（`sum - std`，参考 `run_rolling_policy_search.py`），要求至少 3 周相对 anchor 正 delta；`w5-w6` gate 要求 `holdout_sum > 0` 且两周都为正；通过后跑一次 B_test。
- 观察量：B_test `sum_pnl` 是否 > `42.43`；提交子集实际 `realized_correct_fill_rate_submitted` 是否 >= 0.85；`loss_pnl_sum` 变化。

**G2 — 高 fill 门槛 + 保守 q**
- 机制：在 G1 基础上加 `q shrink ∈ {0.0, 0.1, 0.2}`（`q = (1-shrink)*raw_tree_blend + shrink*0.5`）。测试“更强成交保证 + 更保守方向概率”是否降低 forced-loss 而不过度杀掉 winner。
- 选择与 gate 同 G1。

**G3 — 让 “0.90” 成为真实成交率（Gc 再校准）**
- 机制：先对 `Gc` 做无泄漏再校准，再施加 `>= 0.90` 门槛。再校准用已验证有效的 `p_side`-bin 经验 CDF blend（node 1.5 思路：0.025 宽 `p_side` bin，25% 权重经验 winner-low CDF 混入 hazard `Gc`），校准数据严格早于评估窗口。
- 观察量：再校准后 submitted `Gc` 的 Brier 与 calibration gap 是否下降；`>=0.90` 门槛下实际 fill 是否真的达到约 0.90；B_test `sum_pnl` vs `42.43`。
- 目的：区分“门槛无效”与“门槛有效但 Gc 过度乐观”。

### 2.3 公平对照与成功判据

- 对照：所有 G 节点都在与 anchor 完全相同的 accepted universe 上评估，唯一变化是 fill 门槛（和 G2/G3 的 q/Gc 处理）。
- 成功：B_test `sum_pnl > 42.43`，且 `realized_correct_fill_rate_submitted >= 0.85`，`loss_pnl_sum` 不显著恶化，`w5-w6` 为正。
- 明确负面结论也要记录：若 90% 门槛只是抬高 bid、放大 loser 损失，需在报告里给出 win/loss 分解证据。

---

## 3. 方向二（M 系列）：市价单下单（不考虑手续费）

### 3.1 机制与新成交语义

- 市价：`m` = 该 market 在 `market_t0 + 68s`（开盘后 1 分 08 秒）之前、selected-outcome token 的最后一个 trade 价格。
  - 从 `sell_taker_trades_daily` 按 `condition_id + selected outcome` join，过滤 `trade_time <= market_t0 + 68s`，取 `trade_time` 最大的一条的 `price`。
  - `market_t0` 来自 target frame（`build_expected_return_target.py` 已解析 `market_t0`）。若无 `market_t0`，用该 market 当日 UTC 5 分钟周期起点推导，并在报告写明推导方式。
- 市价单成交语义（无手续费）：市价单必成交于 `m`。
  - 方向正确：收益 `1 - m`。
  - 方向错误：损失 `-m`。
  - 因此 `EV_market = q*(1-m) - (1-q)*m = q - m`。
- 下单规则：`q - m > τ` 才下单，`τ` 为 EV 阈值，可扫描多个。

### 3.2 时间口径与泄漏（关键）

- 使用 `market_t0 + 68s` 之前的最后成交价，意味着决策实际发生在 1:08，比标准 1:00 决策晚 8 秒。本 track 必须把决策时间统一记为 `t = market_t0 + 68s`，并禁止任何 `t` 之后的 trade 进入 `m` 或任何特征。
- `q` 只能用 `t` 之前可得的信息。若沿用现有 `raw_tree_blend` / `p_side`（基于首分钟方向模型，决策在 1:00），需在报告中说明该 q 是 1:00 决策产物，与 1:08 市价组合属于允许的、有明确时间戳的组合，且不得反向使用 1:08 后的价格来改 q。

### 3.3 节点

**M0 — 构建 `m` 并做数据 QA（前置）**
- 产出：每个 accepted 行的 `m`、`m_trade_time`、`has_pre_108_trade`。
- 报告：`m` 覆盖率（有 1:08 前成交的比例）、`m` 分布、`m` 与 `p_side` 的关系、无 `m` 行的占比与处理（默认 abstain，不下单，并计入 coverage）。
- 校验：确认所有 `m_trade_time <= market_t0 + 68s`；无一条晚于门槛。

**M1 — EV 阈值扫描**
- 机制：`EV_market = q - m`，`τ ∈ {0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10}`，`q ∈ {raw_tree_blend, isotonic, raw p_side}`。
- 选择：`w1-w4` robust + `w5-w6` gate；通过后一次 B_test。
- 观察量：B_test `sum_pnl`、`order_coverage`、`win_pnl_sum` / `loss_pnl_sum`、实际方向准确率（因为市价单必成交，PnL 直接由 `q` 校准质量和 `m` 决定）。

**M2 — 与限价 anchor 的公平对照**
- 机制：在 “有合法 `m`” 的相同子集上，同时评估：(a) 市价 M1 最优候选；(b) 限价 anchor（raw_tree_blend/0.85/0.02）。
- 目的：市价单去掉了 winner-fill 的 adverse selection（必成交），但代价是按 `m` 成交而非低点、且错误方向也必成交。只有在同一 universe 上对照，才能判断市价单是否真的优于限价。

**M3 —（延伸，可选）限价/市价混合**
- 机制：当 `q - m` 边际很大且限价 fill 概率低时用市价单；否则用限价单。作为 stretch 节点，仅在 M1/M2 显示市价单有正 edge 时才做。

### 3.4 成功判据

- 成功：在 M2 的共同 universe 上，市价 track 的 B_test `sum_pnl` 高于限价 anchor（同 universe 值），且 `w5-w6` 为正。
- 需重点报告：`m` 覆盖率不足会缩小可交易 universe；若 `q` 在最终月校准漂移，`q - m > τ` 会系统性选到 loser（`m` 偏低但方向错），必须给出 wrong-order 占比与 `avg_loser_cost = mean(m | wrong & submitted)`。

---

## 4. 方向三（S 系列）：限价成交后叠加止损/止盈单

### 4.1 机制与新 PnL 结构

在方向一/anchor 的限价策略之上，成交后再挂一个市价止损/止盈单（无手续费），用 trade 路径判断是否触发。

- 入场：限价单在 `(decision_time, endDate]` 内出现 `price <= b` 的 trade 即成交。
  - `entry_time` = 首个满足 `price <= b` 的 `trade_time`；入场成本 = `b`。
- 止损位：`s = min(0.30, 0.50 * b)`（用户设定）。
- 入场后路径：`ell_post = min(price)`，取 `trade_time > entry_time` 且 `<= endDate` 的 selected-outcome trade。
- 结算规则：
  - 若 `ell_post <= s`：止损触发，按市价 `s` 卖出，realized PnL `= s - b`（对正确和错误方向都适用）。
  - 否则：持有到结算，正确方向 `= 1 - b`，错误方向 `= 0 - b = -b`。
- 直觉：错误方向 token 趋于 0，几乎必然穿过 `s`，止损把损失从 `-b` 收窄到 `s - b`；代价是部分先跌破 `s` 再回升到 1 的 winner 被提前止损，收益从 `1-b` 变成 `s-b`。净效果取决于两者比例，必须回测。

### 4.2 路径重建与执行现实

- 路径重建：对每个成交订单，用 `sell_taker_trades_daily` 中该 `condition_id + selected outcome`、`entry_time < trade_time <= endDate` 的 trade 序列计算入场后最低价及首次跌破 `s` 的成交价。
- 执行现实（必须报告，避免过度乐观）：
  1. 主口径：止损成交价 = `s`。
  2. 保守口径：止损成交价 = 首个 `price <= s` 的实际 trade 价（可能低于 `s`），体现滑点。
  3. 滑点压力：止损成交价 = `s - 0.02`。
  三种口径都要在 B_test 上给出 `sum_pnl`，用区间说明结论稳健性。
- 4 分钟窗口流动性有限，止损可能无法在 `s` 精确成交；因此保守口径与滑点压力是硬性要求，不是可选项。
- 泄漏：止损是成交后的执行规则，`ell_post`、`entry_time` 只能用于回测 PnL，绝不能作为下单决策特征。

### 4.3 节点

**S0 — 路径重建 + 固定止损回测（前置验证）**
- 机制：固定 anchor 限价策略（raw_tree_blend/0.85/0.02），叠加 `s = min(0.30, 0.50*b)`，主口径。
- 报告：相对 anchor 的 `sum_pnl`、`loss_pnl_sum`、`win_pnl_sum` 变化；被止损的错误订单数、被误止损的 winner 数（先跌破 `s` 后结算为 1）、平均止损 PnL。
- 目的：先确认机制方向对不对（loss 是否显著收窄、winner 是否被过度杀掉）。

**S1 — 止损位网格**
- 机制：`s = min(c1, c2 * b)`，`c1 ∈ {0.20, 0.30, 0.40}`，`c2 ∈ {0.40, 0.50, 0.60}`；另加固定止损 `s ∈ {0.20, 0.30}` 作对照。
- 选择：`w1-w4` robust + `w5-w6` gate；通过后一次 B_test，三种执行口径都记录。

**S2 — 止盈叠加**
- 机制：对正确方向 token 增加止盈：价格 `>= tp`（如 `tp ∈ {0.90, 0.95}`）时市价卖出锁定收益。测试是否有助（通常 winner 趋于 1，止盈多半有害，需要证据）。
- 选择与 gate 同 S1。

**S3 —（延伸）最优止损 + 方向一/方向二组合**
- 机制：若 S1 显示止损有稳定正收益，则把最优止损叠加到 G 系列高 fill 门槛策略，或叠加到 M 系列市价单（市价单入场 `m`，止损位 `s = min(0.30, 0.50*m)`）。仅在 S1 成功时执行。

### 4.4 公平对照与成功判据

- 公平对照：S 系列必须在与 anchor 完全相同的订单集合与相同入场成交判定上评估，唯一变化是“成交后是否止损/止盈”，使 delta 完全归因于持仓管理。
- 成功：至少在主口径与保守口径两种下 B_test `sum_pnl > 42.43`，`loss_pnl_sum` 明显下降，且 winner 被误杀带来的 `win_pnl_sum` 损失小于 loss 侧收益；滑点压力口径不出现符号翻转。

---

## 5. 建议执行顺序

1. G0 / M0 / S0 三个前置节点先跑：分别确认 anchor 复现、`m` 可构建且覆盖率可接受、止损路径可重建。任一前置失败则先修数据/接口，不进入排名节点。
2. 方向一 G1 → G3：门槛类改动最直接，先确认 90% 门槛的真实收益与真实成交率。
3. 方向三 S1 → S2：止损直接改 loss 结构，是历史瓶颈（forced-wrong-fill loss）最相关的方向。
4. 方向二 M1 → M2：市价单是最大的语义变化，需在共同 universe 上做公平对照后再判断。
5. 若单方向出现稳定 `> 42.43` 且 `w5-w6` 为正，再做 S3 组合节点。

## 6. 交付物（每个 named 实验）

- `REPORT.md`（做法、baseline vs 结果、B_test 绝对值与相对 `42.43` 的 delta、win/loss 分解、结论）。
- `config_used.yaml`、`feature_manifest.json`、`leakage_check.json`。
- `metrics_bdev.json`、`metrics_btest.json`、`predictions_btest.parquet`。
- 追加一行到 `gc_market_stop_btest_ledger.csv`。

## 7. 硬约束（不可违反）

- 不改 `2mins` 主分支、deploy artifact、live execution 配置；不做 promotion（需单独用户确认）。
- 不改限价基线的 winner-low 定义、forced-wrong-fill 规则、tick、订单 universe；方向二/方向三的新语义只在各自 track 内成立并单独报告。
- B_test 只读一次/候选，禁止用 B_test 选择任何参数。
- 所有拟合与阈值选择只用评估窗口之前的数据；每个实验必须通过 `leakage_check.json`。
- 三个方向的结果不得与 anchor `42.43` 混排名，除非提供同 universe / 同 fills 的公平对照值。
