# Safe Lowest Price Gap：降低 covered gap 的实现方案（实施文档 v1）

> 状态：待实施（仅设计，不含代码改动）
> 日期：2026-06-18
> 适用实验：`price_estimator/safe_lowest_price_gap`（基线脚本）与 `experiments/20260618_safe_lowest_price_gap_fix123`
> 关联设计：[safe_lowest_price_gap_objective_selective_design.md](safe_lowest_price_gap_objective_selective_design.md)

本文把前期诊断结论固化为**可执行的改动清单**。每项改动都给出：动机证据、目标文件与位置、当前值→目标值、预期效果、风险与回滚。
**本文不修改任何代码**，供后续在有数据的环境（Windows / Databricks）实施与重训。

---

## 0. 一句话结论

```text
covered_gap_norm ≈ 0.73 是“模型分辨率天花板”，不是损失/校准没调好。
最大可控杠杆是“降低边际覆盖”（90%→70% 直接降 gap ~0.19）。
其次是“给 FIT 更多、更新鲜的数据”（缩短 calibration tail）。
换损失、加 scale 头、堆同质特征都不是杠杆。
infeasible(26~28%) 样本最终用 p_side 兜底，不丢弃。
```

---

## 1. 背景与当前指标

当前 fix123 在 validation 上（口径：cap 到 p_side、feasible 子集）：

| 指标 | 值 |
|---|---|
| coverage_feasible | 0.907 |
| covered_gap_norm_mean | **0.734** |
| covered_gap_norm_median | 0.801 |
| clamp_over_pside 占比 | 37% |
| abstain 占比 | 0%（机制未触发） |

目标：在可接受的覆盖率下，显著降低 `covered_gap_norm = (p_pred - y) / (p_side - y)`。

---

## 2. 根因证据（决定方案的关键事实）

| 编号 | 证据 | 数值 |
|---|---|---|
| E1 | 预测误差相对房间过大（分辨率天花板） | `sd(f-y)=0.16`，`mean room s=0.247`，比值 **0.65** |
| E2 | 90% 覆盖的边际“税”重 | 90% 覆盖 ≈ `1.28·sd ≈ 0.21` = 房间的 84% |
| E3 | f 对 y 的解释力有限，且随“最低价出现时刻”衰减 | `corr(f,y)=0.65`；前 30s 0.85、120–240s 0.51 |
| E4 | 标签结构：40% 最低价在前 30s，34% 在最后 2 分钟 | 见 §8 数据画像 |
| E5 | FIT 比 CALIB 还小，且时间陈旧 | FIT 7421(47%)、CALIB 8236(53%)；FIT 截止 03-10，VALID 04-11→05-10 |
| E6 | infeasible 几乎全是真实的，非 buffer 造成 | infeasible 26–28%，其中 raw≥p_side 占 24.5–26.7%，buffer 仅 1.4–1.6% |
| E7 | 覆盖率是最大杠杆 | 90%→0.767, 80%→0.658, 75%→0.623, **70%→0.576** |
| E8 | signal filter 对 gap 影响很小 | accepted-only gap 0.751 vs all 0.767（仅 1.6 点） |
| E9 | scale 头是最差的选择性排序信号 | 30% 子集：房间代理 gap 0.63，scale 代理 gap **0.83** |
| E10 | 现 infer 的插值会硬编码 gap 地板 | `raw=f+d·(p_side−f)`，完美模型下 `gap≡d`（中位 0.41） |
| E11 | 归一化 1/s 损失被小房间样本绑架 | normalized 残差 std=14，min −461，max +593 |
| E12 | 指标口径把 abstain/clamp 计入（gap=1） | 占 covered 行 22%；headline 0.735 vs active-only 0.655 |

---

## 3. 实施总览（按性价比排序）

| 优先级 | 变更 | 类型 | 预期 gap 影响 | 风险 |
|---|---|---|---|---|
| P0 | **A. 缩短 calibration tail（31→7）** | config | 间接（更准/更稳） | 低 |
| P0 | **B. 覆盖目标 90%→70%** | config | **直接 −0.19** | 业务需接受 |
| P1 | **C. 启用 sample_filter=accepted_signal** | config | −0.02，主要为对齐部署 | 低 |
| P1 | **D. infeasible→p_side + active-only 指标口径** | code | 口径正确化，启用 abstain | 中 |
| P2 | **E. 修复 infer 边际（去插值地板）** | code | 结构性，去除 0.41 地板 | 中 |
| P2 | **F. 损失数值稳健化（s_floor↑/降权）** | code | 训练更稳 | 低 |
| P3 | **G. 全 feature pack 一次性试验** | data+config | 期望 ~0.02（旁路） | 中 |
| — | **不做：双头 location+scale** | — | 实测更差 | — |

---

## 4. 变更详情

### 变更 A（P0）：缩短 calibration tail，把数据还给 FIT

- **动机**：E5。校准只需 ~1–2k 行定 1 个 `delta` + 36 桶 miss-rate 表 + 选 alpha；当前用 8236 行（>50%）是浪费，且导致 f 训练截止 03-10、上线已 stale 一个月。
- **文件**：`experiments/20260618_safe_lowest_price_gap_fix123/config.yaml`，`split` 段。
- **改动**：
  ```yaml
  split:
    calibration_tail_days: 31   # → 改为 7（或网格 {7, 10, 14}）
    timestamp_column: timestamp
  ```
- **预期**：FIT 从 7421 → ~13000+（+80%），训练数据逼近到 ~04-03，离 validation 仅 ~8 天。
- **校验**：训练日志里打印 fit/calib 行数与时间窗（`fit_window` / `calibration_window` 已在 summary 输出）。确保 calib feasible 行数仍 ≥ ~1500，足以稳定取分位与桶表。
- **风险/回滚**：若 calib 太小导致 `delta` 抖动或某些桶 < `min_bucket_count(30)` → 适度回调到 10–14 天。

### 变更 B（P0）：把边际覆盖目标降到 70%

- **动机**：E2、E7。这是单个最大的 gap 杠杆。
- **文件**：同 config，`objective` 段。
- **改动**：
  ```yaml
  objective:
    optimize_metric: covered_gap_norm_mean
    hard_constraint: coverage_feasible
    min_feasible_coverage: 0.90              # → 0.70
    min_calibration_feasible_coverage: 0.90  # → 0.70
    calibration_coverage_buffer: 0.03        # 保留（calib 端略高于 val 目标，抗漂移）
    max_non_active_share: 0.45               # 视情况放宽/保留
    tie_breaker_metric: covered_gap_norm_median
  ```
- **预期**（validation，全量 feasible，实测）：

  | 目标覆盖 | 实际覆盖 | gap_mean | gap_median |
  |---|---|---|---|
  | 90% | 0.917 | 0.767 | 0.862 |
  | 80% | 0.819 | 0.658 | 0.687 |
  | 75% | 0.772 | 0.623 | 0.632 |
  | **70%** | 0.719 | **0.576** | 0.569 |

- **建议**：把 `delta_quantiles` 网格下沿补到 0.65–0.70 区间，让校准能选到对应 70% 覆盖的 `delta`。当前网格最小 0.50，够用，但可加 `0.65, 0.68`。
- **风险**：70% 是业务可接受下限（用户确认）。低于此不做。

### 变更 C（P1）：启用 accepted_signal 样本过滤（对齐部署）

- **动机**：E8。fix123 当前**无** `sample_filter`，涨跌模型 abstain 的样本进入了训练/评估，与真实部署口径不一致（真实部署只在涨跌模型接受时才出价）。gap 收益小，但**口径正确性**重要。
- **文件**：同 config，新增顶层 `sample_filter` 段（参考 `experiment/20260614_quantile_signal_filter/config.yaml`）。
- **改动**：
  ```yaml
  sample_filter:
    enabled: true
    mode: accepted_signal
    signal:
      t_up: null      # null → 取 deploy manifest 的 t_up=0.5792857
      t_down: null    # null → 取 deploy manifest 的 t_down=0.4314286
  ```
- **机制**：`apply_sample_filter()`（`scripts/price_estimator_common.py:102`）按 `p_up>=t_up | p_up<=t_down` 保留，retention ≈ 70%。train 与 validation 都过滤。
- **预期**：accepted 子集 feasible 77.5%（vs 全量 74%），房间 0.253。gap 0.767→0.751。
- **风险**：训练样本量再降 ~30%，需与变更 A 一起评估（A 增、C 减，净额需看日志）。

### 变更 D（P1）：infeasible 兜底为 p_side + 指标改 active-only 口径

分两部分。

**D1. infeasible 处理（明确化）**
- **动机**：E6。infeasible（`target_safe ≥ p_side`）占 26–28%，几乎全真实，永远无法 covered。
- **决策**：**评估/部署一律输出 `p_pred = p_side`，不丢弃。**
  - 不丢弃的理由：真实交易照样遇到这些样本，丢弃会虚高指标；p_side 是最安全合法兜底。
  - 覆盖率约束只算 **feasible 子集**（`coverage_feasible`，代码现状已正确：`metric_summary` 用 `covered[feasible]`）。
- **现状核对**：`infer_prices()` 已对 `raw>=p_side` 走 `clamp_over_pside→p_side`，infeasible 多数自然落入。**无需新增逻辑**，但需在报告中单列 infeasible 占比与 `max_possible_coverage`（已有）。
- **训练注意**：infeasible 样本**只可**在 quantile/pinball 类损失下保留；**不可**进当前 1/s 归一化损失（见 E11、变更 F）。

**D2. 指标口径：报告 active-only gap，把 abstain/clamp 从 gap 均值剔除**
- **动机**：E12。当前 `covered_gap_norm_mean` 把 abstain/clamp（gap=1）计入，既虚高 headline（0.735 vs active-only 0.655），又让优化器**永不 abstain**（abstain 在该指标里=惩罚），使 §6 选择性机制变死代码。
- **文件**：`train_safe_lowest_price_gap.py`
  - `metric_summary()`（约 L183–L240）：新增/改为同时输出
    - `active_covered_gap_norm_mean`（仅 `action=='active' & covered & feasible`）—— 设为**主优化指标**；
    - 保留 `covered_gap_norm_mean`（全口径）作参考。
  - `candidate_key()`（约 L330）：把排序主键从 `covered_gap_norm_mean` 换成 `active_covered_gap_norm_mean`；约束仍是 `coverage_feasible >= min` 与 `non_active_share <= max`。
  - `config.objective.optimize_metric`：注释说明主指标改为 active-only。
- **预期**：abstain/clamp 不再污染主指标；优化器会在“没把握”的样本上主动 abstain→p_side，把 active 主体报得更紧。实测：active 出价 70% 子集，gap 可到 ~0.63（按房间代理排序，见变更 E 备注）。
- **风险**：需同时设 `non_active_share` 上限（如 ≤0.45），避免“只对极少数样本出价”刷低 gap。已有 `max_non_active_share`，保留即可。

### 变更 E（P2）：修复 infer 边际，去掉插值硬地板

- **动机**：E10。现 infer 为
  ```python
  s_proxy = np.clip(p_side - f, s_floor, None)
  raw = f + delta_norm * s_proxy            # = (1-d)·f + d·p_side
  ```
  对完美模型 `f=y`，`gap_norm ≡ delta_norm`（实测中位 0.41）——**硬编码 gap 地板**；且 `s_proxy=p_side−f` 不是真实房间。
- **文件**：`train_safe_lowest_price_gap.py`，`infer_prices()`（L155–L178）与 `select_calibration_candidate()`（L361–L410）。
- **改动方案（二选一）**：
  - **E-方案1（推荐，简单）**：回到**加性边际**但**分桶**校准——
    `p_pred = ceil_to_tick(f + delta_bucket)`，`delta_bucket = quantile_{cov}(y - f | bucket)`，桶用 `p_side_bucket × market_time_bucket`。消除插值地板，且按局部残差尺度自适应。
  - **E-方案2**：保留单标量但改为**纯加性** `raw = f + delta`，`delta = quantile_{cov}(y - f)`（全局）。最简单，去地板，但不自适应房间。
- **预期**：去掉 0.41 的结构性地板；与变更 B（70% 覆盖）叠加后，active gap 进一步下探。
- **校验**：对比 E-方案1/2 与现插值在相同 `coverage_feasible` 下的 active gap，取低者。
- **风险**：改 infer 同时要改 calibration 里 `delta` 的定义（从 normalized 残差分位→对应方案的残差分位），两处需一致，避免 train/infer 口径错位。

### 变更 F（P2）：损失数值稳健化

- **动机**：E11。`normalized_asym_loss()`（L119–L137）用 `rn=r/s`，`s_floor=0.02`，小房间样本（11% 有 s<0.05）权重 ~1/s² 爆炸，normalized 残差 std=14。
- **文件**：`config.yaml` `loss.s_floor`，可选改 `train_safe_lowest_price_gap.py` 损失。
- **改动**：
  - `loss.s_floor: 0.02 → 0.05`（或 0.08）；或
  - 对 `s < 0.05` 样本在损失里降权（乘 `clip(s/0.05, 0, 1)`），因为它们推理时必被 clamp，无需精拟合。
- **预期**：训练梯度更稳，主体样本拟合不被极端项带偏。
- **风险**：s_floor 过大会削弱对窄房间的贴合；在 {0.05, 0.08} 小网格选。

### 变更 G（P3，旁路实验）：加载全部 feature pack

- **动机**：用户历史经验加秒级特征仅 ~0.02；但仍应一次性试满，确认是否还有空间。
- **可用包**：
  - 一级 31 个：`src/features/registry.py:FEATURE_PACKS`。
  - 二级 12 个：`src/data/second_level_feature_packs.py:SECOND_LEVEL_FEATURE_PACKS`。
- **实施**：在特征构建/ manifest 层启用全部包重建训练集（数据侧流程，非本模型脚本），再训练对比。
- **管理预期**：E1/E3 表明这是天花板问题，期望提升有限（~0.02 级别）。**优先做能真正降 `sd(f-y)` 的“面向最低价”的方向性特征**（下行已实现波动、订单流卖压、前 30s 回撤），而非堆同质包。
- **定位**：作为独立实验分支，不阻塞 P0/P1。

### 明确不做：双头 location + scale MLP

- **动机**：E9。scale 头唯一用途是按不确定度做选择性 abstain，但实测它是**最差**的排序信号（30% 子集 gap 0.83，房间代理只要 0.63）。残差 `|f-y|` 在可观测信号上仅弱异方差（std 0.11–0.15）。gap 由**房间 s** 主导，不是不确定度。
- **结论**：选择性排序用免费的 `p_side - f`（房间代理）或 `f/p_side`（利用率）即可，**不引入第二个头**。

---

## 5. 验收标准

实验在 validation 上报告，需满足：

```text
[硬约束]
- coverage_feasible >= 0.70（变更 B 后的目标）
- side_violation_rate == 0（p_pred <= p_side 恒成立）

[主指标]
- active_covered_gap_norm_mean 显著低于当前 0.655（目标 <= 0.60，期望 ~0.55）
- 全口径 covered_gap_norm_mean 作为参考一并报告

[健康度]
- non_active_share <= 0.45（abstain+clamp 不喧宾夺主）
- infeasible 占比、max_possible_coverage 单列
- fit_window / calibration_window 行数与时间窗，确认 FIT 数据已增大、变新鲜
```

时序无泄漏：保持 `forbidden_columns`，`y/s` 派生量禁作特征；三段切分不打乱。

---

## 6. 建议实验矩阵（在 Databricks 重训）

| 实验 | calib_tail | min_cov | sample_filter | infer 边际 | 指标 | 目的 |
|---|---|---|---|---|---|---|
| R0（基线复跑） | 31 | 0.90 | off | 现插值 | 全口径 | 对齐现状 |
| R1 | **7** | 0.90 | off | 现插值 | 全口径 | 验证变更 A 单独效果 |
| R2 | 7 | **0.70** | off | 现插值 | active-only | 验证变更 B 主杠杆 |
| R3 | 7 | 0.70 | **on** | 现插值 | active-only | 叠加 C 口径对齐 |
| R4 | 7 | 0.70 | on | **E-方案1 分桶加性** | active-only | 去插值地板 |
| R5 | 7 | 0.70 | on | E-方案1 | active-only + **s_floor 0.05** | 叠加 F |

逐步叠加，定位每项真实贡献。R2 应能看到 gap 大幅下降（~0.58 量级）。

---

## 7. 执行与回滚

```text
执行（推荐顺序）：
1. 先改 config（变更 A/B/C）→ 同步到 Databricks → 跑 R1/R2/R3，确认主杠杆。
2. 再改脚本（变更 D 指标口径、E infer、F 损失）→ 跑 R4/R5。
3. 每步对比 summary_metrics.json 的 active/all gap 与 coverage_feasible。

数据位置：训练/验证 parquet 在 Windows/Databricks 侧；本地仅有预测产物，无法本地重训。

回滚：所有 config 改动可逆；脚本改动建议在新实验目录（如 20260619_*）下进行，
     保留 fix123 作为对照，不覆盖。
```

---

## 8. 附：标签数据画像（extracted_labels.csv，23089 行）

```text
lowest_trade_price_next4（原始 y）：mean 0.40, median 0.40, p10 0.11, p90 0.69
  - 仅 1.7% ≤ 0.02、4.3% ≤ 0.05 → 标签多为温和回撤，并非经常崩到 0。

time_to_lowest_trade_sec（4 分钟=240s 内最低价出现时刻）：median 52s
  - [0,30)س: 40.0%   [30,60): 12.9%   [60,120): 18.5%   [120,180): 13.1%   [180,240): 15.0%
  - 含义：40% 最低价在前 30s（较可预测，corr 0.85），
          ~34% 在最后 2 分钟（接近随机游走，corr 0.51）→ 分辨率天花板的来源。

房间 s = p_side - target_safe（feasible）：mean 0.247, median 0.224
  - 11% 的 feasible 行 s < 0.05（窄房间，主导归一化损失数值问题）。

ORACLE（已知 y）covered gap：mean 0.038 —— 仅 tick 粒度造成。
  → 0.038 到 0.73 之间的差距 = 预测分辨率损失，是主战场。
```

---

## 9. 变更—文件 速查

| 变更 | 文件 | 位置/键 |
|---|---|---|
| A | `experiments/.../fix123/config.yaml` | `split.calibration_tail_days` |
| B | 同上 | `objective.min_feasible_coverage` / `min_calibration_feasible_coverage` / `calibration.delta_quantiles` |
| C | 同上 | 新增顶层 `sample_filter` |
| D1 | `train_safe_lowest_price_gap.py` | `infer_prices()` L155、`metric_summary()` L183 |
| D2 | 同上 | `metric_summary()` 新增 `active_covered_gap_norm_mean`、`candidate_key()` L330 |
| E | 同上 | `infer_prices()` L155–178、`select_calibration_candidate()` L361–410 |
| F | `config.yaml` + 脚本 | `loss.s_floor`、`normalized_asym_loss()` L119 |
| G | 数据构建流程 + manifest | `src/features/registry.py`、`src/data/second_level_feature_packs.py` |

> 不做：双头 scale MLP（实测劣于房间代理排序）。
