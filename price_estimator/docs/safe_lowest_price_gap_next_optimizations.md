# Safe Lowest Price Gap：待办优化清单（精简版）

> 日期：2026-06-18
> 当前最佳实验：`experiments/20260618_r2_cov70_active`（active gap **0.516** @ coverage_feasible **0.747**）
> 本文只保留**仍需执行**的优化；已完成项（缩短 calibration tail、覆盖目标 70%、active-only 指标、infeasible→p_side、全量 feature pack）不再列出。
> 仅设计，不改代码；需在有数据的环境（Windows / Databricks）实施重训。

---

## 1. 结论（durable facts）

```text
1) covered_gap_norm 的根因是“模型分辨率天花板”：sd(f - y) ≈ 0.16，约为平均房间 s≈0.25 的 65%。
   90% 覆盖需要的边际 ≈ 1.28·sd ≈ 0.21 = 房间的 84% → gap 被垫高。
2) 覆盖率是最大杠杆（已用）：90%→gap 0.73，70%→active gap 0.52。
3) f 系统性高于真实最低价 y：median(f - y | feasible) ≈ +0.13（R2 诊断同样显示）。
   → 这是 alpha 偏高的直接证据，也是仍可压低 gap 的主入口。
4) 选中 alpha=0.5 处在网格 [0.5,1,1.5,2] 的下边界，且 alpha≥1.0 全部退化
   （f 被推到 ≥ p_side → 全 clamp → gap=1、cov=1）。必须向更低 alpha 搜索。
5) 归一化损失用的 s 代理（p_side - f，clip 到 0.02）不稳：proxy 归一化残差 std≈6.1，
   真实 s 归一化残差 std≈1.4；窄房间样本主导梯度。
6) 模型已逐 p_side 桶算出 local_delta_norm（随桶从 -17 到 +0.5 剧烈变化），
   但推理只用了单一全局 delta_norm=-0.49 → 边际与样本异质性不匹配。
```

---

## 2. 待办优化（按性价比排序）

### H（P0，最高优先）：继续向更低 alpha 搜索

- **动机**：结论 3、4。选中 alpha 在网格下边界，f 仍高于 y；更低 alpha 让 f 自然下移逼近 y，base 报价更紧 → gap 更低，并避开高 alpha 的退化解。
- **文件**：`experiments/<new>/config.yaml`，`loss.alpha_grid`。
- **改动**：
  ```yaml
  loss:
    alpha_grid:
      - 0.01
      - 0.05
      - 0.1
      - 0.2
      - 0.3
      - 0.5
  ```
- **预期**：cov70 下 active gap 从 0.516 进一步下探（目标 < 0.50）。
- **校验**：
  - `epoch_metrics.csv` 中不应再出现 `coverage=1.0 & gap=1.0` 的退化 alpha 行；
  - 选中 alpha 应落在网格**内部**（若仍选 0.01 边界，继续向 0.005 探）；
  - 同步看 `median(f - y | feasible)` 是否从 +0.13 收敛向 0。
- **风险**：alpha 过低→欠预测惩罚不足→f 可能低于 y 过多致覆盖下滑；由校准 delta 与 70% 目标兜回。

### I（P1）：抑制过拟合

- **动机**：编码特征 **1028** 维 / FIT **13897** 行 ≈ 1:13.5，MLP 过参数化（已加全量 feature pack）；fit→val active gap 泛化差 +0.041。
- **文件**：`config.yaml`（特征清单、`model.dropout`、`training.weight_decay`）+ 数据侧特征构建。
- **改动（组合）**：
  1. **特征精选**：用一次性重要性（CatBoost / 置换重要性）把 1028 → ~200–300 维，优先砍同质 term/lag/rolling_z 包；
  2. **加正则**：`training.weight_decay 1e-4 → 1e-3`；`model.dropout [0.10,0.05,0.00] → [0.20,0.10,0.05]`。
- **预期**：缩小泛化差（目标 val−fit active gap < 0.03），让低 alpha 的增益在验证集上可信。
- **校验**：监控 `val_active_gap − fit_active_gap`；val active gap 不得上升（上升说明砍过头）。

### E（P2）：把全局 delta 换成逐桶（per-p_side-bin）加性边际

- **动机**：结论 6。模型已算出 `per_pside_bin_delta_norm`，但推理用单一全局 delta；逐桶边际更贴合各桶残差尺度，并消除“插值地板”（`raw=f+d·(p_side−f)` 对完美模型把 gap 钉在 d）。
- **文件**：`train_safe_lowest_price_gap.py`，`infer_prices()` 与 `select_calibration_candidate()`。
- **改动**：推理改为按样本的 p_side 桶取 `local_delta_norm`（加性、按真实 s 缩放），而非全局 `delta_norm`；校准端同步按桶选分位以满足整体 coverage_feasible≥0.70。
- **预期**：低 p_side / 窄房间桶覆盖更均匀，clamp 占比下降，active gap 再降。
- **风险**：桶样本量小（如低 p_side 桶 feasible 仅个位数）→ 估计噪声大；对 `feasible_count < min_bucket_count` 的桶回退全局 delta。

### F（P2）：损失数值稳健化（修小房间绑架）

- **动机**：结论 5。`s_floor=0.02` 过小，归一化 1/s 在窄房间样本上爆炸（proxy 归一化残差 std≈6.1）。
- **文件**：`config.yaml` `loss.s_floor`，可选改 `normalized_asym_loss()`。
- **改动**：
  - `loss.s_floor: 0.02 → 0.05`（小网格 {0.05, 0.08}）；或
  - 对 `s < 0.05` 样本在损失里降权（乘 `clip(s/0.05, 0, 1)`），因其推理时必被 clamp，无需精拟合。
- **预期**：梯度更稳，主体样本拟合不被极端项带偏，配合 H 更易收敛到低 gap。
- **风险**：s_floor 过大削弱对窄房间贴合；在 {0.05, 0.08} 小网格选。

### C（P2，评估项）：signal_filter 的干净 A/B

- **动机**：R3（含 accepted_signal 过滤）active gap 看似更低，但 abstain 占比升高、训练数据被砍 30% 致过拟合最重，增益**有水分**；需在低 alpha 基线上做干净对照判断是否保留。
- **改动**：以 H 的低 alpha 配置为基线，跑 `sample_filter` on / off 两版，其余一致。
- **判据**：若 on 版在**相同 coverage_feasible** 下 active gap 更低**且** val−fit 泛化差不恶化，则保留；否则关闭（更偏向关闭以保数据量）。

---

## 3. 下一批实验矩阵（仅向前）

| 实验 | 基于 | 变更 | 关键问题 |
|---|---|---|---|
| **R6** | R2 | H：alpha_grid=[0.01,0.05,0.1,0.2,0.3,0.5] | 选中 alpha 是否在内部？active gap 破 0.50？ |
| **R7** | R6 | I：weight_decay↑/dropout↑（或精选特征） | val−fit 泛化差 <0.03 且 gap 不升？ |
| **R8** | R7 | E：逐桶加性 delta | 低 p_side 桶覆盖更均匀、clamp 下降？ |
| **R9** | R7 | F：s_floor 0.05 + 小房间降权 | 训练更稳、gap 再降？ |
| **R10** | R6 | C：signal_filter on/off 对照 | filter 是否真增益（去 R3 水分）？ |

> 逐步叠加，定位每项真实贡献。R6 是关键一步：alpha 仍在边界，最有可能立刻见效。

---

## 4. 验收标准

```text
[硬约束]
- coverage_feasible >= 0.70
- side_violation_rate == 0（p_pred <= p_side 恒成立）
[主指标]
- active_covered_gap_norm_mean 显著低于当前 0.516（目标 <= 0.48）
- 全口径 covered_gap_norm_mean 一并报告（参考）
[健康度]
- non_active_share <= 0.45（abstain+clamp 不喧宾夺主）
- val_active_gap − fit_active_gap < 0.03（泛化）
- 选中 alpha 落在网格内部；无 coverage=1.0&gap=1.0 退化行
- infeasible 占比、max_possible_coverage 单列；infeasible 一律报 p_side
[无泄漏]
- 保持 forbidden_columns；y/s 派生量禁作特征；时间三段切分不打乱
```

---

## 5. 变更—文件 速查

| 变更 | 文件 | 位置/键 |
|---|---|---|
| H | `experiments/<new>/config.yaml` | `loss.alpha_grid` |
| I | `config.yaml` + 数据侧 | `training.weight_decay`、`model.dropout`、特征清单 |
| E | `train_safe_lowest_price_gap.py` | `infer_prices()`、`select_calibration_candidate()`（逐桶 delta） |
| F | `config.yaml` + 脚本 | `loss.s_floor`、`normalized_asym_loss()` |
| C | `config.yaml` | 顶层 `sample_filter`（on/off 对照） |
