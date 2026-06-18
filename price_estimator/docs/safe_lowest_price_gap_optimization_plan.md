# Safe Lowest Price Gap 优化方案（Fix 1–3 实施 + 诊断与分桶报告规范）

> 配套设计文档：`safe_lowest_price_gap_objective_selective_design.md`
> 代码与产物：`price_estimator/safe_lowest_price_gap/`
> 本文目标：把 validation `covered_gap_norm_mean` 从 **0.734** 降到 **~0.50–0.55**，
> 同时保持 `coverage_feasible ≥ 0.90`、`side_violation_rate = 0`。
> **本轮实施 Fix 1 / Fix 2 / Fix 3；Fix 4 推迟（仅留接口与说明）。**

---

## 0. 一句话方案

```text
病根：损失让 f 系统性高估 y（单层已覆盖 ~70%），校准再叠加一个“全局绝对 δ”，
      绝对 δ 对“房间 s 小”的样本是灾难（直接顶到 p_side，gap=1，占 ~37%）。

修法：
  Fix 1  把“全局绝对 δ”换成“随房间缩放的归一化边际” δ_norm·ŝ，
         其中 ŝ = clip(p_side − f, s_floor) 是【推理可见】的房间代理（不能用真实 s，s 含未来标签）。
  Fix 2  把损失两支统一到归一化空间（欠预测也 /s_eff），并调小 alpha，
         让 f 回到 y 的条件中位、把覆盖职责交还给 Fix 1 的边际 → 打破“双层边际”。
  Fix 3  解耦覆盖职责：calibration 覆盖阈 0.93→0.90；主选目标改为 active 子集 gap，
         clamp/abstain 占比作为单独预算约束，避免 gap=1 的兜底样本污染主指标。
  诊断   输出 (y−f)/s 的分布（true_s 与 proxy_ŝ 两版）与每个 p_side 桶的 δ_norm，
         该分布的离散度（q90−q50）≈ 可达 gap 的物理地板，直接预判优化天花板。
  报告   summary_metrics.json 增加“按 p_side 步长分 bin”的逐桶完整统计，便于定位。
```

---

## 1. 现状与根因（简要回顾）

当前 validation（选中配置 `alpha=4, delta=0.072 (q=0.92), bucket_miss_threshold=0.15`）：

| 指标 | calibration | validation |
|---|---|---|
| coverage_feasible | 0.936 | **0.907** ✓ |
| covered_gap_norm_mean | 0.745 | **0.734** ✗ |
| active 子集 gap | 0.634 | 0.629 |
| clamp_over_pside 占比 | 39% | **37%（全 gap=1）** |
| abstain 占比 | 0% | **0%（选择性机制未触发）** |

逐桶（来自 `reports/validation_pside_buckets.md`）：gap 与 p_side **反向**——低 p_side（小房间）桶 gap 0.91–0.92，高 p_side（大房间）桶 0.62。

根因（按影响排序）：

```text
R1 双层边际：alpha=4 的非对称损失已把 f 顶成 y 的高分位（median(f−y)=+0.053，δ=0 时覆盖≈70%），
            校准再叠加绝对 δ=0.072 → p_pred ≈ y + 0.125，gap 主体来源。
R2 量纲不一致：欠预测罚在绝对空间(alpha·|r|)，过预测罚在归一化空间(r/s_eff)。
            大 s 时过预测梯度被 s² 压没→f 上漂(大 gap)；小 s 时 1/s² 巨大但绝对 δ≫s→clamp(gap=1)。
R3 全局加性 δ 与异质 s 不匹配：对所有样本加同一 0.072，小 s 必 clamp、大 s 浪费。→ 37% clamp。
R4 选择性反向：covered_gap_norm_mean 把 clamp/abstain 到 p_side 的 feasible 样本以 gap=1 计入均值，
            “放弃越多 gap 越差”，与设计意图相反。
R5 天花板：即便只覆盖最易 52%，gap 仍 0.385（frontier 实测）→ 存在不可约地板，需 Fix 4。
```

Fix 1/2/3 攻 R1–R4（分配效率与量纲），R5 留给 Fix 4。

---

## 2. 策略总览

| Fix | 攻击根因 | 主要受益桶 | 本轮 | 核心改动文件 |
|---|---|---|---|---|
| **Fix 1** 归一化/缩放边际 `δ_norm·ŝ` | R3, R1 | 低 p_side（小房间） | ✅ | `infer_prices`、`select_calibration_candidate`、`config` |
| **Fix 2** 损失全归一化 + 降 alpha | R2, R1 | 全桶（小 s 最大） | ✅ | `normalized_asym_loss`、`config.loss.alpha_grid` |
| **Fix 3** 解耦覆盖 + 报告口径 | R4, R1 | 全桶 | ✅ | `candidate_key`、`select_*`、`config.objective` |
| 诊断 `(y−f)/s` 分布 + 每桶 δ_norm | 预判 R5 | — | ✅ | 新增函数 + `summary_metrics.json` |
| 分桶 summary（p_side 步长） | 可观测性 | — | ✅ | 新增函数 + `summary_metrics.json` |
| **Fix 4** 更强特征/逐样本不确定性 | R5 | 高 p_side（饱和）| ⏸️ 推迟 | （仅留说明） |

---

## 3. Fix 1：边际随房间 `ŝ` 缩放（关键：用推理可见的房间代理）

### 3.1 为什么不能直接乘真实 `s`

```text
gap_norm = (p_pred − y)/(p_side − y) = (p_pred − y)/s。
要让边际“随房间缩放”，自然想 p_pred = f + δ_norm·s。
但 s = p_side − y 含未来标签 y，【推理时不可见、禁作特征】（见设计文档 §1.3/§3.4）。
→ 必须用推理可见量构造房间代理：ŝ = clip(p_side − f, s_floor, +inf)。
  f 是模型输出（≈ y 的一个定位），p_side 推理已知 → ŝ 推理可算。
  Fix 2 降 alpha 后 f≈median(y)，ŝ≈p_side−median(y)，对真实 s 的代理更准（两 Fix 协同）。
```

### 3.2 校准与推理公式

```text
校准（仅 feasible 行，标签可见）：
  ŝ_cal   = clip(p_side − f, s_floor, +inf)
  r_norm  = (y − f) / ŝ_cal                       # 归一化残差
  δ_norm  = quantile_q( r_norm )                  # 在 q∈delta_quantiles 网格扫

推理（全样本，标签不可见）：
  ŝ       = clip(p_side − f, s_floor, +inf)
  raw     = f + δ_norm · ŝ
  p_pred  = min( ceil_to_tick(raw), p_side )      # 仍硬 cap + 向上取 tick
```

### 3.3 为什么这样能精确控覆盖、又压低 gap

```text
覆盖：covered ⇔ f + δ_norm·ŝ ≥ y ⇔ δ_norm ≥ (y−f)/ŝ = r_norm
     ⇒ P(covered|feasible) = P(δ_norm ≥ r_norm) = q。取 q=0.90 即恰好 90% 边际覆盖。
gap： 对 covered 样本 gap_norm ≈ δ_norm − r_norm_i（当 ŝ≈s）。
     ⇒ 平均 gap ≈ δ_norm − E[r_norm | covered]，本质等于 r_norm 的【离散度】。
     ⇒ 小房间样本拿到的是【小绝对边际】(δ_norm·小ŝ)，不再被绝对 δ 顶满 → clamp 暴跌、低桶 gap 骤降。
     ⇒ 这把“可达 gap 地板”直接等价于诊断段输出的 (y−f)/s 分布的 q90−q50（见 §6/§9）。
```

### 3.4 代码改动（`train_safe_lowest_price_gap.py`）

`infer_prices` 签名由“加性 delta”改为“归一化 delta_norm + s_floor”：

```python
def infer_prices(
    f, p_side, conf_ok, delta_norm, tick_size, tick_tol, s_floor,
) -> PredictionResult:
    f = np.asarray(f, dtype=float)
    p_side = np.asarray(p_side, dtype=float)
    conf_ok = np.asarray(conf_ok, dtype=bool)
    s_proxy = np.clip(p_side - f, s_floor, None)      # 推理可见房间代理
    raw = f + float(delta_norm) * s_proxy             # ← 缩放边际（核心改动）
    ticked = np.maximum(ceil_to_tick(raw, tick_size, tick_tol), 0.0)
    p_pred = np.minimum(ticked, p_side)
    action = np.full(len(f), "active", dtype=object)
    clamp = conf_ok & (raw >= p_side)
    action[clamp] = "clamp_over_pside"
    action[~conf_ok] = "abstain_low_conf"
    p_pred[~conf_ok] = p_side[~conf_ok]
    p_pred[clamp] = p_side[clamp]
    return PredictionResult(p_pred=p_pred, action=action.astype(str), conf_ok=conf_ok)
```

`select_calibration_candidate` 的 δ 生成改为归一化残差分位：

```python
    feasible = y_safe < p_side
    s_floor = float(config["loss"]["s_floor"])
    s_proxy = np.clip(p_side[feasible] - f_cal[feasible], s_floor, None)
    r_norm = (y_safe[feasible] - f_cal[feasible]) / s_proxy     # ← 归一化残差
    if len(r_norm) == 0:
        raise ValueError("No feasible calibration rows available")
    ...
    for q in [float(v) for v in config["calibration"]["delta_quantiles"]]:
        delta_norm = float(np.quantile(r_norm, q))             # ← δ_norm
        bucket_model = fit_bucket_model(calibration, y_safe, p_side, f_cal, delta_norm, config, tolerance)
        for threshold in ...:
            conf_ok = bucket_conf_ok(...)
            pred = infer_prices(f_cal, p_side, conf_ok, delta_norm, tick_size, tick_tol, s_floor)
            ...
```

> 同步：`fit_bucket_model`、`evaluate_with_candidate`、`train_one_alpha` 内对 `infer_prices` 的调用都要补 `s_floor`，
> 并把字段名 `delta` 统一更名为 `delta_norm`（含 `CandidateResult`、frontier 行、checkpoint、summary）。

### 3.5（可选增强）分桶 δ_norm

若单一全局 `δ_norm` 仍残留跨桶覆盖不均，可在 `p_side_bucket × market_time_bucket × selected_side` 上分桶取
`δ_norm_b = quantile_q(r_norm | bucket_b)`（桶样本不足回退全局）。**本轮先用全局 `δ_norm`**，分桶 δ_norm 作为
诊断输出先观察（§6），确认确有必要再启用，避免过拟合校准集。

---

## 4. Fix 2：损失全归一化 + 降 alpha（打破双层边际）

### 4.1 改动

把欠预测分支也搬到归一化空间（训练期 `y` 可见，`s_eff = clip(p_side − y, s_floor)` 是真实房间）：

```python
def normalized_asym_loss(z, y_safe, s_eff, alpha, c, kappa):
    f = torch.sigmoid(z)
    r = f - y_safe
    s = torch.clamp_min(s_eff, 1e-6)
    rn = r / s                                            # ← 残差先归一化
    under = float(alpha) * (torch.sqrt(rn * rn + float(c) * float(c)) - float(c))  # 归一化平滑-|rn|
    t = torch.clamp(rn, min=0.0)
    over_quad = 0.5 * t * t
    over_lin = float(kappa) * (t - 0.5 * float(kappa))
    over = torch.where(t <= float(kappa), over_quad, over_lin)
    return torch.where(r < 0.0, under, over).mean()
```

要点：

```text
- 两支同量纲(都 /s_eff)：欠/过预测在“归一化 gap”同一尺度比较 → f 成为 y 的一致归一化分位，
  不再随 s 增大而上漂(消 R2)。小 s 样本受益最大(此前被绝对量纲与绝对 δ 双杀)。
- 训练用真实 s_eff(标签可见)，与推理用 ŝ(代理)是不同角色：训练度量【真实归一化 gap】才正确，
  代理只用于推理边际。二者不冲突。
```

### 4.2 调小 alpha（把覆盖交还给 δ_norm）

```yaml
loss:
  alpha_grid: [0.5, 1.0, 1.5, 2.0]    # 原 [2,4,8]
```

```text
原 alpha=4 让 f 顶成高分位(单层覆盖 70%)，是 R1 的元凶。
调小后 f→y 的条件中位附近(覆盖≈50%)，剩余覆盖完全由 Fix 1 的 δ_norm·ŝ 提供 → 单层边际，gap 整体下移。
alpha 仍按小网格在 calibration 上选(§5/Fix3)。
```

---

## 5. Fix 3：解耦覆盖职责 + 修正报告/选点口径

### 5.1 calibration 覆盖阈下调

```yaml
objective:
  min_feasible_coverage: 0.90               # validation 硬约束（不变）
  min_calibration_feasible_coverage: 0.90   # 原 0.93 → 0.90
```

```text
0.93 会逼 δ 更高，多买的 margin 直接变成更大 gap。回到 0.90 与最终约束一致，省下的 margin 转为更小 gap。
```

### 5.2 主选目标改为 active 子集 gap + 占比预算

`metric_summary` 增补 active 子集 gap（仅 `action==active & covered_feasible`）：

```python
    active_cov = covered_feasible & (prediction.action == "active")
    active_gap = gap_norm[active_cov]
    metrics["active_covered_gap_norm_mean"]   = float(np.mean(active_gap)) if len(active_gap) else float("nan")
    metrics["active_covered_gap_norm_median"] = float(np.median(active_gap)) if len(active_gap) else float("nan")
    metrics["non_active_share"] = float(((prediction.action != "active")).mean()) if len(prediction.action) else float("nan")
```

`candidate_key` 改为：硬约束 `coverage_feasible ≥ 0.90 且 side_violation=0 且 non_active_share ≤ budget`，
目标最小化 `active_covered_gap_norm_mean`，tie-break 用 median 再用 `−covered_feasible_count`：

```python
def candidate_key(candidate, min_coverage, max_non_active_share):
    m = candidate.metrics
    valid = (
        m["coverage_feasible"] >= min_coverage
        and m["side_violation_rate"] == 0.0
        and m["non_active_share"] <= max_non_active_share
        and math.isfinite(m["active_covered_gap_norm_mean"])
    )
    if not valid:
        return (1.0, -m["coverage_feasible"], float("inf"), float("inf"), float("inf"))
    return (
        0.0,
        m["active_covered_gap_norm_mean"],
        m["active_covered_gap_norm_median"],
        m["non_active_share"],
        -m["covered_feasible_count"],
    )
```

```yaml
objective:
  max_non_active_share: 0.45    # clamp+abstain 占比预算（防止靠兜底刷覆盖）
```

```text
意义：不再让 gap=1 的 clamp/abstain 污染主选指标；同时用占比预算挡住“全靠 p_side 兜底”的退化解。
报告仍保留 overall covered_gap_norm_mean 作对照，但选点以 active 口径为准。
```

---

## 6. 诊断段：`(y−f)/s` 分布 + 每桶 `δ_norm`（本文重点要求之一）

### 6.1 目的

```text
(y−f)/s 的离散度 ≈ 可达 gap 的物理地板（§3.3）。先看清它，就能判断：
  - 现在的 gap 距离地板还有多少“分配损失”可挤（Fix 1–3 能拿回的部分）；
  - 地板本身有多高（只有 Fix 4 能再降的部分）。
同时输出 true_s 版与 proxy_ŝ 版，量化“房间代理”的偏差，验证 Fix 1 代理是否可靠。
```

### 6.2 实现（新增函数）

```python
def _dist_stats(vals: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"count": 0.0}
    out = {
        "count": float(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }
    for q in quantiles:
        out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(vals, q))
    out["spread_q90_q50"] = out.get("q90", float("nan")) - out.get("q50", float("nan"))
    return out


def normalized_residual_diagnostics(y_safe, p_side, f, s_floor, quantiles) -> dict[str, Any]:
    y = np.asarray(y_safe, float); p = np.asarray(p_side, float); ff = np.asarray(f, float)
    feasible = y < p
    yf, pf, fimf = y[feasible], p[feasible], ff[feasible]
    s_true  = np.clip(pf - yf,  s_floor, None)
    s_proxy = np.clip(pf - fimf, s_floor, None)
    r = yf - fimf
    return {
        "feasible_count": float(feasible.sum()),
        "abs_residual":         _dist_stats(r,           quantiles),
        "norm_residual_true_s": _dist_stats(r / s_true,  quantiles),   # 离线真实归一化
        "norm_residual_proxy_s":_dist_stats(r / s_proxy, quantiles),   # 推理实际所用
        "room_s_true":  _dist_stats(s_true,  quantiles),
        "room_s_proxy": _dist_stats(s_proxy, quantiles),
    }


def per_pside_bin_delta_norm(y_safe, p_side, f, edges, s_floor, delta_quantile) -> list[dict[str, float]]:
    y = np.asarray(y_safe, float); p = np.asarray(p_side, float); ff = np.asarray(f, float)
    feasible = y < p
    s_proxy = np.clip(p - ff, s_floor, None)
    r_norm = (y - ff) / s_proxy
    idx = np.digitize(p, edges, right=False)
    rows: list[dict[str, float]] = []
    for b in range(1, len(edges)):
        m = feasible & (idx == b)
        row = {"pside_bin": f"[{edges[b-1]:.2f}, {edges[b]:.2f})",
               "edge_lo": float(edges[b-1]), "edge_hi": float(edges[b]),
               "feasible_count": float(m.sum())}
        if m.any():
            row.update({
                "norm_residual_q50": float(np.quantile(r_norm[m], 0.50)),
                "norm_residual_q90": float(np.quantile(r_norm[m], 0.90)),
                "local_delta_norm": float(np.quantile(r_norm[m], delta_quantile)),  # 该桶若各自定边际
                "mean_room_s_true":  float(np.mean(np.clip(p[m] - y[m], s_floor, None))),
                "mean_room_s_proxy": float(np.mean(s_proxy[m])),
            })
        rows.append(row)
    return rows
```

### 6.3 写入 `summary_metrics.json` 的结构

```json
"normalized_residual_diagnostics": {
  "calibration": { "feasible_count": 6095,
    "abs_residual":          {"count":6095,"mean":...,"std":...,"q01":...,"q50":...,"q90":...,"spread_q90_q50":...},
    "norm_residual_true_s":  {"...":"...","spread_q90_q50": 0.41},
    "norm_residual_proxy_s": {"...":"...","spread_q90_q50": 0.46},
    "room_s_true": {...}, "room_s_proxy": {...} },
  "validation": { "...": "..." }
},
"per_pside_bin_delta_norm": {
  "calibration": [
    {"pside_bin":"[0.20, 0.25)","feasible_count":...,"norm_residual_q50":...,"norm_residual_q90":...,
     "local_delta_norm":...,"mean_room_s_true":...,"mean_room_s_proxy":...},
    "... 每个 p_side 步长桶一行 ..."
  ],
  "validation": [ "..." ]
}
```

> 解读规则：`norm_residual_proxy_s.spread_q90_q50` 就是“当前 f 下、用 90% 覆盖时 active 子集 gap 的近似地板”。
> 若它已 ≈ 现实 gap，说明 Fix 1–3 余量不大、必须上 Fix 4；若远小于现实 gap，说明分配损失大、Fix 1–3 收益高。

---

## 7. `summary_metrics.json`：按 p_side 步长分 bin 的逐桶完整统计（本文重点要求之二）

### 7.1 配置（统一步长，覆盖 0–1）

```yaml
diagnostics:
  pside_bin_step: 0.05                         # 统一步长（更细；可设 0.1）
  normalized_residual_quantiles: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
```

```python
def pside_bin_edges(config) -> list[float]:
    step = float(config["diagnostics"].get("pside_bin_step", 0.05))
    n = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(n + 1)]   # [0.0, step, ..., 1.0]
```

### 7.2 逐桶完整指标（复用 `metric_summary` + action 占比）

```python
def pside_bin_metrics(df, y_safe, p_side, prediction, edges, tolerance) -> list[dict[str, float]]:
    p = np.asarray(p_side, float)
    idx = np.digitize(p, edges, right=False)
    rows: list[dict[str, float]] = []
    for b in range(1, len(edges)):
        m = idx == b
        row = {"pside_bin": f"[{edges[b-1]:.2f}, {edges[b]:.2f})",
               "edge_lo": float(edges[b-1]), "edge_hi": float(edges[b]),
               "sample_count": float(m.sum())}
        if m.any():
            sub = PredictionResult(prediction.p_pred[m], prediction.action[m], prediction.conf_ok[m])
            row.update(metric_summary(y_safe[m], p[m], sub, tolerance))   # 全套覆盖/gap/action 指标
        rows.append(row)
    return rows
```

每个 split（fit / calibration / train / validation）各产一份，写入：

```json
"validation_pside_bin_metrics": [
  {"pside_bin":"[0.40, 0.45)","edge_lo":0.40,"edge_hi":0.45,
   "sample_count":..., "feasible_count":..., "infeasible_count":...,
   "coverage_overall":..., "coverage_feasible":..., "max_possible_coverage":...,
   "covered_gap_norm_mean":..., "covered_gap_norm_median":..., "covered_gap_norm_q25":..., "covered_gap_norm_q75":...,
   "active_covered_gap_norm_mean":..., "active_share":..., "abstain_low_conf_share":..., "clamp_over_pside_share":...,
   "non_active_share":..., "mean_p_pred":..., "mean_p_side":..., "side_violation_rate":...},
  "... 每个 0.05 步长桶一行 ..."
],
"calibration_pside_bin_metrics": [ "..." ],
"train_pside_bin_metrics": [ "..." ],
"fit_pside_bin_metrics": [ "..." ]
```

> 这套替代/补强现有 `reports/validation_pside_buckets.md`：把逐桶 coverage、gap 分位、各 action 占比一次性结构化落 JSON，便于程序化对比和回归。`.md` 可继续由该 JSON 渲染。

---

## 8. 文件级改动清单

```text
config.yaml
  loss.alpha_grid:                      [2,4,8]      → [0.5,1.0,1.5,2.0]            (Fix 2)
  objective.min_calibration_feasible_coverage: 0.93 → 0.90                          (Fix 3)
  objective.max_non_active_share:       (新增) 0.45                                 (Fix 3)
  calibration.delta_quantiles:          含义改为“(y−f)/ŝ 的分位”，建议加 0.88/0.90  (Fix 1)
  diagnostics.pside_bin_step:           (新增) 0.05                                 (报告)
  diagnostics.normalized_residual_quantiles: (新增)                                 (诊断)

train_safe_lowest_price_gap.py
  normalized_asym_loss            欠预测支也 /s_eff                                  (Fix 2)
  infer_prices                    delta→delta_norm·ŝ，签名加 s_floor                 (Fix 1)
  select_calibration_candidate    residual→r_norm=(y−f)/ŝ；delta_norm；调用补 s_floor (Fix 1)
  fit_bucket_model / evaluate_*   对 infer_prices 的调用补 s_floor、改名 delta_norm  (Fix 1)
  metric_summary                  增 active_covered_gap_norm_*、non_active_share     (Fix 3)
  candidate_key                   主目标改 active gap + non_active_share 预算         (Fix 3)
  + normalized_residual_diagnostics / per_pside_bin_delta_norm                       (诊断)
  + pside_bin_edges / pside_bin_metrics                                              (报告)
  main / report dict              写入上述新 JSON 段；CandidateResult.delta→delta_norm
                                  checkpoint.calibration.delta→delta_norm（部署侧同步）
```

> 部署同步提醒：推理侧（`execution_engine/deploy/...`）若读取 checkpoint 的 `calibration.delta` 做 `f+delta`，
> 必须同步改为 `f + delta_norm·clip(p_side−f, s_floor)`，否则线上与离线口径不一致。落地前务必核对部署读取处。

---

## 9. 预期效果：逐桶 `gap_norm` 投影（机制推演，非实测）

> 依据：`calibration_frontier.csv`、`validation_pside_buckets.md` 与上述机制。最终值取决于诊断段输出的
> `(y−f)/s` 真实离散度，须重训后用真实分布校正。

| Bucket (p_side) | 现 gap | F1 后 | F1+F2 | +F3（=本轮全做） | 目标 cov_feasible |
|---|---|---|---|---|---|
| [0.2,0.3) | 0.91 | ~0.80 | ~0.68 | **~0.55–0.62** | ~90% |
| [0.3,0.4) | 0.92 | ~0.80 | ~0.68 | **~0.55–0.62** | ~90% |
| [0.4,0.5) | 0.85 | ~0.76 | ~0.64 | **~0.52–0.58** | ~90% |
| [0.5,0.6) | 0.76 | ~0.70 | ~0.60 | **~0.48–0.55** | ~90% |
| [0.6,0.7) | 0.62 | ~0.60 | ~0.55 | **~0.45–0.52** | ~90% |
| [0.7,0.8) | 0.63 | ~0.61 | ~0.56 | **~0.45–0.52** | ~90% |
| [0.8,0.9) | 0.68 | ~0.65 | ~0.58 | **~0.48–0.55** | 80%→~88%* |
| [0.9,1.0) | 0.62 | ~0.64↑ | ~0.58 | **~0.50–0.58** | 73%→~85%* |
| **整体均值** | **0.734** | ~0.66–0.70 | ~0.58–0.63 | **~0.50–0.55** | ≥0.90 |

```text
* 高 p_side 桶受 sigmoid 饱和限制(R5)，coverage 仅部分回补；要完全修复需 Fix 4。
  形态从“低端高、高端低”的下滑曲线，转为中段最低、两端略高的“微笑”曲线。
  [0.9,1.0) 在 F1 单独时 gap 可能先升(↑)：它在被“再分配”补覆盖，属正常；整体均值仍降。
  不可约地板 ≈ 0.45–0.50（frontier：覆盖 52% 时 gap 仍 0.385），Fix 1–3 改“分配效率”，降不穿此地板。
```

---

## 10. 验证与回归清单（重训后逐项核对）

```text
[正确性]
□ side_violation_rate == 0（所有 split）            □ p_pred ≤ p_side 恒成立
□ coverage_feasible ≥ 0.90（validation）            □ 推理用 ŝ，未触碰 y/s（无泄漏）

[Fix 生效信号]
□ clamp_over_pside_share 显著下降（37% → 目标 <20%）
□ 低 p_side 桶 gap 大幅下降（[0.2,0.4) 从 ~0.91 → <0.65）
□ 逐桶 gap 曲线由“单调下滑”转“微笑”，跨桶 coverage 更均匀
□ overall 与 active covered_gap_norm_mean 均下降；non_active_share ≤ 预算

[诊断解读]
□ norm_residual_proxy_s.spread_q90_q50 记录在案（=可达 gap 地板）
□ proxy_s 与 true_s 的分布差异不大（代理可靠）；否则评估是否启用 §3.5 分桶 δ_norm
□ 若现实 active gap 已逼近该地板 → 收益见顶，排期 Fix 4；否则继续在网格上调 alpha/q
```

---

## 11. Fix 4（本轮推迟，仅留说明）

```text
目标：降低不可约地板 σ_norm（R5）与修高 p_side 桶的 sigmoid 饱和。
候选：
  (a) 在归一化空间直接做目标覆盖率的 pinball 分位回归，得到逐样本边际，替代全局 δ_norm；
  (b) location+scale 双头，scale 头给逐样本不确定性(只作排序/缩放，不做乘性放大)；
  (c) 输出层去饱和：对高 p_side 桶用 p_side·sigmoid 或可学习上界，缓解 f 顶不到位；
  (d) 秒级/微结构特征增强，降低 (y−f) 条件方差。
启动条件：本轮重训后，诊断显示 active gap 已逼近 norm_residual_proxy_s 的 q90−q50 地板。
```

---

## 12. 落地顺序建议

```text
1. 先加【诊断段 §6 + 分桶报告 §7】并在【现有模型】上跑一次 → 拿到真实 (y−f)/s 分布与逐桶基线。
   （不改损失/边际，纯观测，零风险，用于校正 §9 投影与判断 Fix 4 必要性。）
2. 再实施 Fix 1（边际归一化）→ 重训，核对 clamp 暴跌、低桶 gap 下降。
3. 叠加 Fix 2（损失归一化 + 降 alpha）→ 重训，核对整体下移。
4. 叠加 Fix 3（解耦覆盖 + 口径）→ 重训，核对 active gap 主指标与占比预算。
5. 用 §10 清单回归；据诊断地板决定是否排期 Fix 4。
按用户既定流程：代码先同步到 Databricks，再触发 Databricks 侧重训，不在本地先跑。
```
