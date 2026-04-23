# Two-Stage BTC 5-Minute Prediction System — Implementation Plan

> 本文档取代原设计稿，基于与当前代码库（`src/features`, `src/labels`, `src/model`, `src/signal`, `src/sizing`, `config/settings.yaml`）的实际能力和已发现问题重写为可执行的实施计划。

## 0. Confirmed Scope

以下实现边界已确认，后续开发以此为准：

- 两阶段实现以 `BTC/USDT 5m` 为主目标，但配置层同时保留现有其他 horizon，不做一次性删除
- 训练入口允许直接升级为两阶段主入口，不保留旧单阶段训练入口
- 标签体系彻底切换为两阶段定义；`tau` 默认值为 `0.0002`，但应作为统一配置参数，而非硬编码常量
- `core_5m` 中现有特征 pack 全部保留，不因两阶段改造而禁用任何现有特征
- 信号决策彻底切换到 two-stage policy，不再兼容 edge-based 决策
- 执行层默认固定下单 `5` 个合约，不再复用动态 sizing 决策
- 校准拆分为两阶段独立配置：Stage 1 默认不校准，Stage 2 独立校准
- 线上特征预热缓存不属于当前实现阶段，放到下一阶段再做

---

## 1. Objective

构建一个 **两阶段级联模型** 用于 BTC/USDT 5 分钟方向预测：

- **Stage 1 (Active Filter)**: 识别 "tradable windows"（|r_t| > τ），目标高 recall
- **Stage 2 (Direction Classifier)**: 在 Stage 1 认为活跃的样本上判断方向
- **最终输出**: `BUY (YES token)` / `SELL (NO token)` / `NO_TRADE`

### 1.1 PnL 假设（明确边界条件）

执行约定：
- 在 yes_price ≈ 0.50 附近挂限价单
- 默认采用特征预热、`t_0` 触发推理的执行方式
- 每次固定下单 5 个合约
- 持有到 5 分钟窗口结算
- 未成交则放弃，不追单
- 不考虑显式交易成本

在此约定下 PnL 近似对称 ±1：
$$\text{PnL\_per\_trade} \approx 2 \cdot \text{accuracy}_{S2} - 1$$
$$\text{PnL\_per\_sample} = \text{coverage} \cdot (2 \cdot \text{accuracy}_{S2} - 1)$$

**最终优化目标：PnL_per_sample**。

### 1.2 为什么使用两阶段

实验观察：~60% 样本 |r| 非常小，在这类样本上单模型 AUC 仅 0.53。
过滤掉低波动样本后：AUC 升至 0.80+，accuracy 升至 0.80+。

**但离线 80% accuracy 是在 oracle (|r|>τ) 过滤下测得**，线上用 Stage 1 模型过滤会引入 **covariate shift**——Stage 2 在推理时会遇到它训练时未见过的 Stage 1 假阳性样本。解决此问题是本实施计划的核心之一。

---

## 2. Label Design

### 2.1 Return 定义（精确口径）

$$r_t = \frac{\text{close}[t_0 + 4] - \text{open}[t_0]}{\text{open}[t_0]}$$

- `t_0` = 5 分钟网格起点（minute % 5 == 0）
- `close[t_0+4]` = 从 t_0 起第 5 根 1m K 线的收盘价
- 与 [grid_direction.py](src/labels/grid_direction.py) 中 `future_close_offset = minutes - 1 = 4` 保持一致

### 2.2 Stage 1 Label (Active Window)

```
y_stage1 = 1 if |r_t| > τ
         = 0 otherwise
τ = settings.labels.two_stage.active_return_threshold  # default 0.0002
```

### 2.3 Stage 2 Label (Direction)

仅在 Stage 1 label=1 的样本上定义：
```
y_stage2 = 1 if r_t > 0
         = 0 if r_t <= 0
```

### 2.4 Soft Boundary Weighting（关键改进）

τ 附近的样本信号最弱、噪声最大，需要在 Stage 1 训练中降权：

```python
def compute_stage1_weight(abs_return, tau=0.0002):
    """τ 附近窄带内样本降权，防止模型在边界噪声上浪费容量。"""
    narrow_band = tau * 0.3   # 即 [0.7τ, 1.3τ] 区间
    if abs(abs_return - tau) < narrow_band:
        return 0.2
    return 1.0
```

实施要点：
- 作为 Stage 1 训练的 `sample_weight` 列，与现有 [dataset_builder.py](src/data/dataset_builder.py) 的 `DEFAULT_SAMPLE_WEIGHT_COLUMN` 管道兼容
- 不影响 Stage 2 和评估（评估仍使用原始样本）
- `tau` 不扫描优化，但必须从统一配置读取，避免散落在标签、训练、推理和评估代码中

---

## 3. Feature Engineering

### 3.1 两阶段共享特征集

两个 Stage 共用 `core_5m` 特征档案的全部特征（暂不做筛选，后续可根据 Stage 2 的 feature_importance 做剪枝）。

### 3.2 Stage 2 关键补充：Stage 1 Probability as Feature（核心设计）

**解决 covariate shift 的关键机制**（取代硬级联）：

```python
# Stage 2 输入 = 原始 core_5m 特征 + stage1_prob
stage1_prob_oof = get_oof_predictions(stage1_model, X_all)
X_stage2 = pd.concat([X_original, stage1_prob_oof.rename("stage1_prob")], axis=1)
```

**原理**：
- Stage 2 可以学习 `stage1_prob` 对输入可信度的指示作用
- 当 `stage1_prob` 接近阈值（即 Stage 1 的边界样本）时，Stage 2 会自然降低置信度
- 线上推理时同样使用 Stage 1 的输出概率作为 Stage 2 输入，**训练和推理分布一致**

**关键约束**：
- Stage 2 训练时，`stage1_prob` **必须是 OOF 预测**，不能用 Stage 1 在训练集上的 in-sample 概率（否则标签泄漏）
- OOF 概率通过当前代码已有的 `build_walk_forward_splits` + `_build_calibration_oof_predictions` 机制生成

### 3.3 新增特征（作为 `core_5m` profile 的扩展）

在现有 13 个特征包之外新增以下特征。建议新增 1-2 个 feature pack 文件。

#### 3.3.1 5 分钟内子结构特征（新 pack: `intra_5m_structure`）

捕捉 t_0 之前 5 根 1m K 线内部的方向性形态：

```python
# 伪代码
past_close = df["close"].shift(1)
past_open = df["open"].shift(1)

# 最后一根 1m K 线方向（近端信号更强）
last_1m_up = (past_close > past_open).astype(float)

# 前 5 根中上涨 K 线占比
up_ratio_5 = (past_close > past_open).rolling(5).mean()

# 前 5 根中下跌 K 线占比
down_ratio_5 = (past_close < past_open).rolling(5).mean()

# 5 根内最大单根涨幅和跌幅
max_up_ret_5 = (past_close.pct_change(1, fill_method=None)).rolling(5).max()
max_down_ret_5 = (past_close.pct_change(1, fill_method=None)).rolling(5).min()
```

**输出特征**：`last_1m_up`, `up_ratio_5`, `down_ratio_5`, `max_up_ret_5`, `max_down_ret_5`

#### 3.3.2 动量加速度特征（扩展现有 `momentum` pack 或新 pack: `momentum_acceleration`）

一阶动量的变化率，信号更精细：

```python
# 当前已有: ret_1, ret_3, ret_5, ret_10, ret_15
# 新增:
ret_1_accel = df["ret_1"] - df["ret_1"].shift(1)         # 动量加速度
ret_3_accel = df["ret_3"] - df["ret_3"].shift(3)
momentum_reversal = sign(df["ret_1"]) * sign(df["ret_3"]) # -1 表示短中期反向
```

**输出特征**：`ret_1_accel`, `ret_3_accel`, `momentum_reversal`

**注意**：这些特征依赖 `momentum` pack 的输出，实现上应放在 `lagged` pack 之后执行（参考 [lagged.py](src/features/lagged.py) 的 dependency 检查模式）。

### 3.4 特征命名与注册

- 新 pack 添加到 [src/features/registry.py](src/features/registry.py) 的 `FEATURE_PACKS` 字典
- 在 `config/settings.yaml` 的 `core_5m.packs` 列表中加入新 pack 名
- 依赖顺序：`momentum` → `momentum_acceleration` → `lagged`

---

## 4. Training Pipeline

### 4.1 共享时间分割边界（两阶段强一致性）

**所有时间切分必须在全样本时间轴上先定义，再分别裁剪到 Stage 1 / Stage 2 训练集**：

```python
# 1) 用全样本（含 |r|<=τ 的所有 grid 行）定义 time split
full_training_frame = build_training_frame(...)  # 不做任何过滤
splits = purged_chronological_time_window_split(
    full_training_frame,
    validation_window_days=30,
    purge_rows=1,  # 保持不变（用户决定）
)
# splits 中的 train_start/train_end/valid_start/valid_end 是时间坐标

# 2) Stage 1 训练集 = 全样本按 splits 时间边界裁剪
stage1_train = full_training_frame[splits.train_slice]
stage1_valid = full_training_frame[splits.valid_slice]

# 3) Stage 2 训练集 = Stage 1 active 样本按 **同一** 时间边界裁剪
active_mask = (abs_return > τ)
stage2_train = full_training_frame[active_mask][splits.train_slice]
stage2_valid = full_training_frame[active_mask][splits.valid_slice]
```

**实施要点**：
- 需要重构 [train.py](src/model/train.py) 引入 `TwoStageTrainingArtifacts` 数据类
- 保留现有 `purged_chronological_time_window_split` 接口不变
- Walk-forward 同理——在全样本上生成 splits，每个 fold 内按 mask 裁剪两阶段训练集

### 4.2 Walk-Forward 步长优化

当前 [train.py](src/model/train.py#L172) 的 `step_size = valid_rows`（不重叠），BTC regime 切换快，粒度太粗。

**修改**：
```python
walk_forward_splits = build_walk_forward_splits(
    full_training_frame,
    min_train_size=max(train_rows, 1),
    validation_size=max(valid_rows, 1),
    step_size=valid_rows // 2,   # 50% 重叠
    purge_rows=1,
)
```

### 4.3 purge_rows

保持当前默认 `purge_rows=1`。

### 4.4 Training Flow（完整）

```
全样本 OHLCV
   │
   ▼ build_training_frame (含 |r|<=τ 样本)
全样本 TrainingFrame (全部 grid 行)
   │
   ▼ 基于时间的 purged split
切出 [train_slice, valid_slice] 时间边界
   │
   ├─────────────────────────┐
   ▼                          ▼
Stage 1 训练                   Stage 2 训练
   │                           │（延迟到 Stage 1 OOF 产出后）
   │ 输入: 全样本              │
   │ 标签: |r|>τ               │
   │ 权重: soft boundary       │
   │ 模型: LGBM (S1 配置)      │
   │                           │
   ▼                           │
 Stage 1 model                 │
   │                           │
   ▼ walk-forward OOF 预测     │
 stage1_prob_oof ─────────────▶│
                               │ 输入: 仅 |r|>τ 样本
                               │        + stage1_prob 特征
                               │ 标签: r_t > 0
                               │ 模型: LGBM (S2 配置)
                               │
                               ▼
                         Stage 2 model
```

### 4.5 各 Stage 专用超参数（在 `config/settings.yaml` 分开配置）

```yaml
model:
  active_plugins:
    stage1: lightgbm_stage1
    stage2: lightgbm_stage2
  plugins:
    lightgbm_stage1:
      # Stage 1: 波动率/活跃度预测，任务较简单
      n_estimators: 300            # ↓ 从 700
      learning_rate: 0.05
      num_leaves: 10               # ↓ 从 15
      max_depth: 4
      min_child_samples: 100
      subsample: 0.8               # 新增 bagging
      subsample_freq: 1
      colsample_bytree: 0.6
      reg_alpha: 0.5
      reg_lambda: 1.0
      scale_pos_weight: 1.5        # 替代 class_weight=balanced，偏向高 recall
      random_state: 42
    lightgbm_stage2:
      # Stage 2: 方向预测，有效样本少，需要更强正则
      n_estimators: 700
      learning_rate: 0.03
      num_leaves: 15
      max_depth: 4
      min_child_samples: 200       # ↑ 从 100（更保守）
      subsample: 0.8               # 新增 bagging
      subsample_freq: 1
      colsample_bytree: 0.4        # ↓ 从 0.6（更激进列采样）
      reg_alpha: 0.5
      reg_lambda: 1.0
      random_state: 42
      # 不用 class_weight=balanced，Stage 2 的 base rate 可能偏离 0.5

labels:
  two_stage:
    active_return_threshold: 0.0002
```

### 4.6 Calibration

- **Stage 1**: 独立配置，默认不校准，仅用于阈值化
- **Stage 2**: 独立配置，可用现有 `_select_calibrator` 机制自动选择 isotonic/platt/none
- 配置结构应允许按 stage 指定 calibrator，而不是继续使用单一全局 `calibration.active_plugin`

---

## 5. Inference Logic

```python
def predict(x):
    # Stage 1: 判断活跃度
    p_active = stage1_model.predict_proba(x)
    
    if p_active < stage1_threshold:
        return "NO_TRADE"
    
    # Stage 2: 判断方向（核心：加入 stage1_prob 作为特征）
    x_stage2 = concat([x, p_active])  # 与训练保持一致
    p_up = stage2_model.predict_proba(x_stage2)
    p_up_calibrated = stage2_calibrator.transform(p_up) if stage2_calibrator else p_up
    
    # Decision（见第 6 节）
    if p_up_calibrated > buy_threshold:
        return "BUY"       # 买 YES token
    else:
        return "SELL"      # 买 NO token
```

**关键不变量**：训练时 Stage 2 的 `stage1_prob` 来自 OOF，推理时来自 Stage 1 的直接输出。两者分布需要一致性监控（见第 8 节）。

### 5.1 Feature Preheating + `t_0` Trigger（默认执行时序）

当前代码中的大部分特征 pack 已使用 `.shift(1)` 语义，因此 `t_0` 时刻的特征天然只依赖 `t_0-1m` 及之前的已完成 1m K 线。默认执行方案不做额外 1 分钟错位，而是通过 **持续预热特征** 缩短推理延迟：

```
持续运行:
  每次 1m K 线收盘:
    - 更新滚动特征状态
    - 更新 HTF 聚合状态
    - 刷新最新 feature snapshot

t_0 触发 (例如 12:05:00):
  - 读取以 12:04 收盘为止的最新特征快照
  - 执行 Stage 1 / Stage 2 推理
  - 读取 Polymarket 对应市场价格
  - 固定下单 5 个合约
```

该方案的目的不是改变训练标签对齐，而是在 **不损失最新 1m 信息** 的前提下，把线上执行延迟压缩到 `t_0` 后的秒级。

**阶段边界说明**：以上为目标线上架构，但不属于当前实现阶段。本阶段只要求离线训练、评估和两阶段推理逻辑成立；特征预热缓存及 `t_0` 快照读取放到下一阶段实现。

### 5.2 不采用整体 1 分钟错位训练

本版本 **不采用** “所有特征与标签整体错位 1 分钟”的方案。原因：

- 当前特征定义已经保证 `t_0` 时只使用 `t_0-1m` 及更早的数据
- 继续整体错位会丢失最新一根 1m K 线的信息，直接削弱 Stage 1 和 Stage 2 的预测能力
- 预热特征可以解决执行速度问题，不需要通过牺牲信息来换取提前 60 秒计算窗口

因此，离线训练和线上推理继续保持当前时间语义：

- **特征**：使用 `t_0-1m` 及更早的已完成数据
- **标签**：预测 `[t_0, t_0+5m)` 窗口结果
- **执行**：在 `t_0` 触发推理和下单

---

## 6. Decision Layer

### 6.1 基础规则（无 margin rule，最大化 coverage）

**去掉原设计的 margin rule**。只保留 Stage 1 一层过滤，避免双重过滤压缩 coverage：

```
if stage1_prob < stage1_threshold: NO_TRADE
else:
    if p_up > buy_threshold:  BUY (YES)
    else:                     SELL (NO)
```

### 6.2 Base-rate 校正的 buy/sell 阈值

BTC 5 分钟存在微小方向偏差（非 50/50）。用训练集（Stage 1 active 样本上）的 base rate 作为对称中点：

```python
# 在训练阶段计算
base_rate = y_stage2_train.mean()   # 例如 0.517

# 配置到 settings.yaml
buy_threshold = base_rate           # 或 base_rate + small_margin（可后续实验）
```

当前简化方案：`buy_threshold = base_rate`，无 margin。

### 6.3 执行层映射

- `BUY`  → 在 YES token 上挂限价单，price = `yes_price` 或附近
- `SELL` → 在 NO token 上挂限价单，price = `no_price` = `1 - yes_price`
- 下单数量固定为 `5` 个合约，不接入动态仓位管理

需要小幅扩展 [order_router.py](src/execution/order_router.py) 和 [polymarket.py](src/execution/adapters/polymarket.py) 以支持 `side="NO"` 路径。Mapper 已经在 [btc_5m_polymarket.py](src/execution/mappers/btc_5m_polymarket.py) 中返回 `no_token_id` / `no_price`。

### 6.4 默认执行时序

默认执行方案采用 **方案 A**：

1. 在每根 1m K 线收盘时预热并更新特征缓存
2. 在 5 分钟网格点 `t_0` 触发两阶段推理
3. 根据 Stage 2 方向在 YES 或 NO 侧挂单
4. 默认目标价位仍以 `0.50` 附近为中心，不做 stale-feature 提前预测

本版本不引入基于旧特征的提前 1 分钟预测，也不把“开盘前做模型推理”作为默认流程。若后续增加开盘前的被动挂单策略，应作为独立的 execution alpha 模块，不应改变两阶段模型本身的训练与推理时序。

---

## 7. Threshold Selection（Stage 1）

### 7.1 核心原则

Stage 1 阈值 **直接用 OOF 上的 PnL_per_sample 曲线选取**，不用 F_β / F1。

### 7.2 流程

```python
# 在 OOF 验证集上
for threshold in np.arange(0.30, 0.80, 0.02):
    active_mask = stage1_prob_oof > threshold
    coverage = active_mask.mean()
    
    if active_mask.sum() < min_active_samples:
        continue
    
    # 在 active 样本上计算 Stage 2 accuracy
    p_up = stage2_prob_oof[active_mask]
    pred_direction = (p_up > buy_threshold).astype(int)
    true_direction = y_direction_oof[active_mask]
    accuracy = (pred_direction == true_direction).mean()
    
    pnl_per_sample = coverage * (2 * accuracy - 1)
    pnl_per_trade = 2 * accuracy - 1
    
    records.append((threshold, coverage, accuracy, pnl_per_sample, pnl_per_trade))

# 选择最大化 pnl_per_sample 的 threshold
best_threshold = max(records, key=lambda r: r[3])[0]
```

### 7.3 实施

- 新增 `scripts/tune_stage1_threshold.py` 或在 `run_model_experiments.py` 增加扫描逻辑
- 结果写入 `training_report.json`，包含 threshold-PnL 曲线供人工检查

### 7.4 τ 固定性

`tau` 是 **默认值为 0.0002 的配置参数**，本版本不做自动扫描和优化。仅 Stage 1 阈值（即 `p_active` 的 cutoff）参与优化。

---

## 8. Evaluation Framework

### 8.1 Stage 1 Metrics (诊断用)

- `coverage` = 通过率
- `recall` = 在真实 active 样本上的召回
- `precision` = 通过样本中真 active 的占比（影响 Stage 2 输入质量）

这些仅用于诊断，**不作为选型标准**。

### 8.2 Stage 2 Metrics

- `accuracy` = 方向预测准确率（主要指标）
- `balanced_accuracy` = 检测方向偏差
- `confusion_matrix` = 分析 buy/sell 错误分布
- `ece` (Expected Calibration Error) = 校准质量（诊断用）
- `auc` = 排序能力监控（次要指标，不直接用于决策）
- `log_loss` = 概率稳定性监控（次要指标，不直接用于决策）

### 8.3 End-to-End Metrics（选型和报告的主要依据）

| 指标 | 公式 | 用途 |
|------|------|------|
| `coverage` | 交易样本 / 总样本 | 反映交易频率 |
| `trade_accuracy` | 在交易样本上 accuracy | 反映方向质量 |
| `pnl_per_trade` | 2·accuracy − 1 | 单笔期望收益 |
| **`pnl_per_sample`** | coverage · (2·accuracy − 1) | **主要选型目标** |
| `sharpe` | 基于每 5m 收益序列 | 风险调整 |
| `max_drawdown` | 累计 PnL 回撤 | 稳定性 |
| `longest_losing_streak` | 连续亏损次数 | 尾部风险 |

### 8.4 Walk-Forward 报告

每个 WF fold 报告上述所有指标，输出 `mean/min/max`。在 `training_report.json` 中保留所有 fold 明细。

### 8.5 Covariate Shift 监控（新增）

监控 Stage 1 概率在训练和推理阶段的分布：

- 训练时 `stage1_prob_oof` 的分布（mean, std, 分位数）
- 线上 `stage1_prob_live` 的滑动窗口分布
- 若两者 KS 距离 > 0.1 发警报

---

## 9. Model Selection Strategy

### 9.1 网格

- Stage 1 model ∈ {lightgbm_stage1, catboost, logistic}
- Stage 2 model ∈ {lightgbm_stage2, catboost, logistic}
- Stage 1 threshold（通过 §7 流程连续选取）

### 9.2 Selection Criterion

主选择依据：**Walk-Forward `pnl_per_sample_mean`**，平手时按：
1. `pnl_per_sample_min`（稳定性）
2. `trade_accuracy_mean`

### 9.3 次要监控指标

以下指标仍然需要生成、写入报告并持续监控，但 **不作为主决策标准**：

- `log_loss`：用于监控概率输出是否异常波动
- `auc`：用于监控排序能力是否出现退化
- `F1 / F_β`：用于辅助理解 Stage 1 的分类轮廓

主决策标准仍然是 `pnl_per_sample`，其次看稳定性和 `trade_accuracy`。

---

## 10. Implementation Roadmap

### Phase 1: 最小可行改动（P0，直接影响 PnL）

1. [ ] 新增 `src/labels/abs_return.py` 生成 `abs_return` 列供 Stage 1 使用
2. [ ] 修改 [dataset_builder.py](src/data/dataset_builder.py)：为 Stage 1 计算 soft boundary weight（§2.4）
3. [ ] 重构 [train.py](src/model/train.py) 为两阶段：
   - `TwoStageTrainingArtifacts` 数据类
   - `train_two_stage_model()` 函数
   - 共享 time split（§4.1）
   - Stage 1 OOF 概率作为 Stage 2 特征（§3.2）
4. [ ] `config/settings.yaml` 添加 `lightgbm_stage1` / `lightgbm_stage2` 插件配置
   - `tau` 进入统一配置
   - 校准改为 stage-specific 配置
5. [ ] 统一训练入口为 `scripts/train_model.py`，并保留 `scripts/tune_stage1_threshold.py`
6. [ ] Walk-forward step_size = valid_rows // 2（§4.2）
7. [ ] 修改训练/推理 artifact 与信号协议，支持 two-stage policy、`BUY/SELL/NO_TRADE` 三态输出
8. [ ] 移除 5m 主路径中的 edge-based 决策，切换为 §6 的 two-stage decision logic
9. [ ] 执行层固定下单数量为 5 个合约，并打通 `NO` side 下单路径

### Phase 2: 特征增强（P1）

10. [ ] 新增 `src/features/intra_5m_structure.py`（§3.3.1）
11. [ ] 扩展 `momentum` pack 或新建 `momentum_acceleration`（§3.3.2）
12. [ ] 注册到 `FEATURE_PACKS` 和 `settings.yaml`
13. [ ] 为线上信号服务增加特征预热缓存，在 `t_0` 直接读取最新快照推理（§5.1）

### Phase 3: 决策层与执行（P1）

14. [ ] 修改 [signal_service.py](src/services/signal_service.py)：支持两阶段推理
15. [ ] 新 signal policy：`two_stage_policy` 在 [policies.py](src/signal/policies.py) 实现 §6 的决策逻辑
16. [ ] 扩展 [order_router.py](src/execution/order_router.py) 和 Polymarket adapter 支持 `side="NO"`
17. [ ] base_rate 持久化到模型 artifact（训练时计算、推理时读取）

### Phase 4: 评估与监控（P2）

18. [ ] `src/model/evaluation.py` 新增 `compute_pnl_metrics()` 函数
19. [ ] 修改 `scripts/run_model_experiments.py` 按 `pnl_per_sample` 排序，并保留次要监控指标输出
20. [ ] 新增 Stage 1 概率分布漂移监控（§8.5）

---

## 11. Key Design Invariants

1. **共享时间边界**：所有 Stage 的 train/valid 切分在全样本时间轴上统一定义
2. **OOF 严格性**：Stage 2 训练的 `stage1_prob` 必须来自 walk-forward OOF，不得使用 Stage 1 的 in-sample 概率
3. **推理训练一致**：线上 Stage 2 的 `stage1_prob` 输入必须与训练 OOF 的生成方式（同样的模型结构、同样的特征管道）对齐
4. **PnL 是唯一终极指标**：所有模型选型、阈值调优、特征剪枝都以 `pnl_per_sample` 为准
5. **`tau` 配置化但不自动调参**：`tau` 从统一配置读取，默认 `0.0002`；本版本不做自动扫描，避免数据窥探

---

## 12. Non-Goals（本版本不做）

- 概率化 position sizing（当前 fixed_fraction 足够）
- 动态 `tau`（按 rv 自适应）
- 分布预测 / 分位数回归
- Multitask shared backbone
- 订单簿微观结构特征（需要 taker volume 数据，暂不具备）
- 交易成本建模（用户明确不考虑）
- Margin-based NO_TRADE 过滤（避免 coverage 过度压缩）
- 显式 edge-based 策略（用户已验证效果不佳）
- 基于 stale-feature 的提前 1 分钟模型推理

---

## 13. Success Criteria

在 hold-out validation (最近 30 天) 和 walk-forward fold 上：

- `pnl_per_sample_mean` > 当前单模型基线 + 50%（显著优于"全样本单模型"）
- `pnl_per_sample_min` > 0（所有 fold 不亏）
- `trade_accuracy_mean` >= 0.60（考虑 covariate shift 修正后的合理目标）
- Stage 1 概率分布线上/线下 KS < 0.1（无分布漂移）

达到上述标准即认为两阶段架构成功落地。

---

## 14. Summary

与原设计相比的核心变更：

| 维度 | 原设计 | 实施计划 |
|------|--------|----------|
| 级联方式 | 硬门控（Stage 1 过滤 → Stage 2） | 软级联（Stage 1 概率作为 Stage 2 特征） |
| Covariate shift | 未处理 | 通过 OOF stage1_prob 作为 S2 特征 + 训练推理一致性消除 |
| Stage 1 标签 | 纯硬阈值 | 硬阈值 + soft boundary weighting |
| 特征 | 两阶段共用 `core_5m` | 两阶段共用 + 新增 intra_5m / momentum_accel |
| 执行时序 | 未明确 | 特征预热 + `t_0` 触发推理 |
| Time split | 两阶段各自切分 | 全样本统一切分，裁剪到各阶段 |
| Walk-forward 步长 | valid_rows | valid_rows // 2（50% 重叠） |
| 超参数 | 一套通用配置 | Stage 1 / Stage 2 专用配置 |
| Stage 1 阈值选型 | F_β 启发 | OOF PnL_per_sample 曲线直接优化 |
| Stage 2 margin rule | p ∈ [0.45, 0.55] 不交易 | 去除（只保留 Stage 1 过滤） |
| Buy/Sell 阈值 | 对称 0.5 | base-rate 校正 |
| 下单方式 | 未明确 | 固定 5 合约，YES/NO 双向下单 |
| 监控指标 | 禁用部分分类指标 | 次要指标继续生成但不参与主决策 |
| 主指标 | accuracy / F_β | pnl_per_sample |

**一句话**：这是一个 **soft-cascade** 架构，用 OOF Stage 1 概率把活跃度信息无损传递给 Stage 2，同时保留硬阈值作为线上 gate，兼顾两阶段的信息整合和解释性。
