# BTC/USDT 5m 两阶段模型优化实施文档（2026-04-22）

## 1. 文档目标

本文档用于把 2026-04-22 已确认的优化决策固化为一份可执行实施方案，目标有两个：

1. 在不改变核心任务定义的前提下，优先提升 5m 两阶段模型的方向准确率。
2. 在保持架构一致性的前提下，为后续 PnL 改善建立更可靠的数据、标签和训练基线。

本文档只定义本轮要实施的优化范围，不扩展到尚未批准的特征筛选、评估重构或执行层重算。

---

## 2. 已确认的实施边界

以下内容已经确认，后续开发以此为准。

### 2.1 保留的前提

- 主任务仍为 `BTC/USDT`。
- 底层数据频率仍为 `1m`。
- 主训练 horizon 仍为 `5m`。
- Stage 1 仍然使用绝对收益阈值，不改成波动率缩放阈值。
- `settings.yaml` 继续作为单一配置源。
- 标签与特征逻辑继续统一走 `src/labels` 与 `src/features` 共享核心。
- 执行层不允许重算一套独立特征或标签逻辑。

### 2.2 本轮明确要做的事

1. 将 Stage 1 的绝对收益阈值从 `0.0002` 提高到 `0.0003`。
2. 在 `0.0003` 阈值下显式处理样本不均衡问题，当前预估正负样本比约为 `4:1`。
3. 去掉 `threshold_multiplier` 这套方向标签偏移逻辑。
4. 将公开历史数据回补窗口扩展到 `2025-01-01`。
5. 打开 derivatives phase-1 输入，先接入 funding 和 basis。
6. 暂不做特征筛选。
7. 暂不改 walk-forward 方案。
8. 调整 LightGBM 结构，并以 `logloss` 为 early stopping 目标。
9. 校准不作为本轮优先项，不以校准作为提准确率的主路径。
10. 暂不在本轮加入交易成本建模、regime 分层成绩单或分时段成绩单。

### 2.3 本轮明确不做的事

- 不把 Stage 1 改成波动率缩放阈值。
- 不做 buy threshold 与 base rate 解耦的完整重构。
- 不做特征裁剪或 permutation importance 驱动的精简。
- 不改多折 walk-forward 的切分机制。
- 不先做 PnL 扣手续费建模。
- 不做按时段、按 regime 的分析报表增强。

---

## 3. 问题重述

从当前离线结果看，主要矛盾不是执行层，也不是 feature pack 数量本身，而是以下三点：

1. `0.0002` 的 Stage 1 阈值过低，导致 Stage 1 接近常数门控，过滤作用不足。
2. 方向标签额外使用 `threshold_multiplier=1.001`，把方向任务混入了最小涨幅约束，增加了学习难度，也让标签定义分裂为两层门槛。
3. 当前 LightGBM 结构与训练集规模不匹配，同时没有把 `logloss` 作为早停目标，导致训练过程更偏向树容量堆叠，而不是泛化控制。

本轮优化不追求一次性把 PnL 问题全部解决，而是优先做一轮“问题定义纠偏 + 数据扩窗 + 训练稳定性增强”。

---

## 4. 目标定义

### 4.1 Stage 1 目标

Stage 1 继续定义为活跃窗口识别：

$$
y_{stage1} = 1\{ |r_{5m}| > \tau \}
$$

其中：

- $\tau = 0.0003$
- $r_{5m}$ 继续使用当前共享核心中的 5 分钟绝对收益口径

### 4.2 Stage 2 目标

Stage 2 继续只在 Stage 1 正样本上做方向分类：

$$
y_{stage2} = 1\{ close[t_0 + 5m] > open[t_0] \}
$$

这里的关键变更是：

- 不再使用 `threshold_multiplier`。
- 方向标签回归为纯方向定义，不再额外要求超过某个最小涨幅。

这会把“是否值得交易”和“方向朝哪边走”拆回两个独立问题：

- Stage 1 负责过滤小波动样本。
- Stage 2 负责判断方向。

---

## 5. 标签层改动方案

### 5.1 Stage 1 绝对收益阈值调整

统一配置项改为：

```yaml
labels:
  two_stage:
    active_return_threshold: 0.0003
```

该值只能在 `config/settings.yaml` 中维护，训练、推理、评估都从统一配置读取。

### 5.2 去掉 threshold_multiplier

当前 `grid_direction` 标签实现中仍读取 horizon 级别的：

```yaml
label_params:
  threshold_multiplier: 1.001
```

这套逻辑本轮应删除，原因如下：

1. 它和 Stage 1 的绝对收益阈值表达的是同一类“最小有效波动”约束，存在重复建模。
2. 它让 Stage 2 方向标签不再是纯方向标签，降低了任务可解释性。
3. 它会让设置项分散在 horizon label params 与 two-stage settings 两处，不符合单一配置源原则。

实施要求：

- `src/labels/grid_direction.py` 不再读取 `threshold_multiplier`。
- `config/settings.yaml` 中各 horizon 的 `label_params.threshold_multiplier` 删除。
- `label_version` 可保留，但其含义应更新为“纯方向标签版本”。

### 5.3 标签实现原则

标签改动必须满足以下架构约束：

- 不允许在脚本层单独重写标签逻辑。
- 不允许在训练脚本和线上信号逻辑中各自维护一份阈值。
- 所有方向标签与两阶段标签生成逻辑继续由共享核心统一提供。

---

## 6. 样本不均衡处理方案

在 `active_return_threshold = 0.0003` 下，当前预估 Stage 1 正负样本约为 `4:1`。这个比例虽然比 `0.0002` 好，但仍明显偏斜，必须显式处理。

### 6.1 处理目标

本轮目标不是强行把分布改成 `1:1`，而是：

- 让 Stage 1 学到真正的过滤边界；
- 避免模型继续退化成“几乎总是预测 active”；
- 同时尽量保留时间序列原始分布，不做随机过采样打乱时序。

### 6.2 本轮采用的方法

本轮优先采用“权重优先，重采样暂缓”的方案。

#### Stage 1

- 启用类别权重或正负样本权重补偿。
- 不做随机过采样。
- 不做 SMOTE 一类会破坏时间结构的方法。

推荐实现方式：

1. 继续沿用现有 `sample_weight` 管道。
2. 在 Stage 1 训练帧构造时叠加 class-balance 权重。
3. 若当前已有质量权重列，则使用乘法合成：

$$
w_{stage1} = w_{quality} \times w_{class}
$$

其中：

- 正样本权重可设为 `1.0`
- 负样本权重按真实样本比自动计算，初始建议落在 `3.5` 到 `4.5` 区间

原因：当前少数类是“低活跃窗口”，模型更容易忽略这一类，应该提高其损失权重，而不是继续放大多数类。

#### Stage 2

- Stage 2 暂不单独引入新的 class rebalance 机制。
- 先观察在纯方向标签下的天然正负分布是否已足够平衡。
- 若 Stage 2 在新标签下仍明显偏斜，再单独讨论。

### 6.3 为什么先不用校准解决不均衡

校准不能解决 Stage 1 学不到少数类边界的问题。

校准的作用是把已学到的概率映射得更可靠，但如果原始分类器已经偏向多数类，校准只会把这个偏差重新映射，不会凭空创造区分能力。

因此本轮应把精力优先放在：

- 标签拆分正确化
- 类别权重
- 更长历史数据
- 更稳定的训练控制

---

## 7. 数据窗口扩展方案

### 7.1 新的回补起点

本轮将公开历史数据回补起点统一扩展为：

```yaml
data_backfill:
  start_date: "2025-01-01"
```

### 7.2 目标

扩窗的目的不是单纯增加样本数，而是补齐不同市场状态：

- 低波动阶段
- 高波动阶段
- 趋势阶段
- 震荡阶段

当前训练窗口过短，模型更容易把某一段局部结构记成“稳定规律”。把起点扩到 `2025-01-01` 后，目标是显著缓解：

- Stage 1 OOF/validation 概率分布过于集中
- Stage 2 AUC train/validation gap 过大
- 参数对单一尾窗的偶然依赖

### 7.3 derivatives 输入

本轮同意开启 phase-1 的 derivatives 输入，但范围仍受控：

- 启用 funding
- 启用 basis
- 继续关闭 OI
- 继续关闭 options
- 继续关闭 book_ticker

统一配置原则：

- 开关仍由 `config/settings.yaml` 控制。
- 数据加载仍走 `src/data/derivatives`。
- 特征构建仍通过共享 feature pack 进入主训练帧。
- 执行层不直接读 raw derivatives 数据。

---

## 8. 模型与训练策略

### 8.1 本轮参数方向

根据已确认要求，LightGBM 参数调整为：

- `n_estimators = 200`
- `max_depth = 10`
- `num_leaves = 200`

其余参数本轮建议采用“控制过拟合、保留表达能力”的折中方案。

### 8.2 建议配置

#### Stage 1

```yaml
model:
  plugins:
    lightgbm_stage1:
      n_estimators: 200
      learning_rate: 0.03
      num_leaves: 200
      min_child_samples: 200
      subsample: 0.8
      subsample_freq: 1
      colsample_bytree: 0.6
      reg_alpha: 0.5
      reg_lambda: 2.0
      max_depth: 10
      scale_pos_weight: null
      random_state: 42
      early_stopping_rounds: 50
      eval_metric: binary_logloss
```

说明：

- Stage 1 不建议同时硬编码 `scale_pos_weight` 和外部 class sample weight，两者选一个主方案即可。
- 本轮优先建议通过 sample weight 管道处理不均衡，因此 `scale_pos_weight` 可以置空或不启用。

#### Stage 2

```yaml
model:
  plugins:
    lightgbm_stage2:
      n_estimators: 200
      learning_rate: 0.03
      num_leaves: 200
      min_child_samples: 150
      subsample: 0.8
      subsample_freq: 1
      colsample_bytree: 0.5
      reg_alpha: 0.5
      reg_lambda: 2.0
      max_depth: 10
      random_state: 42
      early_stopping_rounds: 50
      eval_metric: binary_logloss
```

### 8.3 为什么用 logloss 做 early stopping

本轮同意用 `logloss` 做早停，原因如下：

1. accuracy 对阈值敏感，不能稳定反映概率质量。
2. 两阶段结构中，Stage 2 概率后续还会参与阈值决策，概率质量比单点分类率更重要。
3. 对当前任务而言，validation accuracy 的小波动很容易来自样本比例变化，而不是模型真的更好。

### 8.4 对当前参数选择的风险说明

`num_leaves = 200` 与 `max_depth = 10` 会显著提高树的表达能力，这本身存在过拟合风险。之所以本轮仍按该方向推进，是因为这是已确认的参数方向。

为控制风险，本轮必须同步做两件事：

1. 用更长历史窗口训练。
2. 启用基于 validation `logloss` 的 early stopping。

如果这两项不同时落地，则大叶子数和深树更可能放大 train/validation gap。

---

## 9. 校准策略

### 9.1 本轮结论

校准不是本轮提升准确率的主路径，优先级下调。

### 9.2 原因

当前主要目标是提高方向准确率，而不是先提高概率可解释性。对于当前问题，准确率下降更可能来自：

- Stage 1 目标过松
- 标签重复加门槛
- 数据历史过短
- 模型训练过程缺少更稳的停止准则

这些问题不先解决，校准带来的收益通常有限。

### 9.3 实施建议

- Stage 1 继续默认不校准。
- Stage 2 本轮也不把校准当成必需项。
- 等新一轮训练基线稳定后，再决定是否恢复仅用于概率阈值优化的 Stage 2 校准。

换言之，本轮的答案是：

> 校准不是没有价值，但在当前阶段没有必要优先做。

---

## 10. 特征策略

### 10.1 本轮原则

本轮不做特征筛选。

### 10.2 理由

当前优先级更高的是：

- 修正标签定义
- 提高 Stage 1 阈值
- 处理 Stage 1 不均衡
- 扩大训练历史
- 加入 funding 与 basis
- 让训练流程使用 logloss early stopping

在这些基础问题未处理前，过早做特征筛选容易把问题误判成“特征太多”，而不是“目标定义和数据跨度不对”。

### 10.3 本轮允许的唯一特征层变化

- 允许开启 funding 相关 pack
- 允许开启 basis 相关 pack
- 其余 pack 保持原状

---

## 11. 配置落库原则

所有本轮新增或修改的业务参数必须统一落到 `config/settings.yaml`，不得散落到脚本或训练函数里。

本轮至少需要在配置层统一维护：

- `labels.two_stage.active_return_threshold = 0.0003`
- data backfill 起点 `2025-01-01`
- derivatives 的 enabled 开关与 funding/basis 子开关
- Stage 1 与 Stage 2 的 LightGBM 结构参数
- early stopping 相关参数

同时需要删除或废弃：

- horizon 级别的 `threshold_multiplier`

如果后续新增 buy threshold、edge threshold 或 class weighting 参数，也必须继续回落到 `settings.yaml`。

---

## 12. 代码改动落点

### 12.1 必改模块

1. `config/settings.yaml`
   - 更新 `active_return_threshold`
   - 更新 backfill 起点
   - 打开 derivatives funding/basis
   - 调整 LightGBM 参数
   - 删除 `threshold_multiplier`

2. `src/labels/grid_direction.py`
   - 去掉 `threshold_multiplier` 读取与应用
   - 保持纯方向标签逻辑

3. `src/model/train.py`
   - 在 Stage 1 训练帧中引入类别不均衡权重合成
   - 保持两阶段共享训练边界

4. `src/model/lightgbm_plugin.py`
   - 增加 `early_stopping_rounds` 与 `eval_metric=logloss` 支持
   - 确保训练过程基于 validation 集早停，而不是只跑固定树数

5. `src/data/derivatives/*`
   - 确认 funding 与 basis 在扩窗后可稳定加载

### 12.2 暂不改模块

- walk-forward split 生成逻辑
- execution 层
- feature selection / feature pruning 相关流程
- PnL 交易成本建模
- 分时段/分 regime 报表模块

---

## 13. 实施顺序

### Phase 1. 标签和配置纠偏

1. 去掉 `threshold_multiplier`
2. 将 `active_return_threshold` 提升到 `0.0003`
3. 清理所有旧配置引用，确保 `settings.yaml` 是唯一配置源

### Phase 2. 训练流程稳定性增强

1. Stage 1 引入 class-balance sample weight
2. LightGBM 支持 validation logloss early stopping
3. 更新 Stage 1 / Stage 2 模型参数

### Phase 3. 数据扩窗与衍生品接入

1. 将回补起点扩到 `2025-01-01`
2. 重新生成 spot 与 derivatives 归档
3. 打开 funding / basis 特征并重新训练

### Phase 4. 基线复验

1. 对比新旧训练报告
2. 重点比较 Stage 1 balanced accuracy
3. 重点比较 Stage 2 validation accuracy 与 ROC AUC
4. 重点比较 end-to-end trade accuracy

---

## 14. 验收标准

本轮验收不以单一 PnL 数字为唯一标准，而以“基线是否变得更合理、更稳”为主。

### 14.1 必须满足

- Stage 1 不再表现为接近常数门控。
- Stage 1 validation `balanced_accuracy` 必须明显高于 `0.50`。
- Stage 2 方向标签不再依赖 `threshold_multiplier`。
- 训练与推理仍然共享同一标签/特征核心。
- 所有关键阈值和模型参数都从 `settings.yaml` 读取。

### 14.2 期望满足

- validation accuracy 明显优于当前基线。
- validation ROC AUC 不低于当前基线，并出现更稳定的提升。
- train/validation gap 收敛，而不是继续扩大。

---

## 15. 本轮结论

本轮优化的核心不是“再堆更多特征”，而是先把问题定义收敛到一套更干净的两阶段结构：

- Stage 1 用 `0.0003` 的绝对收益阈值承担过滤责任。
- Stage 2 回归纯方向分类，不再叠加 `threshold_multiplier`。
- Stage 1 用类别权重解决 `4:1` 不均衡。
- 训练改为围绕 validation `logloss` 做 early stopping。
- 数据窗口扩展到 `2025-01-01`，并正式接入 funding 与 basis。

这套调整完成后，才有资格判断下一步是否需要继续做：

- buy threshold 重构
- 多折 walk-forward
- 特征筛选
- 交易成本建模
- Stage 2 概率校准

在这之前，优先把标签、数据和训练基线做对。