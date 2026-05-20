# PRD：离线标签迁移到 Polymarket 实际 resolved 结果

## 1. 背景

当前项目里同时存在两条标签路径：

1. 主训练样本构建路径仍通过 BTC OHLCV 构造 `target`：
   `target = 1{close[t0 + 4m] >= open[t0]}`，实现位于 `src/labels/grid_direction.py`。
2. 之前已经完成过一条实验路径，用 `scripts/experiments/build_polymarket_resolved_training_frame.py` 把训练 frame 里的 `target` 替换为 Polymarket 实际 resolved 结果。

实验路径已经验证过核心机制：

- 对每个 5m market timestamp 生成 slug：`btc-updown-5m-{epoch}`。
- 从 Gamma 拉取 closed markets。
- 读取 `outcomes` 和 `outcomePrices`。
- 以接近 `1.0` 的 outcome price 判断真实赢家。
- winner 为 `Up` / `YES` 时 `target=1`，winner 为 `Down` / `NO` 时 `target=0`。
- 丢弃 unresolved、ambiguous、missing、non-binary、unknown winner 的市场。
- 保留 `original_target` 和 `label_mismatch` 做审计。
- 可输出 resolved 后的 `development_frame.parquet` 和 `validation_frame.parquet`。

本 PRD 的目标是把这条实验后处理路径迁移为离线训练的一等标签路径，避免未来训练继续默认优化 BTC 原始价格方向标签。

## 2. 目标

BTC/USDT 5m Polymarket 方向预测的 canonical 训练标签必须来自 Polymarket 实际 resolved outcome。

迁移完成后：

- BTC OHLCV 仍然是特征来源。
- 模型训练使用的 `target` 表示 Polymarket 市场最终 resolved 结果。
- BTC 价格派生方向标签只能作为审计或对照诊断，不能作为主训练目标。

主目标保持不变：

```text
selection_score = utility / downside_risk
coverage >= 0.70
```

## 3. 非目标

- 不改变线上 feature builder。
- 不让 execution layer 重新计算 BTC 特征。
- 不改变阈值搜索公式，只强化 `coverage >= 0.70` 硬约束。
- 不在本次迁移中引入新模型族。
- 不立即删除 BTC price-derived label builder；先保留为诊断或兼容 fallback。
- 不使用未 resolved 的实时 Polymarket 市场作为训练标签。

## 4. 产品需求

### 4.1 canonical label source

BTC 5m Polymarket 训练的 canonical binary `target` 必须来自 resolved Polymarket outcome。

要求：

- resolved winner 是 `Up` 或 `YES` 时，`target=1`。
- resolved winner 是 `Down` 或 `NO` 时，`target=0`。
- unresolved、ambiguous、missing、non-binary、unknown winner 的行必须从训练和验证集中剔除。
- 最终训练 frame 继续使用现有 `target` 列名，保持模型训练、评估、阈值搜索兼容。

### 4.2 slug 和 timestamp 映射

每个训练样本必须确定性映射到唯一一个 Polymarket BTC 5m market slug。

slug 规则：

```text
polymarket_slug = btc-updown-5m-{epoch_seconds(market_t0)}
```

timestamp 来源：

- 优先使用 `market_t0`。
- 仅 legacy frame 缺少 `market_t0` 时 fallback 到 `timestamp`。
- timestamp 必须按 UTC 处理。

该映射必须有单元测试覆盖。

### 4.3 resolved market loader

需要把当前实验脚本里的 resolved label 逻辑提升为一等模块，而不是继续藏在 `scripts/experiments` 下。

要求：

- 支持从 Parquet 缓存读取 resolved market labels。
- 支持按 slug 从 Gamma 补拉缺失 closed markets。
- 支持按时间窗口批量拉取 closed markets。
- 支持复用 seed/cache，避免重复请求已 resolved 的 slug。
- 原始 label 审计数据与训练特征分开持久化。

建议新增模块：

```text
src/labels/polymarket_resolved.py
```

建议新增脚本：

```text
scripts/data/step4_features/build_polymarket_resolved_label_store.py
```

### 4.4 label store

新增持久化 label store，不能只依赖某次 experiment output。

建议默认路径：

```text
artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet
```

建议 schema：

```text
polymarket_slug: string
market_t0: timestamp[UTC]
target: int8
polymarket_label_status: string
market_id: string
condition_id: string
question: string
endDate: timestamp/string
closedTime: timestamp/string
closed: bool
umaResolutionStatus: string
outcomes: string/list
outcomePrices: string/list
winner: string
label_version: string
fetched_at: timestamp[UTC]
source: string
```

唯一性要求：

- 每个 `polymarket_slug` 只能有一行。
- duplicate slug 必须 QA fail，除非内容完全一致且可确定性去重。

### 4.5 dataset builder 集成

`build_training_frame()` 必须支持 resolved Polymarket label 作为配置化 label source。

建议配置：

```yaml
objective:
  label: polymarket_resolved_btc_updown_5m
  optimize_metric: selection_score
  min_coverage: 0.70

horizons:
  specs:
    "5m":
      label_builder: polymarket_resolved
      label_params:
        label_version: polymarket_resolved_gamma_v1
        label_store_path: ./artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet
        slug_template: btc-updown-5m-{epoch}
        unresolved_policy: drop
        win_threshold: 0.99
```

集成要求：

- 特征构建逻辑保持不变。
- label builder 只返回 resolved rows。
- delayed feature alignment 继续使用 `market_t0` 和 `feature_timestamp`。
- `abs_return`、`signed_return` 可以继续作为诊断或样本权重输入，但不能定义 `target`。
- 如保留 `original_btc_direction_target`，只能作为审计列，并且必须被 feature schema 排除。

### 4.6 reporting

每个训练报告和 artifact manifest 必须记录 label source。

必需字段：

```text
label_source
label_version
label_store_path
resolved_label_count
unresolved_or_missing_label_count
label_mismatch_count_vs_btc_direction
label_mismatch_rate_vs_btc_direction
polymarket_target_mean
coverage_constraint_min: 0.70
```

报告必须明确写出：

```text
target is Polymarket resolved outcome, not BTC OHLCV direction
```

### 4.7 metrics 和 threshold search

阈值搜索继续优化：

```text
selection_score
```

硬约束：

```text
coverage >= 0.70
```

以下情况必须拒绝结果：

- `coverage < 0.70`
- `threshold_search.best.constraint_satisfied != true`
- `accepted_sample_accuracy <= 0.50`
- `utility <= 0`

### 4.8 可复现性

每个 resolved-label 实验必须保存：

- exact config
- label store path
- label store version/hash
- resolved label build report
- training report
- threshold search output
- cached development/validation split
- git commit hash

resolved label store 必须被视为输入 artifact，不能让训练过程隐式依赖实时网络请求。

## 5. 建议架构

### 5.1 数据流

```text
Binance OHLCV / feature inputs
        |
        v
build_feature_frame()
        |
        v
feature frame keyed by feature_timestamp / market_t0

Polymarket Gamma closed markets
        |
        v
resolved label store keyed by polymarket_slug / market_t0
        |
        v
polymarket_resolved label builder
        |
        v
build_training_frame()
        |
        v
resolved-label TrainingFrame
        |
        v
chronological train / validation split
        |
        v
binary selective model + threshold search
```

### 5.2 模块职责

`src/labels/polymarket_resolved.py`

- 从 resolved label store 构建 label frame。
- 将 market timestamp 转换为 slug。
- 强制 one-to-one slug join。
- 丢弃 unresolved labels。
- 保留必要 audit metadata。

`src/labels/registry.py`

- 注册 `polymarket_resolved` label builder。

`src/data/dataset_builder.py`

- 继续编排 feature/label merge。
- 确保 Polymarket audit columns 不进入 `feature_columns`。
- 输出 label QA metadata。

`scripts/data/step4_features/build_polymarket_resolved_label_store.py`

- 拉取并缓存 closed markets。
- 解析 resolved outcome。
- 写入 label store 和 QA report。

`scripts/experiments/build_polymarket_resolved_training_frame.py`

- 暂时保留为兼容 wrapper。
- first-class label builder 验证通过后标记为 legacy。

## 6. 迁移计划

### Phase 1：提升实验逻辑为共享模块

- 将 `scripts/experiments/build_polymarket_resolved_training_frame.py` 中可复用函数迁移到 `src/labels/polymarket_resolved.py` 或 `src/data/polymarket_labels.py`。
- 实验脚本改成薄 wrapper。
- 增加 slug mapping、outcome parsing、ambiguous labels、label-store join 测试。

### Phase 2：构建 label store

- 新增 label-store builder script。
- 生成覆盖当前 train/validation 时间范围的 BTC 5m resolved label store。
- 保存 label build report，至少包含：
  - requested slug count
  - fetched market count
  - resolved count
  - unresolved count
  - ambiguous count
  - duplicate count
  - target mean

### Phase 3：接入 dataset builder

- 注册 `polymarket_resolved` label builder。
- 在 `horizons.specs.5m.label_params` 下增加 label store 配置。
- 更新 `build_training_frame()` QA，确保 Polymarket audit columns 被排除出 features。
- 确保 `target` 在 split 前就已经替换为 resolved label，而不是 split 后处理。

### Phase 4：训练 resolved-label baseline

- 创建实验配置，例如：

```text
experiments/configs/<timestamp>_polymarket_resolved_label_baseline.yaml
```

- 使用正常 `train_model.py` 路径训练。
- 验证 report 包含 resolved label metadata。
- 与旧 BTC-derived label baseline 做迁移参考比较，但不能声称分数直接同口径可比，因为 target 已经变化。

### Phase 5：废弃 post-processing path

- 保留 `build_polymarket_resolved_training_frame.py` 仅用于 audit/replay。
- 文档说明新实验必须使用 label builder 和 label store。
- 增加测试或 lint，阻止 canonical production experiment config 继续使用 `objective.label: settlement_direction`。

## 7. 测试要求

必需测试：

- `test_polymarket_resolved_slug_mapping_uses_market_t0`
- `test_polymarket_resolved_label_uses_winning_outcome_price`
- `test_polymarket_resolved_label_drops_unresolved_market`
- `test_polymarket_resolved_label_drops_ambiguous_outcome_prices`
- `test_polymarket_resolved_label_store_requires_unique_slug`
- `test_build_training_frame_uses_polymarket_target_not_btc_target`
- `test_polymarket_audit_columns_are_not_features`
- `test_delayed_feature_alignment_preserved_with_resolved_labels`
- `test_threshold_search_requires_coverage_070`
- `test_report_contains_resolved_label_metadata`

现有 `tests/test_polymarket_resolved_training_frame.py` 应保留，并在逻辑提升为共享模块后适配新 import path。

## 8. 验收标准

迁移完成必须同时满足：

1. canonical training config 使用 `label_builder: polymarket_resolved`。
2. 训练 `target` 来自 Polymarket 实际 resolved outcome。
3. BTC price-derived target 不再作为模型训练目标。
4. resolved label store 已保存，并由 config 显式引用。
5. training report 包含 label source、label version、label store path、mismatch diagnostics。
6. 离线/线上 feature logic 保持一致。
7. threshold search 强制 `coverage >= 0.70`。
8. tests pass。
9. 新 resolved-label baseline experiment 已运行并提交。
10. experiment summary 明确说明该结果使用新 label，不能与 BTC-derived-label score 直接同口径比较。

## 9. 风险

- Gamma API 字段可能随市场或时间变化。
- 部分 timestamp 对应 slug 可能缺失。
- 使用 `outcomePrices` 接近 1/0 判断 resolved winner 是实用代理；ambiguous 市场必须 drop，不能猜测。
- 因 unresolved/missing markets 被剔除，训练样本量可能下降。
- BTC-derived feature engineering 仍然有效，但 BTC-derived target 相关诊断必须明确标为 audit only。

## 10. 待决策

- label-store fetching 是否允许在 `train_model.py` 内发生。不允许，作为独立 pre-step，保证可复现。
- `original_btc_direction_target` 是否保留在 training frame。不需要
- canonical production config 缺少 resolved label store 时是否 fail fast。建议：必须 fail fast。

