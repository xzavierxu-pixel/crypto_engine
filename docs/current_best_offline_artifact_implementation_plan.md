# 当前最佳离线 Artifact 构建实现文档

## 背景

目标是在当前代码逻辑不大改的前提下，复用历史离线实验中已经验证过的模型、模型参数、特征配置和时间对齐方案，重新构建当前标签语义下的最佳离线 artifacts。

当前主线已经迁移为：

- 标签：`polymarket_resolved_btc_updown_5m`
- label builder：`polymarket_resolved`
- 覆盖率硬约束：`objective.min_coverage: 0.70`
- 优化目标：`validation_metrics.selection_score`
- 决策对齐：`decision_alignment.enabled: true`，`delayed_feature_offset`

因此历史 BTC OHLCV-derived label 下的最佳 artifact 不能直接作为当前默认 artifact，只能复用其经过实验筛选的配置思想和参数，再用 Polymarket 实际 resolve 结果重新训练、重新验收。

## 当前审查结论

仓库当前分支为 `version1`。工作区只有一个既有未提交文件：

- `execution_engine/execution_engine_flow_timeline.md`

该文件与本实现无关，迁移时不应混入提交。

当前 `config/settings.yaml` 已经是新标签配置：

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
```

当前已生成并可作为新标签 baseline 的 artifact：

```text
artifacts/data_v2/experiments/20260520_polymarket_resolved_label_baseline/report.json
```

验证集指标：

| artifact | label | selection_score | coverage | accepted_sample_accuracy | utility | accepted_count | thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20260520_polymarket_resolved_label_baseline` | Polymarket resolved | `0.5657774529` | `0.7025977504` | `0.6883933676` | `0.2647295126` | `5247` | `t_up=0.540`, `t_down=0.400` |

历史可参考但不能直接采用的最佳类配置：

| artifact / config | label | 口径 | selection_score | coverage | accepted_sample_accuracy | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `20260515_t_plus_1_delayed_validation` | BTC settlement direction | 30d validation | `0.5802688117` | `0.7119528845` | `0.6911073510` | 与当前新标签最接近，主要价值是 T+1 delayed alignment 和当前默认 feature/profile 组合 |
| `20260515_t_plus_1_delayed_online_full_train` | BTC settlement direction | full-train style report | `0.9450449200` | `0.7822246018` | `0.7611225188` | 不是同等 validation acceptance 口径，不能作为默认验收分数 |
| `20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020` | BTC settlement direction | 旧约束 `coverage >= 0.40` | 约 `0.1902780361` | 约 `0.4043545879` | 约 `0.5951923077` | 模型族和参数可迁移，但旧 coverage 与标签语义均不满足当前目标 |

分支参考：

- `main`：仍是旧 `settlement_direction`，且 `min_coverage: 0.40`。
- `baseline`：已有 `min_coverage: 0.70` 和 T+1 delayed alignment，但仍是旧 `grid_direction` 标签。
- 当前 `version1`：已迁移到 Polymarket resolved label，是本次构建的唯一可接受基线。

## 推荐实现目标

生成一个新的当前默认离线 artifact：

```text
artifacts/data_v2/experiments/20260520_polymarket_resolved_best_config_rebuild/
```

它必须满足：

- 使用 Polymarket resolved outcome 作为唯一训练 target。
- 保持当前 feature builder、label builder、threshold search、metrics/report 逻辑不变。
- 复用历史最佳中已经验证过的可迁移配置：
  - T+1 delayed decision alignment。
  - `coverage >= 0.70` threshold search。
  - 当前大特征 profile。
  - 先以 LightGBM baseline 建立可比 artifact。
  - 再迁移历史最强的 `catboost_lgbm_logit_blend + platt_logit` 配置作为候选。
- 不能复用旧标签 artifact 的模型文件、thresholds 或验证分数。

## 候选配置优先级

### 候选 A：当前新标签默认 LightGBM

用途：新标签 baseline 和回归对照。

关键配置：

```yaml
model:
  active_plugin: lightgbm

calibration:
  active_plugin: none

decision_alignment:
  enabled: true
  mode: delayed_feature_offset
  feature_offset_minutes: 1
  order_delay_seconds_after_feature_time: 8
  row_policy: delayed_1m_synthetic_decision_row
```

当前结果已经存在：

```text
selection_score=0.5657774529
coverage=0.7025977504
accepted_sample_accuracy=0.6883933676
```

### 候选 B：历史 best blend 迁移到 Polymarket label

用途：最可能提升当前 artifact 的候选。

来源配置：

```text
experiments/configs/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020.yaml
```

需要迁移的部分：

```yaml
model:
  active_plugin: catboost_lgbm_logit_blend
  plugins:
    catboost_lgbm_logit_blend:
      catboost_weight: 0.9770
      catboost:
        iterations: 1200
        learning_rate: 0.015
        depth: 5
        l2_leaf_reg: 30.0
        random_seed: 42
        loss_function: Logloss
        eval_metric: Logloss
        random_strength: 2.0
        bagging_temperature: 0.5
        allow_writing_files: false
        verbose: false
      lightgbm:
        n_estimators: 1600
        learning_rate: 0.01
        boosting_type: dart
        num_leaves: 31
        min_child_samples: 120
        subsample: 0.6
        subsample_freq: 10
        colsample_bytree: 0.35
        reg_alpha: 1.2
        reg_lambda: 8.0
        max_depth: 6
        scale_pos_weight: null
        verbosity: -1
        random_state: 42

calibration:
  active_plugin: platt_logit
  plugins:
    platt_logit:
      C: 0.2
      max_iter: 1000
```

必须保留当前新标签部分：

```yaml
objective:
  label: polymarket_resolved_btc_updown_5m
  min_coverage: 0.70

horizons:
  specs:
    "5m":
      label_builder: polymarket_resolved
      label_params:
        label_version: polymarket_resolved_gamma_v1
        label_store_path: ./artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet
```

预期：不能假设一定优于 LightGBM。旧 best blend 在旧标签下更偏向高 precision、低 coverage，迁移后必须让 threshold search 在 `coverage >= 0.70` 下重新选择阈值。

### 候选 C：T+1 delayed validation 配置迁移

用途：验证当前默认 LightGBM 与旧 T+1 配置是否实质一致。

来源配置：

```text
experiments/configs/20260515_t_plus_1_delayed_validation.yaml
```

它与当前 `config/settings.yaml` 的主要差别是旧标签语义；模型仍是 LightGBM。当前新标签 baseline 基本已经覆盖这个候选，因此只需要作为对照，不需要优先生成新 artifact。

## 实施步骤

### 1. 固化实验配置

新建实验配置：

```text
experiments/configs/20260520_polymarket_resolved_best_blend_rebuild.yaml
```

生成方式：

- 以当前 `config/settings.yaml` 为底。
- 只替换 `model.active_plugin`、`model.plugins.catboost_lgbm_logit_blend`、`calibration.active_plugin` 和 `calibration.plugins.platt_logit`。
- 不改变 label、feature builder、decision alignment、threshold search、dataset 窗口。

### 2. 复用当前 resolved split 训练

优先使用已经由 Polymarket resolved label 构建的 split，保证与 baseline 可比：

```text
artifacts/data_v2/experiments/20260517_polymarket_resolved_labels_split
```

训练命令：

```powershell
rtk proxy powershell -NoProfile -Command "python scripts\model\train_model.py --cached-split-dir artifacts\data_v2\experiments\20260517_polymarket_resolved_labels_split --output-dir artifacts\data_v2\experiments\20260520_polymarket_resolved_best_blend_rebuild --config experiments\configs\20260520_polymarket_resolved_best_blend_rebuild.yaml"
```

### 3. 验收指标

新 artifact 必须满足：

```text
validation_metrics.coverage >= 0.70
validation_metrics.accepted_sample_accuracy > 0.50
validation_metrics.utility > 0
threshold_search.best.constraint_satisfied == true
label_metadata.label_source == polymarket_resolved
label_metadata.target_semantics 包含 Polymarket resolved outcome
```

主排序：

1. `validation_metrics.selection_score`
2. `validation_metrics.utility`
3. `validation_metrics.coverage`
4. `validation_metrics.accepted_count`
5. 线上可用性和实现复杂度

当前必须打败的新标签 baseline：

```text
selection_score=0.5657774529
utility=0.2647295126
accepted_sample_accuracy=0.6883933676
coverage=0.7025977504
accepted_count=5247
```

### 4. 验证命令

训练后至少运行：

```powershell
rtk proxy powershell -NoProfile -Command "python -m pytest -q tests/test_binary_selective_model.py tests/test_model_pipeline.py tests/test_model_artifacts.py tests/test_polymarket_resolved_training_frame.py"
```

如果代码没有改动，只迁移配置并训练 artifact，可接受 targeted tests 加 artifact/report 审查；若改动模型、calibration、dataset builder 或 label 逻辑，则必须跑全量：

```powershell
rtk proxy powershell -NoProfile -Command "python -m pytest -q"
```

### 5. 记录和提交

实验完成后更新：

```text
experiments/optimization_log.md
```

记录字段：

```text
git_commit
config_path
report_path
primary_metric
signal_coverage
coverage_constraint_satisfied
label_source
target_semantics
before / after selection_score
before / after utility
before / after accepted_sample_accuracy
before / after accepted_count
before / after coverage
```

只提交：

- 新实验配置。
- 优化日志记录。
- 如有必要，提交实现代码变更。

不要把 `execution_engine/execution_engine_flow_timeline.md` 混入提交。

## 接受 / 拒绝规则

接受为当前默认配置的条件：

- 新 artifact 使用 Polymarket resolved label。
- `coverage >= 0.70`。
- `accepted_sample_accuracy > 0.50`。
- `utility > 0`。
- `selection_score > 0.5657774529`。
- report 和 artifact manifest 都记录 resolved label metadata。
- 训练、离线验证、推理和信号生成仍走共享 core。

拒绝条件：

- 任一 label metadata 显示旧 `settlement_direction` 或 `grid_direction`。
- 通过降低 coverage 到 0.70 以下获得更高 score。
- 只复用旧 artifact 的模型或阈值，不重新训练。
- 新配置引入无法在线构建的 feature。
- 新结果没有可复现 config、report 和 commit。

## 默认配置迁移策略

如果候选 B 胜出：

1. 将 `config/settings.yaml` 的模型和 calibration 部分迁移为候选 B。
2. 保持当前 label、coverage、decision alignment、feature profile 不变。
3. 重新生成 canonical artifact，例如：

```text
artifacts/data_v2/experiments/current_default_polymarket_resolved/
```

4. 在优化日志中声明它是当前默认 artifact，并记录 commit hash。

如果候选 B 没有胜出：

1. 保留当前 LightGBM 新标签 baseline 为默认。
2. 在优化日志中记录 best blend 迁移失败的原因。
3. 不因为旧标签历史分数更高而替换当前默认配置。

## 风险点

- 历史 best blend 是旧标签、旧 coverage floor 下的最优；迁移到 Polymarket resolved label 后，概率分布和最佳阈值可能变化。
- `platt_logit` 校准如果在 validation 上调参，validation 分数仍是 threshold-tuned optimistic，应在报告里继续标记。
- 旧 `20260515_t_plus_1_delayed_online_full_train` 的 `0.9450` 不是同等 acceptance validation 口径，不能用它证明当前默认 artifact 优于 baseline。
- `artifacts/` 默认不入 git；可复现性依赖 config、日志、代码 commit 和本地 artifact 路径记录。

## 当前建议

下一步直接做候选 B：用当前 resolved label split 训练 `catboost_lgbm_logit_blend + platt_logit`。这是历史实验中最有信息量、同时当前代码已经支持的模型族迁移，且不需要改 label/feature/threshold 逻辑。

# 新增需求
代码默认值和 execution_engine/config.yaml设置为：min_edge = 0.04，max_buy_price = 0.8
Prewarm需要开启，目的是减少在线的下单时延
解决systemd 调度和配置不一致的问题：execution_engine/config.example.yaml 写 trigger_delay_seconds=68，paper experiment 会用它；但 systemd timer 是 OnCalendar=*:01/5，约等于窗口开始后 60 秒。run_once.py 本身不 sleep，也不使用 schedule.trigger_delay_seconds。我只希望线上能够拿到完整的T结束的数据，所以这部分时间应该是需要统一的，然后prewarm也应该生效才对。