# 当前离线完整流程分析

本文档描述当前代码库中 BTC/USDT 5 分钟方向预测的离线流程。重点是代码实际执行路径，而不是理想化设计。

## 1. 离线流程总览

当前离线主线可以分成七段：

1. 原始数据回填：从 Binance Vision 下载 spot、UM futures、CM futures、option 公共历史数据。
2. 数据标准化和 QA：把 raw CSV 规范化为稳定 Parquet，并生成 schema/QA manifest。
3. 可选二级特征库：用 1s kline、aggTrades、bookTicker、depth 等生成 `second_level` 特征库。
4. 训练样本构建：用共享 `build_training_frame()` 统一完成 OHLCV 规范化、特征构建、标签构建、特征/标签对齐、样本权重和泄漏列排除。
5. 时间切分：按最近 `train_days` / `validation_days` 做 chronological train/validation，并保留 purge gap。
6. 模型训练和阈值搜索：训练 binary selective 模型，校准概率，在 validation 上搜索 `t_up`/`t_down`。
7. 报告和 artifact 持久化：输出模型、校准器、阈值搜索、报告、manifest、诊断切片和缓存 split。

当前主要入口是：

- 数据集构建：`scripts/data/step4_features/build_dataset.py`
- 单次离线训练：`scripts/model/train_model.py`
- 核心样本构建：`src/data/dataset_builder.py`
- 核心训练：`src/model/train.py`
- 指标和阈值搜索：`src/model/evaluation.py`
- 线上共享推理路径：`src/services/signal_service.py`

## 2. 配置驱动

离线流程由 `config/settings.yaml` 驱动，`src/core/config.py` 负责解析为 dataclass。

关键配置：

- `market`: 当前是 Binance `BTC/USDT`、`1m`。
- `horizons.specs.5m`: 当前 5m horizon 使用 `grid_direction` 标签、`core_5m` 特征 profile、`selective_binary_policy` 信号策略。
- `dataset`: 训练时间范围、是否只保留网格行、是否丢弃不完整样本。
- `validation`: 当前 chronological validation 的 `train_days` 和 `validation_days`。
- `decision_alignment`: 当前启用 `delayed_feature_offset`，`feature_offset_minutes: 1`。
- `objective`: 当前 `optimize_metric: selection_score`，`min_coverage` 在现有 `config/settings.yaml` 中是 `0.70`。项目说明要求最低有效结果 `coverage >= 0.70`，实际训练会以配置值 `0.70` 作为阈值搜索硬约束。
- `threshold_search`: `t_up` / `t_down` 搜索范围和步长。
- `sample_weighting`: 用 abs return 线性 ramp 生成样本权重。
- `features.profiles.core_5m.packs`: 控制所有离线特征包。
- `derivatives` 和 `second_level`: 控制衍生品和秒级特征是否进入离线训练。
- `model.active_plugin`: 当前主配置为 `lightgbm`。
- `calibration.active_plugin`: 当前主配置为 `none`。

## 3. 原始数据回填

入口：`scripts/data/step1_acquire/backfill_binance_public_history.py`

脚本读取 `settings.data_backfill`，从 Binance Vision 规划下载请求：

- spot: `klines`、`aggTrades`、`trades`、`bookTicker`
- futures_um: `klines`、mark/index/premium klines、funding、bookTicker、metrics、aggTrades、trades、depth、liquidation
- futures_cm: 同类 futures 数据
- option: BVOLIndex、EOHSummary

输出目录默认来自 `settings.second_level.data_root`，当前是 `./artifacts/data_v2`。raw 数据落在 `raw/<market_family>/<data_type>/<symbol>/...` 下。

该步骤会生成：

- `artifacts/data_v2/manifests/download_manifest.json`
- `artifacts/data_v2/manifests/file_checksums.json`

重要行为：

- 会按 monthly / daily 窗口规划数据。
- 会查询 Binance bucket listing 判断对象是否存在。
- 如果 raw CSV 已存在，会跳过下载。
- 如果启用 checksum，会校验 zip SHA256 后再解压。

## 4. 数据标准化和 QA

入口：`scripts/data/step2_normalize/normalize_binance_public_history.py`

流程：

1. 调用 `normalize_binance_public_history(output_root)`。
2. 把 raw CSV 规范化为 normalized Parquet。
3. 调用 `run_binance_public_qa(output_root)` 生成 QA。

输出：

- normalized Parquet 数据，位于 `artifacts/data_v2/normalized/...`
- `artifacts/data_v2/manifests/schema_manifest.json`
- `artifacts/data_v2/manifests/qa_manifest.json`

离线训练主入口 `train_model.py` 本身不负责 raw 数据标准化；它要求传入已经可读的 OHLCV CSV / Parquet / Feather，或传入缓存好的 split。

## 5. 二级特征库

入口：`scripts/data/step4_features/build_second_level_feature_store.py`

这个步骤是可选的，由 `settings.second_level.enabled` 控制。当前主配置里 `second_level.enabled: false`，所以默认训练不会读取二级特征。

当启用时，脚本可读取：

- spot 1s kline
- spot aggTrades
- spot bookTicker
- depth snapshots
- perp 1s kline
- perp bookTicker
- ETH 1s kline

输出默认路径：

```text
artifacts/data_v2/second_level/version=<feature_store_version>/market=<market>
```

`train_model.py` 中的读取逻辑：

- 如果 `settings.second_level.enabled` 为 true，会优先使用 `--second-level-feature-store`。
- 未显式传入时，使用 `settings.second_level.feature_store_path`。
- 如果默认 store 不存在，会 warning 并跳过；如果用户显式传入但不存在，会报错。
- 加载函数是 `load_sampled_second_level_features(source, second_level_store_path)`，按 1m source 采样对齐。

## 6. 衍生品特征对齐

入口：`src/data/derivatives/feature_store.py`

`train_model.py` 会先调用：

```text
resolve_derivatives_paths()
load_derivatives_frame_from_settings()
```

路径模式：

- `latest`: 从各子配置的 `path` 读取。
- `archive`: 从 `archive_path` 下的 normalized archive 自动加载。

当前主配置：

- `derivatives.enabled: false`
- 各子源 funding / basis / oi / options / book_ticker 的 `enabled` 多为 true，但顶层 false，因此默认不会加入衍生品特征。

如果启用衍生品，`build_feature_frame()` 会通过 `DerivativesFeatureStore.attach_to_spot()` 把衍生品原始 frame 对齐到 spot OHLCV。随后特征包如 `derivatives_funding`、`derivatives_basis`、`derivatives_book_ticker`、`derivatives_oi`、`derivatives_options` 才会基于这些列生成特征。

代码会清理衍生品 helper/meta 列，避免把原始来源字段直接作为模型特征。

## 7. 训练样本构建

核心入口：`src/data/dataset_builder.py::build_training_frame()`

输入：

- raw OHLCV frame
- settings
- horizon name，默认 `5m`
- 可选 derivatives frame
- 可选 second-level features frame

执行顺序：

1. `normalize_ohlcv_frame()` 规范化 OHLCV，排序、时间列标准化、去重、基础校验。
2. `get_horizon_spec(settings, horizon_name)` 解析 horizon。
3. 根据 `decision_alignment` 计算 feature offset。
4. 调用 `build_feature_frame()` 构建特征。
5. 调用 label builder 构建标签。
6. 按 timestamp 或 delayed feature timestamp 合并特征和标签。
7. 合并 grid metadata。
8. 构建 abs/signed return 辅助列。
9. 按 `dataset.train_start` / `dataset.train_end` 过滤时间范围。
10. 推断 feature columns。
11. 执行 feature schema QA 和泄漏列排除。
12. 可选丢弃不完整样本。
13. 计算样本权重。
14. 返回 `TrainingFrame`。

### 7.1 特征构建

核心入口：`src/features/builder.py::build_feature_frame()`

执行顺序：

1. 再次规范化 OHLCV。
2. 读取 horizon 对应 feature profile，例如 `core_5m`。
3. 如果 `settings.derivatives.enabled`，先 attach 衍生品数据。
4. 如果传入 second-level frame，则按 `timestamp` one-to-one merge。
5. 按 profile 中的 `packs` 依次调用 `get_feature_pack(pack_name).transform()`。
6. 删除衍生品 helper/meta 列。
7. 添加 grid columns。
8. 添加 `asset`、`horizon`、`feature_version`。
9. 根据 `select_grid_only` 选择是否只保留 5m 网格行。

当前 `decision_alignment.enabled: true` 时，`build_training_frame()` 调用特征构建时传入 `select_grid_only=False`，因为标签 t0 和特征时间会错开 1 分钟，需要保留非 5m 网格特征行用于合并。

### 7.2 标签构建

当前 5m horizon 使用 `src/labels/grid_direction.py::GridDirectionLabelBuilder`。

标签规则：

```text
y = 1{future_close >= open[t0]}
future_close = close.shift(-horizon.future_close_offset)
```

结合项目说明，当前标签语义是：

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

标签只在 grid t0 行上有效，非 grid 行被置为 NA。最后 `select_grid_rows()` 保留 5m 网格标签行。

### 7.3 Delayed Feature Alignment

当前主配置：

```yaml
decision_alignment:
  enabled: true
  mode: delayed_feature_offset
  feature_offset_minutes: 1
```

因此训练样本对齐逻辑是：

1. 标签 frame 的 `timestamp` 改名为 `market_t0`。
2. 新增 `feature_timestamp = market_t0 + 1 minute`。
3. feature frame 的 `timestamp` 改名为 `feature_timestamp`。
4. 训练样本用 `feature_timestamp` 做 one-to-one merge。
5. 最终训练 frame 的主 `timestamp` 仍设回 `market_t0`。
6. `decision_time` 记录 `feature_timestamp`。

这意味着模型用 t0 后 1 分钟可见的特征来预测 t0 这一轮 Polymarket 结算标签。文档和报告会标记 `validation_threshold_tuned: true` 和 `validation_result_optimistic: true`。

## 8. 特征列选择和泄漏防护

特征列推断：`infer_feature_columns(df)`

规则：

- 排除 `BASE_DATASET_COLUMNS`。
- 排除 raw metadata columns。
- 排除 `raw_`、`source_`、`checksum_` 前缀。
- 排除衍生品 helper/meta 列。

硬禁止泄漏列：

```text
target
future_close
abs_return
signed_return
stage1_target
stage2_target
```

`assert_feature_schema()` 会对 feature columns 做硬检查。`assert_feature_quality()` 还会检查：

- 全空特征
- forbidden metadata features
- label-derived leakage features

如果 `settings.dataset.drop_incomplete_candles: true`，则 `drop_incomplete_samples()` 会删除特征或 target 不完整的样本。

## 9. 样本权重

入口：`compute_sample_weight(abs_return, settings)`

当前主配置启用：

```yaml
sample_weighting:
  enabled: true
  mode: linear_ramp
  min_abs_return: 0.0001
  full_weight_abs_return: 0.0003
  min_weight: 0.35
  max_weight: 1.00
```

逻辑：

- abs return 小于 `min_abs_return` 的样本给 `min_weight`。
- 从 `min_weight` 到 `max_weight` 线性 ramp。
- 达到 `full_weight_abs_return` 后封顶为 `max_weight`。

`TrainingFrame.sample_weight_column` 只有在 `sample_weighting.enabled` 为 true 时才指向样本权重列。

## 10. 单次离线训练入口

入口：`scripts/model/train_model.py`

两种输入模式：

1. `--input`: 传入 OHLCV CSV / Parquet / Feather，脚本现构建 training frame。
2. `--cached-split-dir`: 传入已有 `development_frame.parquet` 和 `validation_frame.parquet`，跳过样本构建。

典型命令：

```powershell
rtk proxy powershell -NoProfile -Command "python scripts/model/train_model.py --input <ohlcv.parquet> --output-dir artifacts/data_v2/experiments/<run_name> --config config/settings.yaml --horizon 5m"
```

脚本流程：

1. 加载 settings。
2. 创建 output dir。
3. 解析 derivatives paths。
4. 决定 train/validation window 天数。
5. 如果是 cached split：
   - 读取 `development_frame.parquet`
   - 读取 `validation_frame.parquet`
   - 用 `load_cached_training_split()` 重新推断 feature columns 并检查 schema
6. 如果是 raw input：
   - 读取 source OHLCV
   - 加载 derivatives frame
   - 可选加载 second-level feature store
   - 调用 `build_training_frame()`
   - 调用 `split_recent_train_validation_frame()`
   - 写出 cached split
7. 对 split 运行 DQC。
8. 调用 `train_binary_selective_model_from_split()`。
9. 写出模型、校准器、报告和诊断文件。

## 11. 时间切分

当前主训练走 `split_recent_train_validation_frame()`。

逻辑：

```text
valid_end_ts = max(timestamp)
valid_start_ts = valid_end_ts - validation_days
train_start_ts = valid_start_ts - train_days
train = [train_start_ts, valid_start_ts)
validation = [valid_start_ts, valid_end_ts]
```

如果 `purge_rows > 0`，会从 train 尾部删除对应行数，避免 train 和 validation 紧邻。

当前配置：

```yaml
validation:
  train_days: 60
  validation_days: 30
```

注意：`dataset.train_window_days` 和 `dataset.validation_window_days` 也存在，但 `train_model.py` 默认使用 `settings.validation.train_days` 和 `settings.validation.validation_days`，除非命令行显式覆盖。

## 12. 模型训练

核心入口：`src/model/train.py::train_binary_selective_model_from_split()`

流程：

1. 根据 `weighted` 参数决定是否使用样本权重。
2. 调用 `_fit_model(train_frame, settings, stage="binary", validation=valid_frame)`。
3. `_fit_model()` 通过 `create_model_plugin()` 创建当前模型插件。
4. 如果是 LightGBM binary，会自动注入：
   - `objective: binary`
   - `scale_pos_weight`: 配置值，若为 null 则按 train 标签比例计算
5. 模型 fit 时传入：
   - train X/y
   - validation X/y
   - sample_weight
   - sample_weight_valid
6. 对 train 和 validation 分别生成 raw probabilities。
7. 创建 calibration plugin，并用 train raw probability + train y fit。
8. 得到校准后的 train/validation `p_up`。

当前主配置 `calibration.active_plugin: none`，所以校准器通常是 no-op。

## 13. 阈值搜索

入口：`src/model/evaluation.py::search_selective_binary_thresholds()`

搜索空间来自 `settings.threshold_search`：

```yaml
t_up_min: 0.45
t_up_max: 0.70
t_down_min: 0.30
t_down_max: 0.55
step: 0.005
```

对每一组候选：

```text
UP       if p_up >= t_up
DOWN     if p_up <= t_down
ABSTAIN  otherwise
```

计算：

- coverage
- accepted_sample_accuracy
- utility
- downside_risk
- selection_score
- accepted_count
- up_prediction_count
- down_prediction_count
- share_up_predictions
- share_down_predictions
- precision_up / precision_down
- roc_auc / brier_score / log_loss

当 `objective.optimize_metric` 是 `selection_score` 时：

- eligibility 只检查 `coverage >= settings.objective.min_coverage`。
- side-share 和 min signal count guardrail 不参与主目标过滤。
- 排名 key 是：
  1. `selection_score`
  2. `utility`
  3. `coverage`
  4. `accepted_count`
  5. `accepted_sample_accuracy`
  6. 阈值离 0.5 更近

同时代码还会跑一次 `side_guarded_best`，强制 `enforce_min_side_share=True`，用于诊断。

如果没有任何候选满足 coverage，代码不会直接失败，而是 fallback 到所有 records 中最优，并在 `threshold_search.best.constraint_satisfied=false` 和 `fallback_reason` 中标记。这一点读报告时必须检查。

## 14. 核心指标公式

入口：`compute_selective_binary_metrics()`

当前 selection score 公式：

```text
coverage = accepted_count / sample_count
accepted_sample_accuracy = correct accepted predictions / accepted_count
utility = coverage * (2 * accepted_sample_accuracy - 1)
downside_risk = sqrt(coverage * (1 - accepted_sample_accuracy))
selection_score = utility / downside_risk
```

如果 `downside_risk == 0`：

- utility > 0: score = `inf`
- utility < 0: score = `-inf`
- utility = 0: score = `0`

报告中 legacy alias 由 `train_model.py::_with_signal_aliases()` 补齐：

- `up_signal_count = up_prediction_count`
- `down_signal_count = down_prediction_count`
- `total_signal_count = accepted_count`
- `signal_coverage = coverage`
- `overall_signal_accuracy = accepted_sample_accuracy`

## 15. 诊断输出

`train_binary_selective_model_from_split()` 会额外生成：

- `threshold_frontier`: 所有阈值候选记录。
- `boundary_slices`: 按 abs return 桶切片。
- `regime_slices`: 按波动、趋势、spread、volume、session 切片。
- `feature_importance`: LightGBM gain/split 或 CatBoost importance。
- `probability_deciles`: validation `p_up` decile 分布。
- `false_up_slices`: UP 错误样本切片。
- `false_down_slices`: DOWN 错误样本切片。
- `probability_summary`: train/validation 概率分布和 KS。
- `probability_reference`: 用于 drift reference 的概率样本。

这些不是主优化目标，但用于解释结果、排查漂移和查找错误集中区域。

## 16. Artifact 和报告

`train_model.py` 会写入 output dir：

```text
<model_plugin>.binary.pkl
<calibration_plugin>.binary.pkl
artifact_manifest.json
report.json
metrics.json
threshold_search.json
threshold_frontier.csv
boundary_slices.csv
regime_slices.csv
feature_importance.csv
probability_deciles.csv
false_up_slices.csv
false_down_slices.csv
probability_reference.json
development_frame.parquet
validation_frame.parquet
data_quality/
```

关键文件：

- `report.json`: 训练和 validation 指标、window、threshold search、feature columns。
- `artifact_manifest.json`: artifact 可加载所需元数据、config hash、feature columns、阈值、data availability、decision alignment、完整 metrics。
- `metrics.json`: 更轻量的 train/validation metrics 和 threshold summary。
- `threshold_search.json`: 阈值搜索配置、best、side_guarded_best。

加载线上 artifact 的入口：

- `src/model/artifacts.py::load_binary_selective_artifacts()`

它从 manifest/report 中读取：

- model plugin name
- calibration plugin name
- feature columns
- `t_up`
- `t_down`
- base rate
- probability reference
- config hash

## 17. 离线和线上一致性

线上共享路径：

- `src/services/signal_service.py`
- `src/services/feature_service.py`
- `src/model/infer.py`
- `src/signal/policies.py`

一致性来源：

- 线上 `SignalService` 使用 `FeatureService(settings)`。
- `FeatureService` 复用 `build_feature_frame()` 构建特征。
- 推理时 `predict_frame()` 使用 artifact 中保存的 `feature_columns`。
- 决策时 `evaluate_selective_binary_signal()` 优先使用 signal context 中的 `t_up` / `t_down`，这些值由 artifact 注入；配置中的 policy 阈值只有在 signal context 缺失时作为 fallback。

因此，离线 artifact 的 feature columns 和 thresholds 是线上推理一致性的关键。

## 18. 批量实验路径

当前还存在旧的 two-stage / family experiment runner：

- `scripts/experiments/run_model_experiments.py`

它会比较 stage1/stage2 模型和衍生品 ablation variant，输出 `experiment_report.json`、`summary.json`、`summary.md`。但该脚本当前优化的是 two-stage end-to-end PnL/precision/coverage，不是当前 AGENTS.md 中定义的 binary selective `selection_score` 主目标。

对当前主目标来说，优先使用：

- `scripts/model/train_model.py`
- `src/model/train.py::train_binary_selective_model_from_split()`
- `src/model/evaluation.py::search_selective_binary_thresholds()`

## 19. 当前流程的关键检查点

每次离线结果验收至少检查：

1. `report.json.validation_metrics.coverage >= objective.min_coverage`。
2. `threshold_search.best.constraint_satisfied == true`。
3. `validation_metrics.selection_score` 是否优于 baseline。
4. `validation_metrics.utility > 0`。
5. `validation_metrics.accepted_sample_accuracy > 0.50`。
6. `accepted_count` 是否足够大。
7. `up_prediction_count` / `down_prediction_count` 是否极端失衡。
8. `artifact_manifest.json.feature_columns` 不含泄漏列。
9. `decision_alignment.coverage_constraint_satisfied == true`。
10. `development_frame.parquet` 和 `validation_frame.parquet` 是否已保存，可复现。

## 20. 当前实现中需要特别注意的点

- `config/settings.yaml` 当前 `objective.min_coverage: 0.70`，与项目说明中的最低 `0.70` 一致。训练实际按 `0.70` 执行。
- `threshold_search.enabled` 存在，但 `train_binary_selective_model_from_split()` 当前总是执行阈值搜索，没有看到按该开关跳过搜索的逻辑。
- 如果 coverage 约束无候选满足，阈值搜索会 fallback 而不是抛错；必须读 `constraint_satisfied`。
- validation 阈值是在 validation 上调出来的，报告中已标记 optimistic；这符合当前项目说明，但不能当成独立 holdout。
- 当前主配置顶层 `derivatives.enabled: false`、`second_level.enabled: false`，虽然 profile 中列了相关 packs，但默认不会真正 attach 衍生品和二级特征数据。
- `build_dataset.py` 只构建训练 frame 和 QA；完整 artifact/report 由 `train_model.py` 生成。
- `run_model_experiments.py` 是 two-stage 实验路径，不应混同为当前 binary selective 主流程。

## 21. 推荐的最小复现流程

从现有 normalized OHLCV 输入开始：

```powershell
rtk proxy powershell -NoProfile -Command "python scripts/model/train_model.py --input <BTCUSDT_1m.parquet> --output-dir artifacts/data_v2/experiments/<run_name> --config config/settings.yaml --horizon 5m"
```

如果已经有缓存 split：

```powershell
rtk proxy powershell -NoProfile -Command "python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/<old_run> --output-dir artifacts/data_v2/experiments/<new_run> --config config/settings.yaml --horizon 5m"
```

验收时优先查看：

```text
artifacts/data_v2/experiments/<run_name>/report.json
artifacts/data_v2/experiments/<run_name>/threshold_search.json
artifacts/data_v2/experiments/<run_name>/artifact_manifest.json
```

主比较字段：

```text
validation_metrics.selection_score
validation_metrics.utility
validation_metrics.accepted_sample_accuracy
validation_metrics.coverage
validation_metrics.accepted_count
validation_metrics.up_prediction_count
validation_metrics.down_prediction_count
threshold_search.best.constraint_satisfied
```
