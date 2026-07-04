# Arbor 研究任务说明：BTCUSDT 5 分钟 Polymarket `sum_pnl` 优化

日期：2026-07-03

## 1. 研究目标

在不改变成交、方向和时间对齐语义的前提下，使用仓库中已经存在的数据，最大化最后一个完整月验证集上的 `sum_pnl`。

本轮固定验证窗口为：

```text
2026-04-11 00:15:00+00:00 至 2026-05-10 23:50:00+00:00
```

训练、校准、特征选择、模型选择和 policy 参数选择只能使用该窗口之前的数据。验证标签只允许在最终评分时读取，不允许用于挑选模型、阈值、分组或 bid policy。若 Arbor 需要反复迭代，应在 2026-04-10 之前的数据内建立内部 chronological development splits；最后一个月应视为冻结的 `B_test`，而不是可反复调参的 `B_dev`。

主指标：

```text
maximize validation.sum_pnl
```

必须同时报告 `accepted_count`、`order_count`、`order_coverage`、`trade_count`、`fill_rate`、`mean_accepted_pnl`、`mean_pnl_filled`、`mean_bid`、`win_pnl_sum`、`loss_pnl_sum`、`correct_fill_rate`、`wrong_fill_forced`，并保留方向和概率校准指标，避免通过意外改变样本宇宙或语义制造虚假提升。

## 2. 可使用的数据

以下本地数据均可读取并用于研究，但标签、未来信息和成交结果字段只能作为训练目标或评估字段，不能作为决策时特征。

### 2.1 `artifacts/`

整个 `artifacts/` 目录可用，重点包括：

- `artifacts/data_v2/raw/`：原始数据缓存。
- `artifacts/data_v2/normalized/`：规范化 Binance/市场数据。
- `artifacts/data_v2/labels/`：resolved Polymarket 标签及标签构建审计。
- `artifacts/data_v2/second_level/`：已启用的秒级特征数据。
- `artifacts/data_v2/datasets/`：训练 frame。现有 BTCUSDT 5m frame 覆盖 2025-12-18 至 2026-05-10。
- `artifacts/data_v2/polymarket_l2/`：Polymarket L2 数据、特征包、manifest 和 QA；现有 common-window 数据主要覆盖 2026-02-13 至 2026-03-27。
- `artifacts/data_v2/reports/`、`artifacts/reports/`：方向模型和其他历史报告。
- `artifacts/logs/`、`artifacts/state/`、`artifacts/aws_*`：执行日志、状态和服务器侧历史，可用于只读诊断；不得把交易结束后才产生的字段回灌为模型特征。

### 2.2 `price_estimator/data/`

该目录全部可用：

- `price_estimator_train.parquet`：15,657 行、1,826 列，2026-02-12 至 2026-04-10。
- `price_estimator_valid.parquet`：7,432 行、1,826 列，2026-04-11 至 2026-05-10。
- `predictions_train.parquet`：方向输出与历史价格估计预测，训练窗口。
- `predictions_validation.parquet`：方向输出与历史价格估计预测，冻结验证窗口。
- `market_refs.parquet`：31,717 个市场引用及 resolved outcome 元数据，覆盖 2025-12-18 至 2026-05-10。
- `extracted_labels.csv`：未来最低成交价及到达时间标签。
- `sell_taker_trades_daily/`：按日成交数据，2026-02-12 至 2026-05-10。

`price_estimator_valid.parquet` 与 `predictions_validation.parquet` 的实际行数是 7,432；部分 expected-return 报告使用 7,468 行的另一套 validation frame。每次实验必须明确输入文件、join key、过滤条件和最终评分行数，不得混用两个 universe 后直接比较。

### 2.3 历史实验与部署 artifact

以下内容可作为代码、配置、模型和 baseline 参考：

- `execution_engine/deploy/baseline/`：当前接受的方向 artifact。
- `execution_engine/deploy/price_estimator_expected_return_h14/`：当前执行模板加载的 price-estimator artifact。
- `price_estimator/expected_return/experiments/`：当前保留的 expected-return、Q、Gc、分组 policy 和 L2 实验。已删除的 `no_leak` 实验不得作为 baseline、训练起点或有效历史结果引用。
- `price_estimator/bid_policy/experiments/`：直接 bid-policy / soft-PnL 实验。
- `price_estimator/safe_lowest_price_gap/`：历史 normalized-gap baseline，仅用于比较或复现实验。
- `price_estimator/experiment/` 与 `price_estimator/upper_bound_mlp/`：历史 quantile、upper-bound、MLP 等探索。

禁止覆盖 deploy artifact、默认配置或 live execution 配置。所有新输出写入独立的 `20260703_<description>` 实验目录。

## 3. 固定成交与 PnL 语义

必须保持当前规则：正确方向的订单只有在 winner-side low 到达 bid 时成交；错误方向的已提交订单强制成交。基础 EV 为：

```text
EV(b | X) = q * Gc(b | X) * (1 - b) - (1 - q) * b
```

其中：

- `q` 是所选方向正确的校准概率；
- `Gc(b|X)` 是正确方向下 `winner_low <= bid` 的概率；
- `b` 是提交 bid；
- 未成交行 PnL 为 0；
- 错误且已提交的订单按 `-b` 计入 PnL；
- `wrong_fill_forced` 必须为 `1.0`。

不得静默改变 bid tick、观察窗口、winner-low 定义、selected-side 映射、强制错误成交规则或订单 universe。

## 4. 当前工作流与 baseline

当前生产链路是：

```text
共享数据/特征
  -> CatBoost 方向模型与 UTC day/session thresholds
  -> selected_side / p_side
  -> expected-return hazard price estimator
  -> min(best_ask - 0.01, expected_return_optimal_bid)
  -> execution engine
```

当前接受的方向 baseline：

```text
experiment: 20260611_catboost_calendar_coordinate_search
validation rows: 7,468
accepted_count: 5,228
coverage: 0.7000535619
accepted accuracy: 0.7073450650
selection_score: 0.6413740846
```

当前唯一确认没有标签泄露、可用于比较的 price-estimator baseline：

```text
experiment: 20260619_expected_return_h14_h2_gc_gt_0p75
validation sum_pnl: 27.44
order_count: 2,597
trade_count: 1,837
order_coverage: 0.4967482785
fill_rate: 0.3513771997
mean_accepted_pnl: 0.0052486611
wrong_fill_forced: 1.0
```

`20260619_expected_return_h14_h2_gc_gt_0p75` 是本轮 Arbor 研究必须复现并超越的 baseline，基准 `validation.sum_pnl = 27.44`。

此前名称包含 `no_leak` 的实验已经删除，不能作为有效结果。确认发生泄露的特征名是 `stage1_sample_weight`。它在 `src/core/constants.py` 中由 `DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN` 定义，并由 `src/data/dataset_builder.py::compute_training_sample_weight` 生成：基础权重依赖 `abs_return`，启用 reversal boost 时还直接读取 `target` 判断 continuation/reversal。因此它包含 label/future-return 信息，只能作为方向模型训练权重，绝对不能进入 direction、calibration、Q、Gc、price-estimator 或 bid-policy 的特征矩阵。即使实验名或旧报告写着 `no_leak`，使用该列作为特征的结果仍属于标签泄露。旧文档中与这些实验相关的 `186.37`、`961.84` 等 PnL 数字全部作废，不得用于排名、目标值、回归测试或 promotion 判断。

## 5. 已经尝试过的方向

### 5.1 方向模型

- CatBoost calendar/session coordinate thresholds。
- reversal、continuation、follow/reversal hybrid、side-specific threshold、sample weighting、ranker、blend、stacked meta model、regime routing。
- 近期 L2 direction feature 实验，包括 market-mid direction 和 common-window 比较。

结果：当前 selective 方向 baseline 在旧目标下较稳定；L2 direction 在 common window 上没有带来正 PnL。目标重设计提出单阈值、UP/DOWN 100% coverage，但尚未被验证和批准为主流程。

### 5.2 概率校准与 Q

- raw `p_side`、isotonic Q、不同 calibration tail 长度。
- XGBoost、CatBoost、LightGBM 及 blend 的 correctness/Q 模型。
- global Q gate 和分桶策略。

结果：简单提高 `min_q` 通常不能改善 PnL；低/中置信度但低 bid 的订单仍可能有正 edge。Q 的 Brier/logloss 与最终 PnL 必须分开评估。

### 5.3 Gc / fill model

- hazard-survival Gc、H2 checkpoint、residual/empirical CDF。
- global、`p_side_bin`、side/hour 分组。
- isotonic Gc、Q+Gc 组合。
- L2 book depth、order flow、trade dynamics、cross-side feature ablation。

结果：fill calibration 仍有明显偏差；部署 H14 的 submitted fill calibration gap 约 `0.1688`。L2 common-window 所有方案均为负 PnL，最佳约 `-10.92`，未推广。

### 5.4 Bid / policy

- expected-return EV gate、global/group min-EV。
- bid offset、bid cap、low-bid、fine bid grid、high-bid。
- `p_side_bin`、selected-side、hour 等 group policy。
- safe-lowest-price-gap、quantile、upper-bound MLP、direct/soft-PnL policy。

结果：收益主要受 bid/fill policy 驱动，而不是单纯方向 accuracy；更高 bid 或更多成交不保证更高 `sum_pnl`，复杂分组又容易在短 calibration tail 上过拟合。

## 6. 当前瓶颈

1. **历史实验存在隐蔽标签泄露**：已删除的 `no_leak` 实验把 `stage1_sample_weight` 用作特征。该列依赖 `abs_return`，在 reversal boost 路径还直接依赖 `target`。仅靠宽泛列名 pattern 不足以防泄露；必须显式禁用该列并追踪每个特征的生成逻辑和 lineage。
2. **验证集复用风险**：最后一个月已经被大量实验查看。它仍可作为用户指定的目标集，但统计意义上不再是完全未见 test。Arbor 必须用 pre-validation rolling splits 做选择，并记录查看冻结月的次数。
3. **样本口径不一致**：`price_estimator/data` 的 validation 文件为 7,432 行，而 H14 报告使用 7,468 行。第一项工作应是锁定 H14 的输入、join、过滤和 scorer，完整复现 `27.44`。
4. **Gc 校准不足**：submitted fill calibration gap 在多次实验中偏大，直接导致 EV 排序和 bid 选择失真。
5. **短 calibration tail 过拟合**：7 天、1,242 行上选择大量分组阈值，容易产生不稳定 policy。
6. **方向、校准和 policy 容易混在一起**：更换方向 universe 后的 PnL 不能直接归因于 price estimator。
7. **L2 时间覆盖较早**：完整 L2 common window 到 2026-03-27，不能直接覆盖最后一个月；在验证月缺少同口径 L2 特征时不能把 L2 实验与非 L2 baseline 当作同一测试。
8. **历史目标与新目标不一致**：当前 baseline 是 selective coverage 约 70%；目标重设计要求方向 coverage 100%。两者必须作为不同 track 报告，不能用一个数字宣称另一个被提升。

## 7. Arbor 推荐研究顺序

1. 建立冻结 scorer：固定 H14 的输入、join、row universe、fill 规则和 metrics schema，首先复现唯一有效 baseline `27.44`。
2. 在 2026-04-10 以前的数据做 rolling/blocked development splits；最后一个月只作有限次数的最终评估。
3. 固定方向输出，先研究 calibration 与 bid policy，避免把方向变化误归因于 price estimator。
4. 优先修正 Gc：按 bid、`p_side`、side、hour 做 reliability 诊断，比较 Brier、校准误差和 downstream PnL。
5. 对分组 policy 使用 shrinkage、最小样本约束和相邻桶平滑，避免 24-hour 或细粒度桶独立过拟合。
6. 再单独开启 100% direction coverage track：单阈值选 UP/DOWN、校准 selected-side correctness，然后重新训练 price estimator；与 70% selective track 分开排名。
7. 若 EV/Gc 路线在多个 development split 上无改进，再尝试 direct-PnL 或 policy-learning，但必须沿用同一个 frozen scorer。

## 8. 泄漏与保护规则

模型特征至少排除：

```text
target, future_*, abs_return, signed_return, stage1_target, stage2_target,
stage1_sample_weight,
chosen_low, correct, winner, pnl, trade_time, endDate, condition_id,
market_id, slug, outcome
```

同时排除匹配下列 pattern 的列，除非它们仅作为 label、join key 或报告字段且不会进入模型矩阵：

```text
target|label|winner|correct|chosen_low|future|closed|endDate|condition|
market_id|question|slug|outcome|fetched|source|time_to|trade_time|
timestamp|date|pnl
```

`stage1_sample_weight` 是明确禁用的泄露特征，不是普通高风险候选项。任何阶段的 `feature_columns`、模型矩阵、scaler、imputer 或 selector 中出现该列，实验必须立即失败。它仅允许通过训练 API 的 `sample_weight` 参数传递给原本设计使用它的方向模型；不得作为数值特征传入模型，也不得用于 calibration、Q、Gc、price-estimator 或 bid-policy。为防止同类问题，其他匹配 `sample_weight` 或 `*_sample_weight` 的字段也应默认拒绝，直到完成生成逻辑审计。审计必须检查列值来源，而不能只检查列名。

Scaler、imputer、encoder、feature selector、calibrator、threshold 和 policy 均不得 fit 冻结验证行。不得修改 `execution_engine/deploy/`、live config、原始数据和既有报告。

## 9. 每个 Arbor 节点的最低交付物

每个实验保存：

- 独立 config 和明确的 experiment id；
- git commit/工作树状态；
- train、calibration、development 和 frozen validation 时间窗口；
- 输入文件、行数、join/filter 规则和 feature list；
- leakage check；
- 方向、校准、Gc reliability 与 PnL metrics；
- predictions parquet、model artifact 和完整报告；
- 对唯一有效 baseline `20260619_expected_return_h14_h2_gc_gt_0p75`、`sum_pnl = 27.44` 的同口径比较；
- 若改变 direction universe，单独报告，不混入 fixed-direction 排名。

停止或推广条件：只有在 frozen scorer、fill 规则和样本 universe 完全一致，且 pre-validation 多个时间 split 表现稳定时，才可称为有效改进。任何 deploy promotion 都需要用户显式批准。
