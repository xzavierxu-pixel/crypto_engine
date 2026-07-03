# Polymarket L2 第一分钟 Feature Pack 与价格标签数据产品需求

## 1. 文档目的

基于 `artifacts/pmdata/poly_l2` 中的 BTC 5 分钟 UP/DOWN Polymarket L2 数据，新增一个可复用、可在线构建、无未来信息的 feature pack，同时独立保存：

1. 每个市场在开盘后第一分钟截止时可见的最后一笔 `last_trade_price`，作为后续下单价格基准；
2. 第一分鐘结束后至市场结算前四分钟内的所选 side 最低成交价及其时间，作为价格最低点标签；
3. 仅使用第一分钟及以前数据生成的 L2 特征，用于方向分类模型和 expected-return/price estimator。

最终业务目标是提升 validation set 的 `sum_pnl`。方向模型仍需满足 `coverage >= 0.70`，并完整报告方向质量、订单数、成交率、价格误差和 PnL，不能用单一中间指标替代最终验证。

本文是实现需求，不代表已经获得 PnL 提升。

指定基准：

| 模块 | 基准实验 |
|---|---|
| 方向分类模型 | `20260611_catboost_calendar_coordinate_search` |
| price estimator / expected-return | `price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth` |

完整数据流必须是：

```text
第一分钟 Polymarket L2 + 补齐后的既有特征包
  -> 方向分类模型
  -> 仅用 train/calibration 拟合的概率校准器
  -> calibrated p_up
  -> selected_side / calibrated p_side
  -> 含 selected-side L2 特征的 price estimator
  -> bid / expected EV / order decision
```

price estimator 不得直接使用未校准的分类概率。

## 1.1 已确认的实现决策

以下事项已经由用户确认，实施时不再作为开放设计项：

- 当前开发起点分支为 `version4`；从该分支创建并切换到新分支 `pmdata`。
- 只补齐两个指定基准模型实际使用的 feature pack 数据，不补齐未被模型引用的数据包。
- side/mirror 映射优先从 `artifacts/data_v2/polymarket_prices` 和现有 Polymarket label 文件获取并交叉验证。
- L2 eligible 母集的最后 30 天作为 validation；validation 前 7 天作为 probability calibration；其余为训练集。
- 第一分钟价格基准使用 `[t0, t0+60s]` 内最后一笔 side 成交。
- 未来最低价使用 `(t0+60s, t0+300s]` 内最低 side 成交价；无成交保持 null。
- 分类概率校准比较 Platt 与 isotonic，并只根据 calibration split 的 log loss/Brier 选择。
- 最终接受以共同时间窗 validation `sum_pnl` 为首要指标，同时要求方向 coverage `>= 0.70`。

## 2. 已知数据现状

数据目录：

```text
artifacts/pmdata/poly_l2
```

用户提供的全量统计：

| event_type | 事件数 |
|---|---:|
| `price_change` | 3,342,885,938 |
| `book` | 111,175,047 |
| `last_trade_price` | 80,432,804 |
| `market_resolved` | 17,924 |
| `tick_size_change` | 1,876 |

时间覆盖：2026-02-13 17:00:00 UTC 至 2026-06-24 13:05:00 UTC。

当前抽样确认的 Parquet schema：

```text
market_slug, timestamp, local_timestamp, event_type,
ask_prices, ask_sizes, bid_prices, bid_sizes,
best_ask, best_bid,
pc_price, pc_size, pc_side,
new_tick_size,
trade_price, trade_size, trade_side, trade_is_mirror,
winning_outcome
```

文件名形如：

```text
btc-updown-5m-1771002000.parquet
```

slug 尾部 epoch 秒定义为 `market_t0`。抽样文件覆盖约 `market_t0 - 8s` 到 `market_t0 + 5m`，因此实现不得假设文件第一条事件就是市场开盘。

## 3. 总体设计约束

### 3.0 开发基线与代码来源

- 当前分支是 `version4`。实现阶段先确认工作区和当前 branch，再从当前 `version4` HEAD 创建并切换到新开发分支 `pmdata`；语义等价于在确认基点后执行 `git switch -c pmdata`。
- 若本地已经存在 `pmdata` 分支，不得覆盖或强制重建；必须先确认它是否指向预期 `version4` 基点。
- 本文更新任务不执行 branch checkout；创建和切换 `pmdata` 属于后续实现步骤。
- 创建分支时记录 `version4` 基点 commit、`pmdata` 初始 commit、原始 L2 schema 和派生 builder version，写入实验 manifest。
- 所有实现和实验提交只进入 `pmdata` 分支，不直接修改 `version4`。
- 原始 `artifacts/pmdata/poly_l2` 不得被修改；派生数据写入版本化输出目录。
- 若执行时当前 branch 不是 `version4`，或工作区存在会与分支创建冲突的未提交改动，必须停止并报告，不能自行丢弃或覆盖用户修改。

### 3.1 单一事实源

- L2 事件解析、side 标准化、时间窗切分和特征计算必须放在共享模块中。
- 离线训练、validation、在线推理和执行侧必须调用相同 builder；不得复制公式。
- 执行层不得重新实现 BTC 特征。Polymarket L2 在线特征也应由共享 builder 或预计算特征服务提供。
- 所有窗口、深度档位、缺失策略、tick 规则和 feature pack 开关必须来自统一配置。

### 3.2 严格分离 feature 与 label

创建三个物理隔离的数据产品：

```text
artifacts/data_v2/polymarket_l2/
  canonical_events/date=YYYY-MM-DD/*.parquet
  first_minute_features/date=YYYY-MM-DD/*.parquet
  first_minute_price_reference/date=YYYY-MM-DD/*.parquet
  future_four_minute_lows/date=YYYY-MM-DD/*.parquet
  manifests/*.json
  qa/*.json
```

- `first_minute_features` 只能包含 `feature_cutoff_time` 及以前可见的信息。
- `first_minute_price_reference` 是在线可见的价格基准，可以作为模型输入或报价锚点，但必须显式配置用途。
- `future_four_minute_lows` 是事后标签，只能用于 target 构建和评估。
- 特征加载器必须拒绝包含 `future_low*`、`winning_outcome`、`market_resolved`、`target`、`correct` 等字段的输入。

### 3.3 可扩展处理

原始数据超过 35 亿行，禁止一次性 concat 全目录到内存。

- 以单市场文件或受控批次流式处理；输出按日期分区。
- 仅读取当前阶段需要的列。
- 可并行处理市场，但输出必须确定性排序并支持断点续跑。
- 临时文件写完并校验后原子 rename。
- 相同 source fingerprint + config hash 重跑必须产生相同结果。
- manifest 记录输入文件数、行数、时间范围、schema hash、配置 hash、代码 commit、失败市场和输出统计。

## 4. 时间语义

对 market slug 中的 epoch 定义：

```text
market_t0           = 5 分钟市场开始时间
feature_cutoff_time = market_t0 + 60 秒
market_end_time     = market_t0 + 300 秒
```

### 4.1 第一分鐘特征窗

默认窗口：

```text
[market_t0, feature_cutoff_time]
```

- 包含 `market_t0` 和截止时刻事件。
- 任何 `timestamp > feature_cutoff_time` 的事件不得参与特征或价格基准。
- `local_timestamp` 只用于延迟和排序诊断，不得代替交易所事件时间作为主切窗字段。
- 同一 `timestamp` 多事件按 `local_timestamp`、稳定源序号排序；仍完全相同时使用确定性的原始行序。
- 若线上实际决策需留安全延迟，配置支持 `availability_lag_ms`，实际 cutoff 为 `market_t0 + 60s - lag`。默认值不得隐藏在代码中。

### 4.2 未来四分钟标签窗

默认窗口：

```text
(feature_cutoff_time, market_end_time]
```

- 严格排除第一分钟截止时刻，避免同一事件同时成为价格基准和 future label。
- 包含结算截止时刻以前的最后事件。
- 如现有 expected-return 的 `decision_time` 实际不是精确 `t0+60s`，必须先输出差异报告，不得静默混用两个窗口。

### 4.3 时间覆盖策略

- **实验总时间覆盖以可用 PM L2 数据为准**。候选母区间是 2026-02-13 17:00:00 UTC 至 2026-06-24 13:05:00 UTC，实际起止点以完成市场、可验证 side 映射和第一分钟/未来四分钟窗口均完整的市场清单为准。
- 不再用“保留 L2 前样本并设置 `pm_l2_available=0`”作为主实验口径；L2 覆盖前的样本不进入本次 L2 实验的 train/calibration/validation。
- 先生成一份不可变的 `l2_eligible_markets` 清单，再以该清单为左表补齐所有其他 feature pack。模型数据集不得由各特征源各自 inner join 决定。
- 先从两个基准 artifact/config 提取实际 feature columns，并解析它们依赖的 feature pack，生成冻结的 `required_feature_packs.json`。只处理该清单中的既有数据包及新增 L2 pack。
- 分类侧清单来源：`20260611_catboost_calendar_coordinate_search` 实际 feature columns。
- price-estimator 侧清单来源：`20260619_expected_return_h2_hazard_smooth` 实际 feature columns。
- 两侧清单取并集并记录每个 pack 的消费模型；未被实际 feature columns 引用的 pack 不补采、不重建、不纳入完成条件。
- 清单内的 BTC 1m/1s、agg trades、derivatives、calendar 或其他 pack，只要在 L2 母区间或 eligible market 上缺失，就必须从其原始数据源补采、重建或合法回填。
- “补齐”必须保持原特征语义和在线可用性；禁止用未来值、双向插值或 validation 统计量填补历史缺口。
- 低频数据只能使用预测时刻已发布的最近值；高频数据缺口按各 pack 原有合法策略重建。无法合法补齐的市场必须进入缺失审计，不能静默丢弃。
- 若某个既有 feature pack 无法达到预设覆盖率，必须分别报告：源数据缺失、构建失败、join 失败和合法缺失。是否禁用该 pack 必须通过配置化消融决定。
- 在完成 required feature pack 覆盖审计后，按时间顺序定义 development/train、probability calibration 和 validation；所有模型共享完全相同的 split 边界和 market IDs。
- validation 固定为 L2 eligible 时间序列的最后 30 个完整 UTC 日；probability calibration 固定为 validation 开始前连续 7 个完整 UTC 日；其余更早 eligible 市场作为训练集。
- 若边界日 L2 覆盖不是完整 UTC 日，向内收缩到最近完整日并在 manifest 记录，不允许用不完整边界日补足 30/7 天。
- validation 和 calibration 边界在任何模型、校准方法、threshold 或 min-EV 调优前冻结。不得为提高结果裁剪低质量市场。
- 报告每个 feature pack 在全区间及 train/calibration/validation 的 market coverage、row coverage、首末可用时间和补齐来源。
- 与旧基准比较时，必须额外在新的 L2 共同时间窗上重跑两套冻结基准，避免把时间窗变化误判为 L2 改善。

## 5. side 标准化与阻断性语义审计

原始数据包含 `trade_is_mirror`，但 `book` 和 `price_change` 的 side/token 语义不能仅凭字段名推断。实现前必须完成并保存 `side_mapping_audit.json`。

映射信息按以下优先级读取：

1. `artifacts/data_v2/polymarket_prices` 中与 slug、condition、token/outcome 对应的 metadata；
2. 当前 Polymarket resolved label 文件，至少包括 `artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet`；
3. 两者共有市场的交叉验证结果；
4. 原始 L2 的 `market_resolved.winning_outcome` 只用于审计，不作为训练 feature。

若目录实际 schema 与上述预期不同，应先生成字段 inventory 并调整显式映射，不允许根据价格高低猜 UP/DOWN。

审计至少包括：

1. 从现有 resolved label/market metadata 获取每个 condition 的 UP、DOWN token 标识和最终 outcome。
2. 验证 non-mirror 与 mirror 价格是否满足互补关系，统计 `abs(p_up + p_down - 1)` 分布。
3. 抽样检查相邻 mirror/non-mirror 成交、盘口最优价及结算前价格是否符合预期。
4. 将 `market_resolved.winning_outcome` 仅用于审计和标签验证，绝不作为特征。
5. 明确 `pc_side`、`trade_side` 表示 maker side、taker side 还是 book side，并通过盘口变动复算验证。
6. 验证 `book` 数组每条记录属于哪个 token；若原始 schema 无法可靠区分 token，必须从数据供应方文档或额外 metadata 解决，禁止猜测。

阻断条件：

- UP/DOWN 映射不能被可靠复算；
- mirror 变换在抽样和全量 QA 上不满足预设容差；
- `book` 或增量无法确定其对应 token；
- 同一市场存在不可解释的时间倒序或跨市场事件。

任一阻断条件未解决时，只能构建已验证字段的数据产品，不能发布完整 L2 feature pack。

标准化事件至少输出：

```text
market_slug, market_t0, condition_id, outcome_side,
event_time, local_time, event_type,
best_bid, best_ask, bid_prices, bid_sizes, ask_prices, ask_sizes,
change_price, change_size, change_side,
trade_price, trade_size, aggressor_side,
tick_size, source_file, source_row_id
```

其中 `outcome_side` 必须是规范化的 `UP` 或 `DOWN`。

## 6. 独立数据产品 A：第一分钟最后成交价

文件：

```text
first_minute_price_reference/date=YYYY-MM-DD/*.parquet
```

每个市场、每个 outcome side 一行：

```text
market_slug
condition_id
market_t0
feature_cutoff_time
outcome_side
last_trade_price_1m
last_trade_time_1m
seconds_since_last_trade_1m
last_trade_size_1m
last_trade_side_1m
trade_count_1m
has_last_trade_1m
source_file
builder_version
```

选择规则：

1. 仅使用 `[market_t0, feature_cutoff_time]` 的 `last_trade_price`。
2. 按规范化 outcome side 分组，选择 event time 最晚的记录。
3. 同 event time 时用 `local_timestamp` 和稳定原始行序决定最后一条。
4. 禁止从未来事件 backfill。
5. side 无成交时价格为 null，`has_last_trade_1m=0`；可另存 cutoff 时刻可见的 `best_bid/best_ask/mid` 作为诊断，但不得伪装成 last trade。
6. UP/DOWN 双边记录应保留，不能只保存当前模型预测 side，因为后续模型可能改变方向。

该表后续可用于：

- 下单绝对价格基准；
- `bid - last_trade_price_1m`、`bid / last_trade_price_1m` 诊断；
- price estimator 的锚点或约束；
- 在线/离线基准价一致性检查。

## 7. 独立数据产品 B：未来四分钟最低价

文件：

```text
future_four_minute_lows/date=YYYY-MM-DD/*.parquet
```

每个市场、每个 outcome side 一行：

```text
market_slug
condition_id
market_t0
feature_cutoff_time
market_end_time
outcome_side
future_low_4m
future_low_time_4m
seconds_to_future_low
future_trade_count_4m
has_future_trade_4m
future_low_source
source_file
builder_version
```

规则：

1. 仅使用 `(feature_cutoff_time, market_end_time]` 内规范化后的 `last_trade_price`。
2. `future_low_4m` 是最小 `trade_price`；同价时保留第一次到达最低价的时间。
3. 不用 `best_bid`、`price_change` 或 book level 冒充成交最低价。
4. 无未来成交时保持 null 并写明原因，不得用 0、1 或结算价填充。
5. 同时保留 UP/DOWN，expected-return 构建阶段再按模型 `selected_side` 连接。
6. 与现有 sell-taker `chosen_low` 做重叠市场对账，报告差异分布、覆盖率和字段语义差异；未经解释不得直接替换旧 label。

必须额外生成 QA：

```text
future_low <= future_window_max_trade_price
future_low_time > feature_cutoff_time
future_low_time <= market_end_time
0.0 < future_low < 1.0
```

## 8. Feature Pack 定义

建议名称：

```text
polymarket_l2_first_minute_v1
```

特征前缀：

```text
pm_l2_1m_
```

所有特征在 `feature_cutoff_time` 可复算。初版控制在约 80–150 个特征，先做低风险、可解释特征，避免直接展开全部价位和高维序列。

### 8.1 数据可用性与质量

- `has_book`, `has_trade`, `has_price_change`
- 各 event type 计数、每秒事件率、活跃秒数
- 首/末事件距 t0/cutoff 秒数
- exchange-local latency 的均值、P50、P95、最大值
- 时间倒序数、重复事件数、异常价格数
- 当前 tick size、窗口内 tick size 变更次数

### 8.2 截止时刻 top-of-book

对 UP、DOWN 分别计算：

- cutoff 前最后可见 `best_bid`, `best_ask`, `mid`, `microprice`
- spread、relative spread、spread ticks
- 最优 bid/ask size
- top-level imbalance `(bid_size-ask_size)/(bid_size+ask_size)`
- cutoff 前 5s/15s/30s/60s 的 mid return、slope、realized volatility
- 距窗口高/低点、最后价格位置

若 `price_change` 是增量，builder 必须从最近合法 `book` 快照开始按顺序 replay，不能把单条增量当完整状态。

### 8.3 深度与盘口形状

配置化深度档：1、2、3、5、10 levels，以及距 mid 1/2/5 cents 或 ticks。

- bid/ask 累计深度及 imbalance
- depth-weighted price、microprice deviation
- book slope/convexity
- 到指定成交量的 VWAP 和冲击成本
- depth concentration、top-level share
- cutoff 前 5s/15s/30s 的 depth depletion/replenishment
- spread widening/narrowing 次数和持续时间

### 8.4 成交与订单流

对 5s、15s、30s、60s 窗口计算：

- trade count、volume、notional
- BUY/SELL count 和 volume imbalance
- 平均/中位/最大 trade size，大单占比
- VWAP、last trade、high、low、range
- trade price 相对 mid/microprice 偏差
- trade arrival intensity、平均 inter-arrival time
- price impact：成交方向与后续短窗 mid 变化
- repeated/mirror 去重前后计数，去重规则必须审计可复算

### 8.5 price_change/order-flow 动态

- bid-side/ask-side 增量次数、增加量、撤单量
- add/cancel ratio、净新增深度
- 最优档改善/恶化次数
- order-flow imbalance（OFI）及 5s/15s/30s/60s 版本
- OFI slope、z-score、burst 指标
- queue depletion/replenishment
- 增量与 mid return 的一致/背离指标

### 8.6 UP/DOWN 跨 side 一致性

- `up_mid + down_mid - 1`
- `up_last + down_last - 1`
- 可成交组合成本：`up_ask + down_ask - 1`
- 可出售组合价值：`up_bid + down_bid - 1`
- 两侧 spread、depth、OFI、trade imbalance 的差和比
- 两侧价格变化相关性及 lead-lag
- 互补价格偏差的均值、P95、窗口末值

所有除法使用统一安全除法和配置化 epsilon；缺失标记与数值填充必须分开。

### 8.7 selected-side 派生特征

基础 feature pack 必须先保存 side-neutral 的 UP/DOWN 特征。方向模型输出 `p_up` 后，expected-return 数据构建器可以派生：

- selected-side last/mid/bid/ask/spread/depth
- selected-side OFI、成交不平衡、波动率
- selected 与 opposite side 的差值/比值
- `p_side - last_trade_price_1m`
- `p_side - mid_1m`
- 价格锚点相对 `p_side` 的可用空间

这些派生不得反向进入产生 `p_up` 的同一阶段，避免循环依赖。

## 9. 配置需求

示例配置结构：

```yaml
polymarket_l2:
  enabled: true
  source_dir: artifacts/pmdata/poly_l2
  canonical_dir: artifacts/data_v2/polymarket_l2/canonical_events
  feature_dir: artifacts/data_v2/polymarket_l2/first_minute_features
  price_reference_dir: artifacts/data_v2/polymarket_l2/first_minute_price_reference
  future_low_dir: artifacts/data_v2/polymarket_l2/future_four_minute_lows
  feature_pack: polymarket_l2_first_minute_v1
  market_duration_seconds: 300
  feature_window_seconds: 60
  availability_lag_ms: 0
  future_low_start_inclusive: false
  future_low_end_inclusive: true
  depth_levels: [1, 2, 3, 5, 10]
  rolling_windows_seconds: [5, 15, 30, 60]
  price_distances: [0.01, 0.02, 0.05]
  safe_divide_epsilon: 1.0e-9
  missing_policy: preserve_sample_with_flags
  side_mapping_version: null
```

禁止硬编码上述业务参数。

## 10. 分类模型集成

L2 feature pack 必须通过共享 feature registry 注册，并在训练 frame 中按 `market_t0`/decision timestamp 一对一 join。

分类模型冻结基准为：

```text
experiment_id: 20260611_catboost_calendar_coordinate_search
config_path: experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
baseline_output_dir: artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_calendar_coordinate_search
```

新实验应保留该基准的 label、模型族和核心训练语义，只增加 L2 feature pack、必要的数据覆盖补齐和显式概率校准。若需要改变其他参数，必须拆成独立消融，不得与 L2 特征效果混在一个实验中。

要求：

- join 后总样本数不变；重复 market key 直接失败。
- 主实验样本来自冻结的 `l2_eligible_markets`。eligible 市场中局部事件缺失时保留原行，设置 availability flag，并使用只在训练集拟合的缺失策略。
- scaler、imputer、selector 只在 development/train 拟合。
- resolved outcome、未来四分钟成交、结算事件不能出现在 feature columns。
- 保持现有 Polymarket resolved label 不变；时间 split 按 4.3 节在 L2 eligible 母集上重新定义并冻结，所有对照实验共用同一 split。
- validation 仍是阈值调优和最终接受集，并明确标记 optimistic。

### 10.1 分类概率校准

- 分类器先输出原始 `raw_p_up`，再由独立 calibration split 拟合概率校准器，输出 `calibrated_p_up`。
- 复用项目已有且可加载 artifact 的校准插件；主候选固定为 Platt/logit 与 isotonic，identity 作为不校准诊断基线。方法选择只能使用冻结的 7 天 calibration 数据。
- Platt 与 isotonic 首先按 calibration log loss 排序，Brier 作为第一 tie-breaker；若两者结论冲突且差异超过预设容差，报告两者 downstream PnL 但不得用 validation 反向选择校准器。
- CatBoost、特征选择、imputer 和概率校准器均不得在 validation 拟合。
- threshold search 使用 `calibrated_p_up`，并保持 coverage `>= 0.70`。
- artifact 必须同时保存分类模型、校准器、校准方法、拟合窗口、输入输出 schema 和校准前后指标。
- 必须报告 raw 与 calibrated 的 Brier、log loss、reliability bucket、ECE、ROC AUC，以及校准后 threshold 下的所有必需方向指标。
- 若校准改善 Brier/log loss 但降低最终 validation `sum_pnl`，不能仅凭校准指标接受。
- `calibrated_p_up` 是 downstream 唯一允许的分类概率；`raw_p_up` 仅保留作诊断。

## 11. Expected-return / hazard 模型集成

目标实验基线：

```text
price_estimator/expected_return/experiments/
20260619_expected_return_h2_hazard_smooth
```

该基准需在冻结的 L2 共同时间窗上原配置重跑，形成 `P0-common-window`，再与 L2 price estimator 比较。原实验报告仍作为历史参考，但不能直接与变化后的时间窗做因果比较。

集成要求：

1. 只接收分类 artifact 输出的 `calibrated_p_up`；令 `p_up = calibrated_p_up`，再更新 `selected_side` 和 `p_side`。禁止把 `raw_p_up` 输入 price estimator。
2. 将 side-neutral L2 特征按 `selected_side` 派生成 selected-side 特征。
3. 将 `last_trade_price_1m` 作为显式价格基准字段，不要覆盖 `p_side` 或现有 `bid`。
4. 将 `future_low_4m` 作为新的、版本化 target 候选，与旧 `chosen_low` 并行保留，先对账后决定是否切换。
5. hazard/CDF 模型只在 correct 且 future low 有效的训练行拟合；wrong side 的 forced-fill PnL 逻辑保持不变。
6. calibration 和 validation 的模型、校准器、特征选择均不能在 validation 拟合。
7. 输出逐样本 `price_reference`, `future_low`, `bid`, `model_fill_prob`, `expected_ev`, `filled`, `realized_pnl` 以便审计。
8. 保存分类概率 lineage：`classifier_experiment_id`、`classifier_artifact_hash`、`probability_calibrator`、`calibration_window`，确保每条 price-estimator 预测可追溯。
9. price estimator 自身的 fill/q calibration 与方向概率校准是两个不同阶段，artifact 和报告中必须分开命名、分开拟合、分开评估。

## 12. 实验与消融计划

为区分方向模型改善和报价模型改善，至少运行：

| 实验 | 方向模型 | expected-return 特征/target | 用途 |
|---|---|---|---|
| A0 | `20260611` 原分类器，原概率链路 | `20260619` 原 price model | L2 共同时间窗冻结复现 |
| A1 | `20260611` + 概率校准 | `20260619` 原 price model | 只测概率校准贡献 |
| A2 | `20260611` + L2 + 概率校准 | `20260619` 原 price model | 只测分类 L2 贡献 |
| A3 | `20260611` + 概率校准 | `20260619` + L2 selected-side 特征 | 只测报价 L2 贡献 |
| A4 | `20260611` + L2 + 概率校准 | `20260619` + L2 selected-side 特征 | 完整方案 |
| A5 | A4 | 去掉 book depth | depth 消融 |
| A6 | A4 | 去掉 order flow | OFI 消融 |
| A7 | A4 | 去掉 trade dynamics | trade 消融 |
| A8 | A4 | 去掉 cross-side 特征 | 双边一致性消融 |

所有实验使用 L2 eligible 母集、同一时间 split、同一 validation market IDs 和相同 PnL 口径。必须提供共同时间窗的 apples-to-apples 对比；原始全量基准只作历史参考，不得混为一个结论。

接受排序：

1. validation `sum_pnl` 提升；
2. direction coverage `>= 0.70`；
3. 正 utility、accepted accuracy > 0.50；
4. order count、trade count 和 PnL 的时间稳定性；
5. fill calibration、泄漏风险和实现复杂度。

必须报告：

- AGENTS.md 规定的全部方向指标；
- `sum_pnl`, `mean_accepted_pnl`, `mean_pnl_filled`；
- `order_count`, `order_coverage`, `trade_count`, `fill_rate`；
- `correct_fill_rate`, `wrong_fill_forced`, `wrong_fill_printed`；
- `mean_bid`, bid 分桶 PnL、UP/DOWN PnL；
- `mean_model_fill_prob`, `submitted_fill_calibration_gap`；
- L2 市场覆盖率、缺失率、各 split 时间覆盖；
- `required_feature_packs.json` 中每个既有 pack 的补齐前/后覆盖率及不可补齐市场数；
- 分类 raw/calibrated 概率的校准指标和 artifact lineage；
- `last_trade_price_1m` 与 bid/future low 的 gap 分布。

不得只因训练集或 full-train PnL 提升而接受。

## 13. 泄漏与一致性检查

必须自动验证：

- feature 最大事件时间 `<= feature_cutoff_time`；
- future low 最小事件时间 `> feature_cutoff_time`；
- `market_resolved` 和 `winning_outcome` 从 feature schema 排除；
- 未来成交 count、未来 low/time、最终 winner 不在 feature columns；
- 所有 rolling window 右端点不超过 cutoff；
- validation 不参与 imputer/scaler/selector/calibration 拟合；
- direction probability calibration 只使用冻结 calibration split，price estimator 只读取 calibrated probability；
- 同一输入市场的 batch builder 与单市场 online builder 逐列一致；
- cutoff 后追加任意事件不会改变第一分钟 feature hash；
- 执行侧不会读取 future-low 数据目录。

## 14. 测试需求

### 14.1 单元测试

- slug epoch 到 `market_t0` 解析；
- UTC 与毫秒时间精度；
- 第一分鐘边界包含性和未来四分钟边界排除性；
- 同 timestamp 稳定排序；
- mirror 到 UP/DOWN 的标准化；
- full book + price_change replay；
- add/cancel、OFI、depth、spread、microprice 公式；
- last trade 选择及无成交缺失策略；
- future low、首次最低点时间及无未来成交策略；
- forbidden column 检查；
- duplicate market key 失败；
- 配置 hash 和幂等输出。

### 14.2 集成测试

- 小型多市场 fixture 从 raw Parquet 生成四类输出；
- 离线 batch 与模拟在线事件流在 cutoff 行完全一致；
- feature join 不改变 dataset 行数和 label；
- selected-side 特征随 UP/DOWN 切换正确；
- expected-return target 只从 future-low 表读取；
- 训练、artifact 保存/加载和推理均识别新 feature pack；
- execution 不访问 label 表且不复制特征公式。

### 14.3 数据 QA

- 每日文件数、市场数、事件数与源数据对账；
- 每个 event type 覆盖和 schema drift；
- 价格范围、size 非负、bid/ask 交叉率；
- UP/DOWN 互补偏差；
- feature/price reference/future low 覆盖；
- market_t0 重复、缺失和跨 split 重叠；
- 随机抽样至少 100 个市场人工复算。

## 15. 预期代码与文件范围

实现阶段预计新增或修改：

```text
src/data/polymarket_l2.py                         # schema、标准化、流式读取
src/features/packs/polymarket_l2_first_minute.py # 共享 feature pack
src/features/registry.py                         # pack 注册
scripts/data/step4_features/build_polymarket_l2_features.py
scripts/model/train_model.py                     # feature frame join
price_estimator/expected_return/build_expected_return_target.py
price_estimator/expected_return/*.py              # selected-side L2 接入
config/settings.yaml 或实验专用统一配置
tests/test_polymarket_l2.py
tests/test_polymarket_l2_feature_parity.py
tests/test_expected_return_polymarket_l2.py
```

实际实现前应根据现有 pack 组织方式确认最终路径，优先小范围接入，不重写现有架构。

## 16. 产物与可复现性

每次完整构建必须保存：

- 精确配置快照；
- source inventory/fingerprint；
- side mapping audit；
- feature manifest 和 forbidden-column report；
- train/calibration/validation coverage report；
- `required_feature_packs.json` 中每个既有 pack 的 backfill manifest 和覆盖审计；
- 分类概率校准 artifact、窗口和 raw/calibrated 对比报告；
- 数据 QA report；
- 训练 report 和逐样本 validation predictions；
- 实验 commit hash；
- 若生成 deploy artifact，记录 accepted offline artifact 和 threshold source。

不得覆盖唯一一份旧实验配置或报告。

## 17. Definition of Done

只有同时满足以下条件，L2 feature pack 才算完成：

1. side/mirror 语义审计通过，UP/DOWN 映射可复算；
2. 已从当前 `version4` HEAD 创建并切换到 `pmdata` 分支，并记录两个分支的基点 commit 与 schema；
3. 第一分鐘价格基准表和未来四分钟最低价表物理隔离并完成 QA；
4. L2 eligible 母市场清单及 `required_feature_packs.json` 已冻结，只对两个基准实际使用的 pack 完成合法补齐和覆盖审计；
5. feature pack 仅使用 cutoff 前信息，追加未来事件不改变特征；
6. 离线与模拟在线特征逐列一致；
7. validation market IDs 和 Polymarket resolved label 未被静默改变；
8. 分类模型以 `20260611_catboost_calendar_coordinate_search` 为基准，概率经独立校准后才进入 price estimator；
9. price estimator 以 `20260619_expected_return_h2_hazard_smooth` 为基准；
10. direction coverage `>= 0.70`；
11. 所有必需方向、校准、价格、成交和 PnL 指标已报告；
12. 共同 L2 时间窗基准、模块隔离实验、完整方案及关键消融可复现；
13. validation `sum_pnl` 与共同时间窗基准完成同口径比较；
14. 明确声明 `sum_pnl` 是否真实提升；若未提升，保留数据产品和诊断，但不晋升模型；
15. 测试通过，无 future/label 字段进入 feature schema；
16. 实验配置、报告和 git commit 足以复现结果。
