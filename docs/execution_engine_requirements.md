# Execution Engine Requirements

## 1. 背景和目标

本需求基于现有 BTC/USDT 5 分钟方向预测项目，不新建一套独立特征或模型体系。线上执行引擎必须复用 `src/` 中已有的特征、推理、信号、Polymarket 映射、风控、审计和幂等能力，并以如下 baseline 作为生产模型输入来源：

```text
artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020
```

线上执行引擎的目标是：

1. 在每个 5 分钟预测窗口开始后，等上一根 4 分钟 kline close 已可用时，从 Binance 实时接口读取所需数据。
2. 构造与 baseline 逻辑一致的实时特征，不从 Binance Vision 下载历史文件作为本次执行输入。
3. 尽可能提前 prewarm 特征计算，最后一批实时数据到达后快速完成 inference。
4. 使用 baseline artifact 中的模型、校准器、特征列和阈值进行预测。
5. 若预测达到 UP/DOWN 阈值，则定位当前最近的 BTC 5 分钟涨跌预测 Polymarket 市场并提交限价单。
6. 每次触发最多提交 2 个限价单，提交后不追单、不撤单、不改价。
7. 部署形态面向 Linux 服务器，入口可被 cron 或 systemd timer 定时调用。

## 2. Baseline 约束

baseline artifact 当前关键信息：

```yaml
model_plugin: catboost_lgbm_logit_blend
calibration_plugin: platt_logit
model_path: artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020/catboost_lgbm_logit_blend.binary.pkl
calibrator_path: artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020/platt_logit.binary.pkl
feature_count: 516
t_up: 0.585
t_down: 0.335
validation_selection_score: 0.19027803605274402
validation_utility: 0.07698289269051321
validation_accepted_sample_accuracy: 0.5951923076923077
validation_coverage: 0.40435458786936235
coverage_constraint_satisfied: true
```

生产执行不得改变 label、horizon、threshold、feature semantics 或模型 artifact。阈值应优先从 artifact manifest / metrics 中读取；配置文件只允许覆盖 artifact 路径、运行模式和执行参数，不应硬编码 `0.5` 或其他业务阈值。

## 3. 范围

### 3.1 本次需要新增

新增仓库根目录下的 `execution_engine/`：

```text
execution_engine/
  README.md
  config.example.yaml
  run_once.py
  prewarm.py
  realtime_data.py
  feature_runtime.py
  polymarket_v2.py
  order_plan.py
  scheduler/
    execution-engine.service.example
    execution-engine.timer.example
    cron.example
  scripts/
    install_linux.sh
    smoke_test.sh
```

实际文件划分可在实现时微调，但必须满足：

1. 独立执行配置不放入 `config/settings.yaml`。
2. 只把 `config/settings.yaml` 作为共享核心 settings 的读取来源。
3. 模型推理、特征 builder、阈值读取、signal policy 尽量复用 `src/`。
4. Polymarket 下单接入 `py-clob-client-v2`，不要继续依赖旧版 `py_clob_client` 作为新执行路径的唯一实现。
5. 文档必须覆盖 Linux 部署、凭证配置、定时任务、日志、审计、故障处理和维护。

### 3.2 不在本次范围

1. 不重新训练模型。
2. 不优化 validation selection_score。
3. 不改变 baseline 特征定义或标签定义。
4. 不引入新的交易策略或仓位管理算法。
5. 不做订单动态追价、撤单、补单或成交后管理。
6. 不让 execution layer 重新实现 BTC 特征逻辑。

## 4. 运行时数据需求

baseline 使用 516 个特征，其中包含：

1. 1m OHLCV / taker buy / trade count 派生特征。
2. 1s second-level 特征，artifact manifest 显示 `second_level.enabled=true`，second-level feature count 为 182。
3. 15m HTF 上下文特征。
4. flow proxy 相关特征。
5. derivatives 在该 baseline 中 `enabled=false`，不要在线上执行中强行接入 funding / basis / OI / options / bookTicker 特征。

实时数据源应来自 Binance 公共实时接口：

1. 1m kline：用于主 OHLCV 特征和 5m grid 对齐。
2. 1s kline 或等价秒级实时聚合：用于 second-level 特征。
3. 如当前特征 builder 需要 `quote_volume`、`trade_count`、`taker_buy_base_volume`、`taker_buy_quote_volume`，实时采集必须提供这些字段。

禁止：

1. 执行时从 Binance Vision 下载 ZIP/CSV 作为实时输入。
2. 使用未来 bar、未来 return、label 派生列或验证阶段不可用字段。
3. 用 execution layer 自己重写和训练不一致的特征公式。

## 5. 时间和触发规则

预测窗口为 Polymarket BTC 5m UP/DOWN 市场窗口。推荐每 5 分钟执行一次：

```text
UTC minute % 5 == 0 后延迟若干秒运行
```

用户需求中的“整 5 分钟多一点点，上一根 4 分钟 kline close available”应实现为可配置参数：

```yaml
schedule:
  interval_minutes: 5
  trigger_delay_seconds: 8
  latest_required_kline_lag_minutes: 1
  max_data_wait_seconds: 20
```

实现含义：

1. 当前预测 `t0` 必须落在 5 分钟 grid 上。
2. 执行前等待 Binance 实时 kline 标记为 closed，避免使用未收盘数据。
3. 若最晚必需数据未到达，允许在 `max_data_wait_seconds` 内轮询。
4. 超时则跳过本窗口，记录审计，不下单。

具体 4 分钟 close 与 5m Polymarket 市场窗口的对齐关系需要实现前确认，见第 14 节。

## 6. Prewarm 需求

执行引擎应支持两阶段：

1. `prewarm.py` 或同进程 prewarm 阶段：在最终 close 数据到达前，拉取并缓存足够长的历史实时数据窗口，预计算大部分 rolling/context 特征。
2. `run_once.py` 最终阶段：等 closed kline 到达后只补齐最后数据，调用共享 `SignalService.predict_from_preheated_snapshot()` 或等价共享服务完成 inference。

prewarm 必须保证：

1. snapshot 的 source timestamp 可审计。
2. 如果最终数据比 prewarm 使用的数据更新，必须刷新相关特征。
3. 不允许因 prewarm 使用未 close bar 导致线上特征比训练更“新”。
4. 如果 prewarm 不可用，可以退化为 run-once 全量构建，但要在日志中标注。

## 7. 推理规则

执行引擎应加载：

```yaml
baseline:
  artifact_dir: artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020
  model_file: catboost_lgbm_logit_blend.binary.pkl
  calibrator_file: platt_logit.binary.pkl
  manifest_file: artifact_manifest.json
```

推理流程：

1. 加载共享 settings，用于 feature builder 和模型 plugin。
2. 从 baseline manifest 读取 feature columns、model plugin、calibration plugin、`t_up`、`t_down`。
3. 通过 `src.model.registry.load_model_plugin()` 加载模型。
4. 通过 `src.calibration.registry.load_calibration_plugin()` 加载校准器。
5. 通过 `src.services.SignalService` 或等价共享路径构造 `Signal`。
6. 使用 signal rule：

```text
p_down = 1 - p_up
UP signal     if p_up >= t_up
DOWN signal   if p_up <= t_down
NO-SIGNAL     otherwise
```

NO-SIGNAL 必须只记录审计，不访问下单路径。

## 8. Polymarket 市场定位

应复用或增强现有 `src.execution.mappers.btc_5m_polymarket.BTC5mPolymarketMapper`：

1. 根据 signal `t0` 定位当前最近的 BTC 5m UP/DOWN 市场。
2. 首选 slug 格式：

```text
btc-updown-5m-<window_start_unix_timestamp>
```

3. 若 slug 未命中，可使用 Gamma API fallback 查 active/未关闭市场。
4. 必须确认市场 active、未 closed、accepting orders。
5. 必须提取 UP/YES token id 和 DOWN/NO token id。

当预测为 UP 时，下单 token 应为 UP/YES token；预测为 DOWN 时，下单 token 应为 DOWN/NO token。

## 9. Polymarket CLOB v2 接入

新执行引擎应接入 `https://github.com/Polymarket/py-clob-client-v2`。

当前 README 显示：

1. 安装包：`py_clob_client_v2`。
2. 主要类型：`ApiCreds`、`ClobClient`、`OrderArgs`、`OrderType`、`PartialCreateOrderOptions`、`Side`。
3. 限价单使用 `create_and_post_order(..., order_type=OrderType.GTC)`。
4. L1 auth 用私钥创建或派生 API credentials。
5. L2 auth 用 `CLOB_API_KEY`、`CLOB_SECRET`、`CLOB_PASS_PHRASE` 做下单、撤单和账户查询。
6. Polygon mainnet chain id 为 `137`，Amoy testnet 为 `80002`。

实现应封装为 `execution_engine/polymarket_v2.py` 或迁移现有 `src.execution.adapters.polymarket`，但要避免破坏现有测试。建议新增 v2 adapter，并保留旧 adapter 的兼容测试。

## 10. 下单规则

若 signal 通过阈值：

1. 获取目标 token 的 order book。
2. 取当前 best bid。
3. 生成两个 BUY GTC limit orders：

```text
order_1_price = min(best_bid, 0.5)
order_1_size  = 5
order_2_price = min(best_bid, 0.5) - 0.1
order_2_size  = 5
```

4. 价格必须按市场 tick size 规整。
5. 若 `order_2_price` 小于 Polymarket 最小价格或不满足 tick size，应按配置决定跳过第二单或 clamp 到最小价格，默认建议跳过第二单并记录原因。
6. 下单后不等待成交、不撤单、不修改订单。
7. 每个预测窗口和 token 只允许提交一次该两单组合，必须有 idempotency key。

建议配置：

```yaml
orders:
  enabled: false
  mode: paper
  side: buy
  order_type: GTC
  first:
    price_cap: 0.5
    offset: 0.0
    size: 5.0
  second:
    price_cap: 0.5
    offset: -0.1
    size: 5.0
  min_price: 0.01
  max_price: 0.99
  tick_size_default: 0.01
  on_invalid_second_order: skip
```

`orders.enabled=false` 和 `mode=paper` 应作为默认值，避免部署时误下真单。

## 11. 配置文件需求

新增独立配置：

```text
execution_engine/config.yaml
```

仓库提交 `config.example.yaml`，真实 `config.yaml` 不提交。建议 `.gitignore` 保护真实凭证配置。

示例结构：

```yaml
baseline:
  artifact_dir: artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020
  settings_path: config/settings.yaml

runtime:
  mode: paper
  timezone: UTC
  audit_log: artifacts/logs/execution_engine/live.jsonl
  summary_dir: artifacts/logs/execution_engine/summaries
  idempotency_store_path: artifacts/state/execution_engine/idempotency.json

binance:
  base_url: https://api.binance.com
  symbol: BTCUSDT
  one_minute_interval: 1m
  one_second_interval: 1s
  request_timeout_seconds: 5
  lookback_minutes: 360
  require_closed_kline: true
  max_clock_skew_seconds: 2

schedule:
  interval_minutes: 5
  trigger_delay_seconds: 8
  max_data_wait_seconds: 20
  prewarm_seconds_before_trigger: 45

polymarket:
  host: https://clob.polymarket.com
  gamma_base_url: https://gamma-api.polymarket.com
  chain_id: 137
  private_key_env: POLYMARKET_PRIVATE_KEY
  api_key_env: CLOB_API_KEY
  api_secret_env: CLOB_SECRET
  api_passphrase_env: CLOB_PASS_PHRASE
  signature_type: null
  funder: null

orders:
  enabled: false
  mode: paper
  first:
    price_cap: 0.5
    offset: 0.0
    size: 5.0
  second:
    price_cap: 0.5
    offset: -0.1
    size: 5.0
  min_price: 0.01
  max_price: 0.99
  tick_size_default: 0.01
  on_invalid_second_order: skip

guards:
  require_market_accepting_orders: true
  require_best_bid: true
  max_orders_per_window: 2
  enforce_idempotency: true
```

## 12. 审计、日志和可观测性

每次运行必须写 JSON summary 和 JSONL audit event，至少包含：

1. run id。
2. UTC run timestamp。
3. target prediction window。
4. Binance 数据窗口、最后 closed kline timestamp、缺失检查结果。
5. baseline artifact path、model plugin、calibration plugin、feature count。
6. `p_up`、`p_down`、`t_up`、`t_down`、signal side。
7. Polymarket slug、market id、token id、best bid、best ask、tick size。
8. 两个订单的计划价格、size、是否提交、CLOB response。
9. skip reason。
10. idempotency key。
11. mode：paper/live。

敏感信息不得写入日志，包括私钥、API key、secret、passphrase。

## 13. 验收标准

实现完成后至少需要：

1. 单元测试：threshold decision UP/DOWN/NO-SIGNAL。
2. 单元测试：order plan 正确生成两单，价格为 `min(best_bid, 0.5)` 和减 `0.1`。
3. 单元测试：无 best bid 时跳过下单。
4. 单元测试：idempotency 阻止同一窗口重复提交。
5. 单元测试：v2 adapter 构造 `OrderArgs`、`Side.BUY`、`OrderType.GTC`。
6. 单元测试：实时数据 normalizer 输出字段满足共享 feature builder。
7. 集成/烟测：paper mode 完整跑通一次，不提交真实订单。
8. 特征一致性检查：用同一段历史数据模拟实时输入，与离线路径的最新 feature row 做列集合、空值、关键数值差异对比。
9. README 部署流程在干净 Linux venv 中可执行。

本需求不要求产生新的 before/after selection_score，因为不改变模型或训练逻辑。若实现过程中修改特征、阈值或模型，则必须重新按项目 experiment protocol 评估并报告 coverage、accepted_sample_accuracy、utility、selection_score。

## 14. 实现前必须确认的问题

1. “上一根 4 分钟 kline close available” 的精确定义是什么？
   - 指每个 5m 市场开始后使用 `t0-1m` 的 1m closed kline 作为最后输入

2. Polymarket 目标市场窗口如何和模型 `t0` 对齐？
   - 模型 label 是 `close[t0 + 4m] >= open[t0]`，而 Polymarket 5m 市场通常有明确 start/end。下单时预测的是当前刚开始的 5m 窗口。举个例子：当前时间10:00到10:04，prewarm就可以开始了,预测polymarket10:05到10:10这个市场的涨跌，等到10：04的close available了，那就是所有数据都ready了，可以开始inference和下单了。下单结束后就可以开始10:10到10:15这个市场的prewarm了，以此类推。

3. Binance 秒级数据源是否确定使用 `/api/v3/klines?interval=1s`？
   - 如果 Binance 现货 REST 对 1s kline 的支持、字段或延迟不满足要求，是否允许用 websocket aggTrade 实时聚合成秒级 kline？尽量使用binance提供的秒级接口。根据项目需求合理选择使用rest API还是websocket

4. 需要多长 lookback 才足够线上构造全部 516 个 baseline 特征？
   - 文档建议先用 360 分钟，但应通过代码扫描 rolling/lag 最大窗口确认。接受按最大特征依赖自动扩大 lookback

5. 生产环境是否允许持久化一个滚动本地缓存？
   - execution engine 可以维护 `artifacts/state/execution_engine/binance_cache.parquet`，减少每次 REST 拉取量

6. Polymarket 凭证和钱包模式是什么？
   - 需要确认 mainnet `chain_id=137`、是否有代理钱包/funder、`signature_type` 是否需要指定，以及环境变量名称。需要指定，请你在之后要生成的README中列出需要我配置的部分

7. 是否默认先部署 paper mode？
   - 直接`mode=live`。

8. 两个限价单的第二单价格如果小于 `0.01` 怎么处理？
   - 选择 clamp 到 `0.01`。

9. `best bid` 应取目标 outcome token 的 best bid
   - UP 用 UP/YES token 的 best bid，DOWN 用 DOWN/NO token 的 best bid。

10. 下单价格是否需要考虑 Polymarket tick size 以外的最小 size / min order amount？
    - 如果 Gamma market 暴露 `orderMinSize` 或 CLOB 返回限制，如果交易所规则要求的最低size小于5，就设置为5，如果高于5，就使用交易所最低size。

11. 同一窗口内如果第一次运行因为数据未就绪跳过，下一分钟是否允许补跑？
    - 允许补跑，补跑截止时间是市场开始前3分钟。比如市场时间是10:05到10:10，补跑最晚可以在10：08结束之前。

12. 当前最近市场找不到时是否允许 fallback 到下一个即将开始的市场？
    - 不允许，避免预测窗口和交易市场错配。

13. live 下单失败是否需要重试？
    -  API 失败重试。建议只对网络瞬断做一次幂等安全重试；业务拒单不重试。

14. `execution_engine` 是否必须完全独立于 `src.execution`？
    - 不完全独立：新目录提供入口、配置、实时数据和部署；共享逻辑继续放在 `src`，避免 duplicated execution logic。

15. 是否需要提交实现 commit？
    - 实现完成后创建一个独立工程 commit，不混入 unrelated user edits。

## 15. README 需要覆盖的部署流程

`execution_engine/README.md` 应包含：

1. Linux 服务器系统依赖安装。
2. Python venv 创建和依赖安装。
3. `py-clob-client-v2` 安装。
4. artifact 和配置文件路径检查。
5. 环境变量配置：

```bash
export POLYMARKET_PRIVATE_KEY=...
export CLOB_API_KEY=...
export CLOB_SECRET=...
export CLOB_PASS_PHRASE=...
```

6. paper mode smoke test：

```bash
python execution_engine/run_once.py --config execution_engine/config.yaml --mode paper --print-json
```

7. live mode 启动前 checklist。
8. cron 配置示例。
9. systemd service/timer 配置示例。
10. 查看日志和 summary。
11. 手动停用执行引擎。
12. 常见故障排查：Binance 数据未就绪、Polymarket 市场未找到、best bid 缺失、凭证错误、重复窗口跳过、模型/特征列不匹配。

## 16. 推荐实现顺序

1. 先实现配置 loader 和 artifact resolver。
2. 实现 Binance realtime normalizer，输出共享 feature builder 所需 schema。
3. 实现实时特征 prewarm / final refresh。
4. 接入 baseline 模型、校准器和阈值，完成 paper signal。
5. 实现 Polymarket v2 adapter 和 market mapper integration。
6. 实现 order plan、idempotency、audit。
7. 写测试和 paper mode smoke test。
8. 写 Linux deploy README 和 scheduler 示例。
9. 最后才启用 live mode。

