# execution_engine 流程与时间线分析

本文基于当前代码分析 `execution_engine` 的线上执行流程，重点说明定时任务、Binance 实时数据、共享特征构建、模型推理、信号决策、Polymarket 市场映射和下单之间的时间关系。

分析结论以当前代码和默认配置为准，核心文件包括：

- `execution_engine/run_once.py`
- `execution_engine/realtime_data.py`
- `execution_engine/feature_runtime.py`
- `execution_engine/order_plan.py`
- `execution_engine/polymarket_v2.py`
- `execution_engine/prewarm.py`
- `execution_engine/config.py`
- `execution_engine/config.example.yaml`
- `execution_engine/scheduler/execution-engine.timer.example`
- `execution_engine/scripts/run_paper_experiment.py`
- `config/settings.yaml`
- `src/features/builder.py`
- `src/data/second_level_features.py`
- `src/signal/policies.py`

## 1. 总体定位

`execution_engine` 是一个薄执行层，不是新的训练或特征系统。它负责在固定时间点：

1. 读取执行配置和 baseline artifact。
2. 拉取 Binance 1m、1s、aggTrades 实时数据。
3. 按目标 Polymarket 5m 市场窗口对齐数据。
4. 调用 `src/` 共享特征和模型推理路径生成 `p_up`。
5. 调用共享信号策略得到 YES、NO 或 NO-SIGNAL。
6. 映射到 Polymarket BTC 5m market slug。
7. 读取订单簿并生成限价买单计划。
8. 在 live 且 orders enabled 时提交订单。
9. 写 audit log 和 summary JSON。

它不应重新实现 BTC 特征、标签或训练逻辑。当前主路径确实复用了共享核心：

```text
run_once.py
  -> RuntimeInferenceEngine
     -> src.data.second_level_features.build_second_level_feature_store
     -> src.data.second_level_features.sample_second_level_feature_store
     -> src.features.builder.build_feature_frame
     -> src.model.infer.predict_frame
  -> src.signal.policies.evaluate_selective_binary_signal
```

## 2. 当前关键配置

`execution_engine/config.example.yaml` 中的执行层默认值：

```yaml
schedule:
  interval_minutes: 5
  trigger_delay_seconds: 68
  max_data_wait_seconds: 20
  prewarm_seconds_before_trigger: 45

thresholds:
  t_up: null
  t_down: null

orders:
  enabled: false
  mode: paper
```

`config/settings.yaml` 中当前与时间线最相关的配置：

```yaml
decision_alignment:
  enabled: true
  mode: delayed_feature_offset
  feature_offset_minutes: 1
  order_delay_seconds_after_feature_time: 8
  row_policy: delayed_1m_synthetic_decision_row

horizons:
  specs:
    "5m":
      minutes: 5
      grid_minutes: 5
      label_builder: grid_direction
      label_params:
        label_version: settlement_direction_t0_open_to_t4_close_tie_up_v2
      feature_profile: core_5m
      signal_policy: selective_binary_policy
```

最重要的当前语义：

```text
feature_offset_minutes = 1

market_t0 / signal_t0 = T
decision_time         = T + 1 minute
feature_timestamp     = T + 1 minute
market window         = [T, T+5m)
```

也就是说，当前线上特征行不是 timestamp 为 `T` 的行，而是 timestamp 为 `T+1m` 的 synthetic decision row。这个延迟对齐是为了在 `T+1m` 附近已经拿到 `[T, T+1m)` 内的数据后，再交易仍然处于进行中的 Polymarket `[T, T+5m)` 市场。

## 3. 标签语义

项目标签规则保持不变：

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

对 5 分钟市场窗口 `[T, T+5m)`：

```text
label open         = 1m candle at T 的 open
label future close = 1m candle at T+4m 的 close
label outcome      = close[T+4m] >= open[T]
```

execution layer 在线上不计算标签。它只生成交易时的信号和订单。线上 summary 里的 `signal.t0` 对应 Polymarket 市场窗口起点 `T`，不是标签计算时间。

## 4. 入口：run_once.py

主入口是 `run_once(config_path, mode_override=None, target_window_start=None)`。

流程：

```text
load_execution_config(config_path)
mode = mode_override or config.runtime.mode
target_window_start = provided target or current_5m_window_start()
baseline = load_baseline_artifact(config.baseline)
settings = load_settings(config.baseline.settings_path)
feature_offset_minutes = settings.decision_alignment.feature_offset_minutes if enabled else 0
audit = AuditService(...)
```

`current_5m_window_start()` 会把当前 UTC 时间向下取整到最近的 5 分钟边界：

```text
12:00:00 - 12:04:59.999 -> 12:00:00
12:05:00 - 12:09:59.999 -> 12:05:00
```

如果没有显式传入 `target_window_start`，运行开始时间决定目标市场窗口。若任务启动太晚，例如 `12:04:30Z`，仍会映射到 `[12:00, 12:05)`。当前代码没有显式的 max window age guard。

## 5. 定时任务

### 5.1 systemd timer 示例

`execution_engine/scheduler/execution-engine.timer.example`：

```ini
[Timer]
OnCalendar=*:01/5
AccuracySec=1s
RandomizedDelaySec=0
Unit=execution-engine.service
```

这个 timer 会在每小时的 `00:01, 00:06, 00:11, ...` 触发 service。换成 5m 市场窗口表示：

```text
12:01:00Z -> current_5m_window_start = 12:00:00Z
12:06:00Z -> current_5m_window_start = 12:05:00Z
```

service 示例：

```ini
ExecStart=/opt/crypto_engine/.venv/bin/python execution_engine/run_once.py --config execution_engine/config.yaml
```

注意：`run_once.py` 本身不会根据 `schedule.trigger_delay_seconds` sleep。systemd 示例实际延迟由 `OnCalendar=*:01/5` 决定，约等于窗口开始后 60 秒触发。

### 5.2 paper experiment 脚本

`execution_engine/scripts/run_paper_experiment.py` 使用 `next_trigger_time()`，它会读取或接收 `delay_seconds`：

```text
trigger = current_5m_window_start(now) + delay_seconds
if now >= trigger:
  trigger = next 5m window start + delay_seconds
```

因此 paper experiment 默认会用 `config.schedule.trigger_delay_seconds`。在 `config.example.yaml` 中这个值是 `68`，也就是窗口开始后 68 秒触发：

```text
12:01:08Z -> target_window_start = 12:00:00Z
```

### 5.3 prewarm

`execution_engine/prewarm.py` 不是 service 示例中自动调用的主路径。它可以提前拉取 runtime frames、构建 feature frame，并可选把 minute、second、aggTrades、features 写到 cache output。配置里有 `prewarm_seconds_before_trigger`，但当前 systemd 示例没有对应 prewarm unit。

## 6. Polymarket 市场窗口映射

`build_btc_5m_slug(t0, offset_windows=0)`：

```text
ts = floor t0 to minute
window_start = floor ts to 5m boundary + offset_windows * 5m
window_end = window_start + 5m
slug = btc-updown-5m-<window_start unix timestamp>
```

当前 `run_once()` 使用：

```text
slug, window_start, window_end = build_btc_5m_slug(signal.t0, offset_windows=0)
```

所以 `signal.t0` 直接映射到同一个 Polymarket 窗口：

```text
signal.t0      = 2026-05-20T12:00:00Z
market slug    = btc-updown-5m-<12:00:00Z unix timestamp>
market window  = [2026-05-20T12:00:00Z, 2026-05-20T12:05:00Z)
```

## 7. Binance 数据拉取

`BinanceRealtimeClient.fetch_runtime_frames(end_time=None)` 拉取三类数据：

```text
1m klines
1s klines
aggTrades
```

基本步骤：

1. `end = end_time or server_time()`
2. `lookback_start = end - lookback_minutes`
3. 读取本地 parquet cache：`minute.parquet`、`second.parquet`、`agg_trades.parquet`
4. 从 cache 最新时间减少量 overlap 开始增量拉取：
   - minute overlap: 2 minutes
   - second overlap: 10 seconds
   - aggTrades overlap: 10 seconds
5. 1m 和 1s kline 用 `filter_closed_klines()` 过滤已闭合 candle。
6. aggTrades 按 timestamp 或 fromId 拉取，并按 `agg_trade_id` 去重。
7. 合并 cache 和 fresh，写回 cache。

`filter_closed_klines()` 的规则：

```text
keep rows where close_time <= server_time
```

因此它不会把尚未闭合的 1m 或 1s kline 放进 runtime frames。

## 8. 数据对齐：wait_for_signal_runtime_frames

`run_once()` 调用：

```python
minute_frame, second_frame, agg_trades_frame, frame_alignment =
    binance.wait_for_signal_runtime_frames(
        signal_t0=target_window_start,
        max_wait_seconds=config.schedule.max_data_wait_seconds,
        feature_offset_minutes=feature_offset_minutes,
    )
```

对 `market_t0 = T` 和当前 `feature_offset_minutes = 1`：

```text
decision_time / target              = T + 1m
required_latest_closed_minute       = T
required_latest_closed_second       = T + 59s
required_latest_agg_trade           = T + 59s - max_agg_trade_lag_seconds
```

默认 `max_agg_trade_lag_seconds = 2` 时：

```text
required_latest_agg_trade = T + 57s
```

代码会等待直到数据满足对齐要求，最长等待 `max_data_wait_seconds`。如果 `require_agg_trade_through_last_second=true`，aggTrades 有自己的较短等待上限 `agg_trade_wait_seconds`。

## 9. finalize_runtime_frames_for_signal

`finalize_runtime_frames_for_signal()` 是时间线最关键的函数。

输入：

```text
minute_frame
second_frame
agg_trades_frame
signal_t0 = T
feature_offset_minutes = 1
```

计算：

```text
market_t0       = floor(signal_t0 to minute) = T
target          = decision_timestamp(T, offset=1) = T+1m
required_minute = expected_latest_closed_minute(T, offset=1) = T
required_second = target - 1s = T+59s
required_agg    = required_second - max_agg_trade_lag_seconds
```

过滤：

```text
safe_minute = minute rows with timestamp <= required_minute
safe_second = second rows with timestamp <  target
safe_agg    = agg trades with timestamp <  target
```

校验：

```text
safe_minute 必须包含 required_minute
safe_second 最新 timestamp 必须 >= required_second
safe_agg 最新 timestamp 必须 >= required_agg
  当 require_agg_trade_through_last_second=true 时强制
```

然后追加 synthetic decision row：

```text
timestamp  = target = T+1m
OHLCV      = NaN
close_time = NaT
```

返回的 `alignment` 会记录：

```text
signal_t0
market_t0
decision_time
decision_alignment_mode
feature_offset_minutes
row_policy
feature_timestamp
market_window_start
market_window_end
required_latest_closed_minute
required_latest_closed_second
required_latest_agg_trade
minute_latest
second_latest
agg_trade_latest
agg_trade_lag_seconds
post_signal_second_rows_dropped
post_signal_agg_trade_rows_dropped
synthetic_decision_row
```

当前配置下，健康 summary 应满足：

```text
signal.t0                         == T
signal.decision_time              == T+1m
signal.feature_timestamp          == T+1m
signal.row_policy                 == delayed_1m_synthetic_decision_row
signal.required_latest_closed_minute == T
signal.required_latest_closed_second == T+59s
signal.minute_latest              == T
signal.second_latest              >= T+59s and < T+1m
signal.agg_trade_latest           >= T+57s and < T+1m
market.window_start               == T
market.window_end                 == T+5m
```

## 10. 为什么需要 synthetic decision row

共享特征构建器以 1m OHLCV frame 为 backbone，并在 grid timestamp 上产生特征行。线上在 `T+1m` 做决策时，并不存在一个真实的 1m candle，其 open time 正好是 `T+1m` 且已经闭合。

因此 runtime 层追加一行 synthetic row：

```text
timestamp = T+1m
OHLCV = NaN
```

作用是让 `src.features.builder.build_feature_frame()` 可以在 `T+1m` 这个决策时间点生成一行 grid feature row。它不是 Binance 真实 candle，也不是未来数据。

特征是否安全取决于各 feature pack 是否只使用历史列，例如多数 pack 使用 `shift(1)` 和 backward rolling。在当前代码结构下：

- 1m 数据真实可用到 `T`。
- 1s 数据真实可用到 `T+59s`。
- aggTrades 可用到 `T+59s` 附近，默认允许最多 2 秒 lag。
- synthetic row 自身的 OHLCV 是 NaN，避免把未闭合或未来 candle 注入推理。

## 11. 特征构建流程

`RuntimeInferenceEngine.build_feature_frame()`：

```text
decision_frame = minute_frame[["timestamp"]]

if settings.second_level.enabled:
  second_store = build_second_level_feature_store(
    kline_frame=second_frame,
    agg_trades_frame=agg_trades_frame,
    feature_profile=settings.second_level.get_profile_payload(),
  )
  sampled_second = sample_second_level_feature_store(decision_frame, second_store)
else:
  sampled_second = empty frame

feature_frame = build_feature_frame(
  minute_frame,
  settings,
  horizon_name="5m",
  select_grid_only=...,
  second_level_features_frame=sampled_second,
)

validate baseline.feature_columns exist
```

### 11.1 当前 second_level 状态

当前 `config/settings.yaml`：

```yaml
second_level:
  enabled: false
```

所以默认线上不会构建 1s/aggTrades second-level feature store，也不会把 second-level 特征 merge 进 1m feature frame。即使 `execution_engine` 拉取了 1s 和 aggTrades，它们当前主要用于满足 runtime 对齐能力和将来启用 second-level 的路径。

如果启用 `second_level.enabled=true`，运行时会：

1. 用 1s kline 构建 per-second feature store。
2. 可选用 aggTrades 添加事件结构特征。
3. 用 `sample_second_level_feature_store()` 对每个 decision timestamp 做 `merge_asof(direction="backward")`。
4. 只取 `sl_` 开头的训练特征列。

这意味着 second-level 特征采样是向后看的，不会主动取 decision timestamp 之后的数据。

### 11.2 1m 特征路径

`src.features.builder.build_feature_frame()` 做以下事情：

1. 规范化 OHLCV frame。
2. 根据 `horizon_name="5m"` 读取 horizon spec。
3. 根据 `feature_profile=core_5m` 读取 feature pack 列表。
4. 如果 derivatives enabled，附加 derivatives feature store。
5. 如果 second-level sampled frame 非空，按 timestamp one-to-one merge。
6. 依次执行 profile 中的 feature packs。
7. 删除 derivatives helper columns。
8. 添加 grid columns、asset、horizon、feature_version。
9. 根据 `select_grid_only` 选择 5m grid rows。

当前 `core_5m` 包含大量特征包，例如 momentum、volatility、path_structure、candle_structure、intra_5m_structure、derivatives_*、lagged、time 等。execution layer 不复制这些公式。

## 12. 推理行选择

`run_once()` 调用：

```python
result = inference.predict(
    minute_frame,
    second_frame,
    agg_trades_frame,
    signal_t0=pd.Timestamp(target_window_start),
    use_latest_available_before_signal=False,
    runtime_context=frame_alignment,
)
```

`RuntimeInferenceEngine.predict()` 中的行选择逻辑：

```text
market_t0 = signal_t0 = T
context_feature_timestamp = runtime_context["feature_timestamp"]

if context_feature_timestamp is not None and not use_latest_available_before_signal:
  target_t0 = context_feature_timestamp
else:
  target_t0 = market_t0

select row where feature_frame.timestamp == target_t0
```

当前配置下：

```text
target_t0 = feature_timestamp = T+1m
```

但生成的 `Signal` 仍然使用：

```text
signal.t0 = signal_t0 = T
```

因此 summary 中会同时出现：

```text
signal.t0            = market window start = T
signal.market_t0     = T
signal.decision_time = T+1m
signal.feature_timestamp = T+1m
```

这是预期行为。

## 13. 阈值和信号策略

artifact 加载逻辑在 `execution_engine/artifacts.py`：

```text
model_plugin
calibration_plugin
model_path
calibrator_path
feature_columns
t_up
t_down
```

阈值选择：

```text
effective t_up   = config.thresholds.t_up   if not null else artifact.t_up
effective t_down = config.thresholds.t_down if not null else artifact.t_down
```

`src.signal.policies.evaluate_selective_binary_signal()`：

```text
p_down = 1 - p_up

YES / UP signal  if p_up >= t_up
NO / DOWN signal if p_up <= t_down
NO-SIGNAL        otherwise
```

因此线上阈值不硬编码 0.5。若配置中 `thresholds.t_up/t_down` 为 null，会使用 artifact 阈值。

## 14. 市场、订单簿和订单计划

若信号策略返回 `should_trade=false`：

```text
summary.skipped += decision.reason
return summary
```

若有交易信号：

1. `build_btc_5m_slug(signal.t0)` 映射到 Polymarket market slug。
2. `PolymarketV2Adapter.get_market_by_slug(slug)` 通过 Gamma API 查询市场。
3. guard 检查 market 是否 active、closed、accepting_orders。
4. 根据 side 选择 token：
   - YES -> `market.yes_token_id`
   - NO -> `market.no_token_id`
5. `get_orderbook(token_id)` 读取 CLOB order book。
6. `build_two_limit_order_plan()` 生成最多两条 BUY limit orders。

订单价格逻辑：

```text
if best_bid exists:
  raw_price = min(best_bid, leg.price_cap) + leg.offset
else:
  raw_price = min(best_ask, leg.price_cap) + leg.offset - tick_size

price = floor_to_tick(raw_price, tick_size)
```

edge guard：

```text
fair_value = p_up if YES else 1 - p_up
edge = fair_value - price
skip if edge < min_edge
skip if price > max_buy_price
skip if spread > max_spread
skip if notional > max_order_notional
```

真实提交订单需要同时满足：

```text
mode == "live"
config.orders.enabled == true
```

否则会写入 order plan 和 skipped reason，但不会提交订单。

## 15. 幂等性

live 提交时使用 `IdempotencyStore`。key 由窗口、token、side 和 leg 组成：

```text
<window_start ISO>:<token_id>:<side>:<leg>:two_limit_plan
```

这避免同一窗口同一 token 同一 leg 重复提交。注意 key 使用 `window_start`，不是 `decision_time`。

## 16. Audit 与 Summary

`run_once()` 会记录 audit events：

```text
binance_frames_loaded
signal_generated
decision_evaluated
execution_skipped
market_mapped
order_plan_created
order_submitted
```

summary 文件写到：

```text
config.runtime.summary_dir / <signal_t0>.json
```

排查时间线时优先看这些字段：

```text
signal.t0
signal.market_t0
signal.decision_time
signal.feature_timestamp
signal.decision_alignment_mode
signal.feature_offset_minutes
signal.row_policy
signal.required_latest_closed_minute
signal.required_latest_closed_second
signal.required_latest_agg_trade
signal.minute_latest
signal.second_latest
signal.agg_trade_latest
signal.agg_trade_lag_seconds
signal.post_signal_second_rows_dropped
signal.post_signal_agg_trade_rows_dropped
market.slug
market.window_start
market.window_end
signal.t_up
signal.t_down
signal.artifact_t_up
signal.artifact_t_down
decision.reason
orders
submitted
```

## 17. 当前主时间线示例

以 Polymarket 市场 `[12:00:00Z, 12:05:00Z)` 为例，当前 `feature_offset_minutes=1`：

```text
12:00:00Z
  Polymarket BTC 5m market starts.
  market_t0 / signal_t0 = 12:00:00Z

12:00:00Z - 12:00:59.999Z
  Binance 1m candle open time 12:00 正在形成。
  Binance 1s rows 12:00:00 ... 12:00:59 陆续形成。
  Polymarket market 已经开始交易。

12:01:00Z 附近
  systemd timer 示例触发 run_once。
  或 paper experiment 在 12:01:08Z 触发。
  current_5m_window_start() 仍返回 12:00:00Z。

run_once data wait phase
  目标 market_t0 = 12:00:00Z
  decision_time = 12:01:00Z
  required_latest_closed_minute = 12:00:00Z
  required_latest_closed_second = 12:00:59Z
  required_latest_agg_trade = 12:00:57Z when max_agg_trade_lag_seconds=2

finalize phase
  只保留 timestamp < 12:01:00Z 的 second/agg 数据。
  只保留 timestamp <= 12:00:00Z 的 minute 数据。
  追加 synthetic decision row:
    timestamp = 12:01:00Z
    OHLCV = NaN

feature phase
  共享 feature builder 构建 feature_frame。
  RuntimeInferenceEngine 选择 feature_frame.timestamp == 12:01:00Z 的行。

prediction phase
  model 输出 p_up。
  Signal:
    t0 = 12:00:00Z
    market_t0 = 12:00:00Z
    decision_time = 12:01:00Z
    feature_timestamp = 12:01:00Z

decision phase
  p_up >= t_up   -> YES
  p_up <= t_down -> NO
  otherwise      -> NO-SIGNAL

market phase
  slug = btc-updown-5m-<12:00:00Z unix timestamp>
  market window = [12:00:00Z, 12:05:00Z)

order phase
  如果 YES/NO 通过并且 guards 通过，生成 BUY limit order。
  只有 live + orders.enabled=true 才提交。
```

当前线上语义可以概括为：

```text
在窗口 T 开始约 1 分钟后，
使用已闭合到 T 的 1m candle 和 T+59s 前的秒级数据，
预测并交易仍在进行的 Polymarket [T, T+5m) 市场。
```

## 18. 与 feature_offset_minutes=0 的差异

代码也支持 `feature_offset_minutes=0`。若关闭 delayed alignment：

```text
market_t0 / signal_t0 = T
decision_time         = T
feature_timestamp     = T
required_minute       = T-1m
required_second       = T-1s
row_policy            = exact_signal_t0_with_synthetic_decision_row
```

这表示在窗口刚开始后，只用窗口开始前的数据预测 `[T, T+5m)`。

但当前 `config/settings.yaml` 启用的是：

```text
feature_offset_minutes = 1
row_policy = delayed_1m_synthetic_decision_row
```

因此排查当前 summary 时应按 delayed 时间线判断，不应按 exact T 时间线判断。

## 19. 线上/线下一致性说明

一致的部分：

- 线上使用 `src.features.builder.build_feature_frame()`。
- 线上使用相同 feature profile：`core_5m`。
- 线上模型推理使用 `src.model.infer.predict_frame()`。
- 线上 selective binary policy 使用 `src.signal.policies.evaluate_selective_binary_signal()`。
- 阈值来自 artifact 或 config override。
- 线上不计算 label。

需要特别审计的差异：

```text
offline training row at T:
  历史数据集中 T candle 已经完整存在。
  label = close[T+4m] >= open[T]

online delayed row at T+1m:
  signal.t0 = T
  selected feature row timestamp = T+1m synthetic row
  真正的 1m 市场数据可用到 T
  真正的 1s/agg 数据可用到 T+59s 附近
  预测市场仍是 [T, T+5m)
```

这不是单纯的“预测窗口开始前方向”，而是“窗口开始后约 1 分钟的延迟决策”。评估和回测必须明确这种对齐方式，否则会把线上实际交易语义和离线标签语义混在一起。

## 20. 风险和检查点

### 20.1 触发过晚

如果 `run_once` 在窗口内很晚启动，例如 `12:04:30Z`，`current_5m_window_start()` 仍返回 `12:00:00Z`。当前没有 guard 阻止临近结算才交易同一市场。

建议增加配置化 guard：

```text
skip if now - decision_time > max_decision_age_seconds
skip if market_window_end - now < min_seconds_before_market_end
```

### 20.2 schedule 配置未统一驱动 systemd

`config.schedule.trigger_delay_seconds` 被 paper experiment 使用，但 systemd 示例没有读取这个配置。生产部署如果依赖 systemd，需要确保 `OnCalendar` 与配置中的预期延迟一致。

### 20.3 second_level disabled

当前 `second_level.enabled=false`。若 baseline artifact 的 feature columns 包含 `sl_` 特征，runtime 会在 `_validate_feature_columns()` 报 missing feature。若 artifact 不包含 `sl_` 特征，则 1s/aggTrades 当前不影响模型输入。

### 20.4 synthetic row 的 NaN 传播

synthetic row 的 OHLCV 是 NaN。任何没有 shift 的 feature 如果直接使用当前 row OHLCV，可能在 runtime row 产生 NaN。模型输入最终是否可接受取决于模型插件和训练时是否也包含相同对齐策略。应监控 runtime feature row 的 NaN ratio，并对关键 baseline features 做抽样检查。

### 20.5 aggTrades 延迟

默认要求 aggTrades 最新时间至少到 `required_latest_agg_trade`。若 Binance aggTrades 延迟超过 `max_agg_trade_lag_seconds`，本轮会 raise 而不是降级交易。这是保守行为，但会降低线上覆盖率。

### 20.6 阈值覆盖

如果 `execution_engine/config.yaml` 中显式设置了 `thresholds.t_up/t_down`，它会覆盖 artifact 阈值。线上 summary 应记录并检查：

```text
signal.t_up == signal.artifact_t_up
signal.t_down == signal.artifact_t_down
```

除非正在做明确的 live threshold experiment。

## 21. 推荐审计清单

每个 live/paper summary 至少检查：

```text
signal.t0 == market.window_start
market.window_end == signal.t0 + 5m
signal.feature_offset_minutes == 1
signal.row_policy == delayed_1m_synthetic_decision_row
signal.decision_time == signal.t0 + 1m
signal.feature_timestamp == signal.t0 + 1m
signal.required_latest_closed_minute == signal.t0
signal.minute_latest == signal.t0
signal.required_latest_closed_second == signal.t0 + 59s
signal.second_latest >= signal.t0 + 59s
signal.second_latest < signal.t0 + 1m
signal.agg_trade_latest >= signal.required_latest_agg_trade
signal.agg_trade_latest < signal.t0 + 1m
```

若有订单：

```text
decision.side in {YES, NO}
YES uses yes_token_id
NO uses no_token_id
order side is BUY
order price respects tick_size and guards
idempotency key uses market window_start, token_id, side, leg
```

若要判断是否存在时间泄漏，不要只看 `feature_timestamp`。应同时看：

```text
row_policy
feature_offset_minutes
required_latest_closed_minute
required_latest_closed_second
minute_latest
second_latest
agg_trade_latest
post_signal_second_rows_dropped
post_signal_agg_trade_rows_dropped
```

当前 delayed 模式下，`feature_timestamp = signal.t0 + 1m` 是预期行为。真正需要确认的是输入数据没有超过 `decision_time`，并且市场映射仍是 `signal.t0` 对应的 `[T, T+5m)`。

## 22. 最终结论

当前 `execution_engine` 主流程是：

```text
每 5 分钟窗口 T 开始约 1 分钟后触发，
拉取并校验 Binance 数据到 T / T+59s，
追加 timestamp=T+1m 的 synthetic decision row，
调用共享 core 构建 5m feature row，
用 artifact 模型输出 p_up，
用 artifact/config 阈值生成 YES/NO/NO-SIGNAL，
把 signal.t0=T 映射到 Polymarket [T,T+5m) 市场，
在 live + orders.enabled 时提交 BUY limit order。
```

这条路径保持了特征公式和信号策略的共享核心复用，但时间语义是延迟 1 分钟决策。后续任何验证、回测或线上审计都应显式标注：

```text
market_t0 = T
decision_time = T+1m
feature_timestamp = T+1m
market_window = [T, T+5m)
```

