# execution_engine 完整流程与时间线说明

本文档解释当前 `execution_engine` 的线上执行流程，重点说明 5 分钟窗口、Binance 数据时间、特征行时间、Polymarket 市场窗口之间的对齐关系，并分析当前实现是否存在线上漂移风险。

## 1. 总体结论

当前执行链路不是独立重写特征或模型，而是：

```text
Binance 实时数据
  -> execution_engine.realtime_data 规范化为共享 schema
  -> execution_engine.feature_runtime 调用 src/ 共享特征 builder
  -> baseline artifact 模型 + calibrator 推理
  -> src.signal.policies 阈值判断
  -> Polymarket BTC 5m slug 映射
  -> order_plan 生成两档限价单
  -> paper/live 提交或跳过
```

核心对齐规则是：

```text
要交易 Polymarket 窗口 [T, T+5m)
模型 signal_t0 = T
线上实际使用的最后一根 1m 特征行 = T-1m 这一分钟 bar
该 bar 的 timestamp = T-1m
该 bar 的 close_time = T-1ms
因此它只应在 T 之后才被线上使用
```

举例：

```text
目标市场: 2026-05-10 10:05:00Z 到 10:10:00Z
signal_t0: 2026-05-10 10:05:00Z
最后可用 1m bar:
  timestamp/open_time = 2026-05-10 10:04:00Z
  close_time          = 2026-05-10 10:04:59.999Z
线上触发建议时间:
  2026-05-10 10:05:08Z 到 10:05:20Z
推理时选择:
  feature_timestamp < signal_t0 的最后一行，即 10:04:00Z
Polymarket slug:
  btc-updown-5m-<10:05:00Z unix timestamp>
```

在这个定义下，当前实现的主路径没有看到“使用未来 OHLCV/未来收益/label 派生列”的直接证据；但存在几个需要明确监控的漂移风险：

1. 配置里的 `thresholds.t_up/t_down` 如果非空，会覆盖 artifact 阈值，可能造成线上阈值与训练报告不一致。
2. `second-level` 线上只使用 Binance `1s kline + aggTrades`，而 artifact manifest 显示 derivatives disabled，但包含 flow proxy 与 second-level 特征；如果训练时某些 second-level 扩展源在线上不可用，会以缺失/0 填充形式产生分布漂移。
3. 手动或 timer 在错误时间触发时，`current_5m_window_start()` 会按当前时间取最近 5m 窗口；如果运行太晚，可能仍映射到已经接近结束的市场。
4. `feature_timestamp` 是 bar 开盘时间，不是数据可用时间；审计时必须同时看 Binance latest closed 数据和执行触发时间，不能误以为 `10:04:00` 在 `10:04:00` 已可用。

## 2. 入口和配置加载

主入口是 `execution_engine/run_once.py` 的 `run_once()`。

运行参数：

```bash
python execution_engine/run_once.py --config execution_engine/config.yaml --mode paper --print-json
```

流程开始时会读取：

```text
execution_engine/config.py::load_execution_config
execution_engine/artifacts.py::load_baseline_artifact
src.core.config::load_settings
```

关键配置：

```yaml
baseline:
  artifact_dir: execution_engine/deploy/baseline
  settings_path: config/settings.yaml

runtime:
  mode: paper

binance:
  lookback_minutes: 360
  require_closed_kline: true

schedule:
  interval_minutes: 5
  trigger_delay_seconds: 8
  max_data_wait_seconds: 20

thresholds:
  t_up: null
  t_down: null
```

注意：当前 `config.example.yaml` 里 `thresholds.t_up/t_down` 是非空示例值。代码逻辑是“配置非空则覆盖 artifact，否则用 artifact”。如果生产目标是完全复现 artifact 阈值，应把生产 `config.yaml` 里的阈值设为 `null`。

## 3. 目标窗口如何确定

`run_once()` 接受可选参数 `target_window_start`。如果没有显式传入，就调用：

```text
current_5m_window_start(now)
```

该函数会把当前 UTC 时间向下取整到最近的 5 分钟边界。

例子：

```text
当前时间 10:05:08Z -> target_window_start = 10:05:00Z
当前时间 10:09:59Z -> target_window_start = 10:05:00Z
当前时间 10:10:00Z -> target_window_start = 10:10:00Z
```

之后：

```text
signal_t0 = target_window_start
slug = btc-updown-5m-<signal_t0 unix timestamp>
```

也就是说，当前实现交易的是 `signal_t0` 开始的 Polymarket 5m 市场，`build_btc_5m_slug(signal.t0, offset_windows=0)` 没有向后偏移一个窗口。

## 4. Binance 实时数据时间语义

`BinanceRealtimeClient` 拉取三类数据：

```text
1m kline      -> 主 OHLCV 特征和 5m grid
1s kline      -> second-level kline 特征
aggTrades     -> second-level event enrichment
```

`normalize_binance_klines()` 把 Binance kline 的 `open_time` 改名为共享字段 `timestamp`，同时保留 `close_time`。

因此一根 1m kline 的含义是：

```text
timestamp = 10:04:00Z
close_time = 10:04:59.999Z
open/high/low/close/volume = 10:04:00Z 到 10:04:59.999Z 内的完整数据
```

`filter_closed_klines(frame, server_time)` 只保留：

```text
close_time <= server_time
```

所以只要 `server_time` 来自 Binance server time，并且 `require_closed_kline=true`，线上不会使用未收盘 kline。

## 5. 特征构建完整链路

`RuntimeInferenceEngine.predict()` 调用：

```text
RuntimeInferenceEngine.build_feature_frame()
```

内部步骤：

```text
1. decision_frame = minute_frame[["timestamp"]]
2. build_second_level_feature_store(1s kline, aggTrades)
3. sample_second_level_feature_store(decision_frame, second_store)
4. src.features.builder.build_feature_frame(
     minute_frame,
     settings,
     second_level_features_frame=sampled_second
   )
5. 校验 baseline.feature_columns 全部存在
6. predict_frame(feature_frame, model, calibrator, feature_columns)
```

这里最重要的是第 3 步。

`sample_second_level_feature_store()` 使用：

```text
pd.merge_asof(..., direction="backward")
```

含义是：对每个 1m decision timestamp，选择时间小于等于该 timestamp 的最近一条 second-level 特征。

例子：

```text
decision timestamp: 10:04:00Z
可用 1s second_store:
  10:03:58Z
  10:03:59Z
  10:04:00Z
  10:04:01Z

采样结果:
  选择 10:04:00Z，不会选择 10:04:01Z
```

因此 second-level 对齐方向是向后取最近值，不会从 decision timestamp 之后取未来秒级数据。

## 6. 推理时选择哪一行特征

`run_once()` 调用推理时固定传入：

```python
use_latest_available_before_signal=True
signal_t0=pd.Timestamp(target_window_start)
```

`RuntimeInferenceEngine.predict()` 在这种模式下不会要求 `feature_timestamp == signal_t0`，而是选择：

```text
feature_timestamp < signal_t0 的最后一行
```

举例：

```text
target_window_start / signal_t0 = 10:05:00Z
feature_frame timestamps:
  09:59:00Z
  10:00:00Z
  10:01:00Z
  10:02:00Z
  10:03:00Z
  10:04:00Z

选择 row:
  10:04:00Z

Signal 中记录:
  signal.t0 = 10:05:00Z
  decision_context.timestamp = 10:05:00Z
  decision_context.feature_timestamp = 10:04:00Z
```

这个设计是为了满足“用上一根已收盘的 1m bar 预测当前刚开始的 5m Polymarket 市场”的线上约束。

## 7. 10:00 到 10:15 的完整时间线示例

下面假设 timer 在每个 5m 边界后 20 秒运行 `run_once`，prewarm 在边界前运行。

### 7.1 预测 10:05-10:10 市场

```text
10:00:00Z
  10:00-10:05 市场开始。
  可以开始为 10:05-10:10 市场做 prewarm。

10:04:00Z 到 10:04:59.999Z
  形成最后一根输入 1m bar。
  这根 bar 的 timestamp 是 10:04:00Z。

10:05:00Z 后
  Binance server_time 已超过 10:04:59.999Z。
  filter_closed_klines 允许 10:04:00Z 这一行进入 runtime frame。

10:05:08Z 到 10:05:20Z
  run_once 触发。
  current_5m_window_start = 10:05:00Z。
  feature row = 10:04:00Z。
  signal.t0 = 10:05:00Z。
  slug = btc-updown-5m-<10:05:00Z unix>。
  如果 p_up >= t_up，下 YES/UP token。
  如果 p_up <= t_down，下 NO/DOWN token。
  否则 NO-SIGNAL。

10:05:20Z 之后
  当前实现只提交 GTC limit order。
  不追单、不撤单、不改价。
```

### 7.2 预测 10:10-10:15 市场

```text
10:05:00Z
  10:05-10:10 市场开始。
  可以开始为 10:10-10:15 市场 prewarm。

10:09:00Z 到 10:09:59.999Z
  最后一根输入 1m bar。

10:10:08Z 到 10:10:20Z
  run_once 触发。
  signal_t0 = 10:10:00Z。
  feature_timestamp = 10:09:00Z。
  slug = btc-updown-5m-<10:10:00Z unix>。
```

## 8. Polymarket 市场映射

信号通过阈值后，`run_once()` 调用：

```text
build_btc_5m_slug(signal.t0, offset_windows=0)
```

该函数将 `signal.t0` 向下取整到 5m 边界，生成：

```text
btc-updown-5m-<window_start_unix_timestamp>
```

然后：

```text
PolymarketV2Adapter.get_market_by_slug(slug)
```

如果市场不存在、未 active、closed、或不 accepting_orders，则跳过下单并写 audit/summary。

方向映射：

```text
模型 p_up >= t_up  -> Decision side = YES -> yes_token_id
模型 p_up <= t_down -> Decision side = NO  -> no_token_id
```

下单前会读取目标 token 的 order book，取 best bid 或 best ask fallback，然后生成最多两档 BUY GTC limit order。

## 9. 线上与离线特征是否一致

从代码路径看，当前实现做到了关键的一致性：

```text
线上主特征:
  execution_engine.feature_runtime
    -> src.features.builder.build_feature_frame

线上 second-level:
  execution_engine.feature_runtime
    -> src.data.second_level_features.build_second_level_feature_store
    -> src.data.second_level_features.sample_second_level_feature_store

线上推理:
  src.model.infer.predict_frame

线上 signal policy:
  src.signal.policies.evaluate_selective_binary_signal
```

也就是说，execution layer 没有自己重写 BTC 特征公式，符合“共享 core 是单一事实来源”的要求。

需要注意的是，线上数据源与离线数据源仍可能有分布差异：

```text
离线 second-level store:
  可能来自已落盘的历史 1s/agg/book/扩展源

线上 runtime:
  当前只拉 Binance 1m kline、1s kline、aggTrades
  没有显式接入 book/depth/cross-market/ETH/derivatives
```

如果 baseline feature columns 需要某些线上没有的列，当前 `_validate_feature_columns()` 会直接报错。若列存在但由缺失值填 0，则不会报错，但可能产生分布漂移。当前 artifact manifest 显示：

```text
feature_count = 516
second_level.enabled = true
second_level.feature_count = 182
derivatives.enabled = false
```

因此 derivatives 不应成为当前线上特征缺口；主要风险集中在 second-level profile 是否与训练时完全一致、aggTrades 是否稳定覆盖、以及 1s kline REST 数据是否完整。

## 10. 是否存在未来函数或时间泄漏

基于当前读取到的代码，主执行链路的防未来机制包括：

```text
1. Binance kline 使用 close_time <= server_time 过滤闭合 bar。
2. 推理选择 feature_timestamp < signal_t0，而不是 signal_t0 或之后。
3. second-level 采样使用 merge_asof direction="backward"。
4. feature_frame 使用 baseline.feature_columns，缺列直接报错。
5. execution_engine 不重新计算 label，不读取 target/future_close。
```

因此，按推荐 timer 在 5m 边界之后运行时，没有看到明显未来函数。

但有一个容易误判的点：

```text
feature_timestamp = 10:04:00Z
```

这并不表示线上在 `10:04:00Z` 就知道该行完整 OHLCV。它表示 10:04 这一分钟 bar 的开盘时间。该行在线上真正可用时间应接近：

```text
10:05:00Z 之后
```

所以审计时不能只比较 `feature_timestamp` 和 `signal_t0`，还要确认 `run_timestamp >= close_time`。

## 11. 线上漂移风险分析

### 11.1 阈值漂移

`RuntimeInferenceEngine.__init__()` 的逻辑是：

```text
self.t_up = baseline.t_up if config.thresholds.t_up is None else config.thresholds.t_up
self.t_down = baseline.t_down if config.thresholds.t_down is None else config.thresholds.t_down
```

这意味着配置可以覆盖 artifact 阈值。

风险：

```text
训练报告/manifest 阈值: t_up=0.585, t_down=0.335
生产 config 阈值:       t_up=0.5425, t_down=0.44
```

如果生产配置确实这样设置，线上 signal coverage、accepted accuracy、selection_score 都会偏离 artifact 报告。严格复现实验时，应将生产 `thresholds` 设为 `null`，让 artifact 统一控制阈值。

### 11.2 时间窗口漂移

当前 `run_once()` 默认交易当前 5m 窗口：

```text
now = 10:05:20Z -> signal_t0 = 10:05:00Z -> 交易 10:05-10:10 市场
```

如果 timer 严格在边界后十几秒运行，这是预期行为。

风险场景：

```text
now = 10:09:40Z
current_5m_window_start = 10:05:00Z
仍会映射 10:05-10:10 市场
```

此时市场即将结束，价格、流动性、成交机会都已与训练/验证假设不同。建议在文档和监控中增加“最晚允许触发时间”，例如窗口开始后 180 秒内，否则跳过。

### 11.3 数据源漂移

线上依赖 Binance public REST：

```text
/api/v3/klines interval=1m
/api/v3/klines interval=1s
/api/v3/aggTrades
```

风险：

```text
1. REST 返回延迟导致 latest closed bar 缺失。
2. aggTrades 分页或限流导致最近事件不完整。
3. 本地 cache 合并时，如果缓存损坏或旧 schema 残留，可能影响 runtime frame。
```

当前实现会等到三类 frame 都非空，但没有在 `wait_for_closed_runtime_frames()` 中强校验“latest minute 是否达到 signal_t0-1m”。因此 summary/audit 里的 `minute_latest`、`second_latest`、`agg_trade_latest` 很重要。

### 11.4 缺失值填充漂移

`sample_second_level_feature_store()` 对 `sl_` 列做：

```text
replace inf -> NaN -> fillna(0.0)
```

如果线上某段秒级数据缺失，部分 second-level 特征可能被填成 0。这个行为可能与训练时缺失处理一致，也可能造成线上分布集中到 0。建议在 summary 中长期监控：

```text
second-level sl_ 列 NaN/zero ratio
has_agg_trade_enrichment
agg_trade_gap_flag
kline_gap_flag
```

### 11.5 模型与依赖版本漂移

baseline 使用 pickle artifact：

```text
catboost_lgbm_logit_blend.binary.pkl
platt_logit.binary.pkl
```

如果生产 venv 中 `scikit-learn`、`lightgbm`、`catboost` 版本与 artifact 保存环境差异过大，可能出现加载警告或推理行为差异。当前 README 已提示 `scikit-learn==1.7.2`。

## 12. 推荐审计字段

每次 summary/audit 中已经包含部分关键字段。排查时间漂移时，应重点看：

```text
signal.t0
signal.feature_timestamp
minute_latest
second_latest
agg_trade_latest
p_up / p_down
t_up / t_down
artifact_t_up / artifact_t_down
market.slug
market.window_start / market.window_end
decision.reason
orders_enabled / mode
submitted
```

判断是否正确对齐的简单规则：

```text
feature_timestamp == signal.t0 - 1 minute
minute_latest >= signal.t0 - 1 minute
second_latest >= signal.t0 - 1 second 或至少覆盖到 signal.t0 附近
market.window_start == signal.t0
t_up/t_down 与 artifact_t_up/artifact_t_down 是否按预期一致
```

注意：`second_latest` 可能大于 `signal.t0`，因为 runtime 拉取的是当前 lookback 窗口内的最新秒级数据；最终模型行仍由 `feature_timestamp < signal_t0` 控制，不会选择 signal_t0 之后的 feature row。

## 13. 建议改进项

不改变模型、不改变特征语义的前提下，建议优先做这些低风险增强：

1. 在 `run_once()` 中显式校验 `feature_timestamp == signal_t0 - 1 minute`，否则跳过并记录原因。
2. 增加“窗口开始后最晚执行秒数”配置，避免在市场快结束时仍对当前窗口下单。
3. 生产配置将 `thresholds.t_up/t_down` 设为 `null`，除非明确要做线上阈值覆盖实验。
4. 在 summary 中增加 second-level 覆盖率、零值率、最新数据滞后秒数。
5. 增加一个离线回放测试：用同一段历史数据模拟 runtime 输入，比较线上构建出的最后一行 feature 与离线 feature store 对齐后的行。

## 14. 最终判断

当前 `execution_engine` 主路径的时间线设计是合理的：

```text
用 T-1m 已闭合 bar 的特征
在 T 之后预测并交易 [T, T+5m) Polymarket 市场
second-level 只向后 asof 对齐
特征公式复用 src/ 共享 builder
```

因此没有看到明显的线上未来函数。

但当前实现还不能完全排除线上漂移，主要风险不是 label 泄漏，而是：

```text
阈值配置覆盖 artifact
错误触发时间导致市场窗口过晚
实时 Binance 秒级/aggTrades 覆盖不足
second-level 缺失值填 0 带来的分布漂移
依赖版本与 artifact 环境不一致
```

上线前应把这些风险纳入 summary 检查和 smoke test，而不是只看是否成功提交订单。
