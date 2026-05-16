# execution_engine 完整流程与时间线说明

本文档说明当前 `execution_engine` 的线上执行流程，以及离线标签、Binance 实时数据、线上特征行、模型信号和 Polymarket BTC 5 分钟市场之间的时间对齐关系。

最重要的当前实现事实是：线上推理现在使用 **`signal_t0` 上的 synthetic decision row**。这和旧文档里“选择 `signal_t0` 之前最后一行特征”的描述不同。

## 1. 当前主流程

```text
systemd timer / 手动运行
  -> execution_engine/run_once.py
  -> 读取 execution_engine/config.yaml
  -> 读取 baseline artifact 和 config/settings.yaml
  -> 拉取并对齐 Binance runtime frames
  -> 在 signal_t0 追加 synthetic 1m decision row
  -> 调用 src/ 共享 feature builder 构建特征
  -> artifact model + calibrator 推理
  -> 调用共享 selective binary signal policy
  -> 将 signal_t0 映射到 Polymarket BTC 5m slug
  -> 生成最多两档 BUY limit orders
  -> 仅在 mode=live 且 orders.enabled=true 时提交订单
  -> 写 audit 和 summary JSON
```

execution layer 不重新实现 BTC 特征公式，也不重新计算 label。线上特征路径复用共享 core：

```text
execution_engine.feature_runtime.RuntimeInferenceEngine
  -> src.data.second_level_features.build_second_level_feature_store
  -> src.data.second_level_features.sample_second_level_feature_store
  -> src.features.builder.build_feature_frame
  -> src.model.infer.predict_frame
  -> src.signal.policies.evaluate_selective_binary_signal
```

## 2. 离线标签定义

当前项目标签保持不变：

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

对 5 分钟 grid 点 `T`：

```text
offline t0              = T
offline input row time  = T 及之前的历史行
label open              = T 这一分钟 candle 的 open
label future close      = T+4m 这一分钟 candle 的 close
对应市场窗口           = [T, T+5m)
```

示例：

```text
T = 2026-05-14T12:00:00Z
label 比较:
  open  at 2026-05-14T12:00:00Z
  close at 2026-05-14T12:04:00Z candle close

Polymarket window:
  2026-05-14T12:00:00Z <= market < 2026-05-14T12:05:00Z
```

这是训练和评估用的标签。`execution_engine` 线上不计算这个标签。

## 3. 线上目标窗口

`run_once()` 选择目标窗口：

```text
target_window_start = current_5m_window_start()
signal_t0           = target_window_start
slug                = btc-updown-5m-<signal_t0 unix timestamp>
```

`build_btc_5m_slug(signal.t0, offset_windows=0)` 会把信号映射到正好从 `signal_t0` 开始的 Polymarket 市场。

示例：

```text
run starts at       2026-05-14T12:00:20Z
signal_t0           2026-05-14T12:00:00Z
Polymarket slug     btc-updown-5m-1778760000
market window       [2026-05-14T12:00:00Z, 2026-05-14T12:05:00Z)
```

当前执行路径没有向后偏移一个窗口。

## 4. Binance Runtime 数据对齐

`BinanceRealtimeClient.wait_for_signal_runtime_frames()` 会把所有 runtime 输入对齐到 `signal_t0`。

对目标 `T = signal_t0`，需要的最新安全输入是：

```text
required_latest_closed_minute = T - 1 minute
required_latest_closed_second = T - 1 second
required_latest_agg_trade     = T - 1 second - max_agg_trade_lag_seconds
```

推理前代码会过滤：

```text
safe_minute = minute rows with timestamp <= T-1m
safe_second = second rows with timestamp <  T
safe_agg    = agg trades with timestamp <  T
```

然后强校验：

```text
safe_minute 必须包含 T-1m
safe_second 最新 timestamp 必须 >= T-1s
safe_agg 最新 timestamp 必须 >= required_latest_agg_trade
  当 require_agg_trade_through_last_second=true 时
```

如果这些输入不满足要求，本轮会 raise，不会为该窗口写正常 summary 或下单。

## 5. Synthetic Decision Row 的含义

安全输入检查通过后，`finalize_runtime_frames_for_signal()` 会在 1m frame 末尾追加一行 synthetic decision row：

```text
timestamp  = T
OHLCV      = NaN
close_time = NaT
```

它的目的，是让共享 feature builder 在目标市场起点 `T` 上生成一行可推理的 grid decision row。它不是 Binance 真实 candle，也不包含未来 OHLCV。

summary 中会记录：

```text
row_policy        = exact_signal_t0_with_synthetic_decision_row
feature_timestamp = T
```

关键解释：

```text
feature_timestamp == signal_t0
```

并不表示模型用了 `T` 这一分钟尚未收盘的 OHLCV。它表示被选中的特征行是 timestamp 标记为 `T` 的 synthetic decision row。真实可用市场数据在构建前已经被限制为：

```text
1m data      through T-1m
1s data      through T-1s
aggTrades    through T-1s, subject to max_agg_trade_lag_seconds
```

因此，当前线上审计不能只看 `feature_timestamp == signal_t0` 就判断泄漏；必须同时看 `row_policy` 和 required/latest pre-signal 字段。

## 6. Runtime 特征行选择

当前 `run_once()` 调用推理时使用：

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

因为 `use_latest_available_before_signal=False`，`RuntimeInferenceEngine.predict()` 会选择：

```text
feature_frame.timestamp == signal_t0
```

这行之所以存在，是因为 runtime 层已经追加了 synthetic decision row。

代码里仍保留旧模式：

```text
use_latest_available_before_signal=True
```

该模式会选择 `feature_timestamp < signal_t0` 的最后一行。但当前 live `run_once()` 主路径不使用这个模式。

## 7. 完整时间线示例

以 `[12:00:00Z, 12:05:00Z)` 市场为例：

```text
11:59:00Z - 11:59:59.999Z
  最后一根必需的 1m candle 形成。

11:59:59Z
  最后一秒必需的 1s kline / agg trade 时间目标。

12:00:00Z
  Polymarket BTC 5m 市场开始。
  signal_t0 = 12:00:00Z。

12:00:08Z - 12:00:20Z
  timer 通常触发 run_once。
  Binance closed data 被拉取和过滤。
  需要满足:
    minute_latest    = 11:59:00Z
    second_latest    >= 11:59:59Z
    agg_trade_latest >= 11:59:59Z，除非 lag config 允许更晚缺口

  runtime 追加 synthetic 1m decision row:
    timestamp = 12:00:00Z
    OHLCV     = NaN

  共享 feature builder 生成 decision feature row。
  模型对 signal_t0=12:00:00Z 输出 p_up。
  signal policy 决定 YES、NO 或 NO-SIGNAL。
  若信号通过，映射到 slug btc-updown-5m-1778760000。
```

线上语义是：

```text
只用 T 之前已经安全可用的数据，预测并交易 [T, T+5m) 市场。
```

## 8. Summary 对齐审计字段

排查线上/线下对齐时，应同时看这些字段：

```text
signal.t0
signal.feature_timestamp
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
signal.t_up / signal.t_down
signal.artifact_t_up / signal.artifact_t_down
decision.reason
submitted
```

当前工作流下，一个健康的对齐 summary 应满足：

```text
signal.row_policy == exact_signal_t0_with_synthetic_decision_row
signal.feature_timestamp == signal.t0
signal.required_latest_closed_minute == signal.t0 - 1 minute
signal.required_latest_closed_second == signal.t0 - 1 second
signal.minute_latest == signal.t0 - 1 minute
signal.second_latest >= signal.t0 - 1 second
market.window_start == signal.t0
```

aggTrades 对齐规则：

```text
agg_trade_latest >= required_latest_agg_trade
```

当 `require_agg_trade_through_last_second=true` 时必须满足。

## 9. 阈值来源

有效阈值选择逻辑是：

```text
effective t_up   = artifact.t_up   if config.thresholds.t_up   is null else config.thresholds.t_up
effective t_down = artifact.t_down if config.thresholds.t_down is null else config.thresholds.t_down
```

因此 `execution_engine/config.yaml` 可以有意覆盖 artifact 阈值。若线上要复现 artifact 行为，应设置：

```yaml
thresholds:
  t_up: null
  t_down: null
```

除非正在明确做 live threshold experiment，否则 recent summary 应检查：

```text
signal.t_up == signal.artifact_t_up
signal.t_down == signal.artifact_t_down
```

## 10. Signal 与订单映射

二分类信号规则：

```text
p_down = 1 - p_up

YES / UP signal  if p_up >= t_up
NO / DOWN signal if p_up <= t_down
NO-SIGNAL        otherwise
```

通过阈值后映射到 Polymarket token：

```text
YES -> yes_token_id
NO  -> no_token_id
```

order planner 最多生成两档 BUY limit orders。只有同时满足以下条件才会真实提交：

```text
runtime mode is live
orders.enabled is true
```

paper mode 仍会写 signal、market、order-plan 和 skipped 字段，但不会提交订单。

## 11. 线上与线下一致性

当前线上路径和离线共享 core 保持一致的部分：

```text
Feature formulas:
  online uses src.features.builder.build_feature_frame

Second-level feature logic:
  online uses src.data.second_level_features builders and asof sampler

Model inference:
  online uses src.model.infer.predict_frame

Signal policy:
  online uses src.signal.policies.evaluate_selective_binary_signal

Label logic:
  online does not calculate labels
```

主要差异是预期差异：

```text
offline training row at T:
  完整历史数据集中，T 这一分钟真实 OHLCV 已经存在
  label 是 close[T+4m] >= open[T]

online decision row at T:
  timestamp=T 的 synthetic row 被追加
  真实 pre-signal 数据只允许到 T-1m / T-1s
  特征必须只能从 pre-signal 历史中计算
```

这要求 feature builder 对 decision row 不依赖当前窗口真实 OHLCV。synthetic row 里的 NaN OHLCV 是防止把未完成或未来 `T` candle 注入线上推理的保护机制。

## 12. 泄漏与漂移检查

在声明 live 性能改善前，至少检查：

```text
features 中没有 target、future_close、abs_return、signed_return、
stage1_target、stage2_target 或其他 label-derived 字段。

runtime feature 没有使用未完成的 T candle OHLCV。

timestamp >= T 的 second-level 数据没有进入被选中的决策行。

market.window_start == signal.t0。

Polymarket outcome 评估优先使用 summary 中实际 market slug。

阈值来自 artifact，或来自明确记录的 config override。
```

最近检查过的 2026-05-14 / 2026-05-15 live summaries 显示：

```text
signal_t0 == market.window_start
row_policy == exact_signal_t0_with_synthetic_decision_row
feature_timestamp == signal_t0
minute_latest == signal_t0 - 1 minute
second_latest and agg_trade_latest before signal_t0
```

这些证据支持当前 summary 的线上时间对齐是正确的。

## 13. 已知运行风险

### 阈值漂移

如果生产 config 覆盖 artifact 阈值，live coverage 和 accepted accuracy 会和 artifact report 明显不同。应把这种覆盖视为实验并记录。

### 触发过晚

`current_5m_window_start()` 会把当前时间向下取整到 5 分钟边界。如果 run 在窗口内很晚才开始，例如 `12:04:30Z`，仍会映射到 `[12:00, 12:05)` 市场。这和接近开盘时交易的市场状态不同。

建议增加 guard：

```text
skip if now - signal_t0 exceeds a configured maximum age
```

### Runtime 数据覆盖不足

现在 engine 会在 pre-signal minute/second/agg 数据不足时 raise。这比在不完整对齐上交易更安全，但会降低 coverage，需要监控失败次数。

### Synthetic Row 特征语义

任何隐式依赖当前行 OHLCV 的特征都要谨慎。因为 synthetic row 的 OHLCV 是 NaN，这类特征应该变成 NaN/imputed，而不是静默使用未来值。应监控关键 runtime features 的 NaN/zero ratio。

### 数据源漂移

线上数据来自 live Binance REST/cache，而离线训练数据可能来自历史采集任务。两者仍可能有分布差异。Polymarket settlement 源也可能在极近边界处和 Binance label 不一致，尽管最近检查样本里二者匹配。

## 14. 推荐 Smoke Test

对单个 live 或 paper summary，检查：

```text
signal.t0:                            T
signal.row_policy:                    exact_signal_t0_with_synthetic_decision_row
signal.feature_timestamp:             T
signal.required_latest_closed_minute: T-1m
signal.minute_latest:                 T-1m
signal.required_latest_closed_second: T-1s
signal.second_latest:                 >= T-1s and < T
signal.agg_trade_latest:              >= required_latest_agg_trade and < T
market.window_start:                  T
```

如果信号通过，还要检查：

```text
decision.side in {YES, NO}
order token matches decision.side
idempotency key uses window_start, token_id, side, and leg
```

## 15. 最终解释

当前 workflow 是：

```text
在 5 分钟边界 T 刚过后，
只使用 T 之前已经安全闭合/可用的 Binance 数据，
追加 timestamp=T 的 synthetic decision row，
预测 Polymarket [T, T+5m) 市场，
并只在 artifact/config 阈值接受信号时交易 YES 或 NO。
```

因此，在当前 summary 里看到 `feature_timestamp == signal_t0` 是预期行为，不能单独据此判断未来数据泄漏。真正证明对齐的是：

```text
row_policy == exact_signal_t0_with_synthetic_decision_row
required/latest pre-signal data fields 对齐
market.window_start == signal.t0
```
