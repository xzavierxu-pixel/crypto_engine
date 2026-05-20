# Live Online Pipeline Monitor - 2026-05-20

## Scope

Server: `aws-poly`

Monitor window:

```text
start_utc: 2026-05-20T10:27:56.286410+00:00
end_utc:   2026-05-20T11:27:56.286410+00:00
```

Raw monitor output on server:

```text
/home/ubuntu/opt/crypto_engine/artifacts/reports/execution_engine/live_monitor_one_hour_20260520.json
/home/ubuntu/opt/crypto_engine/artifacts/reports/execution_engine/live_monitor_one_hour_20260520.log
```

The monitor used Polymarket resolved outcomes as the primary truth source. Binance BTC direction was recorded only as a diagnostic.

## Deployment State

Code on server:

```text
commit: 4255f41
runtime.mode: live
orders.enabled: true
orders.mode: live
```

Threshold state at the end of monitoring:

```text
execution_engine/config.yaml:
  thresholds.t_up: null
  thresholds.t_down: null

artifact thresholds:
  t_up: 0.62
  t_down: 0.415
```

This is the intended configuration: live execution loads accepted thresholds from the deploy artifact.

Systemd timers were active:

```text
prewarm timer:   T+00:23
execution timer: T+01:08
```

## One-Hour Metrics

```text
window_count: 12
accepted_count: 9
coverage: 0.7500
abstain_count: 3
YES signals: 4
NO signals: 5

Polymarket resolved windows: 8
accepted resolved windows: 8
correct accepted resolved: 3
accepted resolved accuracy: 0.3750

Binance-direction accuracy on accepted: 0.4444
Polymarket-vs-Binance mismatch count: 2
```

Offline accepted validation reference:

```text
coverage: 0.7001874665
accepted_sample_accuracy: 0.6909542934
selection_score: 0.5748509217
thresholds: t_up=0.62, t_down=0.415
```

The one-hour live result is therefore materially below offline validation accuracy, even though live coverage is above the offline minimum.

## Probability Distribution

One-hour live `p_up` distribution:

```text
mean: 0.4942
std:  0.1891
p10:  0.2438
p50:  0.4589
p90:  0.7529
min:  0.2089
max:  0.7679
```

Deploy artifact offline full-train reference:

```text
mean: 0.5153
std:  0.2127
p10:  0.2296
p50:  0.5225
p90:  0.7991
```

The live distribution is slightly lower and less dispersed, but the larger issue is not only distribution shift. Several accepted signals with clear model confidence resolved in the opposite direction.

## Per-Window Outcomes

| T0 UTC | p_up | Signal | Polymarket outcome | Binance direction | Correct vs Polymarket | Notes |
|---|---:|---|---|---|---|---|
| 10:30 | 0.2340 | NO | YES | YES | no | High-confidence NO failed. |
| 10:35 | 0.6305 | YES | NO | NO | no | Just above UP threshold, failed. |
| 10:40 | 0.3827 | NO | NO | NO | yes | Correct. |
| 10:45 | 0.4753 | abstain | unresolved in monitor | YES | n/a | Correctly no trade under artifact thresholds. |
| 10:50 | 0.3321 | NO | YES | NO | no | Polymarket/Binance mismatch; model matched Binance, lost on settlement. |
| 10:55 | 0.2089 | NO | NO | NO | yes | Correct. |
| 11:00 | 0.3853 | NO | YES | YES | no | High-confidence NO failed. |
| 11:05 | 0.7552 | YES | NO | NO | no | High-confidence YES failed. |
| 11:10 | 0.5839 | abstain | unresolved in monitor | NO | n/a | Correctly no trade under artifact thresholds. |
| 11:15 | 0.7679 | YES | YES | NO | yes | Polymarket/Binance mismatch; correct on settlement. |
| 11:20 | 0.7322 | YES | unresolved in monitor | YES | n/a | Submitted. |
| 11:25 | 0.4424 | abstain | unresolved in monitor | unresolved | n/a | Correctly no trade under artifact thresholds. |

## Pipeline Health

Data alignment:

```text
minute_ok: 12 / 12
second_ok: 12 / 12
max_agg_lag_seconds: 0.758
configured max_agg_trade_lag_seconds: 2.0
```

No evidence was found that the low accuracy is caused by missing T-minute data, stale 1-second data, or aggTrade lag.

Order path:

```text
order responses: 18
successful responses with orderID: 18
matched status count in immediate lookup: 6
live status count in immediate lookup: 9
```

The order submission path is functioning. Second-leg orders are often low-price GTC orders and remain `LIVE` rather than immediately matching. This is expected behavior under the current two-leg policy.

## Problems Found

### 1. Live accuracy is far below offline validation

The live accepted resolved accuracy in this one-hour window was:

```text
3 / 8 = 37.5%
```

This is far below the accepted offline validation accuracy of about `69.1%`.

This cannot be explained by data alignment or order submission failures. The model was directionally wrong on multiple high-confidence accepted samples.

### 2. Recent threshold override caused earlier extra low-confidence trades

Before this one-hour monitor, the server config had been changed at `2026-05-20T09:20:25Z` to:

```yaml
thresholds:
  t_up: 0.55
  t_down: 0.45
```

That was not the accepted artifact threshold. It increased coverage and admitted weaker signals. I restored the config before this one-hour monitor:

```yaml
thresholds:
  t_up: null
  t_down: null
```

The one-hour monitor therefore used artifact thresholds again, but performance remained weak.

### 3. Polymarket settlement differs from BTC OHLCV direction in some windows

There were 2 Polymarket-vs-Binance direction mismatches during the monitor.

The important example is `10:50`:

```text
model signal: NO
Binance direction: NO
Polymarket resolved outcome: YES
```

Under the project objective this is a wrong live trade, because the target label is Polymarket resolved outcome. BTC direction can only be a diagnostic.

### 4. Current summary status lookup is incomplete for some matched orders

Some `order_statuses` entries contain null status fields even when the initial response had `success=true` and an order ID. This looks like a CLOB lookup normalization/availability issue after immediate match or market resolution. It does not appear to block order placement, but it weakens monitoring clarity.

## Likely Causes

1. Short live sample, but the observed miss rate is too large to ignore.
2. Current market regime is not matching the offline validation regime.
3. The model is overconfident in some short-horizon reversal conditions. Examples:
   - `10:30`: `p_up=0.234`, predicted NO, resolved YES.
   - `11:05`: `p_up=0.755`, predicted YES, resolved NO.
4. Polymarket settlement mismatch versus Binance OHLCV direction adds noise, but it does not explain all misses.
5. There is no feature-level online drift artifact yet, only probability distribution reference. That limits diagnosis of which input features shifted.

## Actions Already Taken

1. Restored `aws-poly` thresholds to artifact mode:

```yaml
thresholds:
  t_up: null
  t_down: null
```

2. Confirmed execution still uses:

```text
artifact thresholds: 0.62 / 0.415
```

3. Confirmed timers remain active and aligned:

```text
prewarm:   T+00:23
execution: T+01:08
```

4. Confirmed the live order path returns successful responses and order IDs.

## Recommended Fixes

### Immediate

1. Do not loosen thresholds live.
2. Keep `thresholds.t_up/t_down` as `null` so the engine uses artifact thresholds.
3. Reduce live risk until a larger live sample recovers:
   - set `orders.first.size` and `orders.second.size` lower, or
   - temporarily disable live orders and continue paper/shadow.
4. Continue monitoring until at least 100 accepted resolved samples are available.

### Monitoring

1. Add a live report job that writes:

```text
coverage
accepted_sample_accuracy
YES/NO split
Polymarket-vs-Binance mismatch rate
p_up bucket accuracy
order response success rate
matched/live/canceled order status counts
```

2. Improve order status lookup so immediately matched orders always record a non-null status or a clear lookup error.

### Modeling

1. Build and store offline feature distribution references for the 1016 deployed features.
2. During live inference, store feature row summary or feature drift metrics.
3. Analyze the false accepted samples by:
   - p_up bucket
   - YES vs NO
   - minute of hour/session
   - high volatility versus low volatility
   - Polymarket-vs-Binance mismatch
4. Re-run threshold search on recent live/paper resolved samples separately from the offline validation artifact. Do not overwrite the accepted artifact until the new validation protocol is defined.

## Recommendation

Do not treat the current live run as healthy. The pipeline is operational, but model live accuracy is not acceptable in this monitored hour.

The safest operating mode is to keep the engine deployed but reduce exposure or return to paper/shadow until more resolved live samples confirm recovery. If live continues, keep artifact thresholds and monitor every resolved market.
