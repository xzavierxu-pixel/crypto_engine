# aws-poly post-deploy loss analysis

Analysis date: 2026-05-18

Server: `aws-poly`

Deployment inspected:

- Repo path: `/home/ubuntu/opt/crypto_engine`
- Branch: `version1`
- HEAD: `0204dcbb01dd0919df49b83dc17f3c624c962bab`
- Commit time: 2026-05-17 12:06:00 +0800
- Commit subject: `resolved data integrity issue`
- Analysis window used: 2026-05-17 04:06:00 UTC through the last available execution summary at 2026-05-18 00:35:00 UTC

Secrets were used only to authenticate CLOB queries on the server. No secret values were printed or copied.

## Executive summary

The live strategy lost money primarily because the execution layer is suffering from severe adverse selection.

The model signals after the deploy were worse than offline validation, but they were still directionally positive:

- Offline validation accepted-sample accuracy in deployed artifact: `0.7611`
- Online signal accuracy after deploy, before fill filtering: `0.6611`
- Online filled-window signal accuracy: `0.5481`
- Online unfilled-window signal accuracy: `1.0000`
- Filled order accuracy by event: `0.4846`
- Filled order accuracy by shares: `0.4989`
- Average fill price: `0.5372`
- Realized PnL from matched resolved fills: `-46.678948 USDC`

The most important finding is:

> The strategy was correct often enough at the signal level, but the limit order policy preferentially filled the losing predictions and missed many winning predictions.

This is visible in the fill split:

- Accepted signals with resolved outcomes: `180`
- Correct accepted signals: `119 / 180 = 66.11%`
- Filled windows: `135`, accuracy `54.81%`
- Unfilled windows: `45`, accuracy `100.00%`
- Wrong signals filled: `100%`
- Correct signals filled: `62.18%`

So the live fills do not represent the offline accepted-sample distribution. The live strategy is currently trading a negatively selected subset of the model's predictions.

## Current live configuration

The inspected `execution_engine/config.yaml` had:

```yaml
runtime:
  mode: live
orders:
  enabled: true
  first:
    price_cap: 0.7
    offset: 0.0
    size: 5.0
  second:
    price_cap: 0.7
    offset: -0.1
    size: 5.0
guards:
  max_orders_per_window: 2
```

The deployed artifact thresholds were:

```text
t_up = 0.535
t_down = 0.405
```

The order planner submits up to two BUY limit orders per signal. It uses the selected Polymarket token's best bid as the quote reference when available:

```text
first price  = min(best_bid, 0.70)
second price = min(best_bid, 0.70) - 0.10
```

This means the engine does not currently require expected value versus market price. A YES signal with `p_up = 0.56` can still place a BUY order at `0.70` if the market best bid is high enough. For binary payout tokens, a simple expected-value gate is:

```text
YES buy edge = p_up - price
NO buy edge  = (1 - p_up) - price
```

The current live policy can buy at prices above the model-implied probability. That is structurally dangerous even before fill selection is considered.

## Data sources and method

Evidence used:

- Execution summaries: `artifacts/logs/execution_engine/summaries/*.json`
- Audit log: `artifacts/logs/execution_engine/live.jsonl`
- Deployed artifact metrics:
  - `execution_engine/deploy/baseline/report.json`
  - `execution_engine/deploy/baseline/metrics.json`
  - `execution_engine/deploy/baseline/threshold_search.json`
- CLOB authenticated trade history queried with `execution_engine/secrets.env`
- Binance 1m klines for actual labels

Actual direction was calculated with the project label rule:

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

For matched BUY fills:

```text
cost   = shares * fill_price
payout = shares if token side matches actual side else 0
pnl    = payout - cost
```

Open orders at analysis time: `0`.

## Aggregate results

```text
summary_count:                  226
first_t0:                       2026-05-17T05:15:00+00:00
last_t0:                        2026-05-18T00:35:00+00:00
signals_should_trade:           180
signal_accuracy:                0.6611111111
filled_window_signal_count:     135
filled_window_signal_accuracy:  0.5481481481
unfilled_window_signal_count:   45
unfilled_window_signal_accuracy:1.0000000000
engine_orders_submitted:        360
orders_with_any_fill:           242
fill_events:                    293
filled_windows:                 135
filled_shares:                  1217.22491
gross_cost_buy:                 653.946622
gross_payout_buy:               607.267674
realized_pnl:                   -46.678948
fill_accuracy:                  0.4846416382
weighted_fill_accuracy_shares:  0.4988952075
avg_fill_price:                 0.5372438706
```

## Offline versus online

Deployed artifact validation metrics:

```text
validation sample_count:              7471
validation coverage:                  0.7822246018
validation accepted_sample_accuracy:  0.7611225188
validation selection_score:           0.9450449200
validation utility:                   0.4085129166
validation downside_risk:             0.4322682530
validation up_prediction_count:       3394
validation down_prediction_count:     2450
```

Online after deploy:

```text
online accepted signals:       180
online accepted accuracy:      0.6611111111
online YES/NO signals:         roughly balanced
online filled-window accuracy: 0.5481481481
online fill accuracy by shares:0.4988952075
```

There are two separate gaps:

1. Model generalization gap: offline `76.1%` accepted accuracy fell to online `66.1%`.
2. Execution selection gap: online `66.1%` signal accuracy fell to `54.8%` at filled windows and `49.9%` by filled shares.

The second gap is the more direct cause of the loss.

## PnL breakdown

By decision side:

```text
NO  fills=146 shares=615.9674 cost=334.1905 pnl=-23.2050 weighted_acc=0.5049 avg_px=0.5425
YES fills=147 shares=601.2575 cost=319.7561 pnl=-23.4739 weighted_acc=0.4928 avg_px=0.5318
```

The loss was symmetric across YES and NO. This is not primarily a side-balance issue.

By order leg:

```text
first  fills=155 shares=677.2580 cost=396.4105 pnl=-24.1256 weighted_acc=0.5497 avg_px=0.5853
second fills=138 shares=539.9669 cost=257.5361 pnl=-22.5533 weighted_acc=0.4352 avg_px=0.4769
```

The second leg was materially worse. Lower-priced passive orders are not automatically safer here; they are more likely to be hit when the market is moving against the selected token.

By maker/taker role:

```text
MAKER fills=279 shares=1147.2249 cost=618.3466 pnl=-51.0789 weighted_acc=0.4945 avg_px=0.5390
TAKER fills=14  shares=70.0000   cost=35.6000  pnl=+4.4000  weighted_acc=0.5714 avg_px=0.5086
```

Most fills were maker fills and maker fills produced the loss. This strongly supports the adverse-selection explanation.

By fill price bucket:

```text
0.0-0.1 pnl=-0.10  weighted_acc=0.000 avg_px=0.020
0.1-0.2 pnl=+3.64  weighted_acc=0.500 avg_px=0.135
0.2-0.3 pnl=+5.45  weighted_acc=0.364 avg_px=0.265
0.3-0.4 pnl=-3.00  weighted_acc=0.320 avg_px=0.344
0.4-0.5 pnl=-12.53 weighted_acc=0.351 avg_px=0.436
0.5-0.6 pnl=-18.35 weighted_acc=0.466 avg_px=0.529
0.6-0.7 pnl=-23.30 weighted_acc=0.554 avg_px=0.615
0.7-0.8 pnl=+1.50  weighted_acc=0.707 avg_px=0.700
```

Mid-price fills, especially `0.4-0.7`, produced most of the loss. These are precisely the fills that look cheap enough to get executed but still require a strong conditional edge after fill selection.

By model confidence bucket:

```text
0.5-0.6 pnl=-2.60  weighted_acc=0.435 avg_px=0.457
0.6-0.7 pnl=-11.04 weighted_acc=0.507 avg_px=0.529
0.7-0.8 pnl=+11.46 weighted_acc=0.570 avg_px=0.541
0.8-0.9 pnl=-35.00 weighted_acc=0.382 avg_px=0.588
0.9-1.0 pnl=-9.50  weighted_acc=0.333 avg_px=0.650
```

The highest-confidence live predictions performed poorly in this sample. That can happen from calibration drift, regime shift, data mismatch, or overconfident model outputs. It means size should not be increased from raw model confidence until live calibration is verified.

Worst UTC hours:

```text
2026-05-17T12:00 pnl=-29.40 shares=80.0 weighted_acc=0.125
2026-05-17T17:00 pnl=-22.00 shares=70.0 weighted_acc=0.286
2026-05-17T11:00 pnl=-14.81 shares=36.0 weighted_acc=0.167
2026-05-17T18:00 pnl=-12.80 shares=90.0 weighted_acc=0.444
2026-05-17T15:00 pnl=-12.00 shares=50.0 weighted_acc=0.400
2026-05-17T23:00 pnl=-10.25 shares=50.0 weighted_acc=0.400
2026-05-17T10:00 pnl=-9.25  shares=50.0 weighted_acc=0.200
2026-05-18T00:00 pnl=-6.60  shares=50.0 weighted_acc=0.400
```

This suggests time-regime or volatility-regime filters may be useful, but the current sample is too short to choose those filters without a broader replay.

## Worst windows

Largest single-window losses:

```text
2026-05-17T06:05Z side=NO  actual=YES p_up=0.1904 fills=3 shares=15 cost=8.50 pnl=-8.50
2026-05-17T11:25Z side=YES actual=NO  p_up=0.7018 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T12:35Z side=YES actual=NO  p_up=0.8144 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T15:00Z side=NO  actual=YES p_up=0.1627 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T15:10Z side=YES actual=NO  p_up=0.7215 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T16:50Z side=NO  actual=YES p_up=0.0727 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T17:40Z side=NO  actual=YES p_up=0.2670 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T19:50Z side=YES actual=NO  p_up=0.5749 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T20:30Z side=YES actual=NO  p_up=0.8535 fills=2 shares=10 cost=6.50 pnl=-6.50
2026-05-17T20:55Z side=YES actual=NO  p_up=0.6823 fills=2 shares=10 cost=6.50 pnl=-6.50
```

These losses are consistent with a two-leg order plan that lets both orders fill on incorrect windows.

## Root causes

### 1. Passive limit orders are selecting bad outcomes

The clearest issue is that wrong signals fill more reliably than correct ones.

Correct accepted signals filled only `62.18%` of the time. Wrong accepted signals filled `100%` of the time.

This is typical when using passive bids in a fast binary market. If the prediction is correct, the market price often moves away and the bid does not fill. If the prediction is wrong, the market moves toward or through the bid and fills it.

### 2. The order price is not constrained by model-implied fair value

Current order logic uses `min(best_bid, price_cap) + offset`. It does not require:

```text
price <= model_probability - edge_buffer
```

For YES:

```text
fair_value = p_up
```

For NO:

```text
fair_value = 1 - p_up
```

Buying above fair value has negative model EV even if the signal direction passes the threshold.

### 3. Offline objective optimizes direction selection, not tradable execution PnL

The project objective is currently selection score from accepted prediction accuracy and coverage. It does not include:

- market price at decision time
- spread
- probability of fill
- conditional accuracy after fill
- stale/late order fills
- per-window max loss from multiple orders
- settlement mechanics

The offline validation can look strong while live PnL is negative if the execution policy trades a biased subset.

### 4. Online signal accuracy is lower than offline validation

Offline validation accepted accuracy was `76.11%`; online signal accuracy was `66.11%`.

Likely contributors:

- Validation was threshold-tuned and explicitly marked optimistic.
- The live period was short and may be a different regime.
- The model may be overconfident in live high-confidence buckets.
- The post-deploy data path uses live Binance/Gamma/CLOB timing, where small alignment differences matter more.
- The validation target is direction, while the live market is entered at `t0+1m`, after price has already moved.

### 5. Two orders per window double exposure in bad windows

The live policy can place two 5 USDC orders per signal. Many worst windows lost `6.5` to `8.5` USDC because both legs were filled on the wrong side.

The second leg had lower weighted accuracy and lost almost as much as the first leg despite lower average price.

### 6. No fill-aware post-trade feedback loop exists

The system logs signals and submitted orders, but the trading decision does not currently learn from:

- fill versus no-fill outcome
- order age
- market price at fill
- whether filled windows have different accuracy than unfilled windows
- PnL by price bucket
- PnL by confidence bucket
- PnL by time regime

Without this loop, threshold improvements may not translate to live profitability.

## Recommended immediate actions

1. Keep the live timers stopped until execution is fixed.

2. Do not resume live trading with the current two-leg passive order policy.

3. Add an EV gate before submitting any BUY:

```text
YES: require p_up - order_price >= min_edge
NO:  require (1 - p_up) - order_price >= min_edge
```

Start with a conservative `min_edge`, for example `0.05` to `0.10`, and tune it in replay.

4. Reduce live exposure while testing:

```yaml
guards:
  max_orders_per_window: 1
orders:
  first:
    size: 1.0
  second:
    size: 0.0
```

5. Disable the second leg until it proves positive in a fill-aware replay.

6. Add order expiration / cancellation behavior:

- Cancel any unfilled order after a short TTL.
- Never leave bids resting deep into a window when the price path has already invalidated the signal.
- Cancel all outstanding orders before the next 5m window.

7. Add live-trading kill switches:

- max daily loss
- max consecutive losing filled windows
- max filled notional per hour
- max orders per hour
- automatic paper-mode switch when live fill accuracy falls below threshold

## Required engineering fixes

### Fill-aware backtest

Build a replay that joins:

- offline signal probabilities
- Polymarket best bid/ask at decision time
- chosen order prices
- actual fill/no-fill assumptions
- settlement outcome
- realized PnL

The acceptance metric for live deployment should not only be `selection_score`. It should include:

```text
expected_pnl
realized/replayed_pnl
fill_rate
filled_window_accuracy
weighted_fill_accuracy_by_shares
max_drawdown
loss_per_wrong_window
```

### Price-aware thresholding

The decision should depend on both model probability and market price:

```text
trade YES only if p_up >= t_up and p_up - buy_price >= min_edge
trade NO only if p_down >= t_down and p_down - buy_price >= min_edge
```

This should live in unified config, not hardcoded.

### Execution report

Every live/paper run should produce an execution report with:

```text
signal_t0
decision_side
p_up
p_down
market_slug
best_bid
best_ask
order_price
order_size
order_id
fill_status
filled_size
avg_fill_price
actual_side
gross_cost
payout
pnl
fill_latency_seconds
order_age_seconds
```

### Fill reconciliation job

Add a periodic or manual reconciliation script that:

1. Loads submitted order IDs from `live.jsonl`.
2. Queries CLOB trades using authenticated credentials.
3. Matches taker and maker fills by order ID.
4. Computes settlement direction from the shared label rule.
5. Writes a daily PnL report.

This analysis was done manually; it should become a repeatable script before trading resumes.

### Online/offline consistency check

For each live window, store enough data to reproduce the exact model input row:

- feature timestamp
- raw minute/second/aggtrade latest timestamps
- feature values or feature hash
- model artifact hash
- thresholds
- decision context

Then add a replay test:

```text
live summary + stored raw frames -> same p_up/p_down as live
```

## Model and validation fixes

1. Do not judge live readiness only from validation accepted accuracy.

2. Add recent rolling validation slices:

- last 1 day
- last 3 days
- last 7 days
- high-volatility windows
- low-volatility windows
- high-spread Polymarket windows
- late-entry windows after `t0+1m`

3. Recalibrate probabilities on the delayed-alignment validation set.

4. Investigate high-confidence failures. In this live sample, confidence `0.8-1.0` was loss-making:

```text
0.8-0.9 pnl=-35.00 weighted_acc=0.382 avg_px=0.588
0.9-1.0 pnl=-9.50  weighted_acc=0.333 avg_px=0.650
```

5. Add probability bucket reports for online live data, not only offline validation.

6. Consider raising thresholds only after price-aware EV gating exists. Raising thresholds alone will not fix adverse selection if passive fills remain biased.

## Execution policy alternatives

### Conservative maker policy

Only place maker bids when:

```text
order_price <= fair_value - edge_buffer
spread <= max_spread
time_remaining >= min_time_remaining
best_bid/ask is fresh
```

Use one order, small size, short TTL.

### Taker-only micro policy

Use taker orders only when the visible ask is still positive EV:

```text
YES: p_up - best_ask >= edge_buffer
NO:  p_down - best_ask_for_no >= edge_buffer
```

This avoids maker adverse selection but pays spread. The sample showed small positive taker PnL, but there were only 14 taker fills, so this is not enough evidence by itself.

### No-trade unless market disagrees with model

Treat Polymarket price as a strong baseline. Only trade when the model and market differ materially:

```text
abs(model_probability - market_probability) >= min_disagreement
```

This is more appropriate than trading every threshold-passing direction.

### Dynamic sizing

Size should be based on edge and realized live reliability, not raw confidence:

```text
size = base_size * clipped((fair_value - price - edge_buffer) / scale)
```

Set hard caps per window and per hour.

## Suggested implementation order

1. Keep live disabled.
2. Add a reconciliation script and daily PnL report.
3. Add EV gate and max one order per window.
4. Disable second leg.
5. Add fill-aware replay using historical live summaries and CLOB fills.
6. Run paper mode for at least one full day with the new reporting.
7. Resume live only with 1 USDC size and kill switches.
8. Increase size only after live filled-window PnL is positive over a meaningful sample.

## Bottom line

The current loss is not explained by YES/NO imbalance. YES and NO both lost roughly the same amount.

The loss is best explained by a combination of:

1. Online signal accuracy lower than optimistic offline validation.
2. Passive maker orders filling a negatively selected subset.
3. Order prices not constrained by model fair value.
4. Two-leg exposure doubling losses in wrong windows.
5. No live fill-aware objective or kill switch.

The most urgent fix is to make execution price-aware and fill-aware. The model can still have useful directional information, but the current order policy converts that information into negative PnL.
