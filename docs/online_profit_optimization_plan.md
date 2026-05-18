# Online profit optimization plan

Created: 2026-05-18

Goal: improve live Polymarket execution profitability for the BTC 5m direction strategy.

This plan combines findings from:

- `docs/aws_poly_loss_analysis_2026-05-18.md`
- `docs/aws_poly_profit_analysis_2026-05-15_2300_sgt.md`

## Executive Summary

The next optimization target should be live realized PnL, not only offline direction accuracy.

The two live periods show the core rule:

```text
profitable period:
  weighted_fill_accuracy = 0.5909
  average_fill_price     = 0.4107
  pnl                    = +79.30 USDC

losing period:
  weighted_fill_accuracy = 0.4989
  average_fill_price     = 0.5372
  pnl                    = -46.68 USDC
```

For binary payout tokens, the rough break-even condition is:

```text
weighted_fill_accuracy > average_fill_price
```

The strategy made money when it bought cheap enough claims with enough realized correctness. It lost money when it bought more expensive claims while filled-window accuracy deteriorated.

The current execution layer does not enforce this relationship. It places orders from best bid / price cap without checking model-implied fair value:

```text
current:
  first price  = min(best_bid, price_cap)
  second price = min(best_bid, price_cap) - 0.10

missing:
  YES edge = p_up - order_price
  NO edge  = (1 - p_up) - order_price
```

The immediate priority is therefore:

1. Make execution price-aware.
2. Make reporting fill-aware.
3. Run a short paper-only profit test with exchange-compatible minimum order size.
4. Only then resume model/threshold optimization.

Current user constraint:

```text
paper-only test duration: up to 1 hour
early stop condition: paper PnL > 25 USDT/USDC-equivalent
live trading: disabled
minimum order size: 5 contracts
single-order notional cap: 4 USDT/USDC-equivalent
order TTL / cancellation: out of scope for this iteration
kill switch: out of scope for this iteration
```

## Objective

Primary paper objective for this iteration:

```text
maximize paper_realized_or_replayed_pnl during the 1-hour paper window
```

Subject to:

```text
paper mode only
test stops after 1 hour or when paper PnL > 25 USDT/USDC-equivalent
weighted_fill_accuracy > average_fill_price over the paper test window
no live orders are submitted
```

Offline model objective remains useful, but it is not sufficient for trading decisions. The paper acceptance report must include:

```text
realized_pnl
expected_pnl
average_fill_price
filled_window_signal_accuracy
weighted_fill_accuracy_by_shares
fill_rate
max_drawdown
loss_per_wrong_window
```

## What We Learned

### Profitable Window

Window:

```text
2026-05-15 23:00 to 2026-05-16 09:00 SGT
```

Important details:

```text
feature timing: T+1
model: LightGBM
calibration: none
t_up: 0.535
t_down: 0.405
signal_accuracy: 0.7632
filled_window_signal_accuracy: 0.6538
weighted_fill_accuracy: 0.5909
average_fill_price: 0.4107
realized_pnl: +79.30 USDC
```

Why it worked:

- Online signal accuracy matched offline validation.
- Average fill price was low.
- Second leg bought very cheap claims.
- Both YES and NO sides were profitable.
- Every observed price bucket was profitable.

### Losing Window

Window:

```text
after 2026-05-17 deploy through 2026-05-18 00:35 UTC
```

Important details:

```text
feature timing: T+1
model: LightGBM
calibration: none
t_up: 0.535
t_down: 0.405
signal_accuracy: 0.6611
filled_window_signal_accuracy: 0.5481
weighted_fill_accuracy: 0.4989
average_fill_price: 0.5372
realized_pnl: -46.68 USDC
```

Why it failed:

- Online signal accuracy fell.
- Filled subset was worse than all accepted signals.
- Wrong signals filled more reliably than correct signals.
- Average fill price was too high for the realized accuracy.
- Two-leg exposure doubled losses in wrong windows.
- There was no EV gate.

## Non-Negotiable Live Trading Rule

No order should be submitted unless it passes a model-price edge test.

For a YES buy:

```text
fair_value = p_up
edge = fair_value - order_price
require edge >= min_edge
```

For a NO buy:

```text
fair_value = 1 - p_up
edge = fair_value - order_price
require edge >= min_edge
```

Initial conservative config:

```yaml
execution_edge:
  enabled: true
  min_edge: 0.08
  max_price: 0.55
  max_spread: 0.08
```

These numbers should be tuned by replay. For this iteration, they should be tested in paper mode only.

## Phase 0: Keep Live Trading Disabled

Status: required before any code or config changes.

Actions:

1. Keep `execution-engine.timer` disabled.
2. Keep `execution-engine-prewarm.timer` disabled unless paper-only.
3. Do not resume live mode during this iteration.

Reason:

The current live order policy can be profitable in favorable conditions but has no guard against the losing condition:

```text
weighted_fill_accuracy <= average_fill_price
```

## Phase 1: Build Repeatable Live Reconciliation

Purpose: make every live/paper run measurable.

Deliverable:

```text
execution_engine/scripts/reconcile_polymarket_fills.py
```

Inputs:

```text
artifacts/logs/execution_engine/live.jsonl
artifacts/logs/execution_engine/summaries/*.json
execution_engine/secrets.env
Binance 1m klines
```

Outputs:

```text
artifacts/reports/execution_engine/reconciliation/<date>.json
artifacts/reports/execution_engine/reconciliation/<date>.md
```

Required metrics:

```text
window_start
window_end
signal_count
accepted_signal_count
submitted_order_count
filled_order_count
filled_window_count
signal_accuracy
filled_window_signal_accuracy
unfilled_window_signal_accuracy
fill_accuracy_by_event
weighted_fill_accuracy_by_shares
average_fill_price
gross_cost
gross_payout
realized_pnl
pnl_by_side
pnl_by_leg
pnl_by_price_bucket
pnl_by_confidence_bucket
pnl_by_hour
open_order_count
unmatched_trade_count
```

Acceptance:

- The script must reproduce:
  - profitable window: approximately `+79.30 USDC`
  - losing window: approximately `-46.68 USDC`
- It must not print secrets.
- It must clearly separate:
  - engine-matched fills
  - account-level extra trades
  - unresolved trades

## Phase 2: Add EV Gate to Order Planning

Purpose: prevent buying claims above model-implied fair value.

Config proposal:

```yaml
execution_edge:
  enabled: true
  min_edge: 0.08
  max_buy_price: 0.55
  max_spread: 0.08
  apply_to_first_leg: true
  apply_to_second_leg: true
```

Order validation:

```text
YES order:
  fair_value = p_up
  edge = p_up - order_price

NO order:
  fair_value = 1 - p_up
  edge = (1 - p_up) - order_price
```

Skip reason examples:

```text
edge_below_minimum
price_above_max_buy_price
spread_too_wide
missing_quote
```

Report each skipped order with:

```text
side
p_up
p_down
order_price
fair_value
edge
min_edge
best_bid
best_ask
spread
reason
```

Acceptance:

- Unit tests for YES/NO edge calculations.
- Unit tests for skipped orders.
- Paper-mode report shows expected PnL improvement versus old policy.

## Phase 3: Paper-Only Exposure Settings

Polymarket minimum order size is 5 contracts, so the paper test must use at least 5 contracts per submitted order.
The user-requested single-order cap is enforced as notional cost:

```text
order_notional = order_price * order_size <= 4.0
```

With the 5-contract minimum, this means any 5-contract order above `0.80` must be skipped. The current paper config is stricter: `max_buy_price=0.35`, and sizes to a maximum `4.0` USDC notional per simulated order.

Paper-only config:

```yaml
orders:
  enabled: true
  mode: paper
  first:
    price_cap: 0.35
    offset: 0.0
    size: 8.0
  second:
    price_cap: 0.30
    offset: 0.0
    size: 0.0
guards:
  max_orders_per_window: 1
```

Rationale:

- The profitable period's second leg worked because it was cheap.
- The losing period's second leg was negative.
- Disable second leg until replay proves it is positive under an EV gate.

Do not open live trading after this paper run. The only required output is a report to the user.

## Phase 4: Paper Test Stop Rule

Purpose: run a short controlled paper test and stop as soon as the target is met.

Test rule:

```text
run paper mode for at most 1 hour
if paper realized/replayed PnL > 25 USDT/USDC-equivalent, stop immediately
after stopping, report results to the user
do not switch to live
```

Report required:

```text
start_time
end_time
elapsed_minutes
paper_signal_count
paper_order_count
paper_filled_or_simulated_count
paper_realized_or_replayed_pnl
average_fill_price
weighted_fill_accuracy
pnl_by_side
pnl_by_leg
stop_reason
```

Acceptance:

- Paper test runs no longer than 1 hour.
- Test stops early if PnL exceeds 15.
- No live orders are submitted.
- User receives a concise result report.

## Phase 5: Deferred Items

The following are intentionally out of scope for this iteration:

- Order TTL.
- Automatic cancellation.
- Kill switch.
- Live promotion.

## Phase 6: Fill-Aware Replay

Purpose: optimize execution policy before any future longer paper run or live trading discussion.

Replay data:

- Past execution summaries.
- CLOB order books if available.
- Actual authenticated fills.
- Binance settlement labels.

Replay variants:

```text
baseline current two-leg policy
one-order policy
second-leg disabled
min_edge = 0.03 / 0.05 / 0.08 / 0.10 / 0.12
max_buy_price = 0.40 / 0.45 / 0.50 / 0.55 / 0.60
confidence buckets
time-of-day buckets
spread filters
```

Ranking objective:

```text
1. realized/replayed PnL
2. max drawdown
3. weighted_fill_accuracy - average_fill_price
4. filled_window count
5. simplicity
```

Do not choose a policy that only improves offline selection_score but worsens live PnL.

## Phase 7: Model and Threshold Work

Only after the paper-only execution plan produces interpretable reports.

Work items:

1. Online probability bucket calibration.
2. Recent rolling validation:
   - last 1 day
   - last 3 days
   - last 7 days
   - high-volatility regimes
   - low-volatility regimes
3. Compare T+1 and T+2 with live-style execution replay.
4. Re-tune thresholds against filled-window PnL, not only accepted accuracy.
5. Add market price features only if they are available online and pass leakage checks.

Important:

T+2 had better offline validation:

```text
T+2 validation accepted_sample_accuracy = 0.8086
T+2 selected thresholds = t_up 0.595 / t_down 0.37
```

But T+2 must not be assumed better live until it passes fill-aware replay. Later entry may reduce tradable edge if market price has already moved.

## Proposed Implementation Order

### Step 1

Create reconciliation script.

Expected result:

```text
Can reproduce profitable and losing windows from logs + CLOB fills.
```

### Step 2

Add execution EV gate.

Expected result:

```text
Orders with negative model-price edge are skipped.
```

### Step 3

Disable second leg and use the exchange minimum paper order size.

Expected result:

```text
Only one 5-contract paper order can be generated per accepted signal.
```

### Step 4

Run the paper-only 1-hour test.

Expected result:

```text
Stop after 1 hour or when paper PnL > 25 USDT/USDC-equivalent.
```

### Step 5

Report the result to the user.

Expected result:

```text
The user receives PnL, signal quality, average price, and stop reason.
```

### Step 6

Continue only if the user explicitly asks for a longer paper run or live deployment.

Acceptance:

```text
no live mode is enabled by default
```

## Concrete Initial Config Recommendation

For the first paper-only candidate:

```yaml
execution_edge:
  enabled: true
  min_edge: 0.08
  max_buy_price: 0.35
  max_spread: 0.08
  max_order_notional: 4.0
  apply_to_first_leg: true
  apply_to_second_leg: true

orders:
  enabled: true
  mode: paper
  first:
    price_cap: 0.35
    offset: 0.0
    size: 5.0
  second:
    price_cap: 0.30
    offset: 0.0
    size: 0.0

guards:
  max_orders_per_window: 1
  enforce_idempotency: true

paper_test:
  max_duration_minutes: 60
  stop_when_pnl_gt: 25.0
  report_to_user: true
```

Do not switch to live as part of this plan.

## Success Criteria

The optimization is working only if the paper report shows:

```text
realized_pnl > 0
weighted_fill_accuracy > average_fill_price
filled_window_signal_accuracy stable or improving
average_fill_price controlled
loss per wrong window capped
```

Paper test success criteria:

```text
duration <= 60 minutes
paper_pnl > 25 USDT/USDC-equivalent, or report final PnL at 60 minutes
no live orders submitted
```

## What Not To Do

Do not:

- Resume the old two-leg 5 USDC live strategy.
- Optimize only offline accepted accuracy.
- Increase size from raw model confidence.
- Assume T+2 is better live only because offline validation is better.
- Ignore average fill price.
- Treat unfilled correct signals as realized edge.
- Open live trading from this paper-only experiment.

## Bottom Line

The path to higher online收益 is not just a better classifier. The classifier already showed useful signal in the profitable window. The missing layer is execution discipline.

The next version should make every order answer this question before submission:

```text
Is the model's estimated probability sufficiently higher than the actual price we are paying?
```

If the answer is no, the engine should skip.

The target behavior is:

```text
fewer trades
lower average fill price
higher edge per fill
smaller loss per wrong window
clear stop after 1 hour or PnL > 25
```

That is the most direct route to improving online收益.
