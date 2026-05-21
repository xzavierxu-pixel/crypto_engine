# Live Profit Window Comparison - 2026-05-15/16 vs 2026-05-20/21

## 1. Windows

Profitable window requested by user:

```text
SGT: 2026-05-15 23:00 to 2026-05-16 09:00
UTC: 2026-05-15 15:00 to 2026-05-16 01:00
```

Recent loss window:

```text
UTC: 2026-05-20 15:23:40 to 2026-05-21 00:23:40
```

Reports:

```text
artifacts/reports/execution_engine/live_profit_analysis_20260515_2300_20260516_0900_sgt.json
artifacts/reports/execution_engine/bot_matched_trades_20260515_2300_20260516_0900_sgt.json
artifacts/reports/execution_engine/live_loss_analysis_last_9h.json
artifacts/reports/execution_engine/bot_matched_trades_last_9h.json
```

The matched-trades reports use CLOB trade history matched back to bot order IDs. They are more reliable than summary-only reports because old summaries often recorded submitted live orders that were not fully filled.

## 2. High-Level Result

### 2026-05-15/16 profitable window

```text
resolved cycles: 102
accepted cycles: 76
coverage: 74.51%
bot orders: 152
matched bot orders: 86
matched fill rows: 107
matched shares: 429.966259
matched avg price: 0.4109
matched accuracy: 64.49%
matched PnL: +103.30
ROI on cost: +58.47%
```

### 2026-05-20/21 loss window

```text
resolved cycles: 106
accepted cycles: 71
coverage: 66.98%
bot orders: 72
matched bot orders: 68
matched fill rows: 73
matched shares: 339.985302
matched avg price: 0.6299
matched accuracy: 60.27%
matched PnL: -9.15
ROI on cost: -4.28%
```

## 3. Main Finding

The newer setup did not lose only because the model was worse. The biggest measurable difference is execution price.

```text
old matched avg price: 0.4109
new matched avg price: 0.6299
price increase: +0.2190
```

The old window needed only about 41.1% accuracy to break even. The new window needed about 63.0% accuracy to break even. Actual matched accuracy dropped from 64.49% to 60.27%, but the larger issue is that the new execution price moved the break-even threshold above realized accuracy.

In other words:

```text
old: 64.49% accuracy vs 41.09% break-even -> strongly positive
new: 60.27% accuracy vs 62.99% break-even -> negative
```

## 4. Execution Differences

### Old window had two legs

```text
first leg:
  matched fills: 60
  accuracy: 71.67%
  avg price: 0.4918
  PnL: +54.60
  ROI: +43.54%

second leg:
  matched fills: 47
  accuracy: 55.32%
  avg price: 0.2931
  PnL: +48.70
  ROI: +94.94%
```

The second leg was not accurate, but it was cheap. Its break-even was only about 29.3%, so 55.3% realized accuracy produced very high EV.

### New window had only first leg

```text
first leg:
  matched fills: 73
  accuracy: 60.27%
  avg price: 0.6299
  PnL: -9.15
  ROI: -4.28%
```

The new setup removed the cheap optionality from the second leg and paid much higher prices for the first leg.

### Maker/taker behavior changed

Old window:

```text
maker fills: 103
maker avg price: 0.4096
maker PnL: +97.05
taker fills: 4
taker avg price: 0.4375
taker PnL: +6.25
```

New window:

```text
maker fills: 28
maker avg price: 0.6622
maker PnL: +8.85
taker fills: 45
taker avg price: 0.6133
taker PnL: -18.00
```

This suggests the newer execution style is paying up and crossing or matching aggressively more often. That creates adverse selection: more orders fill, but many fill at prices that require a higher hit rate than the model delivered.

## 5. Model/Market Regime Comparison

Both windows share the same structural failure mode: if the first-minute direction continues, the strategy wins; if the remaining four minutes reverse, it loses.

Old profitable window, matched fills:

```text
first-minute continuation:
  fills: 67
  accuracy: 95.52%
  avg price: 0.421
  PnL: +146.19

first-minute reversal:
  fills: 40
  accuracy: 12.50%
  avg price: 0.393
  PnL: -42.89
```

New loss window, matched fills by BTC 5m agreement:

```text
model agrees with BTC 5m direction:
  fills: 45
  accuracy: 93.33%
  avg price: 0.650
  PnL: +64.95

model disagrees with BTC 5m direction:
  fills: 27
  accuracy: 7.41%
  avg price: 0.601
  PnL: -71.10
```

The reversal problem existed in both versions. The old version survived it because continuation wins were bought cheaply and because the second leg had very low entry prices. The new version pays much more, so the same reversal failure mode overwhelms the continuation gains.

## 6. Threshold / Coverage Difference

Old live summaries used:

```text
t_up: 0.535
t_down: 0.405
coverage in requested window: 74.51%
```

New live summaries used:

```text
t_up: 0.62
t_down: 0.415
coverage in recent loss window: 66.98%
```

The newer artifact was optimized offline for validation selection_score with coverage near 70%, but the realized live coverage in the loss window was below 70%. This does not prove the artifact is bad, but it shows the live regime did not match the offline acceptance distribution.

## 7. Why The Older Version Made Money

The old window was profitable because several factors aligned:

1. Entry prices were much lower.
2. Two-leg execution captured cheap optionality, especially the 0.30 leg.
3. Most matched fills were maker-style fills, not aggressive taker fills.
4. First-minute continuation periods generated very large gains.
5. The reversal regime still lost money, but not enough to offset cheap continuation wins.

## 8. Why The Newer Version Underperformed

The newer setup underperformed because:

1. Average entry price increased from 0.4109 to 0.6299.
2. The required break-even accuracy rose to about 63.0%, while realized accuracy was only 60.27%.
3. The second leg was disabled, removing a large source of positive optionality.
4. Taker-like matched fills were much more common and were negative EV in the recent window.
5. The model still over-follows first-minute direction and still loses badly on post-first-minute reversals.
6. Offline validation improvement did not directly optimize execution price, maker fill quality, or realized EV after price.

## 9. Recommended Optimization Direction

### 9.1 Separate model quality from execution quality

Future reports must include both:

```text
model-side accuracy and selection_score
realized fill EV after price
maker/taker split
matched avg price
fill rate by leg
PnL by first-minute continuation/reversal
```

### 9.2 Restore low-risk optionality, but behind config

Do not blindly turn the old second leg back on. Instead, test config-controlled variants:

```yaml
orders:
  first:
    enabled: true
    price_mode: fixed_or_capped
    max_price: 0.50
  second:
    enabled: false
    price_mode: fixed
    price: 0.30
```

Candidate experiments:

```text
A: current first leg only
B: first leg capped at 0.50
C: second leg 0.30 enabled, first leg capped
D: maker-only mode with no aggressive crossing
E: current model signal plus min_edge filter
```

### 9.3 Add realized-EV filter

Direction confidence alone is not enough. Add a disabled-by-default filter:

```text
edge = predicted_win_probability - order_price
trade only if edge >= min_edge
```

This should be evaluated on actual matched trades, not only submitted orders.

### 9.4 Continue reversal-risk work

The reversal risk PRD remains valid. However, reversal filtering alone will not fix paying too much. Reversal risk should be combined with price/edge controls.

## 10. Current Conclusion

The newer model may or may not be directionally worse, but the evidence here says the larger regression is execution EV:

```text
old: lower price, two legs, mostly maker, +103.30 actual matched PnL
new: higher price, one leg, many taker fills, -9.15 actual matched PnL
```

The next improvement should optimize realized EV after fill price, not just offline `selection_score`.
