# aws-poly profitable window analysis

Analysis date: 2026-05-18

Requested window:

- Singapore time: 2026-05-15 23:00 to 2026-05-16 09:00
- UTC: 2026-05-15 15:00 to 2026-05-16 01:00

Server: `aws-poly`

Secrets were used only to authenticate CLOB trade queries on the server. No secret values were printed or copied.

## Important timestamp notes

The first available execution summary inside the requested window is:

```text
2026-05-15T15:15:00+00:00
2026-05-15 23:15 Singapore time
```

So the analysis covers the available engine summaries from `23:15 SGT` through `09:00 SGT`.

Relevant repo commits around this period:

```text
2026-05-15 23:29 +0800  2be13b4  Implement T+1 delayed feature alignment
2026-05-16 09:42 +0800  127776a  profitable version1
2026-05-16 10:04 +0800  673a4d9  Advance delayed feature alignment to T+2
```

The requested window ends at `2026-05-16 09:00 SGT`, before the T+2 commit. The summaries in this window show:

```text
feature_offset_minutes = 1
row_policy = delayed_1m_synthetic_decision_row
```

So this profitable window was a T+1 feature-timing window, not T+2.

## PnL reconciliation result

Using the same conservative method as the loss analysis, I matched CLOB fills back to execution-engine submitted order IDs from the summaries.

Result:

```text
summary_count:                 102
first_t0:                      2026-05-15T15:15:00+00:00
last_t0:                       2026-05-16T01:00:00+00:00
signals_should_trade:          76
signal_accuracy:               0.7631578947
filled_window_signal_count:    52
filled_window_signal_accuracy: 0.6538461538
unfilled_window_signal_count:  24
unfilled_window_signal_accuracy: 1.0000000000
engine_orders_submitted:       152
orders_with_any_fill:          88
fill_events:                   109
filled_windows:                52
filled_shares:                 439.966259
gross_cost_buy:                180.686866
gross_payout_buy:              259.989540
realized_pnl:                  +79.302674 USDC
fill_accuracy_by_event:        0.5871559633
weighted_fill_accuracy:        0.5909306332
average_fill_price:            0.4106834611
```

This exact matched-order method gives `+79.30 USDC`, not `>100 USDC`.

Possible reasons your observed number was higher:

- Polymarket UI may have included mark-to-market value, not only resolved payout.
- The UI may have included orders whose `t0` was just outside the requested window.
- Some account-level trades are not cleanly attributable to execution-engine summary order IDs.
- The requested human time range may have been interpreted with slightly different boundaries.
- The account could have had position changes from orders submitted before the first available summary.

For cause analysis, the matched-order number is the safest because it ties each fill to a model signal and settlement label.

## Why this window made money

There were three main reasons.

### 1. The model was actually in line with offline validation

In the later loss analysis, online signal accuracy after deploy was only about `66.11%`.

In this profitable window:

```text
online signal accuracy = 76 /? accepted signals = 0.7631578947
```

That is almost identical to the deployed artifact's offline validation accepted-sample accuracy:

```text
offline validation accepted_sample_accuracy = 0.7611225188
```

So during this window, the model's directional edge generalized well.

### 2. The average fill price was low

The average fill price was:

```text
average_fill_price = 0.4106834611
```

For a binary payout token, the rough break-even win rate is the average buy price:

```text
break_even_accuracy ~= average_fill_price
```

So this window only needed about `41.07%` share-weighted correctness to break even.

Actual share-weighted correctness was:

```text
weighted_fill_accuracy = 0.5909306332
```

That is a very large margin:

```text
0.5909 - 0.4107 = +0.1802
```

In plain terms: the strategy was buying average 41-cent claims that paid out 1 USDC about 59% of the time by shares.

That is why the PnL was positive.

### 3. The filled subset was still good enough

There was still adverse selection:

```text
overall signal accuracy:        0.7632
filled-window signal accuracy:  0.6538
unfilled-window signal accuracy:1.0000
```

Correct unfilled signals still existed. However, unlike the later loss period, the filled subset remained profitable because:

- filled-window accuracy was still above 65%
- average fill price was only about 41%
- both YES and NO sides contributed positive PnL
- the losing windows generally lost about 4 USDC, while many winning windows made 6 USDC or more

The later losing period had the opposite setup:

```text
later loss period avg_fill_price:        0.5372
later loss period weighted_fill_accuracy:0.4989
```

That combination is structurally negative.

## PnL breakdown

### By decision side

```text
NO:
  fills:       59
  shares:      229.974836
  cost:        94.139439
  pnl:         +35.852961
  weighted_acc:0.5652461907
  avg_price:   0.4093466948

YES:
  fills:       50
  shares:      209.991423
  cost:        86.547427
  pnl:         +43.449713
  weighted_acc:0.6190592841
  avg_price:   0.4121474380
```

This was not a one-sided win. Both YES and NO made money.

YES was slightly better, but NO also had a strong edge because the average price was low.

### By order leg

```text
first leg:
  fills:       61
  shares:      259.984800
  cost:        127.892400
  pnl:         +42.100000
  weighted_acc:0.6538551485
  avg_price:   0.4919226047

second leg:
  fills:       48
  shares:      179.981459
  cost:        52.794466
  pnl:         +37.202674
  weighted_acc:0.5000356176
  avg_price:   0.2933328038
```

The second leg only won about 50% by shares, but it bought at an average price of `0.2933`, so it was still highly profitable.

This is an important contrast with the later loss period. A second leg can be profitable when it is genuinely cheap. It becomes dangerous when it is filled at mid/high prices or when adverse selection pushes its conditional accuracy below its price.

### By maker/taker role

```text
MAKER:
  fills:       105
  shares:      419.966259
  cost:        171.936866
  pnl:         +73.052674
  weighted_acc:0.5833552928
  avg_price:   0.4094063804

TAKER:
  fills:       4
  shares:      20.000000
  cost:        8.750000
  pnl:         +6.250000
  weighted_acc:0.7500000000
  avg_price:   0.4375000000
```

Maker fills were profitable here. That does not mean passive maker fills are always safe. It means that in this window, passive fills were cheap enough and accurate enough.

In the later loss period, maker fills were the main loss source.

## Price bucket analysis

```text
price 0.2-0.3:
  fills=11 shares=29.998610 pnl=+7.200389 weighted_acc=0.5000 avg_price=0.2600

price 0.3-0.4:
  fills=37 shares=149.982849 pnl=+30.002285 weighted_acc=0.5000 avg_price=0.3000

price 0.4-0.5:
  fills=9 shares=45.000000 pnl=+9.600000 weighted_acc=0.6667 avg_price=0.4533

price 0.5-0.6:
  fills=52 shares=214.984800 pnl=+32.500000 weighted_acc=0.6512 avg_price=0.5000
```

Every price bucket was profitable.

The key is not that every bucket had extremely high accuracy. The cheap buckets only needed low win rates:

- At `0.30`, 50% correctness is very profitable.
- At `0.50`, the strategy needed above 50%, and it achieved about 65%.

There were no filled buckets above `0.6` in this matched data. That is a major difference from the later loss period where many fills happened around `0.6-0.7`.

## Confidence bucket analysis

```text
confidence 0.5-0.6:
  fills=16 shares=59.986683 pnl=-4.094485 weighted_acc=0.3334 avg_price=0.4017

confidence 0.6-0.7:
  fills=54 shares=214.983863 pnl=+57.545873 weighted_acc=0.6744 avg_price=0.4067

confidence 0.7-0.8:
  fills=29 shares=119.995713 pnl=+20.351286 weighted_acc=0.5834 avg_price=0.4138

confidence 0.8-0.9:
  fills=9 shares=40.000000 pnl=+3.000000 weighted_acc=0.5000 avg_price=0.4250

confidence 0.9-1.0:
  fills=1 shares=5.000000 pnl=+2.500000 weighted_acc=1.0000 avg_price=0.5000
```

Most profit came from model confidence `0.6-0.8`, not from extreme confidence.

This matters because the later loss period had very poor results in high-confidence buckets. Raw confidence should not be trusted as a sizing signal until live calibration is much better understood.

## Hourly SGT breakdown

```text
2026-05-15 23:00 SGT: pnl=-5.2038  weighted_acc=0.1998 avg_price=0.4080
2026-05-16 00:00 SGT: pnl=-5.5490  weighted_acc=0.2500 avg_price=0.3888
2026-05-16 01:00 SGT: pnl=+22.0000 weighted_acc=1.0000 avg_price=0.4500
2026-05-16 02:00 SGT: pnl=+6.0000  weighted_acc=1.0000 avg_price=0.4000
2026-05-16 03:00 SGT: pnl=+15.5000 weighted_acc=0.7778 avg_price=0.4333
2026-05-16 04:00 SGT: pnl=+3.9038  weighted_acc=0.4546 avg_price=0.3836
2026-05-16 05:00 SGT: pnl=+6.5013  weighted_acc=0.5556 avg_price=0.4111
2026-05-16 06:00 SGT: pnl=+16.7500 weighted_acc=0.6250 avg_price=0.4156
2026-05-16 07:00 SGT: pnl=+9.2004  weighted_acc=0.6000 avg_price=0.4160
2026-05-16 08:00 SGT: pnl=+14.2000 weighted_acc=0.7500 avg_price=0.3950
2026-05-16 09:00 SGT: pnl=-4.0000  weighted_acc=0.0000 avg_price=0.4000
```

The first two hours were losing. The profit came mostly from sustained positive performance from `01:00` through `08:00 SGT`.

## Best windows

Largest positive windows:

```text
2026-05-16 04:15 SGT side=NO  actual=NO  p_up=0.3769 shares=10 pnl=+6.60 avg_price=0.34
2026-05-16 06:05 SGT side=YES actual=YES p_up=0.5470 shares=10 pnl=+6.50 avg_price=0.35
2026-05-16 00:10 SGT side=YES actual=YES p_up=0.6307 shares=9.997 pnl=+6.448 avg_price=0.355
2026-05-16 08:20 SGT side=YES actual=YES p_up=0.7243 shares=10 pnl=+6.20 avg_price=0.38
2026-05-16 01:15 SGT side=YES actual=YES p_up=0.6041 shares=10 pnl=+6.00 avg_price=0.40
2026-05-16 01:45 SGT side=YES actual=YES p_up=0.6094 shares=10 pnl=+6.00 avg_price=0.40
2026-05-16 02:40 SGT side=YES actual=YES p_up=0.7432 shares=10 pnl=+6.00 avg_price=0.40
2026-05-16 03:50 SGT side=NO  actual=NO  p_up=0.3725 shares=10 pnl=+6.00 avg_price=0.40
2026-05-16 03:55 SGT side=YES actual=YES p_up=0.6270 shares=10 pnl=+6.00 avg_price=0.40
2026-05-16 04:45 SGT side=NO  actual=NO  p_up=0.3534 shares=10 pnl=+6.00 avg_price=0.40
```

The best windows are mostly full-size wins bought around `0.34-0.40`.

## Worst windows

Largest negative windows:

```text
2026-05-15 23:20 SGT side=YES actual=NO  p_up=0.6678 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 00:05 SGT side=YES actual=NO  p_up=0.7358 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 03:35 SGT side=NO  actual=YES p_up=0.1640 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 04:30 SGT side=NO  actual=YES p_up=0.3269 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 05:50 SGT side=YES actual=NO  p_up=0.7196 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 06:20 SGT side=NO  actual=YES p_up=0.2164 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 06:35 SGT side=NO  actual=YES p_up=0.2647 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 06:45 SGT side=YES actual=NO  p_up=0.6102 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 07:55 SGT side=YES actual=NO  p_up=0.5760 shares=10 pnl=-4.00 avg_price=0.40
2026-05-16 08:00 SGT side=NO  actual=YES p_up=0.1232 shares=10 pnl=-4.00 avg_price=0.40
```

Losses were capped around `-4 USDC` for many bad full-size windows because the average entry price was around `0.40`. In the later loss period, many losing windows cost `-6.5 USDC` or worse because the average price was much higher.

## Comparison with later losing period

| Metric | Profitable window | Later loss window |
|---|---:|---:|
| Signal accuracy | `0.7632` | `0.6611` |
| Filled-window signal accuracy | `0.6538` | `0.5481` |
| Weighted fill accuracy | `0.5909` | `0.4989` |
| Average fill price | `0.4107` | `0.5372` |
| PnL | `+79.3027` | `-46.6789` |

The most important difference is:

```text
profitable window: weighted_fill_accuracy 0.5909 > avg_fill_price 0.4107
losing window:     weighted_fill_accuracy 0.4989 < avg_fill_price 0.5372
```

That is the cleanest explanation of the PnL difference.

## Interpretation

This profitable period was not just luck from one side of the market:

- YES made money.
- NO made money.
- First leg made money.
- Second leg made money.
- Maker fills made money.
- Every observed price bucket made money.

The setup was favorable because the model was accurate and entries were cheap.

However, the unfilled-window accuracy was still `100%`, meaning adverse selection was still present. The system still missed correct predictions. It just overcame that adverse selection because the fills it did get were cheap and accurate enough.

## What this suggests for future trading

The live strategy should not simply try to reproduce this exact time window. Instead, it should enforce the conditions that made it profitable:

1. Buy only when model fair value is materially above price:

```text
YES edge = p_up - order_price
NO edge  = (1 - p_up) - order_price
```

2. Keep average entry price low unless live filled accuracy is demonstrably high.

3. Add a hard price/EV gate. The profitable window had no matched fills above `0.6`; many profitable entries were around `0.30-0.50`.

4. Do not size from raw model confidence alone. The best PnL came from confidence `0.6-0.8`, while later high-confidence periods performed badly.

5. Keep separate reports for:

- signal accuracy
- filled-window signal accuracy
- fill accuracy by shares
- average fill price
- realized PnL

6. Treat any period where:

```text
weighted_fill_accuracy <= average_fill_price
```

as a stop-trading condition.

## Bottom line

This window was profitable because the model accuracy matched offline expectations and the average fill price was low. The strategy bought claims at about `0.41` that settled correctly about `0.59` of the time by shares.

The later loss period failed for the opposite reason: the model was weaker, the filled subset was much worse, and the average entry price was too high.
