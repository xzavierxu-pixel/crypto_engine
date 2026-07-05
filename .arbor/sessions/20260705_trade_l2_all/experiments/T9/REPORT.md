# Experiment T9 - data-driven segment abstention

## Hypothesis

Chronologically learned negative-PnL cells can remove loss exposure from the frozen 42.43 anchor without using B_test to define buckets.

## What Changed

Added a frozen abstention table over `p_side`, anchor bid, or their joint cells. Quantile edges, confidence bounds, and rejection decisions use only preceding rolling observations; the final table is frozen from w1-w6 before one B_test read.

## What Did Not Change

- Direction artifact: accepted selective direction output
- Calibration method: anchor `raw_tree_blend`
- Fill semantics: correct fills iff winner low reaches bid; wrong submitted orders forced filled
- Accepted universe: 5,228 accepted rows from 7,468 samples

## Data Windows

- Train/calibration: expected-return train data ending before 2026-04-11
- B_dev: expanding chronological w1-w6
- B_test: 2026-04-11 00:21:00+00:00 through 2026-05-10 23:51:00+00:00

## Leakage Check

- Forbidden columns intersection: `[]`
- Feature cutoff check: inherited frozen anchor; abstention uses only decision-time `p_side` and bid
- Fit-on-validation check: B_test not used for quantiles, thresholds, cell selection, or candidate selection

## Metrics

| Split | sum_pnl | delta_vs_anchor | order_count | order_coverage | win_pnl | loss_pnl | fill_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_dev w1-w6 | 330.82 | n/a | n/a | n/a | n/a | n/a | n/a |
| B_test | 42.43 | -0.00 | 1980 | 0.3787 | 358.87 | -316.44 | 0.1447 |

## B_test Required Result

- selected variant/aggressiveness: `bid_only/conservative`
- btest_sum_pnl: 42.43
- btest_delta_vs_42p43: -0.00
- saved_loss: -0.00
- killed_wins: 0.00
- order_count change: 0
- order_coverage change: 0.0000
- wrong submitted rate: 0.2561
- avg loser bid: 0.6241
- submitted fill calibration gap: 0.1447

## Diagnosis

The abstained B_test cells realized 0.00; positive means abstention killed net wins, negative means it saved net losses. Full segment distributions and rolling stability are saved alongside this report.

## Decision

- Continue / modify / stop: modify or stop
- Reason: sum_pnl did not improve over the frozen anchor
