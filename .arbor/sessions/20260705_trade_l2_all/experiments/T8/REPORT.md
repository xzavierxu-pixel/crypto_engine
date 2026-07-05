# Experiment T8 - two-minute trade/L2 auxiliary

## Hypothesis

A two-minute profitable-fill score can identify forced-wrong exposure in the first-minute anchor, but may only veto orders and never replace its side or bid.

## What Did Not Change

- First-minute anchor: raw_tree_blend, Gc floor 0.85, min EV 0.02
- Fill semantics: correct fills iff winner_low <= bid; every wrong submitted order is forced filled
- Accepted universe and anchor side/bid construction: unchanged

## Frozen Selection

- Candidate: `anchor_no_auxiliary`
- w1-w4 sum: 218.50; w5-w6 holdout: 112.32
- B_test used in selection: no; frozen B_test evaluation count: 1

## B_test

| policy | sum_pnl | delta vs 42.43 | orders | win_pnl | loss_pnl |
|---|---:|---:|---:|---:|---:|
| anchor | 42.43 | -0.00 | 1980 | 358.87 | -316.44 |
| auxiliary gate | 42.43 | -0.00 | 1980 | 358.87 | -316.44 |

- Two-minute profitable-fill Brier: 0.193578
- Wrong submitted rate: 0.256061
- Auxiliary veto count: 0

## Leakage and Deployability

All model fitting is chronological and trade aggregation enforces `trade_time <= two-minute decision_time`. The score is not available at the first-minute decision, so this is a research-only delayed-decision auxiliary and cannot be inserted into current live first-minute execution without an explicit timing redesign and user approval.

## Decision

The auxiliary changed B_test PnL by 0.00 versus the reconstructed anchor. No automatic promotion.
