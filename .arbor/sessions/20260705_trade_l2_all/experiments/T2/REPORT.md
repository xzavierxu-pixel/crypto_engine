# Experiment T2 - joint profitable-fill baseline

## Hypothesis

Directly modeling `P(correct and winner_low <= bid | X,bid)` can remove the independence error in `q * Gc` and improve submitted-action PnL.

## What Changed

A bid-monotone joint classifier and 5-cent action grid were evaluated in an isolated experiment using pre-decision trade-path, available L2/liquidity, and legal BTC features.

## What Did Not Change

- Direction artifact: frozen selected side and threshold-accepted universe from each source frame
- Calibration method: fixed blended q; no submitted-action calibration (reserved for T3)
- Fill semantics: correct fills iff `winner_low <= bid`; wrong submitted orders are forced filled
- Accepted universe: 5,228 B_test rows from 7,468 total rows

## Data Windows

- Train: 2026-02-12 00:36 UTC through 2026-04-10 23:41 UTC
- Calibration: none for raw T2
- B_dev: w1-w4 policy selection; w5-w6 frozen rolling holdout
- B_test: 2026-04-11 00:16 UTC through 2026-05-10 23:51 UTC

## Leakage Check

- Forbidden columns intersection: empty
- Feature cutoff check: every Polymarket trade aggregation enforces `trade_time <= decision_time`
- Fit-on-validation check: no; policy was frozen before one B_test evaluation

## Frozen Policy

- Model: `joint_monotone`
- Conditional fill floor: `0.60`
- Minimum EV: `0.03`
- Bid grid: legal 5-cent ticks from 0.05 through 0.90

## Metrics

| Split | sum_pnl | delta_vs_anchor | order_count | order_coverage | win_pnl | loss_pnl | fill_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_dev w1-w4 | 237.70 | n/a | 3,232 | n/a | 663.50 | -425.80 | n/a |
| rolling w5-w6 | 127.40 | n/a | 1,495 | n/a | 304.30 | -176.90 | n/a |
| B_test | 12.90 | -29.53 | 2,631 | 0.352303 | 474.75 | -461.85 | -0.177341 |

Rolling PnL was `62.85, 47.55, 58.15, 69.15, 55.35, 72.05` for w1-w6.

## B_test Required Result

- btest_sum_pnl: 12.90
- btest_delta_vs_42p43: -29.53
- btest_wrong_submitted_rate: 0.295325
- btest_avg_loser_bid: 0.594402
- btest_submitted_fill_calibration_gap: -0.177341
- joint-event Brier (all legal actions): 0.158978
- base `q * Gc` joint-event Brier: 0.154138
- submitted joint-event Brier: 0.266013
- submitted joint-event calibration gap: -0.155609
- q Brier: 0.203802
- win/loss decomposition: +474.75 / -461.85

## Diagnosis

The joint model did not improve calibration or B_test PnL. Its all-grid Brier is worse than factorized `q * Gc`, and its submitted profitable-fill predictions remain overconfident by 0.1556. Gross wins were almost fully cancelled by 777 forced-wrong fills.

## Decision

Stop this raw T2 candidate and do not promote it. B_test `12.90` is `29.53` below the current `42.43` anchor. Retain the artifacts as the baseline for T3 submitted-action calibration.
