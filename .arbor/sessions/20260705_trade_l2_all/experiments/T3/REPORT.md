# Experiment T3 - submitted-action calibration

## Hypothesis

Calibrating the joint profitable-fill probability only on actions the frozen T2 policy would submit will reduce its overconfidence and forced-wrong loss exposure.

## What Changed

The frozen `platt` map (blend `0.5`) recalibrates T2 joint probability. Calibrated EV and conditional fill checks may remove an order.

## What Did Not Change

- Direction artifact: frozen T2 selected side and accepted universe
- Bid: never changed; T3 cannot create an order T2 did not submit
- Fill semantics: correct fills iff `winner_low <= bid`; wrong submitted orders are forced filled
- B_test accepted universe: 5228 rows

## Data Windows

- Calibration: chronological out-of-sample w1-w6 submissions, all before B_test
- B_dev: w1-w4 method/blend selection; w5-w6 frozen holdout
- B_test: 2026-04-11 through 2026-05-10 UTC

## Leakage Check

- Forbidden columns intersection: empty
- Feature cutoff check: inherited from T2 (`trade_time <= decision_time`)
- Fit-on-validation check: each rolling fold uses strictly earlier OOS folds; B_test was evaluated once after freeze

## Metrics

| Split | sum_pnl | order_count | win_pnl | loss_pnl | joint Brier |
|---|---:|---:|---:|---:|---:|
| w1 | 62.85 | 688 | 138.40 | -75.55 | 0.244780 |
| w2 | 51.70 | 840 | 172.95 | -121.25 | 0.234083 |
| w3 | 55.10 | 688 | 144.40 | -89.30 | 0.231661 |
| w4 | 63.45 | 632 | 136.25 | -72.80 | 0.230037 |
| w5 | 49.60 | 647 | 126.50 | -76.90 | 0.238888 |
| w6 | 69.30 | 610 | 131.65 | -62.35 | 0.250137 |
| B_test | 4.20 | 2132 | 382.55 | -378.35 | 0.252889 |

## B_test Required Result

- raw T2 vs calibrated T3 sum_pnl: 12.90 vs 4.20 (delta -8.70)
- btest_delta_vs_42p43: -38.23
- raw T2 vs T3 order_count: 2631 vs 2132
- raw T2 vs T3 loss_pnl_sum: -461.85 vs -378.35
- raw vs calibrated joint Brier on T2 submitted universe: 0.266013 vs 0.252889
- btest_wrong_submitted_rate: 0.298311
- btest_avg_loser_bid: 0.594890
- btest_submitted_fill_calibration_gap: -0.128760

## Diagnosis

Calibration reduced forced-wrong loss magnitude and improved joint-event Brier. It removed 499 of 2631 raw T2 orders.

## Decision

T3 does not beat the 42.43 anchor. No promotion is authorized by this experiment.
