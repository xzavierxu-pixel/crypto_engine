# Experiment T1 - submitted-action calibration audit

## Hypothesis

Aggregate fill statistics hide whether degradation is caused by winner-fill overprediction or forced-wrong loss concentration. This audit leaves the frozen trade-conditioned candidate unchanged and decomposes its submitted actions.

## What Changed

Only reporting changed. A reusable audit validates fill/PnL invariants and emits bid, q, q-by-bid, UTC-day, correctness, selected-liquidity, and opposite-liquidity diagnostics.

## What Did Not Change

- Direction artifact: accepted legacy direction output used by the frozen candidate
- Calibration method: frozen `0.5 * raw_tree_blend + 0.5 * q_trade`
- Fill semantics: correct orders fill iff `winner_low <= bid`; wrong submitted orders are forced filled
- Accepted universe: 5,228 rows
- Policy: frozen node 1.1, Gc floor 0.90, minimum EV 0.05

## Data Windows

- Train: 2026-02-12 00:35 UTC through 2026-04-10 23:40 UTC
- Calibration: historical candidate's pre-B_test training/calibration protocol; no fitting in T1
- B_dev / rolling: w1-w6 historical frozen results
- B_test: 2026-04-11 00:15 UTC through 2026-05-10 23:50 UTC

## Leakage Check

- Forbidden columns intersection for policy fitting: empty; T1 does not fit or choose actions
- Feature cutoff check: trade liquidity proxy uses only `trade_time <= decision_time`
- Fit-on-validation check: no fitting, threshold selection, or bucket-policy tuning

## Metrics

| Split | sum_pnl | delta_vs_anchor | order_count | order_coverage | win_pnl | loss_pnl | fill_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_dev w1-w4 | 290.31 | n/a | unavailable | unavailable | unavailable | unavailable | unavailable |
| rolling w5-w6 | 138.58 | n/a | unavailable | unavailable | unavailable | unavailable | unavailable |
| B_test | 14.78 | -27.65 | 2,842 | 0.5436 | 502.42 | -487.64 | -0.2452 |

The reported fill gap is `realized correct-fill rate - predicted winner-fill probability` on correct submitted orders. The all-submitted forced-fill gap is -0.1514, but it mixes the distinct forced-wrong mechanism and is not used as the primary calibration diagnostic.

## B_test Required Result

- btest_sum_pnl: 14.78
- btest_delta_vs_42p43: -27.65
- btest_wrong_submitted_rate: 0.284659 (809 / 2,842)
- btest_avg_loser_bid: 0.602769
- btest_submitted_fill_calibration_gap: -0.245168
- q_brier: 0.200796
- Gc / correct submitted fill Brier: 0.280702
- correct submitted: predicted fill 0.916589, realized fill 0.671422, PnL +502.42
- wrong submitted: forced fill 1.0, average bid 0.602769, PnL -487.64

## Bucket Diagnosis

- The largest price exposure, bid 0.60-0.70, lost 13.89 (win +200.23, loss -214.12) across 1,131 orders.
- q 0.70-0.80 lost 8.71 (win +223.27, loss -231.98) across 1,228 orders; confidence did not protect against forced-wrong exposure.
- Winner-fill probability is overpredicted in every material bid/q bucket. Correct-fill calibration gaps range from about -0.20 to -0.41.
- The lowest and third selected-side trade-liquidity quartiles lost 10.39 and 10.69 respectively. The lowest opposite-side liquidity quartile lost 9.42. There is no monotone liquidity-only explanation.
- Full tables are in `submitted_action_buckets.csv`; detailed frozen actions are in `predictions_btest.parquet`.

## Diagnosis

The dominant failure is the combination of severe correct-fill overprediction and concentrated forced-wrong losses, not a lack of gross winner PnL. The policy predicted 91.66% fill for correct submissions but realized 67.14%; meanwhile 809 wrong orders were necessarily filled and lost 487.64. Gross wins of 502.42 barely cover those losses. This explains why strong rolling PnL did not transfer to the month-long B_test.

## Decision

- Continue / modify / stop: modify the Gc/submitted calibration and loss-exposure mechanism; do not promote this candidate
- Reason: B_test 14.78 is 27.65 below the 42.43 anchor, with a -0.245 correct-fill calibration gap and near-total cancellation of gross wins by forced-wrong losses

Promotion still requires explicit user approval; this audit does not alter live execution.
