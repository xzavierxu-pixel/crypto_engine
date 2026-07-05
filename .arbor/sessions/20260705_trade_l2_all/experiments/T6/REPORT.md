# Experiment T6 - loss-exposure constrained bid policy

## Hypothesis
Max-bid, q, liquidity-disagreement, and loser-exposure vetoes can reduce forced-wrong-fill losses while preserving most winner PnL.

## What Changed
Applied an OR-veto gate to the frozen T4 hazard-only analytic bid. The loser proxy is `bid * (1-q) * (1+abs(log1p(selected_liquidity)-log1p(opposite_liquidity)))`.

## What Did Not Change
- Direction, q model, Gc model, accepted universe, analytic bid objective, and forced-wrong-fill semantics are unchanged.
- T4 hazard-only was used because it was the T4 w1-w4 winner and T3 was unavailable at dispatch.

## Selection and Windows
- w1-w4 selected `mb1.00_q0.60_ld99.00_lp99.00` without B_test; w5-w6 were confirmation only.
- B_test: ['2026-04-11 00:16:00+00:00', '2026-05-10 23:51:00+00:00']; evaluated exactly once after freeze.

## Metrics
| B_test variant | sum_pnl | delta vs 42.43 | orders | win_pnl | loss_pnl | wrong rate | avg loser bid |
|---|---:|---:|---:|---:|---:|---:|---:|
| unconstrained | 8.47 | -33.96 | 2691.0 | 478.38 | -469.91 | 0.287625 | 0.607119 |
| constrained | 5.51 | -36.92 | 2644.0 | 469.17 | -463.66 | 0.287821 | 0.609277 |

## Leakage Check
Forbidden intersection empty; trade cutoff `trade_time <= decision_time`; all model and policy choices precede B_test.

## Diagnosis
The gate changed B_test PnL by -2.96, rescued 6.25 of loss PnL, and changed win PnL by -9.21. B_test sum PnL is 5.51, delta -36.92 versus the fixed 42.43 anchor.

## Decision
No automatic promotion; user approval is required even if the candidate exceeds the anchor.
