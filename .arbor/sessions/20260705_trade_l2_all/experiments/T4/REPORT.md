# Experiment T4 - trade/L2 empirical Gc shrink

## Hypothesis

Trade-path bucket statistics can reduce hazard Gc overconfidence while shrink blends and caps retain row-level hazard ranking.

## What Changed

Only Gc was corrected. Chronological train-only buckets use p_side, candidate bid, selected/opposite 60-second liquidity, and selected-side 60-second sell-pressure slope. Full empirical, hazard-only, three shrink blends, and an empirical upper cap were compared.

## What Did Not Change

- Direction artifact and accepted universe: unchanged legacy candidate (5228 accepted B_test rows)
- q calibration: frozen 0.5 raw-tree blend + 0.5 trade q
- Policy thresholds: Gc floor 0.90, minimum EV 0.05, 0.01 tick
- Fill semantics: correct fills iff winner_low <= bid; wrong submitted orders are forced filled

## Selection

- Selected on w1-w4 only: `hazard_only`
- Rule: maximum sum(PnL) - standard deviation(PnL)
- w1-w4 sum: 210.10; w5-w6 holdout sum: 88.55
- B_test was evaluated once after freezing the selected variant. Required ablations were evaluated in the same frozen pass.

## B_test Result

| variant | sum_pnl | delta vs 42.43 | orders | win_pnl | loss_pnl | submitted Gc Brier | fill gap (pred-actual) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hazard_only | 4.65 | -37.78 | 1411 | 257.95 | -253.30 | 0.176445 | 0.114956 |
| full_empirical | -1.15 | -43.58 | 1602 | 285.81 | -286.96 | 0.215738 | 0.171143 |
| shrink_blend_0.25 | -9.21 | -51.64 | 1150 | 199.55 | -208.76 | 0.151907 | 0.084104 |
| shrink_blend_0.50 | -10.00 | -52.43 | 960 | 154.04 | -164.04 | 0.165285 | 0.100183 |
| shrink_blend_0.75 | -6.26 | -48.69 | 932 | 140.56 | -146.82 | 0.196401 | 0.137904 |
| upper_cap_0.90 | -9.18 | -51.61 | 765 | 122.23 | -131.41 | 0.144661 | 0.076571 |


## Leakage Check

- Forbidden feature intersection: empty
- Bucket edges/statistics: fit on chronological training rows only
- Trade cutoff: trade_time <= decision_time
- B_test in fitting/selection: no

## Diagnosis and Decision

The frozen selected variant produced B_test sum_pnl 4.65, delta -37.78 versus the 42.43 anchor. Promotion is not automatic and requires user approval.
