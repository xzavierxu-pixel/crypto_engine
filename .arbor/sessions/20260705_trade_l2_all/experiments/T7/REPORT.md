# Experiment T7 - anchor-safe challenger

## Hypothesis

A trade/L2 challenger should replace the validated 42.43 anchor only in decision-time cells where the chronological reward-delta posterior has a lower credible bound above zero.

## What Changed

T4 was selected as challenger because its w1-w4 B_dev PnL (280.96) exceeded T2 and T3. A frozen normal-inverse-gamma posterior gate defaults every row to the anchor.

## What Did Not Change

- Direction, accepted universe, labels, bids produced by each source policy, and fill semantics
- Correct fills iff `winner_low <= bid`; wrong submitted orders are forced filled
- Anchor policy: raw-tree blend, Gc floor 0.85, minimum EV 0.02

## Leakage Check

- Every rolling posterior uses only earlier fold reward deltas
- T2/T3/T4 challenger choice and gate hyperparameters use w1-w4 only
- w5-w6 are frozen holdout; B_test was run once after freeze

## Rolling Metrics

| split | anchor | challenger | safe | replacements | replacement delta |
|---|---:|---:|---:|---:|---:|
| w1 | 49.62 | 73.30 | 49.62 | 0 | 0.00 |
| w2 | 45.34 | 65.30 | 61.40 | 779 | 16.06 |
| w3 | 55.53 | 67.15 | 64.08 | 700 | 8.55 |
| w4 | 68.01 | 75.21 | 75.21 | 736 | 7.20 |
| w5 | 56.49 | 61.30 | 61.30 | 632 | 4.81 |
| w6 | 55.83 | 67.80 | 67.80 | 723 | 11.97 |

## B_test Required Result

- anchor B_test: 42.43
- challenger B_test: 4.65
- safe-gated B_test: 4.65
- safe delta vs anchor: -37.78
- replacement count: 2018
- replacement win/loss PnL: 256.31 / -251.53
- replacement realized delta vs anchor counterfactual: -37.78
- safe total win/loss PnL: 257.95 / -253.30

## Diagnosis and Decision

The posterior gate did not improve the frozen anchor. Recording B_test does not authorize promotion or deployment.
