# Experiment T5 - side-specific UP/DOWN path features

## Hypothesis
Explicit opposite-token state and UP/DOWN relative pressure improve adverse-selection representation beyond selected/opposite aliases.

## What Changed
Added pre-decision UP and DOWN path statistics, sell-pressure measures, trade-count/price-level depth proxies, UP/DOWN ratios, and complement deviations. Evaluated legacy, q-only, Gc-only fill-selection, and joint ablations.

## What Did Not Change
- Direction artifact: legacy selected side and accepted universe
- Calibration method: frozen raw-tree blend structure
- Fill semantics: correct may not fill; every wrong submitted order is forced filled
- Accepted universe: threshold_accepted == true
- Bid policy: gc_floor=0.90, min_ev=0.05

## Data Windows
- B_dev: chronological rolling w1-w6; choice used w1-w4 only and w5-w6 confirmed
- B_test: 2026-04-11T00:16:00+00:00..2026-05-10T23:51:00+00:00; evaluated once after freeze

## Leakage Check
- Forbidden columns intersection: []
- Feature cutoff: every source trade has trade_time <= decision_time
- Fit-on-validation: passed; all q/Gc models fit on corresponding training rows

## Metrics
| Variant | B_test sum_pnl | delta vs 42.43 | q Brier | Gc Brier | orders | win PnL | loss PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 8.47 | -33.96 | 0.205156 | 0.139284 | 2691.0 | 478.38 | -469.91 |
| q_side | 19.39 | -23.04 | 0.204960 | 0.139284 | 2666.0 | 476.74 | -457.35 |
| gc_side | 26.42 | -16.01 | 0.205156 | 0.125540 | 2204.0 | 412.44 | -386.02 |
| q_gc_side | 26.35 | -16.08 | 0.204960 | 0.125540 | 2185.0 | 408.89 | -382.54 |

## B_test Required Result
- selected_variant: legacy
- btest_sum_pnl: 8.47
- btest_delta_vs_42p43: -33.96
- btest_wrong_submitted_rate: 0.287625
- btest_avg_loser_bid: 0.607119
- btest_submitted_fill_calibration_gap: 0.230098

## Diagnosis
Relative to the same-run legacy ablation (8.47), selected T5 changed PnL by 0.00. The q-only and Gc-only rows isolate correctness-probability quality from bid/fill selection; no direction or universe change can explain the delta.

## Decision
- Stop promotion; retain the ablation evidence.
