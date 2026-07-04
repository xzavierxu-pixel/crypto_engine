# Deep pre-B_test rolling research report

## Outcome

The best valid frozen B_test result improved from **27.44 to 42.43**. The requested
target of 100 was not reached, so no model, deploy artifact, default config, or live
execution behavior was promoted.

The improvement came from a correctness-probability ensemble, not from changing the
direction universe:

- `q = 0.50 * raw selected-side probability + 0.25 * LightGBM q + 0.25 * CatBoost q`;
- `Gc >= 0.85`;
- minimum expected value `0.02`;
- unchanged accepted rows, H2 hazard model, tick grid, and forced fill for every
  submitted wrong prediction.

Frozen B_test metrics for this candidate:

| Metric | Baseline | Candidate |
|---|---:|---:|
| sum_pnl | 27.44 | 42.43 |
| accepted_count | 5,228 | 5,228 |
| order_count | 2,597 | 1,980 |
| order_coverage | 0.49675 | 0.37873 |
| trade_count | — | 1,564 |
| fill_rate | — | 0.29916 |
| win_pnl_sum | — | 358.87 |
| loss_pnl_sum | — | -316.44 |
| submitted fill calibration gap | — | 0.14487 |

## Rolling protocol

All available pre-B_test rows were used chronologically. Six expanding-window H2
models were retrained, each with its own pre-window fit/calibration data:

| Fold | Validation window | H2 expected-return PnL |
|---|---|---:|
| w1 | 2026-02-27..2026-03-05 | 46.61 |
| w2 | 2026-03-06..2026-03-12 | 14.16 |
| w3 | 2026-03-13..2026-03-19 | 20.13 |
| w4 | 2026-03-20..2026-03-26 | 34.03 |
| w5 | 2026-03-27..2026-04-02 | 33.64 |
| w6 | 2026-04-03..2026-04-10 | 52.18 |

w1-w4 were used for policy selection. w5-w6 remained untouched until the candidate
was fixed. The selected q policy scored 218.50 on w1-w4 and 112.32 on w5-w6,
versus exact-policy baselines of 144.14 and 85.71. A later expanded q grid found a
nearby candidate with 221.78 and 127.07, but it was not used for another B_test
attempt.

## Directions tested

The run evaluated 3,844 policy/model combinations plus six independently trained
rolling hazard models:

1. Raw, isotonic, logistic, LightGBM, CatBoost, and blended correctness models.
2. Gc floors, EV admission thresholds, q shrinkage, and ensemble-disagreement
   penalties.
3. 14/21/28-day recency models versus expanding-history models.
4. Direct action-value regression over 17 bid actions.
5. Fixed and affine empirical bid policies.
6. Joint direction classifiers and bid policies.

Key negative results:

- uncertainty penalties did not help; the selected penalty was zero;
- 14/21/28-day models did not beat expanding-history models;
- direct-PnL regression scored 136.60 on tune folds, below the comparable EV-policy
  family;
- empirical fixed/affine bids were materially weaker;
- joint direction optimization improved all six pre-test weeks, but its B_test result
  was only 29.48, despite changing 590 directions.

## Cross-month failure analysis

The tested 42.43 candidate had weekly B_test PnL of `10.69`, `15.38`, `-0.19`, and
`16.55`. Baseline weekly PnL was `15.12`, `11.24`, `-1.51`, and `2.59`.

The principal failure is not one catastrophic week or a broken fill rule. Instead,
the approximately 50-70 PnL seen in each recent pre-test week compressed to roughly
0-16 per week throughout the next month. Week three also had lower accepted
direction accuracy (`0.6819`), but weeks one and two retained accuracy above `0.719`
and still produced only modest PnL. This indicates decay in conditional order value:
the model still predicts direction reasonably, but q/Gc no longer separates enough
high-value orders to produce the pre-test payoff spread.

The H2 submitted-fill calibration gap was already large before B_test (roughly
0.10-0.19 by fold), and remained 0.145 on B_test. Repeating parameter searches on
the same q/Gc factorization is unlikely to bridge the remaining 57.57 PnL gap.

## Counterfactual direction data

Both UP and DOWN lows were reconstructed from raw sell-taker trades between each
row's decision time and market end. The reconstructed selected-side low matched the
existing target on 99.20% of pre-test rows with mean absolute error 0.00026. For
joint-direction evaluation, the original `chosen_low` was retained whenever the side
was unchanged; reconstructed lows were used only for flipped sides.

## Leakage and evaluation discipline

- `stage1_sample_weight` was explicitly forbidden.
- Every checkpoint feature list was checked for forbidden exact columns,
  `future_*`, and names containing `sample_weight`.
- Fit, calibration, and validation windows were chronological.
- Labels were used only on rows preceding the corresponding validation window.
- B_test was not used in grid ranking. Two milestone evaluations were made for
  materially different mechanisms: the selected-side q redesign and the later joint
  direction redesign.

## Recommendation

The current best evidence-backed candidate is the 42.43 selected-side q ensemble,
but it should not be promoted yet. Reaching 100 likely requires a new source of
forward information or a redesigned online/adaptive estimation protocol, not more
static hyperparameter search. The next falsifiable direction is walk-forward online
updating evaluated with multiple simulated month boundaries; it must be specified
without using the frozen April-May labels for parameter choice.

## Main artifacts

- `rolling_hazard_summary.csv`: six H2 rolling folds
- `rolling_policy_search.csv`: initial 840 q/policy candidates
- `recency_q_search.csv`: 1,440 recency candidates
- `joint_direction_search.csv`: 840 joint direction candidates
- `uncertainty_policy_search.csv`: uncertainty-aware candidates
- `direct_pnl_summary.json`: direct action-value result
- `tested_btest_weekly_diagnostics.csv`: decomposition of already-tested policies
- `btest_milestone_once.json`: best B_test result, 42.43
- `joint_direction_btest_milestone.json`: joint-direction B_test result, 29.48
- `.coordinator/idea_tree.json`: durable Arbor state
