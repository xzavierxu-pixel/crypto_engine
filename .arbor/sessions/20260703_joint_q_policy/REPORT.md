# Arbor research report: joint correctness model and bid policy

## Outcome

The requested frozen validation target was **not reached**. A new feature-conditional
correctness model exceeded `sum_pnl=100` on both chronological B_dev folds, but
failed on the final-month B_test. It is rejected and nothing was promoted.

| Split | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Earlier B_dev, 2026-03-14..2026-03-27 | 82.34 | 129.79 | +47.45 |
| Main B_dev, 2026-03-28..2026-04-10 | 88.38 | 129.97 | +41.59 |
| Frozen B_test, 2026-04-11..2026-05-10 | 27.44 | 9.99 | -17.45 |

## Method

The run trained selected-side correctness classifiers only on rows preceding each
validation window. It compared raw probability, isotonic and logistic calibration,
LightGBM, CatBoost, and tree blends. These q estimates were crossed with shrinkage,
Gc floors, and minimum-EV admission thresholds: 504 B_dev policy combinations in
total. Selection maximized worst-fold improvement, then mean improvement.

The selected candidate used:

- LightGBM feature-conditional correctness probability q;
- no shrinkage toward 0.5;
- strict winner-fill probability floor `Gc >= 0.80`;
- minimum expected value `0.005`;
- unchanged direction decisions, accepted universe, hazard Gc model, bid grid, and
  wrong-order forced-fill semantics.

LightGBM improved q calibration on both development folds. Main-fold Brier changed
from raw `0.19113` to `0.18186`; earlier-fold Brier changed from `0.20201` to
`0.19467`. On B_test its Brier was `0.20805`, and the PnL improvement did not
generalize.

## B_test diagnostics

- accepted count: 5,228
- order count: 3,217; order coverage: 0.61534
- trade count: 2,501; fill rate: 0.47839
- win PnL: 565.54; loss PnL: -555.55
- `sum_pnl`: 9.99
- `wrong_fill_forced`: 1.0
- submitted fill calibration gap: 0.12767

The dominant bottleneck is temporal generalization, not development-set capacity.
The feature-conditional model increased B_dev PnL substantially and consistently,
yet the next month erased the gain. Another parameter search on the same two folds
is therefore not justified. A direct-PnL model should first be assessed with more
rolling forward folds. Joint direction optimization also requires counterfactual
UP and DOWN low-price labels; the current selected-side frame contains only
`chosen_low` for the already selected side, so changing direction with it would be
an invalid evaluation.

## Leakage and promotion status

The 576 checkpoint features were checked against forbidden exact names,
`future_*`, and any name containing `sample_weight`. `stage1_sample_weight` was
not used. Labels were used only as training targets on pre-validation rows.

The final B_test was evaluated exactly once after robust two-fold selection. The
candidate lost to baseline, so no merge, deploy artifact, config, default workflow,
or live execution behavior was changed.

## Artifacts

- `dev_search.csv`: all 504 B_dev combinations
- `dev_summary.json`: winner, top candidates, and calibration metrics
- `final_btest_once.json`: single frozen B_test result
- `run_joint_q_policy.py`: reproducible experiment runner
- `.coordinator/idea_tree.json`: Arbor state and rejected/deferred directions
