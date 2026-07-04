# Experiment 5

**Hypothesis**: Mechanism: temporally pooled empirical winner-low CDF with optional p_side stratification replacing the overconfident contextual hazard
Hypothesis: winner-low quantiles are stable while contextual Gc selection is overconfident; a low-variance empirical CDF should choose lower, better-calibrated bids across months
Observable: improve Gc Brier and robust PnL on w1-w4, remain positive on w5-w6, and reduce submitted fill calibration gap
Conflicts: node 1.1 added more contextual Gc capacity and worsened late calibration; this deliberately removes capacity and pools the stable target distribution

**Score**: 272.94

**Insight**: Low-variance empirical Gc stratified by p_side beat the contextual hazard on untouched PnL (148.22 vs 128.49); all-history p-bin CDF improved Gc Brier on all six folds, validating selection-overconfidence as the bottleneck.

**Result**: Selected 28-day p-bin empirical Gc, floor 0.80, min_ev 0.005: tune 272.94, holdout 148.22; B_test reserved for milestone.
