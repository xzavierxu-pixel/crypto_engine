# Experiment 2.1

**Hypothesis**: Mechanism: multiscale Polymarket path-signature and liquidity-shape feature bank
Hypothesis: fixed-window moments discard ordering, acceleration, side-switching, and price-impact structure that separates profitable fills from forced wrong fills
Observable: q AUC and late-fold Gc Brier improve together, with w1-w4 robust sum_pnl above 290.31 and positive w5-w6 uplift
Conflicts: prior node 1 used count/last/mean/min/max/range/std/slope; this counters by representing event order, duration-weighted path shape, impact, and selected-opposite lead-lag

**Score**: 288.89

**Insight**: Path signatures improved q and Gc Brier on all six folds, but robust B_dev remained below 290.31 and frozen B_test reached only 28.82; calibration improvement alone did not improve forced-fill PnL.

**Result**: Winner raw_path plus path Gc, floor 0.85, min_ev 0.02; tune 288.89, holdout 139.00, B_test 28.82; rejected.
