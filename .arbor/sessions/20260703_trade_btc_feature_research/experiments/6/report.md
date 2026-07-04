# Experiment 6

**Hypothesis**: Mechanism: covariate-shift-resistant minimal q model using only p_side and pre-decision trade state, fitted before a temporal calibration tail
Hypothesis: the 576-feature q model and trade augmentation degrade on the final month because stale BTC relationships dominate; removing shifting features and calibrating on the latest train tail should preserve only transportable error signals
Observable: improve q Brier and PnL over raw p_side on w1-w4 and every untouched w5-w6 fold using both hazard and empirical Gc
Conflicts: node 1's broad trade model improved rolling folds but final-month q Brier worsened to 0.2088; this counters by feature minimization and temporal calibration

**Score**: 199.52

**Insight**: Feature minimization reduced some q drift, but improvements were not uniform; the best 25% minimal-q blend scored 199.52 tune and 78.79 holdout, below broader trade q.

**Result**: Rejected on B_dev; no new B_test evaluation.
