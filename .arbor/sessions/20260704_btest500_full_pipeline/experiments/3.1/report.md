# Experiment 3.1

**Hypothesis**: Mechanism: full-information contextual action-value learner over a discrete legal bid grid
Hypothesis: factorized q-times-Gc EV creates wrong credit assignment under drift; supervised potential outcomes for every bid can learn context-dependent action ranking while fixing q and fill semantics
Observable: exceed 290.31 robust B_dev sum_pnl with positive uplift on late folds and lower forced-wrong-fill loss
Conflicts: prior node 3 notes direct regression failed; this counters with per-action potential outcomes, fold-robust policy selection, and explicit action ranking

**Score**: 17.13460190063472

**Insight**: Full-information contextual action-value regression generalized positively to w5-w6 (47.99) but collapsed on w1-w4 (robust 17.13), indicating severe reward-model instability and action-value shrinkage.

**Result**: 21 legal bid actions, 22 decision-time features, forbidden-feature intersection empty; tune sum 23.03, tune robust 17.13, holdout sum 47.99; B_test not used.
