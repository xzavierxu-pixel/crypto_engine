# Experiment 3.2

**Hypothesis**: Mechanism: chronological Bayesian optimization of a low-dimensional q-conditioned bid policy
Hypothesis: the profitable policy surface is nonlinear but low-dimensional; Gaussian-process search across nested B_dev folds can identify stable q-to-bid controls without refitting direction q
Observable: exceed 290.31 robust B_dev sum_pnl and improve every late-fold score under one frozen policy parameterization
Conflicts: node 2.2 found a single q gate insufficient; this searches a structured q-conditioned bid curve and exposure controls jointly

**Score**: 164.29521496589103

**Insight**: Gaussian-process optimization of a q-conditioned bid curve materially underperformed the 290.31 B_dev trunk; low-dimensional policy tuning cannot repair the fixed q/Gc economics.

**Result**: 80 evaluations; tune robust 164.30; holdout sum recorded in bayesian_bid_policy_summary.json; B_test not used.
