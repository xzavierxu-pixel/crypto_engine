# Experiment 1

**Hypothesis**: Mechanism: supervised selected-side correctness ensemble trained only on pre-validation rows
Hypothesis: feature-conditional q should reduce the calibration and regime errors that made isotonic policy gains collapse on B_test
Observable: positive worst-fold sum_pnl delta across both chronological B_dev folds and improved Brier score
Conflicts: prior policy-only search overfit B_dev; this counters via learned feature-conditional correctness

**Score**: 129.97

**Insight**: Feature-conditional LGBM correctness probability generalized across both B_dev folds: 129.97 and 129.79, with strict Gc floor 0.80 and min EV 0.005.

**Result**: Robust worst-fold delta +41.59 over fold baselines; selected for the single frozen B_test evaluation.
