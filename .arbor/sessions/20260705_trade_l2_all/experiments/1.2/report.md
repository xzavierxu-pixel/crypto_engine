# Experiment 1.2

**Hypothesis**: Mechanism: T1 submitted-action calibration audit stratified by bid, q, liquidity, and correctness
Hypothesis: aggregate fill metrics hide whether month degradation comes from correct-fill overprediction or forced-wrong loss concentration
Observable: calibration gaps and B_test PnL decomposition identify a dominant failure bucket without changing policy
Conflicts: none - diagnostic-only experiment

**Score**: 290.31

**Insight**: Frozen trade-Gc candidate is overconfident on submitted correct fills: predicted 0.91659 versus realized 0.67142; bid 0.60-0.70 and q 0.70-0.80 dominate B_test losses.

**Result**: T1 diagnostic completed: B_test sum_pnl 14.78, delta -27.65, 809 wrong submissions, calibration gap -0.24517; 4 tests passed.
