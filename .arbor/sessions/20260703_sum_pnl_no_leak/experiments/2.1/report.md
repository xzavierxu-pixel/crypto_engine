# Experiment 2.1

**Hypothesis**: Mechanism: Calibration-tail empirical p_side CDF policy
Hypothesis: A fully empirical Gc eliminates neural calibration bias and may improve bids when recent data represents the next regime.
Observable: B_dev sum_pnl exceeds 88.38 with finite fill reliability and stable order coverage.
Conflicts: node 1.1 retained H2 ranking; this removes the neural Gc entirely.

**Score**: 51.87

**Insight**: Calibration-tail empirical Gc discarded useful H2 row-level ranking and cut B_dev PnL nearly in half despite high order coverage.

**Result**: B_dev 51.87; 2278 orders.
