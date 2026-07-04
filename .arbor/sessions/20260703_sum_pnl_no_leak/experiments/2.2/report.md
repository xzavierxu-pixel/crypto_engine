# Experiment 2.2

**Hypothesis**: Mechanism: Fit-window empirical p_side CDF policy
Hypothesis: A larger historical fit window should reduce empirical-CDF variance enough to offset temporal drift.
Observable: B_dev sum_pnl beats the calibration-tail empirical policy and maintains at least 100 orders.
Conflicts: node 2.1 prioritizes recency; this prioritizes sample size.

**Score**: 54.03

**Insight**: Fit-window empirical Gc achieved a low calibration gap 0.0928 but poor PnL 54.03, proving aggregate calibration alone is insufficient for bid ranking.

**Result**: B_dev 54.03; 2381 orders.
