# Experiment 1.3

**Hypothesis**: Mechanism: Coarse p_side-bin empirical-CDF blend
Hypothesis: Wider 0.10 bins should lower calibration variance while a blend preserves H2 row-level structure.
Observable: B_dev sum_pnl and fill calibration improve beyond the 0.05-bin blend or reveal a bias-variance limit.
Conflicts: node 1.1 favored 0.05 bins; this tests whether its gain is variance-limited.

**Score**: 80.24

**Insight**: 0.10 bins over-pooled and selected min_ev 0.01; B_dev fell to 80.24 and calibration gap returned to 0.1737.

**Result**: B_dev -8.14 vs baseline; 1305 orders.
