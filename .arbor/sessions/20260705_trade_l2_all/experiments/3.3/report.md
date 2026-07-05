# Experiment 3.3

**Hypothesis**: Mechanism: T4 trade/L2 bucket shrinkage and caps applied to hazard Gc ranking
Hypothesis: preserving row ranking while shrinking overconfident conditional fill probabilities improves action selection
Observable: empirical, hazard-only, shrink-blend, and upper-cap variants are compared on rolling and B_test PnL plus Gc Brier
Conflicts: prior fully empirical Gc lost row-level ranking; this explicitly retains it

**Score**: 280.96

**Insight**: Empirical Gc shrink variants underperformed the frozen hazard-only policy; selected hazard-only by w1-w4.

**Result**: B_dev w1-w4 280.96, w5-w6 129.10; B_test 4.65 versus 42.43 anchor.
