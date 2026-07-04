# Experiment 1.4

**Hypothesis**: Mechanism: Pooled 0.20 p_side-bin empirical-CDF blend
Hypothesis: Aggressive pooling should expose whether p_side conditioning adds real Gc signal or only tail noise.
Observable: B_dev sum_pnl remains competitive with a smaller calibration gap and fewer fallback bins.
Conflicts: node 1.1 used narrower bins; this intentionally removes most conditional granularity.

**Score**: 86.66

**Insight**: 0.20 bins reduced the calibration gap to 0.1622 but lost PnL; p_side conditioning is useful and aggressive pooling removes signal.

**Result**: B_dev -1.72; 1626 orders.
