# Experiment 1.1

**Hypothesis**: Mechanism: T0 frozen reproduction of node 1 and 1.1 feature-policy variants
Hypothesis: exact code and data reuse will distinguish genuine rolling reproducibility from historical implementation drift
Observable: B_dev and w1-w6 match prior results within tolerance and a first complete B_test artifact row is produced
Conflicts: none - this is the required falsifiable baseline

**Score**: 290.31

**Insight**: T0 exactly reproduced rolling node1/node1.1, but B_test fell to 40.00/14.78; trade-conditioned Gc increased wrong submissions and loss exposure under month drift.

**Result**: T0 complete: node1 w1-w4 266.34 and B_test 40.00; node1.1 w1-w4 290.31, w5-w6 138.58, B_test 14.78; exactly-once guard and leakage checks passed.
