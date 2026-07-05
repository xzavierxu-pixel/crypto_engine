# Experiment 4

**Hypothesis**: Mechanism: Correctness-q admission gate jointly optimized with a fixed discrete bid, without a Gc factorization
Hypothesis: High-confidence direction subsets can monetize 73% accuracy while avoiding the 0.2265 Gc calibration error
Observable: Robust w1-w4 sum_pnl exceeds 89.99 and the frozen policy passes both w5-w6 weeks
Conflicts: node 2 showed p_side-only grouping is weak; this uses learned correctness ensembles as the admission signal

**Score**: 49.64

**Insight**: Correctness q alone selects accurate rows but cannot predict low-price winner fills; joint correctness-and-fill modeling is required.

**Result**: w1-w4 sum_pnl 49.64, below 89.99; q-gate mechanism rejected.
