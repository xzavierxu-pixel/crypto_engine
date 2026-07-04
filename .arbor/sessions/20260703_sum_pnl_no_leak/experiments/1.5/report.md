# Experiment 1.5

**Hypothesis**: Mechanism: Fine 0.025 p_side-bin empirical-CDF blend
Hypothesis: Finer conditioning may capture heterogeneous fill curves that the 0.05 bins average away despite higher variance.
Observable: B_dev sum_pnl exceeds 94.18 without increasing the submitted-fill calibration gap.
Conflicts: node 1.1 supports blending; this probes the opposite granularity direction from nodes 1.3 and 1.4.

**Score**: 95.84

**Insight**: Fine 0.025 bins with 25% empirical blend improved B_dev to 95.84 and reduced the calibration gap to 0.1296; this is the best current node.

**Result**: B_dev +7.46; 2160 orders; order coverage 0.8424.
