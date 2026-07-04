# Experiment 1.1

**Hypothesis**: Mechanism: Cross-fitted p_side-bin empirical-CDF blend
Hypothesis: Blending H2 Gc with calibration-only empirical CDFs should correct systematic overprediction while retaining row-level ranking.
Observable: B_dev sum_pnl exceeds 88.38 and submitted-fill calibration gap falls below 0.17.
Conflicts: none - prior historical runs were scored on B_test; this uses a clean pre-validation holdout.

**Score**: 94.18

**Insight**: A 25% calibration-only empirical-CDF blend reduced submitted fill calibration gap from 0.1733 to 0.1310 and improved B_dev sum_pnl from 88.38 to 94.18; gains came with higher order coverage 0.834.

**Result**: B_dev +5.80; 2,138 orders; wrong_fill_forced=1.0; fixed direction universe; no forbidden feature.
