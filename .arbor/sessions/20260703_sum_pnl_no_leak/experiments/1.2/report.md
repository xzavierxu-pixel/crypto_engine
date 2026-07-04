# Experiment 1.2

**Hypothesis**: Mechanism: Beta-posterior upper cap on p_side-bin Gc
Hypothesis: Capping overconfident H2 curves with calibration-only empirical uncertainty bounds should remove false-positive EV orders without collapsing coverage.
Observable: B_dev sum_pnl exceeds 88.38 with reduced loss_pnl_sum magnitude and at least 100 orders.
Conflicts: none - tests conservative probability correction rather than blending.

**Score**: 89.79

**Insight**: The beta upper-cap correction reduced the fill calibration gap to 0.1397 but improved B_dev only 1.41, underperforming the smoother empirical blend.

**Result**: B_dev +1.41; 1,570 orders; wrong_fill_forced=1.0; fixed direction universe; no forbidden feature.
