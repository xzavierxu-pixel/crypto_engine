# Experiment 3.2

**Hypothesis**: Mechanism: T3 submitted-subset calibration maps for the T2 joint score
Hypothesis: calibrating only actions the frozen policy would submit corrects selection-induced overconfidence
Observable: raw-versus-calibrated B_test comparison lowers calibration gap or loss without excessive order collapse
Conflicts: none - calibration is fit only on the chronological calibration window

**Score**: 233.1

**Insight**: Submitted-action Platt calibration improved joint Brier and reduced loss exposure, but removed profitable orders and lowered B_test PnL.

**Result**: B_dev w1-w4 233.10, w5-w6 118.90; B_test 4.20, delta -38.23 vs anchor and -8.70 vs raw T2.
