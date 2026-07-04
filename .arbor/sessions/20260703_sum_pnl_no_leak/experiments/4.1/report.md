# Experiment 4.1

**Hypothesis**: Mechanism: Calibration-tail isotonic q with retrained H2 Gc
Hypothesis: Monotone correctness calibration should improve EV ranking by aligning q with realized selected-side correctness.
Observable: B_dev Brier improves and H14-style sum_pnl exceeds 88.38.
Conflicts: Gc blend node 1.1 improved fill probabilities; this independently corrects correctness probabilities.

**Score**: 88.38

**Insight**: The current H14 runner ignores the checkpoint q_calibrator and reuses raw frame p_side, so isotonic q produced no H14 change; integrating calibrated q requires an explicit downstream interface change.

**Result**: B_dev unchanged at 88.38; do not claim q calibration improvement from this path.
