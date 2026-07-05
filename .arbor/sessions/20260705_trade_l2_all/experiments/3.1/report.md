# Experiment 3.1

**Hypothesis**: Mechanism: T2 monotone bid-conditioned model of P(correct and winner_low<=bid given X,bid)
Hypothesis: joint profitable-fill prediction removes independence error between direction correctness and winner fill
Observable: joint-event Brier and B_test sum_pnl improve over T0 with complete win-loss decomposition
Conflicts: prior joint-winfill attempt lacked the full trade/L2 representation and mandated submitted calibration

**Score**: 237.7

**Insight**: Joint profitable-fill worsened submitted-action calibration and B_test despite positive rolling folds.

**Result**: B_dev w1-w4 sum 237.70; holdout w5-w6 127.40; B_test 12.90 versus 42.43 anchor.
