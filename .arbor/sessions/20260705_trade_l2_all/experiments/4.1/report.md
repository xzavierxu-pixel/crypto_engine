# Experiment 4.1

**Hypothesis**: Mechanism: T6 loser-exposure constrained optimizer with bid cap, q floor, and liquidity disagreement gate
Hypothesis: optimizing expected reward without an explicit loser-cost proxy submits expensive wrong orders under drift
Observable: wrong-submitted rate and average loser bid fall with a favorable B_test delta versus the unconstrained challenger
Conflicts: prior fixed q gate barely changed B_test; multi-signal loss constraints target actual exposure

**Score**: 279.134

**Insight**: Loss constraints selected only q>=0.60; stricter caps/gates sacrificed more winning PnL than loss exposure they rescued.

**Result**: B_dev robust 279.134; B_test 5.51, delta -36.92 vs anchor and -2.96 vs unconstrained.
