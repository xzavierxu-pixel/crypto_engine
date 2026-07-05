# Experiment 4.2

**Hypothesis**: Mechanism: T7 posterior-lower-bound gate choosing between anchor and best trade/L2 challenger per segment
Hypothesis: selective replacement captures stable challenger gains while reverting uncertain segments to the 42.43 anchor
Observable: safe-gated B_test beats or preserves anchor and reports replacement count plus replaced win/loss PnL
Conflicts: none - this is the required promotion-safe composition layer

**Score**: 250.31

**Insight**: Posterior safe gate collapsed toward the T4 challenger under month shift and failed to preserve anchor PnL.

**Result**: B_dev w1-w4 250.31, w5-w6 129.10; B_test anchor 42.43, challenger 4.65, safe 4.65 with 2018 replacements.
