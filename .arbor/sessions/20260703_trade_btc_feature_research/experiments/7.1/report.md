# Experiment 7.1

**Hypothesis**: Mechanism: prequential full-universe direction model retrained daily on all markets settled before the current UTC day
Hypothesis: full-universe value is real on rolling folds but static direction accuracy drops at the final boundary; online direction updates should preserve the enlarged action space while adapting side selection
Observable: exceed node 7 on w1-w4 robust sum_pnl and both untouched weeks, with improved daily direction accuracy after day one
Conflicts: node 8 adapted q only inside the legacy universe; this adapts the upstream side decision that determines both correctness and which low distribution applies

**Score**: 260.53

**Insight**: Daily online direction adaptation modestly improved accuracy but reduced robust tune PnL versus static full-universe direction; tune 260.53, holdout 165.61.

**Result**: Rejected on B_dev; no additional B_test used.
