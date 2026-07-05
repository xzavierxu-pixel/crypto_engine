# Experiment 2.1

**Hypothesis**: Mechanism: T9 lower-confidence-bound abstention tables over p_side, bid, and joint cells
Hypothesis: cells with consistently negative development PnL can be removed without hard-coding the observed B_test month
Observable: three aggressiveness levels report B_test delta, saved loss, killed wins, coverage change, and rolling stability
Conflicts: prior node 2.2 used fixed q>=0.55; this learns cell decisions chronologically

**Score**: 330.82

**Insight**: Chronological LCB segment abstention found no stable negative cells after freezing all six folds; the B_test anchor was preserved with zero abstentions, showing fixed low-confidence bans are unsupported by the available development evidence.

**Result**: T9 completed: w1-w6 sum 330.82, B_test 42.43, delta 0.00, all 3x3 frozen variants reported; no orders rejected.
