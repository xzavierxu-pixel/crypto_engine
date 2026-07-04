# Experiment 4

**Hypothesis**: Mechanism: bid-expanded direct realized-PnL regression with conservative quantile objective over legal pre-decision market state
Hypothesis: q and Gc probability errors compound under distribution shift; directly learning the asymmetric payoff surface should favor lower bids that retain upside while limiting forced wrong-fill losses
Observable: exceed node 1 on robust w1-w4 sum_pnl and both untouched w5-w6 weeks with materially lower mean submitted bid
Conflicts: node 2 modeled a joint probability then reconstructed EV; direct payoff supervision removes probability-calibration dependence entirely

**Score**: 68.22

**Insight**: Direct PnL regression lowered bids but failed to learn stable action ranking: tune sum 68.22 with a negative week, far below probability-factorized policies.

**Result**: Rejected on B_dev; no B_test used.
