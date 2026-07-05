# Experiment 2

**Hypothesis**: Mechanism: Cross-fitted empirical action-value table that directly estimates PnL for each bid from q, side, and time context
Hypothesis: Direct conditional PnL avoids compounding q and Gc calibration errors that nearly cancelled validation wins and losses
Observable: Cross-fitted w1-w4 sum_pnl materially beats the best factorized EV policy and passes w5-w6
Conflicts: prior direct-PnL failed on first-minute data; the two-minute horizon changes fill distributions and this version uses cross-fitted discrete action values

**Score**: 33.27

**Insight**: Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal.

**Result**: w1-w4 sum_pnl 33.27, below 89.99; mechanism rejected.
