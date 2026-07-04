# Experiment 2

**Hypothesis**: Mechanism: bid-expanded joint win-and-fill classifier r(b,X)=P(correct and winner_low<=b|X) replacing the q times Gc independence decomposition
Hypothesis: Gc calibration drift shows correctness and fill depth are conditionally coupled; directly estimating their joint event should rank bids by realized EV more stably across months
Observable: improve robust w1-w4 sum_pnl and remain above the node-1 base-Gc policy on both untouched w5-w6 weeks without using B_test
Conflicts: node 1.1 improved PnL but degraded Gc Brier after w2; joint-event supervision removes the unstable conditional division

**Score**: 241.36

**Insight**: Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2.

**Result**: Rejected on B_dev; no B_test used.
