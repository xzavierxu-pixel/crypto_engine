# Experiment 3

**Hypothesis**: Mechanism: contemporaneous market-implied correctness anchor derived from selected and opposite pre-decision trade prices with complement normalization
Hypothesis: learned q models retain stale month-specific relationships, while the live market price aggregates current regime information and should reduce cross-month q drift
Observable: market-price or blended q improves Brier and robust PnL on w1-w4 and beats node 1 on both untouched w5-w6 weeks
Conflicts: prior L2 market-mid experiments used a shorter common window; this uses trade-derived anchors across the full rolling period and explicit complement normalization

**Score**: 166.13

**Insight**: Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side.

**Result**: Rejected on B_dev: tune 166.13, untouched 65.09; no B_test used.
