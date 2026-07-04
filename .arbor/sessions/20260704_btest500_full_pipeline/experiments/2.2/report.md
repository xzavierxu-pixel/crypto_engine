# Experiment 2.2

**Hypothesis**: Mechanism: precision-constrained order admission with bid exposure caps
Hypothesis: calibrated q and Gc still submit too many medium-confidence high-loss orders; explicit correctness floors and bid caps should reduce forced wrong-fill loss faster than win PnL
Observable: robust w1-w4 sum_pnl exceeds 290.31, w5-w6 remain positive, and loss_pnl_sum falls without collapsing win_pnl_sum
Conflicts: node 2.1 improved calibration but B_test stayed 28.82; this converts calibrated probabilities into explicit downside control

**Score**: 290.79

**Insight**: Explicit q>=0.55 gating raised robust B_dev from 288.89 to 290.79 and holdout to 139.86, but removed only 195 B_test rows and improved frozen PnL merely from 28.82 to 29.27.

**Result**: raw_path plus path Gc, floor 0.85, min_ev 0.02, min_q 0.55; B_test 29.27; rejected.
