# Experiment 3.5

**Hypothesis**: Mechanism: T8 two-minute trade/L2 score used only as first-minute auxiliary feature or no-order gate
Hypothesis: later-horizon market path can identify first-minute loser exposure without replacing the production horizon
Observable: B_test loss_pnl_sum falls while first-minute direction and accepted universe remain fixed
Conflicts: standalone two-minute replacement is explicitly excluded

**Score**: 218.5

**Insight**: Two-minute auxiliary vetoes did not identify first-minute loser exposure robustly; B_dev selected no auxiliary.

**Result**: B_dev w1-w4 218.50, w5-w6 112.32; B_test retained anchor exactly at 42.43 with zero vetoes.
