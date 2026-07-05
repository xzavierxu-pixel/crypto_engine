## Codebase

Working directory: C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.1
**Hypothesis**:
Mechanism: T2 monotone bid-conditioned model of P(correct and winner_low<=bid given X,bid)
Hypothesis: joint profitable-fill prediction removes independence error between direction correctness and winner fill
Observable: joint-event Brier and B_test sum_pnl improve over T0 with complete win-loss decomposition
Conflicts: prior joint-winfill attempt lacked the full trade/L2 representation and mandated submitted calibration

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 3.1 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 3.1 --split btest`
- **Dataset info**: chronological rolling w1-w6 under .arbor/sessions/20260703_prefinal_rolling/folds; frozen B_test expected_return_validation.parquet with 7468 samples/5228 accepted
- **Baseline score**: 42.43
- **Current trunk score**: 42.43

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.2, done, score=290.3] Frozen trade-Gc candidate is overconfident on submitted correct fills: predicted 0.91659 versus realized 0.67142; bid 0.60-0.70 and q 0.70-0.80 dominate B_test losses.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.1-<brief-description>/`.
