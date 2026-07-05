## Codebase

Working directory: C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-2.1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 2.1
**Hypothesis**:
Mechanism: T9 lower-confidence-bound abstention tables over p_side, bid, and joint cells
Hypothesis: cells with consistently negative development PnL can be removed without hard-coding the observed B_test month
Observable: three aggressiveness levels report B_test delta, saved loss, killed wins, coverage change, and rolling stability
Conflicts: prior node 2.2 used fixed q>=0.55; this learns cell decisions chronologically

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-2.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 2.1 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-2.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 2.1 --split btest`
- **Dataset info**: chronological rolling w1-w6 under .arbor/sessions/20260703_prefinal_rolling/folds; frozen B_test expected_return_validation.parquet with 7468 samples/5228 accepted
- **Baseline score**: 42.43
- **Current trunk score**: 42.43

Use B_dev for final experiment scoring. Do NOT use B_test.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/2.1-<brief-description>/`.
