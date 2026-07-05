## Codebase

Working directory: C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-1.1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1.1
**Hypothesis**:
Mechanism: T0 frozen reproduction of node 1 and 1.1 feature-policy variants
Hypothesis: exact code and data reuse will distinguish genuine rolling reproducibility from historical implementation drift
Observable: B_dev and w1-w6 match prior results within tolerance and a first complete B_test artifact row is produced
Conflicts: none - this is the required falsifiable baseline

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-1.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 1.1 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-1.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 1.1 --split btest`
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

Save results to `results/1.1-<brief-description>/`.
