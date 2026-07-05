## Codebase

Working directory: C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.3

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.3
**Hypothesis**:
Mechanism: T4 trade/L2 bucket shrinkage and caps applied to hazard Gc ranking
Hypothesis: preserving row ranking while shrinking overconfident conditional fill probabilities improves action selection
Observable: empirical, hazard-only, shrink-blend, and upper-cap variants are compared on rolling and B_test PnL plus Gc Brier
Conflicts: prior fully empirical Gc lost row-level ranking; this explicitly retains it

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.3/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 3.3 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-3.3/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 3.3 --split btest`
- **Dataset info**: chronological rolling w1-w6 under .arbor/sessions/20260703_prefinal_rolling/folds; frozen B_test expected_return_validation.parquet with 7468 samples/5228 accepted
- **Baseline score**: 42.43
- **Current trunk score**: 42.43

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.2, done, score=290.3] Frozen trade-Gc candidate is overconfident on submitted correct fills: predicted 0.91659 versus realized 0.67142; bid 0.60-0.70 and q 0.70-0.80 dominate B_test losses. | [2, pending] Children findings: [2.1, done, score=330.8] Chronological LCB segment abstention found no stable negative cells after freezing all six folds; the B_test anchor was preserved with zero abstentions, showing fixed low-confidence bans are unsupported by the available development evidence.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.3-<brief-description>/`.
