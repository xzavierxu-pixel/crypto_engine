## Codebase

Working directory: C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-4.1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 4.1
**Hypothesis**:
Mechanism: T6 loser-exposure constrained optimizer with bid cap, q floor, and liquidity disagreement gate
Hypothesis: optimizing expected reward without an explicit loser-cost proxy submits expensive wrong orders under drift
Observable: wrong-submitted rate and average loser bid fall with a favorable B_test delta versus the unconstrained challenger
Conflicts: prior fixed q gate barely changed B_test; multi-signal loss constraints target actual exposure

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-4.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 4.1 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\arbor-worktrees\20260705_trade_l2_all-node-4.1/.arbor/sessions/20260705_trade_l2_all/run_all_trade_l2_experiments.py --experiment 4.1 --split btest`
- **Dataset info**: chronological rolling w1-w6 under .arbor/sessions/20260703_prefinal_rolling/folds; frozen B_test expected_return_validation.parquet with 7468 samples/5228 accepted
- **Baseline score**: 42.43
- **Current trunk score**: 42.43

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=290.3] T0 exactly reproduced rolling node1/node1.1, but B_test fell to 40.00/14.78; trade-conditioned Gc increased wrong submissions and loss exposure under month drift. | [1.2, done, score=290.3] Frozen trade-Gc candidate is overconfident on submitted correct fills: predicted 0.91659 versus realized 0.67142; bid 0.60-0.70 and q 0.70-0.80 dominate B_test losses. | [2, pending] Children findings: [2.1, done, score=330.8] Chronological LCB segment abstention found no stable negative cells after freezing all six folds; the B_test anchor was preserved with zero abstentions, showing fixed low-confidence bans are unsupported by the available development evidence. | [3, pending] Children findings: [3.1, done, score=237.7] Joint profitable-fill worsened submitted-action calibration and B_test despite positive rolling folds. | [3.2, done, score=233.1] Submitted-action Platt calibration improved joint Brier and reduced loss exposure, but removed profitable orders and lowered B_test PnL. | [3.3, done, score=281] Empirical Gc shrink variants underperformed the frozen hazard-only policy; selected hazard-only by w1-w4. | [3....
- 4: T6 constraints and T7 safe replacement did not preserve anchor PnL.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/4.1-<brief-description>/`.
