## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 2
**Hypothesis**:
Mechanism: Cross-fitted empirical action-value table that directly estimates PnL for each bid from q, side, and time context
Hypothesis: Direct conditional PnL avoids compounding q and Gc calibration errors that nearly cancelled validation wins and losses
Observable: Cross-fitted w1-w4 sum_pnl materially beats the best factorized EV policy and passes w5-w6
Conflicts: prior direct-PnL failed on first-minute data; the two-minute horizon changes fill distributions and this version uses cross-fitted discrete action values

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split dev --node-id 2`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split test --node-id 2`
- **Dataset info**: Read-only two-minute expected-return train/validation data; rolling folds w1-w4 tune, w5-w6 gate; validation milestone only
- **Baseline score**: 59.9
- **Current trunk score**: 89.99

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=89.99] Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/2-<brief-description>/`.
