## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 4
**Hypothesis**:
Mechanism: Correctness-q admission gate jointly optimized with a fixed discrete bid, without a Gc factorization
Hypothesis: High-confidence direction subsets can monetize 73% accuracy while avoiding the 0.2265 Gc calibration error
Observable: Robust w1-w4 sum_pnl exceeds 89.99 and the frozen policy passes both w5-w6 weeks
Conflicts: node 2 showed p_side-only grouping is weak; this uses learned correctness ensembles as the admission signal

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split dev --node-id 4`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split test --node-id 4`
- **Dataset info**: Read-only two-minute expected-return train/validation data; rolling folds w1-w4 tune, w5-w6 gate; validation milestone only
- **Baseline score**: 59.9
- **Current trunk score**: 89.99

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=89.99] Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500. | [2, done, score=33.27] Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/4-<brief-description>/`.
