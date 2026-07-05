## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 5.1
**Hypothesis**:
Mechanism: Boundary-focused calibration search over the validated joint-event model
Hypothesis: The cycle-4 winner hit q extrapolation and H-scale boundaries, so extending those ranges can correct remaining conservative bias without retraining
Observable: w1-w4 robust sum_pnl exceeds 115.45 and w5-w6 stay positive
Conflicts: none - refines the validated node 5 mechanism using its observed boundary behavior

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split dev --node-id 5.1`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split test --node-id 5.1`
- **Dataset info**: Read-only two-minute expected-return train/validation data; rolling folds w1-w4 tune, w5-w6 gate; validation milestone only
- **Baseline score**: 59.9
- **Current trunk score**: 115.45

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=89.99] Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500. | [2, done, score=33.27] Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal. | [4, done, score=49.64] Correctness q alone selects accurate rows but cannot predict low-price winner fills; joint correctness-and-fill modeling is required. | [5, done, score=115.5] Joint profitable-fill modeling improved every tune week and holdout to 53.15; it is the first mechanism to bypass the q/Gc factorization successfully.
- 5: Joint profitable-fill modeling improved every tune week and holdout to 53.15; it is the first mechanism to bypass the q/Gc factorization successfully.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/5.1-<brief-description>/`.
