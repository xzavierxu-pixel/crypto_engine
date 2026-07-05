## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 7
**Hypothesis**:
Mechanism: Bid-specific joint-event classifiers with post-hoc monotone cumulative probabilities
Hypothesis: Separate classifiers let low- and high-bid fill events use different feature interactions that a single bid-conditioned tree underfits
Observable: w1-w4 sum_pnl exceeds 124.80 and both w5-w6 remain positive
Conflicts: node 5 used one shared bid-conditioned model; this increases action-specific capacity while preserving chronology

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split dev --node-id 7`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_optimize_500/run_policy_search.py --split test --node-id 7`
- **Dataset info**: Read-only two-minute expected-return train/validation data; rolling folds w1-w4 tune, w5-w6 gate; validation milestone only
- **Baseline score**: 59.9
- **Current trunk score**: 124.8

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=89.99] Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500. | [2, done, score=33.27] Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal. | [4, done, score=49.64] Correctness q alone selects accurate rows but cannot predict low-price winner fills; joint correctness-and-fill modeling is required. | [5, done, score=115.5] Children findings: [5.1, done, score=122.6] Boundary refinement improved B_dev to 122.60, but holdout stayed near 50; calibration is no longer the main bottleneck. | [5.2, done, score=113.4] One-cent interpolation reduced robustness; coarse action discretization is not the limiting factor. | [6, done, score=124.8] Blending direct and factorized profitable-fill probabilities added a small gain, but holdout remained near 50; model capacity by bid is the next bottleneck.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/7-<brief-description>/`.
