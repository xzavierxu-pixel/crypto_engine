## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.4
**Hypothesis**:
Mechanism: cross-fitted quantile contextual bandit for downside-aware admission of the validated analytic bid
Hypothesis: node 3.3 showed contextual action changes are unstable, but context may still rank the downside of a fixed validated action; lower-quantile realized-PnL prediction should reject forced-loss regimes without perturbing bid selection
Observable: robust w1-w4 sum_pnl exceeds 290.31, all six folds remain positive, and loss_pnl_sum falls while retaining most win_pnl_sum
Conflicts: node 2.2 q-only gating barely moved B_test and node 3.3 selected zero action deviation; this models the full realized reward distribution of the fixed order rather than correctness alone

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_gc_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `milestone-only explicit runner`
- **Dataset info**: rolling weekly B_dev w1-w6 through 2026-04-10; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 221.78
- **Current trunk score**: 290.31

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=260.5] Settlement-safe daily direction updating remained stable on B_dev but scored -8.84 on B_test; full-universe accuracy 0.644 is too low for forced-wrong-fill economics. | [2, pending] Children findings: [2.1, done, score=288.9] Path signatures improved q and Gc Brier on all six folds, but robust B_dev remained below 290.31 and frozen B_test reached only 28.82; calibration improvement alone did not improve forced-fill PnL. | [2.2, done, score=290.8] Explicit q>=0.55 gating raised robust B_dev from 288.89 to 290.79 and holdout to 139.86, but removed only 195 B_test rows and improved frozen PnL merely from 28.82 to 29.27. | [3, pending] Children findings: [3.1, done, score=17.13] Full-information contextual action-value regression generalized positively to w5-w6 (47.99) but collapsed on w1-w4 (robust 17.13), indicating severe reward-model instability and action-value shrinkage. | [3.2, done, score=164.3] Gaussian-process optimization of a q-conditioned bid curve materially underperformed the 290.31 B_dev trunk; low-dimensional policy tuning cannot repair the fixed q/Gc economics. | [3.3, done, score=190.8] An...
- 3: Children findings: [3.1, done, score=17.13] Full-information contextual action-value regression generalized positively to w5-w6 (47.99) but collapsed on w1-w4 (robust 17.13), indicating severe reward-model instability and action-value shrinkage. | [3.2, done, score=164.3] Gaussian-process optimization of a q-conditioned bid curve materially underperformed the 290.31 B_dev trunk; low-dimensional policy tuning cannot repair the fixed q/Gc economics. | [3.3, done, score=190.8] Analytic anchoring prevented the catastrophic action-value collapse but did not beat the factorized-EV trunk: robust B_dev 190.75, holdout 100.63, frozen B_test 24.30. The selected d0 variant never changed the analytic action, showing the contextual reward signal was useful only as a conservative score perturbation.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.4-<brief-description>/`.
