## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.2
**Hypothesis**:
Mechanism: chronological Bayesian optimization of a low-dimensional q-conditioned bid policy
Hypothesis: the profitable policy surface is nonlinear but low-dimensional; Gaussian-process search across nested B_dev folds can identify stable q-to-bid controls without refitting direction q
Observable: exceed 290.31 robust B_dev sum_pnl and improve every late-fold score under one frozen policy parameterization
Conflicts: node 2.2 found a single q gate insufficient; this searches a structured q-conditioned bid curve and exposure controls jointly

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_gc_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `milestone-only explicit runner`
- **Dataset info**: rolling weekly B_dev w1-w6 through 2026-04-10; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 221.78
- **Current trunk score**: 290.31

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=260.5] Settlement-safe daily direction updating remained stable on B_dev but scored -8.84 on B_test; full-universe accuracy 0.644 is too low for forced-wrong-fill economics. | [2, pending] Children findings: [2.1, done, score=288.9] Path signatures improved q and Gc Brier on all six folds, but robust B_dev remained below 290.31 and frozen B_test reached only 28.82; calibration improvement alone did not improve forced-fill PnL. | [2.2, done, score=290.8] Explicit q>=0.55 gating raised robust B_dev from 288.89 to 290.79 and holdout to 139.86, but removed only 195 B_test rows and improved frozen PnL merely from 28.82 to 29.27.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.2-<brief-description>/`.
