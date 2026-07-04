## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 7.1
**Hypothesis**:
Mechanism: prequential full-universe direction model retrained daily on all markets settled before the current UTC day
Hypothesis: full-universe value is real on rolling folds but static direction accuracy drops at the final boundary; online direction updates should preserve the enlarged action space while adapting side selection
Observable: exceed node 7 on w1-w4 robust sum_pnl and both untouched weeks, with improved daily direction accuracy after day one
Conflicts: node 8 adapted q only inside the legacy universe; this adapts the upstream side decision that determines both correctness and which low distribution applies

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `disabled until robust milestone`
- **Dataset info**: rolling expanding weekly folds w1-w6; w1-w4 selection, w5-w6 untouched gate, final month protected
- **Baseline score**: 221.78
- **Current trunk score**: 221.78

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=266.3] Children findings: [1.1, done, score=290.3] Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved. | [2, done, score=241.4] Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2. | [3, done, score=166.1] Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side. | [4, done, score=68.22] Direct PnL regression lowered bids but failed to learn stable action ranking: tune sum 68.22 with a negative week, far below probability-factorized policies. | [5, done, score=272.9] Low-variance empirical Gc stratified by p_side beat the contextual hazard on untouched PnL (148.22 vs 128.49); all-history p-bin CDF improved Gc Brier on all six folds, validating selection-overconfidence as the bottleneck. | [6, done, score=199.5] Feature minimization reduced som...
- 7: Full-universe direction coverage unlocked substantial additional value: tune 283.59 and untouched 173.10, with both holdout weeks above 85 PnL; the legacy 70% mask was constraining the action space.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/7.1-<brief-description>/`.
