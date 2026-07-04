## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 6
**Hypothesis**:
Mechanism: covariate-shift-resistant minimal q model using only p_side and pre-decision trade state, fitted before a temporal calibration tail
Hypothesis: the 576-feature q model and trade augmentation degrade on the final month because stale BTC relationships dominate; removing shifting features and calibrating on the latest train tail should preserve only transportable error signals
Observable: improve q Brier and PnL over raw p_side on w1-w4 and every untouched w5-w6 fold using both hazard and empirical Gc
Conflicts: node 1's broad trade model improved rolling folds but final-month q Brier worsened to 0.2088; this counters by feature minimization and temporal calibration

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `disabled until robust milestone`
- **Dataset info**: rolling expanding weekly folds w1-w6; w1-w4 selection, w5-w6 untouched gate, final month protected
- **Baseline score**: 221.78
- **Current trunk score**: 221.78

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=266.3] Children findings: [1.1, done, score=290.3] Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved. | [2, done, score=241.4] Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2. | [3, done, score=166.1] Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side. | [4, done, score=68.22] Direct PnL regression lowered bids but failed to learn stable action ranking: tune sum 68.22 with a negative week, far below probability-factorized policies. | [5, done, score=272.9] Low-variance empirical Gc stratified by p_side beat the contextual hazard on untouched PnL (148.22 vs 128.49); all-history p-bin CDF improved Gc Brier on all six folds, validating selection-overconfidence as the bottleneck.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/6-<brief-description>/`.
