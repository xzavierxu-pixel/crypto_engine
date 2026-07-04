## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 4
**Hypothesis**:
Mechanism: bid-expanded direct realized-PnL regression with conservative quantile objective over legal pre-decision market state
Hypothesis: q and Gc probability errors compound under distribution shift; directly learning the asymmetric payoff surface should favor lower bids that retain upside while limiting forced wrong-fill losses
Observable: exceed node 1 on robust w1-w4 sum_pnl and both untouched w5-w6 weeks with materially lower mean submitted bid
Conflicts: node 2 modeled a joint probability then reconstructed EV; direct payoff supervision removes probability-calibration dependence entirely

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `disabled until robust milestone`
- **Dataset info**: rolling expanding weekly folds w1-w6; w1-w4 selection, w5-w6 untouched gate, final month protected
- **Baseline score**: 221.78
- **Current trunk score**: 221.78

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=266.3] Children findings: [1.1, done, score=290.3] Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved. | [2, done, score=241.4] Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2. | [3, done, score=166.1] Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/4-<brief-description>/`.
