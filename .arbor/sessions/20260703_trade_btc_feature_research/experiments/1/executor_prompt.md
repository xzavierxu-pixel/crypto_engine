## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1
**Hypothesis**:
Mechanism: decision-time cross-market feature bank combining Polymarket pre-decision trade microstructure with expanded BTC second-level regime signals
Hypothesis: conditional order value decays across months because q and Gc omit contemporaneous liquidity/regime state; legal trailing features should reduce q/Gc calibration drift
Observable: improve w1-w4 robust sum_pnl and q/Gc Brier, then remain positive and beat baseline on untouched w5-w6
Conflicts: none - prior nodes varied temporal weighting and policy uncertainty but did not add new market-state observations

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `disabled until robust milestone`
- **Dataset info**: rolling expanding weekly folds w1-w6; w1-w4 selection, w5-w6 untouched gate, final month protected
- **Baseline score**: 221.78
- **Current trunk score**: 221.78

Use B_dev for final experiment scoring. Do NOT use B_test.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1-<brief-description>/`.
