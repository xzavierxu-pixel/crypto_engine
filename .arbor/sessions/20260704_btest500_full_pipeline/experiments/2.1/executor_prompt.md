## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 2.1
**Hypothesis**:
Mechanism: multiscale Polymarket path-signature and liquidity-shape feature bank
Hypothesis: fixed-window moments discard ordering, acceleration, side-switching, and price-impact structure that separates profitable fills from forced wrong fills
Observable: q AUC and late-fold Gc Brier improve together, with w1-w4 robust sum_pnl above 290.31 and positive w5-w6 uplift
Conflicts: prior node 1 used count/last/mean/min/max/range/std/slope; this counters by representing event order, duration-weighted path shape, impact, and selected-opposite lead-lag

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_gc_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `milestone-only explicit runner`
- **Dataset info**: rolling weekly B_dev w1-w6 through 2026-04-10; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 221.78
- **Current trunk score**: 290.31

Use B_dev for final experiment scoring. Do NOT use B_test.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/2.1-<brief-description>/`.
