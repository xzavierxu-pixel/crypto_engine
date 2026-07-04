## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1.1
**Hypothesis**:
Mechanism: monotone residual Gc calibrator trained on bid-expanded correct orders using base hazard CDF plus pre-decision trade state
Hypothesis: the remaining month drift is miscalibrated conditional fill probability; conditioning the hazard CDF on observed liquidity should improve bid ranking without changing fill semantics
Observable: lower Gc Brier on every rolling fold and improve robust w1-w4 sum_pnl while preserving positive w5-w6 uplift
Conflicts: none - node 1 improved q while holding Gc fixed, so this attacks the remaining probability component

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py`
- **Evaluation command (B_test, do not use for routine experiments)**: `disabled until robust milestone`
- **Dataset info**: rolling expanding weekly folds w1-w6; w1-w4 selection, w5-w6 untouched gate, final month protected
- **Baseline score**: 221.78
- **Current trunk score**: 221.78

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=266.3] Strictly pre-decision Polymarket trade features improved q Brier/logloss on all six folds and raised w1-w4 sum_pnl from 144.14 to 266.34; untouched w5-w6 improved by 42.78. Broad BTC expansion was inconsistent and not selected.
- 1: Strictly pre-decision Polymarket trade features improved q Brier/logloss on all six folds and raised w1-w4 sum_pnl from 144.14 to 266.34; untouched w5-w6 improved by 42.78. Broad BTC expansion was inconsistent and not selected.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1.1-<brief-description>/`.
