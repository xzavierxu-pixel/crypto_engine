## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1
**Hypothesis**:
Mechanism: End-to-end as-of-time shift rebuild with immutable source inputs and a three-minute selected-side low target
Hypothesis: Recomputing every direction and pricing input at market_t0+2m removes the representation mismatch that would arise from merely shifting labels while preserving the accepted policy mechanism
Observable: Frozen w1-w4 selection, positive w5-w6 gate, and one final B_test sum_pnl directly comparable with 42.43
Conflicts: none - attacks an unexplored decision-time axis while retaining the validated 20260703 policy

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_shift/run_two_minute_pipeline.py --phase pretest`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260704_two_minute_shift/run_two_minute_pipeline.py --phase btest`
- **Dataset info**: Existing raw Binance and Polymarket trade data; chronological pretest w1-w6 with w1-w4 selection and w5-w6 untouched gate; frozen Apr-May validation B_test once
- **Baseline score**: 218.5
- **Current trunk score**: 218.5

Use B_dev for final experiment scoring. Do NOT use B_test.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1-<brief-description>/`.
