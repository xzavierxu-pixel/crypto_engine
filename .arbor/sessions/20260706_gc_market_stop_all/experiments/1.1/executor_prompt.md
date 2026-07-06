## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1.1
**Hypothesis**:
Mechanism: Exact frozen anchor replay using raw_tree_blend, Gc floor 0.85, and min_ev 0.02
Hypothesis: Reusing the accepted preparation and bid functions on the frozen train/test files should reproduce 42.43 and validate the environment before any search
Observable: B_test sum_pnl matches 42.43 within floating-point tolerance and all required accounting fields are emitted
Conflicts: none - mandatory preflight before changing policy mechanics

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 1.1 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 1.1 --split test`
- **Dataset info**: w1-w4 tune; w5-w6 untouched gate; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 218.5
- **Current trunk score**: 218.5

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=42.43] Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap. | [1.2, done, score=202.9] Nominal Gc>=0.90 raised realized submitted correct-fill to the quarantined diagnostic value 0.801 but only one of four tune weeks beat anchor; selection gate failed. | [1.3, done, score=202.9] q shrinkage did not win; shrink=0 remained best and only one of four tune weeks beat anchor. | [1.4, done, score=198.9] 15% empirical p_side-bin CDF blend improved grid Brier but zero of four tune weeks beat anchor; calibration improvement did not convert to PnL. | [2, pending] Children findings: [2.1, done, score=0] M0 found legal market prices for the majority of accepted rows with zero post-cutoff joins; B_test m coverage is recorded in QA. | [2.2, done, score=390.5] Market EV selected raw_tree_blend with tau=0; w5-w6 were positive and B_test sum_pnl was 122.63 under market-order semantics. | [2.3, done, score=390.5] On 5,164 rows with legal m, market PnL 122.63 beat the paired limit anchor 45.41 by 77.22. | [2.4, done, score=236.9] Sel...
- 1: Children findings: [1.1, done, score=42.43] Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap. | [1.2, done, score=202.9] Nominal Gc>=0.90 raised realized submitted correct-fill to the quarantined diagnostic value 0.801 but only one of four tune weeks beat anchor; selection gate failed. | [1.3, done, score=202.9] q shrinkage did not win; shrink=0 remained best and only one of four tune weeks beat anchor. | [1.4, done, score=198.9] 15% empirical p_side-bin CDF blend improved grid Brier but zero of four tune weeks beat anchor; calibration improvement did not convert to PnL.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1.1-<brief-description>/`.
