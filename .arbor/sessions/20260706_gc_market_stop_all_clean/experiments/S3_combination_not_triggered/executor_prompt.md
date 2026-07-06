## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.4
**Hypothesis**:
Mechanism: Conditional composition of the best validated stop with a winning high-fill or market-entry policy
Hypothesis: Composition is justified only if S1 independently shows stable positive holding-management value, preventing a failed stop from contaminating a strong entry policy
Observable: Launch only when S1 passes its rolling and B_test criteria; otherwise persist an explicit not-triggered artifact with zero B_test reads
Conflicts: S1 failed all four tune-week deltas, so the prerequisite is false and this node must be pruned

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 3.4 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 3.4 --split test`
- **Dataset info**: w1-w4 tune; w5-w6 untouched gate; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 218.5
- **Current trunk score**: 218.5

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=42.43] Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap. | [1.2, done, score=202.9] Nominal Gc>=0.90 raised realized submitted correct-fill to the quarantined diagnostic value 0.801 but only one of four tune weeks beat anchor; selection gate failed. | [1.3, done, score=202.9] q shrinkage did not win; shrink=0 remained best and only one of four tune weeks beat anchor. | [1.4, done, score=198.9] 15% empirical p_side-bin CDF blend improved grid Brier but zero of four tune weeks beat anchor; calibration improvement did not convert to PnL. | [2, pending] Children findings: [2.1, done, score=0] M0 found legal market prices for the majority of accepted rows with zero post-cutoff joins; B_test m coverage is recorded in QA. | [2.2, done, score=390.5] Market EV selected raw_tree_blend with tau=0; w5-w6 were positive and B_test sum_pnl was 122.63 under market-order semantics. | [2.3, done, score=390.5] On 5,164 rows with legal m, market PnL 122.63 beat the paired limit anchor 45.41 by 77.22. | [2.4, done, score=236.9] Sel...
- 3: Children findings: [3.1, done, score=150.1] Fixed stop cut B_test losses from -316.44 to -176.83 but false-stopped 227 winners, reducing wins enough for net PnL 20.91; conservative/pressure variants were worse. | [3.2, done, score=170.7] Best stop was fixed 0.20, yet zero tune weeks beat the identical-order anchor; S1 failed before holdout/B_test. | [3.3, done, score=132.2] Take-profit overlays further reduced rolling PnL and the S1 prerequisite had already failed.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.4-<brief-description>/`.
