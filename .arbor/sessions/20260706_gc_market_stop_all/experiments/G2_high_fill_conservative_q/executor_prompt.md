## Codebase

Working directory: C:\Users\ROG\Desktop\crypto_engine_version1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1.3
**Hypothesis**:
Mechanism: Joint high-fill action constraint and shrinkage of selected-side correctness toward one half
Hypothesis: Conservative q can reject expensive high-floor bids whose apparent EV depends on overconfident correctness, reducing forced-wrong loss while preserving filled winners
Observable: The w1-w4 robust winner passes positive w5-w6 and improves win/loss decomposition relative to G1
Conflicts: G0 showed Gc optimism rather than q-only error; this combines orthogonal q conservatism with the mandated fill constraint

## Evaluation Info

- **Evaluation command (B_dev)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 1.3 --split dev`
- **Evaluation command (B_test, do not use for routine experiments)**: `rtk python C:\Users\ROG\Desktop\crypto_engine_version1/.arbor/sessions/20260706_gc_market_stop_all/run_experiments.py --node 1.3 --split test`
- **Dataset info**: w1-w4 tune; w5-w6 untouched gate; frozen B_test 2026-04-11..2026-05-10
- **Baseline score**: 42.43
- **Current trunk score**: 42.43

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, pending] Children findings: [1.1, done, score=42.43] Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap.
- 1: Children findings: [1.1, done, score=42.43] Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1.3-<brief-description>/`.
