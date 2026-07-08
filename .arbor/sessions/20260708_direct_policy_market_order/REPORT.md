# Research Report — Direct Policy Market Order 0708

Implemented `arbor_research_0708.md` as a standalone Arbor session on the full `1:08` market-order universe. The deploy direction feature manifest stayed fixed at 569 columns, no deploy artifact or live config changed, and B_test was recorded once for every named node.

## Final B_test

| experiment_id               | btest_status   |   btest_sum_pnl | holdout_passed   |
|:----------------------------|:---------------|----------------:|:-----------------|
| DP0_frame_qa                | qa_once        |        nan      |                  |
| DP1_static_ev_baseline      | evaluated_once |         50.1197 | True             |
| DP2_direct_policy_regret_ce | evaluated_once |        137.081  | True             |
