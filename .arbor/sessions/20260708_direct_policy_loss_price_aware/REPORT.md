# Research Report - Direct Policy Loss Price Aware 0708

Implemented `arbor_research_0708_v2.md` as an isolated Arbor session on the 1:08 market-order universe. The deploy feature manifest stayed fixed for DP0-DP4; DP5 added research-only price/base features; DP6 added research-only price-context features; B_test was recorded once per named node.

## Final B_test

| experiment_id                     | btest_status    |   btest_sum_pnl | holdout_passed   |
|:----------------------------------|:----------------|----------------:|:-----------------|
| DP0_frame_qa_v2                   | qa_once         |        nan      |                  |
| DP1_replay_v1_reference           | evaluated_once  |         50.1197 | True             |
| DP2_soft_reward_ce                | evaluated_once  |        146.587  | True             |
| DP3_cross_side_regret_weighted_ce | evaluated_once  |        142.832  | True             |
| DP5_xgb_direct_pnl_objective      | evaluated_once  |         86.3519 | True             |
| DP6_optional_price_context_model  | evaluated_once  |        127.772  | True             |
| DP4_price_aware_utility_layer     | evaluated_once  |         58.6323 | True             |
| DP7_stable_universe_diagnostic    | diagnostic_once |        nan      |                  |
