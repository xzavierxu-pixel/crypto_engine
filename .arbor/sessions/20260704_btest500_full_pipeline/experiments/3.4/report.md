# Executor report 3.4

## Idea
Cross-fitted quantile contextual bandit for downside-aware admission of the validated analytic bid.

## Changes
Added full-information mean/q30/median action-reward models and B_dev-only retention selection over the fixed raw_tree_blend analytic bid.

## Baseline vs Result
Current trunk robust B_dev: 290.31. Candidate robust B_dev: 212.27530699783776. Candidate frozen B_test sum_pnl: 21.45; previous best: 42.43.

## Score
212.27530699783776

## Insights
The winner retained 90% of orders using predicted mean reward per unit bid. The filter remained positive on all six dev folds but failed to generalize to B_test.