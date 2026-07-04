# Executor report 3.7

## Idea
Settlement-safe Bayesian exponential-weights aggregation of frozen bid experts into a posterior consensus action.

## Changes
Added MAP, weighted mean, weighted median, and top-five consensus bids with daily full-information posterior updates.

## Baseline vs Result
Current trunk robust B_dev: 290.31. Candidate robust B_dev: 208.16607284730318. Candidate frozen online B_test sum_pnl: 37.00; previous best: 42.43.

## Score
208.16607284730318

## Insights
Weighted-median consensus was stable on all dev folds but increased B_test exposure and forced-loss PnL. Thompson's stochastic single-expert path remained better on B_test.