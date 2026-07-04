# Executor report 3.5

## Idea
Hierarchical Bayesian posterior over bid rewards with partial pooling across q, side, hour, and bid cells.

## Changes
Added time-decayed conjugate-style reward aggregates, q/side/hour hierarchical shrinkage, posterior standard errors, and analytic-EV/posterior blending over 84 legal bids.

## Baseline vs Result
Current trunk robust B_dev: 290.31. Candidate robust B_dev: 197.35939355257207. Candidate frozen B_test sum_pnl: 26.33; previous best: 42.43.

## Score
197.35939355257207

## Insights
The selected policy used all history, q-only pooling, 25% posterior blend, and a 0.5 posterior uncertainty penalty. More contextual hierarchy was rejected by B_dev.