# Executor report 3.3

## Idea
Analytic-EV-anchored contextual action-value ensemble with lower-confidence-bound ranking.

## Changes
Added five-member Huber LightGBM action-value ensemble, analytic EV anchoring, disagreement penalty, and bounded action deviation.

## Baseline vs Result
Current trunk robust B_dev: 290.31. Candidate robust B_dev: 190.75222443628013. Candidate B_test sum_pnl: 24.30; previous best: 42.43.

## Score
190.75222443628013

## Insights
The frozen winner used d0, so action deviations were rejected by B_dev selection. Anchoring stabilized standalone contextual regression but did not improve generalization.