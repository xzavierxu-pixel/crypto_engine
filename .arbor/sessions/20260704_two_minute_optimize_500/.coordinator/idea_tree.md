# Idea Tree

**Baseline**: 59.9 | **Trunk**: 124.8

## ROOT: Optimize the two-minute shifted BTCUSDT Polymarket pricing strategy for validation sum_pnl, targeting at least 500. Use the immutable datasets and folds from 20260704_two_minute_shift; maximize B_dev sum_pnl over w1-w4 with w5-w6 as untouched gate; validation/B_test is milestone-only after candidate freeze; up to 100 cycles and 6-hour GPU/CPU real-run budget; auto performance-first; do not overwrite existing data, deploy, configs, or sessions; no commits or dependency installs. [DONE]

**Insight**: Children findings: [1, done, score=89.99] Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500. | [2, done, score=33.27] Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal. | [4, done, score=49.64] Correctness q alone selects accurate rows but cannot predict low-price winner fills; joint correctness-and-fill modeling is required. | [5, done, score=115.5] Children findings: [5.1, done, score=122.6] Boundary refinement improved B_dev to 122.60, but holdout stayed near 50; calibration is no longer the main bottleneck. | [5.2, done, score=113.4] One-cent interpolation reduced robustness; coarse action discretization is not the limiting factor. | [6, done, score=124.8] Blending direct and factorized profitable-fill probabilities added a small gain, but holdout remained near 50; model capacity by bid is the next bottleneck. | [7, done, score=94.3] Bid-specific models underperformed the shared bid-conditioned model; splitting actions reduced statistical strength.

### 1: Mechanism: Broad constrained search over q ensembles, Gc floors, EV thresholds, and bid-grid transformations
Hypothesis: The current policy operates in a mismatched action region, evidenced by 14.7% order coverage and a 0.2265 submitted-fill gap despite 73.65% direction accuracy
Observable: w1-w4 sum_pnl exceeds 59.9 and w5-w6 remain positive without changing direction rows
Conflicts: prior frozen-policy result scored -0.75 on validation; this counters by re-optimizing the shortened-window action space [DONE] (score: 89.99)

**Insight**: Gc power remapping and LGBM q reduced the fill mismatch; all tune weeks were positive and holdout reached 35.85, but factorized EV remains far below 500.

**Result**: w1-w4 sum_pnl 89.99; w5-w6 sum_pnl 35.85; gate passed.

**Branch**: 2mins

### 2: Mechanism: Cross-fitted empirical action-value table that directly estimates PnL for each bid from q, side, and time context
Hypothesis: Direct conditional PnL avoids compounding q and Gc calibration errors that nearly cancelled validation wins and losses
Observable: Cross-fitted w1-w4 sum_pnl materially beats the best factorized EV policy and passes w5-w6
Conflicts: prior direct-PnL failed on first-minute data; the two-minute horizon changes fill distributions and this version uses cross-fitted discrete action values [DONE] (score: 33.27)

**Insight**: Empirical p_side action tables cannot isolate profitable orders; fixed-bid economics are negative without a stronger correctness admission signal.

**Result**: w1-w4 sum_pnl 33.27, below 89.99; mechanism rejected.

**Branch**: 2mins

### 3: Mechanism: Recency-weighted monotone fill calibration with group-specific correction by side, q bucket, and session
Hypothesis: Correcting the 0.2265 Gc overprediction at the decision boundary should admit bids whose realized fill probability supports positive EV
Observable: Fill Brier and reliability improve on every tune fold while sum_pnl rises and the holdout gate remains positive
Conflicts: prior static H2 calibration remained biased; this counters with walk-forward local calibration instead of another global hazard fit [PENDING]

### 4: Mechanism: Correctness-q admission gate jointly optimized with a fixed discrete bid, without a Gc factorization
Hypothesis: High-confidence direction subsets can monetize 73% accuracy while avoiding the 0.2265 Gc calibration error
Observable: Robust w1-w4 sum_pnl exceeds 89.99 and the frozen policy passes both w5-w6 weeks
Conflicts: node 2 showed p_side-only grouping is weak; this uses learned correctness ensembles as the admission signal [DONE] (score: 49.64)

**Insight**: Correctness q alone selects accurate rows but cannot predict low-price winner fills; joint correctness-and-fill modeling is required.

**Result**: w1-w4 sum_pnl 49.64, below 89.99; q-gate mechanism rejected.

**Branch**: 2mins

### 5: Mechanism: Monotone joint-event gradient booster for P(correct and winner_low<=bid | X,bid) with direct bid-level EV
Hypothesis: Modeling the profitable fill event jointly removes conditional-Gc error and learns which accurate directions can fill at favorable prices
Observable: Chronological w1-w4 sum_pnl exceeds 89.99 with improved joint-event Brier and positive w5-w6
Conflicts: nodes 2 and 4 separated admission from fill; this directly models their interaction at each bid [DONE] (score: 115.5)

**Insight**: Children findings: [5.1, done, score=122.6] Boundary refinement improved B_dev to 122.60, but holdout stayed near 50; calibration is no longer the main bottleneck. | [5.2, done, score=113.4] One-cent interpolation reduced robustness; coarse action discretization is not the limiting factor.

**Result**: w1-w4 sum_pnl 115.45; w5-w6 53.15; gate passed.

**Branch**: 2mins

#### 5.1: Mechanism: Boundary-focused calibration search over the validated joint-event model
Hypothesis: The cycle-4 winner hit q extrapolation and H-scale boundaries, so extending those ranges can correct remaining conservative bias without retraining
Observable: w1-w4 robust sum_pnl exceeds 115.45 and w5-w6 stay positive
Conflicts: none - refines the validated node 5 mechanism using its observed boundary behavior [DONE] (score: 122.6)

**Insight**: Boundary refinement improved B_dev to 122.60, but holdout stayed near 50; calibration is no longer the main bottleneck.

**Result**: w1-w4 122.60; w5-w6 50.70; gate passed.

**Branch**: 2mins

#### 5.2: Mechanism: Monotone interpolation of joint-event probabilities from 5-cent training actions to the native 1-cent bid grid
Hypothesis: Removing coarse action quantization recovers PnL without changing the learned event ranking
Observable: w1-w4 sum_pnl exceeds 122.60 and w5-w6 remain positive
Conflicts: none - isolates action discretization within validated node 5 [DONE] (score: 113.4)

**Insight**: One-cent interpolation reduced robustness; coarse action discretization is not the limiting factor.

**Result**: w1-w4 113.43, below 122.60; rejected.

**Branch**: 2mins

### 6: Mechanism: Probability-level ensemble of direct joint-event boosting and factorized q-times-H2 fill estimates
Hypothesis: Blending structurally different profitable-fill forecasts reduces model-specific calibration and ranking errors
Observable: w1-w4 robust sum_pnl exceeds 122.60 with positive w5-w6
Conflicts: node 1 factorization and node 5 joint modeling each worked partially; this combines rather than replaces their complementary signals [DONE] (score: 124.8)

**Insight**: Blending direct and factorized profitable-fill probabilities added a small gain, but holdout remained near 50; model capacity by bid is the next bottleneck.

**Result**: w1-w4 124.80; w5-w6 50.20; gate passed.

**Branch**: 2mins

### 7: Mechanism: Bid-specific joint-event classifiers with post-hoc monotone cumulative probabilities
Hypothesis: Separate classifiers let low- and high-bid fill events use different feature interactions that a single bid-conditioned tree underfits
Observable: w1-w4 sum_pnl exceeds 124.80 and both w5-w6 remain positive
Conflicts: node 5 used one shared bid-conditioned model; this increases action-specific capacity while preserving chronology [DONE] (score: 94.3)

**Insight**: Bid-specific models underperformed the shared bid-conditioned model; splitting actions reduced statistical strength.

**Result**: w1-w4 94.30; below 124.80; rejected.

**Branch**: 2mins
