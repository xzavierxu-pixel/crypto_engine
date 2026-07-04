# Idea Tree

**Baseline**: 88.38 | **Trunk**: 88.38

## ROOT: Maximize chronological validation sum_pnl beyond 100 by jointly improving selected-side correctness probability models and bid policy; B_test protected from iteration; baseline B_dev 88.38/82.34 and B_test 27.44; no label-derived or sample-weight features; 4-hour real-run budget [DONE]

**Insight**: Children findings: [1, done, score=130] Feature-conditional LGBM correctness probability generalized across both B_dev folds: 129.97 and 129.79, with strict Gc floor 0.80 and min EV 0.005.

### 1: Mechanism: supervised selected-side correctness ensemble trained only on pre-validation rows
Hypothesis: feature-conditional q should reduce the calibration and regime errors that made isotonic policy gains collapse on B_test
Observable: positive worst-fold sum_pnl delta across both chronological B_dev folds and improved Brier score
Conflicts: prior policy-only search overfit B_dev; this counters via learned feature-conditional correctness [DONE] (score: 130)

**Insight**: LGBM q improved both B_dev folds and calibration, but frozen B_test sum_pnl was only 9.99; temporal generalization remains the binding bottleneck.

**Result**: Rejected after single B_test: 9.99 versus baseline 27.44 and target 100.

### 2: Mechanism: direct conditional PnL ranking model with chronological cross-fitting
Hypothesis: learning realized policy value directly can capture interactions omitted by factorized q-times-Gc EV
Observable: robust B_dev sum_pnl above the factorized baseline on both folds
Conflicts: none - attacks an objective-representation axis not previously tested [PRUNED]

**Insight**: [Pruned: Not launched after B_test exposed severe temporal drift; direct-PnL work needs additional forward dev folds before another frozen-test evaluation.]

### 3: Mechanism: joint direction and bid policy search using side-specific outcome models
Hypothesis: replacing fixed direction decisions can recover PnL lost by confident but systematically wrong side selection
Observable: higher direction accuracy and robust sum_pnl without changing the fill rule or universe
Conflicts: requires side-specific low-price data absent from current selected-side frame; defer until data availability is established [PRUNED]

**Insight**: [Pruned: Current selected-side dataset lacks counterfactual opposite-side chosen-low data required for valid joint direction/bid evaluation.]
