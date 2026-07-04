# Idea Tree

**Baseline**: 88.38 | **Trunk**: 88.38

## ROOT: Maximize B_dev sum_pnl, preserve frozen final-month B_test, beat valid baseline 27.44 without leakage; stage1_sample_weight forbidden; 10 cycles/4 hours; isolated commits allowed, no merge to pmdata. [DONE]

**Insight**: Children findings: [1, pending] Children findings: [1.1, done, score=94.18] A 25% calibration-only empirical-CDF blend reduced submitted fill calibration gap from 0.1733 to 0.1310 and improved B_dev sum_pnl from 88.38 to 94.18; gains came with higher order coverage 0.834. | [1.2, done, score=89.79] The beta upper-cap correction reduced the fill calibration gap to 0.1397 but improved B_dev only 1.41, underperforming the smoother empirical blend. | [1.3, done, score=80.24] 0.10 bins over-pooled and selected min_ev 0.01; B_dev fell to 80.24 and calibration gap returned to 0.1737. | [1.4, done, score=86.66] 0.20 bins reduced the calibration gap to 0.1622 but lost PnL; p_side conditioning is useful and aggressive pooling removes signal. | [1.5, done, score=95.84] Fine 0.025 bins with 25% empirical blend improved B_dev to 95.84 and reduced the calibration gap to 0.1296; this is the best current node. | [2, pending] Children findings: [2.1, done, score=51.87] Calibration-tail empirical Gc discarded useful H2 row-level ranking and cut B_dev PnL nearly in half despite high order coverage. | [2.2, done, score=54.03] Fit-window empirical Gc achieved a low calibration gap 0.0928 but poor Pn...

### 1: Mechanism: Calibration-corrected Gc family
Hypothesis: Correcting the systematic submitted-fill overprediction should restore EV ranking because B_dev and B_test both show about 0.17 calibration gaps.
Observable: Higher B_dev sum_pnl than 88.38 with lower submitted-fill calibration gap and unchanged direction universe.
Conflicts: none - attacks the documented Gc calibration bottleneck. [PENDING]

**Insight**: Children findings: [1.1, done, score=94.18] A 25% calibration-only empirical-CDF blend reduced submitted fill calibration gap from 0.1733 to 0.1310 and improved B_dev sum_pnl from 88.38 to 94.18; gains came with higher order coverage 0.834. | [1.2, done, score=89.79] The beta upper-cap correction reduced the fill calibration gap to 0.1397 but improved B_dev only 1.41, underperforming the smoother empirical blend. | [1.3, done, score=80.24] 0.10 bins over-pooled and selected min_ev 0.01; B_dev fell to 80.24 and calibration gap returned to 0.1737. | [1.4, done, score=86.66] 0.20 bins reduced the calibration gap to 0.1622 but lost PnL; p_side conditioning is useful and aggressive pooling removes signal. | [1.5, done, score=95.84] Fine 0.025 bins with 25% empirical blend improved B_dev to 95.84 and reduced the calibration gap to 0.1296; this is the best current node.

#### 1.1: Mechanism: Cross-fitted p_side-bin empirical-CDF blend
Hypothesis: Blending H2 Gc with calibration-only empirical CDFs should correct systematic overprediction while retaining row-level ranking.
Observable: B_dev sum_pnl exceeds 88.38 and submitted-fill calibration gap falls below 0.17.
Conflicts: none - prior historical runs were scored on B_test; this uses a clean pre-validation holdout. [DONE] (score: 94.18)

**Insight**: A 25% calibration-only empirical-CDF blend reduced submitted fill calibration gap from 0.1733 to 0.1310 and improved B_dev sum_pnl from 88.38 to 94.18; gains came with higher order coverage 0.834.

**Result**: B_dev +5.80; 2,138 orders; wrong_fill_forced=1.0; fixed direction universe; no forbidden feature.

#### 1.2: Mechanism: Beta-posterior upper cap on p_side-bin Gc
Hypothesis: Capping overconfident H2 curves with calibration-only empirical uncertainty bounds should remove false-positive EV orders without collapsing coverage.
Observable: B_dev sum_pnl exceeds 88.38 with reduced loss_pnl_sum magnitude and at least 100 orders.
Conflicts: none - tests conservative probability correction rather than blending. [DONE] (score: 89.79)

**Insight**: The beta upper-cap correction reduced the fill calibration gap to 0.1397 but improved B_dev only 1.41, underperforming the smoother empirical blend.

**Result**: B_dev +1.41; 1,570 orders; wrong_fill_forced=1.0; fixed direction universe; no forbidden feature.

#### 1.3: Mechanism: Coarse p_side-bin empirical-CDF blend
Hypothesis: Wider 0.10 bins should lower calibration variance while a blend preserves H2 row-level structure.
Observable: B_dev sum_pnl and fill calibration improve beyond the 0.05-bin blend or reveal a bias-variance limit.
Conflicts: node 1.1 favored 0.05 bins; this tests whether its gain is variance-limited. [DONE] (score: 80.24)

**Insight**: 0.10 bins over-pooled and selected min_ev 0.01; B_dev fell to 80.24 and calibration gap returned to 0.1737.

**Result**: B_dev -8.14 vs baseline; 1305 orders.

#### 1.4: Mechanism: Pooled 0.20 p_side-bin empirical-CDF blend
Hypothesis: Aggressive pooling should expose whether p_side conditioning adds real Gc signal or only tail noise.
Observable: B_dev sum_pnl remains competitive with a smaller calibration gap and fewer fallback bins.
Conflicts: node 1.1 used narrower bins; this intentionally removes most conditional granularity. [DONE] (score: 86.66)

**Insight**: 0.20 bins reduced the calibration gap to 0.1622 but lost PnL; p_side conditioning is useful and aggressive pooling removes signal.

**Result**: B_dev -1.72; 1626 orders.

#### 1.5: Mechanism: Fine 0.025 p_side-bin empirical-CDF blend
Hypothesis: Finer conditioning may capture heterogeneous fill curves that the 0.05 bins average away despite higher variance.
Observable: B_dev sum_pnl exceeds 94.18 without increasing the submitted-fill calibration gap.
Conflicts: node 1.1 supports blending; this probes the opposite granularity direction from nodes 1.3 and 1.4. [DONE] (score: 95.84)

**Insight**: Fine 0.025 bins with 25% empirical blend improved B_dev to 95.84 and reduced the calibration gap to 0.1296; this is the best current node.

**Result**: B_dev +7.46; 2160 orders; order coverage 0.8424.

### 2: Mechanism: Shrunk empirical bid-policy family
Hypothesis: Directly estimating realized bid utility with hierarchical shrinkage should outperform plug-in EV when Gc probability errors dominate decisions.
Observable: Higher multi-split B_dev sum_pnl with stable order coverage and no validation-fitted parameters.
Conflicts: none - changes the decision representation rather than only recalibrating Gc. [PENDING]

**Insight**: Children findings: [2.1, done, score=51.87] Calibration-tail empirical Gc discarded useful H2 row-level ranking and cut B_dev PnL nearly in half despite high order coverage. | [2.2, done, score=54.03] Fit-window empirical Gc achieved a low calibration gap 0.0928 but poor PnL 54.03, proving aggregate calibration alone is insufficient for bid ranking. | [2.3, done, score=29.23] Coarse direct empirical bins produced both poor calibration and poor PnL; direct empirical Gc is not competitive without H2 blending.

#### 2.1: Mechanism: Calibration-tail empirical p_side CDF policy
Hypothesis: A fully empirical Gc eliminates neural calibration bias and may improve bids when recent data represents the next regime.
Observable: B_dev sum_pnl exceeds 88.38 with finite fill reliability and stable order coverage.
Conflicts: node 1.1 retained H2 ranking; this removes the neural Gc entirely. [DONE] (score: 51.87)

**Insight**: Calibration-tail empirical Gc discarded useful H2 row-level ranking and cut B_dev PnL nearly in half despite high order coverage.

**Result**: B_dev 51.87; 2278 orders.

#### 2.2: Mechanism: Fit-window empirical p_side CDF policy
Hypothesis: A larger historical fit window should reduce empirical-CDF variance enough to offset temporal drift.
Observable: B_dev sum_pnl beats the calibration-tail empirical policy and maintains at least 100 orders.
Conflicts: node 2.1 prioritizes recency; this prioritizes sample size. [DONE] (score: 54.03)

**Insight**: Fit-window empirical Gc achieved a low calibration gap 0.0928 but poor PnL 54.03, proving aggregate calibration alone is insufficient for bid ranking.

**Result**: B_dev 54.03; 2381 orders.

#### 2.3: Mechanism: Coarse-bin fit-window empirical CDF policy
Hypothesis: Combining the large fit window with 0.10 bins should suppress sparse-bin variance in direct empirical bids.
Observable: B_dev sum_pnl exceeds both direct empirical siblings with lower downside loss.
Conflicts: nodes 2.1 and 2.2 isolate time-window choice; this adds hierarchical pooling by coarsening bins. [DONE] (score: 29.23)

**Insight**: Coarse direct empirical bins produced both poor calibration and poor PnL; direct empirical Gc is not competitive without H2 blending.

**Result**: B_dev 29.23; 1246 orders.

### 3: Mechanism: Robust chronological policy-selection family
Hypothesis: Selecting policies on worst-fold or shrinkage-adjusted utility should reduce short-tail overfitting that makes single-tail PnL unstable.
Observable: Better worst-fold and mean B_dev sum_pnl without using the frozen month.
Conflicts: none - attacks selection variance rather than model form. [PENDING]

**Insight**: Children findings: [3.1, done, score=65.96] The 0.025-bin blend reduced the earlier-fold calibration gap from 0.1471 to 0.1284 but lowered H14-style sum_pnl from 82.34 to 65.96; calibration improvement did not translate into robust PnL uplift.

#### 3.1: Mechanism: Earlier blocked-fold falsification of the 0.025-bin blend
Hypothesis: A real Gc calibration improvement should preserve positive uplift when the entire fit/calibration/dev sequence is shifted back two weeks.
Observable: The blend beats its fold-specific H14 baseline on 2026-03-14 through 2026-03-27 without accessing B_test.
Conflicts: node 1.5 won one B_dev split; this tests whether that gain is temporal overfit. [DONE] (score: 65.96)

**Insight**: The 0.025-bin blend reduced the earlier-fold calibration gap from 0.1471 to 0.1284 but lowered H14-style sum_pnl from 82.34 to 65.96; calibration improvement did not translate into robust PnL uplift.

**Result**: Earlier fold delta -16.38. Node 1.5 is rejected for B_test/promotion because uplift failed chronological falsification.

### 4: Mechanism: Selected-side probability calibration family
Hypothesis: Calibrating q before EV computation should correct the loss-side term that Gc-only methods cannot fix.
Observable: Improved B_dev Brier/logloss and sum_pnl with unchanged direction decisions.
Conflicts: none - attacks q rather than Gc. [PENDING]

**Insight**: Children findings: [4.1, done, score=88.38] The current H14 runner ignores the checkpoint q_calibrator and reuses raw frame p_side, so isotonic q produced no H14 change; integrating calibrated q requires an explicit downstream interface change.

#### 4.1: Mechanism: Calibration-tail isotonic q with retrained H2 Gc
Hypothesis: Monotone correctness calibration should improve EV ranking by aligning q with realized selected-side correctness.
Observable: B_dev Brier improves and H14-style sum_pnl exceeds 88.38.
Conflicts: Gc blend node 1.1 improved fill probabilities; this independently corrects correctness probabilities. [DONE] (score: 88.38)

**Insight**: The current H14 runner ignores the checkpoint q_calibrator and reuses raw frame p_side, so isotonic q produced no H14 change; integrating calibrated q requires an explicit downstream interface change.

**Result**: B_dev unchanged at 88.38; do not claim q calibration improvement from this path.
