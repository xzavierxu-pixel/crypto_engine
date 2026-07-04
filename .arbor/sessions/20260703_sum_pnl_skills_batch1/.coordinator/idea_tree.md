# Idea Tree

**Baseline**: 88.38 | **Trunk**: 88.38

## ROOT: Continue no-leak sum_pnl research using five disabled skills as mechanisms; 10 evaluations across two chronological B_dev folds; frozen B_test; stage1_sample_weight forbidden [DONE]

**Insight**: Children findings: [1, done, score=80.6] Explicit q integration works mechanically but monotone calibration alone is temporally weak. | [2, done, score=97.21] Shrinkage is the only robust B_dev mechanism in this batch, but it failed the frozen-month verification and is rejected. | [3, done, score=69.62] Fixed clipping destroyed useful q dispersion and underperformed. | [4, done, score=80.01] Intraday residual offsets did not stabilize policy PnL. | [5, done, score=79.5] Grouped residual calibration overfit and was worst on the earlier fold.

### 1: Mechanism: Chronological isotonic selected-side q calibration
Hypothesis: Replacing raw p_side with calibration-only monotone correctness estimates should repair the ignored q interface and improve EV ranking.
Observable: Higher sum_pnl than fold-specific H14 baselines on both B_dev folds.
Conflicts: prior node 4.1 said H14 ignored q; this counters via an explicit q-to-EV path. [DONE] (score: 80.6)

**Insight**: Explicit q integration works mechanically but monotone calibration alone is temporally weak.

**Result**: Pure isotonic q underperformed both fold baselines (80.60 main; 59.68 earlier).

### 2: Mechanism: Raw/isotonic shrinkage ensemble for q
Hypothesis: Blending calibrated and raw correctness probabilities should reduce isotonic variance under temporal shift while preserving monotone correction.
Observable: Better worst-fold and mean B_dev sum_pnl than pure isotonic q.
Conflicts: prior node 3.1 exposed temporal instability; this counters via shrinkage. [DONE] (score: 97.21)

**Insight**: Shrinkage is the only robust B_dev mechanism in this batch, but it failed the frozen-month verification and is rejected.

**Result**: Raw/isotonic 50/50 shrink improved both B_dev folds (97.21 vs 88.38; 90.25 vs 82.34), but final B_test was -12.14 vs 27.44.

### 3: Mechanism: Conservative bounded q probabilities
Hypothesis: Clipping calibrated q to a defensible interval should limit extreme EV decisions caused by sparse calibration plateaus.
Observable: Lower downside loss and higher worst-fold sum_pnl than unbounded isotonic q.
Conflicts: prior node 2.2 showed calibration error alone is insufficient; this targets decision sensitivity. [DONE] (score: 69.62)

**Insight**: Fixed clipping destroyed useful q dispersion and underperformed.

**Result**: Bounded isotonic q scored 69.62 main and 54.92 earlier.

### 4: Mechanism: Shrunk UTC-session residual calibration
Hypothesis: Six-hour cyclical session offsets fitted before policy selection should correct recurring intraday correctness drift without changing direction decisions.
Observable: Higher two-fold mean sum_pnl with stable order coverage.
Conflicts: none - attacks temporal calibration structure not previously tested. [DONE] (score: 80.01)

**Insight**: Intraday residual offsets did not stabilize policy PnL.

**Result**: UTC-session residual q scored 80.01 main and 62.32 earlier.

### 5: Mechanism: Hierarchically shrunk p_side-bin residual calibration
Hypothesis: Grouped reliability residuals over p_side bins should capture local q bias while shrinkage prevents sparse-bin overfit.
Observable: Higher worst-fold sum_pnl and competitive Brier diagnostics on both B_dev folds.
Conflicts: prior Gc p_side bins were unstable; this applies grouping to q with explicit shrinkage. [DONE] (score: 79.5)

**Insight**: Grouped residual calibration overfit and was worst on the earlier fold.

**Result**: p_side-bin residual q scored 79.50 main and 47.46 earlier.
