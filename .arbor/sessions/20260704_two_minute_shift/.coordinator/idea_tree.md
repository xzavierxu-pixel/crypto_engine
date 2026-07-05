# Idea Tree

**Baseline**: 218.5 | **Trunk**: 218.5

## ROOT: Rebuild the accepted first-minute pipeline at decision_time=market_t0+2m using only features timestamped before decision; price target is selected-side minimum over [decision_time,endDate]; retain the 20260703 prefinal rolling policy and chronological protocol; maximize validation sum_pnl; B_test may be evaluated exactly once after pretest freeze; never overwrite existing data or deploy artifacts. [DONE]

**Insight**: Children findings: [1, done, score=59.9] Two-minute direction accuracy improved, but the shortened-window Gc calibration gap reached 0.2265 and validation PnL fell to -0.75.

### 1: Mechanism: End-to-end as-of-time shift rebuild with immutable source inputs and a three-minute selected-side low target
Hypothesis: Recomputing every direction and pricing input at market_t0+2m removes the representation mismatch that would arise from merely shifting labels while preserving the accepted policy mechanism
Observable: Frozen w1-w4 selection, positive w5-w6 gate, and one final B_test sum_pnl directly comparable with 42.43
Conflicts: none - attacks an unexplored decision-time axis while retaining the validated 20260703 policy [DONE] (score: 59.9)

**Insight**: Two-minute direction accuracy improved, but the shortened-window Gc calibration gap reached 0.2265 and validation PnL fell to -0.75.

**Result**: Pretest gate passed at 30.38 on w5-w6; final validation sum_pnl was -0.75 versus 42.43 baseline.

**Branch**: 2mins
