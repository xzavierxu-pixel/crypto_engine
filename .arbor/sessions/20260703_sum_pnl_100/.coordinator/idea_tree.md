# Idea Tree

**Baseline**: 88.38 | **Trunk**: 88.38

## ROOT: Run exactly 100 no-leak B_dev policy experiments; maximize robust chronological sum_pnl; target frozen B_test sum_pnl >=100; B_test only once for final selected candidate; stage1_sample_weight forbidden; 4-hour cap [DONE]

**Insight**: Children findings: [1, pending] Children findings: [1.1, done, score=88.9] Recorded as part of the complete two-fold interaction search. | [1.2, done, score=78.56] Recorded as part of the complete two-fold interaction search. | [1.3, done, score=64.83] Recorded as part of the complete two-fold interaction search. | [1.4, done, score=36.69] Recorded as part of the complete two-fold interaction search. | [1.5, done, score=89.4] Recorded as part of the complete two-fold interaction search. | [1.6, done, score=81.24] Recorded as part of the complete two-fold interaction search. | [1.7, done, score=66.26] Recorded as part of the complete two-fold interaction search. | [1.8, done, score=37.29] Recorded as part of the complete two-fold interaction search. | [1.9, done, score=88.38] Recorded as part of the complete two-fold interaction search. | [1.10, done, score=80.01] Recorded as part of the complete two-fold interaction search. | [1.11, done, score=68.76] Recorded as part of the complete two-fold interaction search. | [1.12, done, score=37.07] Recorded as part of the complete two-fold interaction search. | [1.13, done, score=89.99] Recorded as part of the complete two-fold interacti...

### 1: Mechanism: Robust two-fold joint search over q shrinkage, Gc floor, and EV admission
Hypothesis: Jointly controlling correctness calibration strength, fill-confidence gating, and EV admission should expose a temporally stable policy missed by one-axis experiments.
Observable: Among exactly 100 B_dev candidates, select one with positive delta on both chronological folds and the highest worst-fold delta.
Conflicts: prior node 2 improved both B_dev folds but failed B_test; this counters via explicit robust two-fold selection across the complete policy interaction. [DONE] (score: 98.1)

**Insight**: Joint search found a two-fold B_dev winner, but it failed the frozen month; no candidate is promoted.

**Result**: 100 candidates completed; experiment 37 selected by worst-fold delta; frozen B_test=5.48 vs 27.44 baseline.

#### 1.1: Mechanism: Joint policy candidate 1 with q-alpha=0, Gc-floor=0.65, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.9)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.90 (delta +0.52); earlier=75.98 (delta -6.36); worst_delta=-6.36

#### 1.2: Mechanism: Joint policy candidate 2 with q-alpha=0, Gc-floor=0.65, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 78.56)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=78.56 (delta -9.82); earlier=69.04 (delta -13.30); worst_delta=-13.30

#### 1.3: Mechanism: Joint policy candidate 3 with q-alpha=0, Gc-floor=0.65, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 64.83)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=64.83 (delta -23.55); earlier=73.61 (delta -8.73); worst_delta=-23.55

#### 1.4: Mechanism: Joint policy candidate 4 with q-alpha=0, Gc-floor=0.65, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 36.69)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=36.69 (delta -51.69); earlier=64.32 (delta -18.02); worst_delta=-51.69

#### 1.5: Mechanism: Joint policy candidate 5 with q-alpha=0, Gc-floor=0.7, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.4)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.40 (delta +1.02); earlier=72.23 (delta -10.11); worst_delta=-10.11

#### 1.6: Mechanism: Joint policy candidate 6 with q-alpha=0, Gc-floor=0.7, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 81.24)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=81.24 (delta -7.14); earlier=72.68 (delta -9.66); worst_delta=-9.66

#### 1.7: Mechanism: Joint policy candidate 7 with q-alpha=0, Gc-floor=0.7, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 66.26)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=66.26 (delta -22.12); earlier=75.47 (delta -6.87); worst_delta=-22.12

#### 1.8: Mechanism: Joint policy candidate 8 with q-alpha=0, Gc-floor=0.7, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 37.29)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=37.29 (delta -51.09); earlier=65.61 (delta -16.73); worst_delta=-51.09

#### 1.9: Mechanism: Joint policy candidate 9 with q-alpha=0, Gc-floor=0.75, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.38)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.38 (delta +0.00); earlier=76.93 (delta -5.41); worst_delta=-5.41

#### 1.10: Mechanism: Joint policy candidate 10 with q-alpha=0, Gc-floor=0.75, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 80.01)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=80.01 (delta -8.37); earlier=86.33 (delta +3.99); worst_delta=-8.37

#### 1.11: Mechanism: Joint policy candidate 11 with q-alpha=0, Gc-floor=0.75, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 68.76)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=68.76 (delta -19.62); earlier=82.34 (delta +0.00); worst_delta=-19.62

#### 1.12: Mechanism: Joint policy candidate 12 with q-alpha=0, Gc-floor=0.75, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 37.07)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=37.07 (delta -51.31); earlier=69.82 (delta -12.52); worst_delta=-51.31

#### 1.13: Mechanism: Joint policy candidate 13 with q-alpha=0, Gc-floor=0.8, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.99)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.99 (delta +1.61); earlier=87.26 (delta +4.92); worst_delta=+1.61

#### 1.14: Mechanism: Joint policy candidate 14 with q-alpha=0, Gc-floor=0.8, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 84.44)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=84.44 (delta -3.94); earlier=85.56 (delta +3.22); worst_delta=-3.94

#### 1.15: Mechanism: Joint policy candidate 15 with q-alpha=0, Gc-floor=0.8, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 62.08)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=62.08 (delta -26.30); earlier=90.13 (delta +7.79); worst_delta=-26.30

#### 1.16: Mechanism: Joint policy candidate 16 with q-alpha=0, Gc-floor=0.8, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 41.63)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=41.63 (delta -46.75); earlier=75.86 (delta -6.48); worst_delta=-46.75

#### 1.17: Mechanism: Joint policy candidate 17 with q-alpha=0, Gc-floor=0.85, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 90.12)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=90.12 (delta +1.74); earlier=92.49 (delta +10.15); worst_delta=+1.74

#### 1.18: Mechanism: Joint policy candidate 18 with q-alpha=0, Gc-floor=0.85, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 72.45)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=72.45 (delta -15.93); earlier=92.78 (delta +10.44); worst_delta=-15.93

#### 1.19: Mechanism: Joint policy candidate 19 with q-alpha=0, Gc-floor=0.85, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 45.17)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=45.17 (delta -43.21); earlier=79.10 (delta -3.24); worst_delta=-43.21

#### 1.20: Mechanism: Joint policy candidate 20 with q-alpha=0, Gc-floor=0.85, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 38.16)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=38.16 (delta -50.22); earlier=78.51 (delta -3.83); worst_delta=-50.22

#### 1.21: Mechanism: Joint policy candidate 21 with q-alpha=0.25, Gc-floor=0.65, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.88)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.88 (delta -4.50); earlier=71.40 (delta -10.94); worst_delta=-10.94

#### 1.22: Mechanism: Joint policy candidate 22 with q-alpha=0.25, Gc-floor=0.65, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.14)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.14 (delta -0.24); earlier=70.33 (delta -12.01); worst_delta=-12.01

#### 1.23: Mechanism: Joint policy candidate 23 with q-alpha=0.25, Gc-floor=0.65, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 81.71)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=81.71 (delta -6.67); earlier=73.71 (delta -8.63); worst_delta=-8.63

#### 1.24: Mechanism: Joint policy candidate 24 with q-alpha=0.25, Gc-floor=0.65, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 61.79)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=61.79 (delta -26.59); earlier=66.24 (delta -16.10); worst_delta=-26.59

#### 1.25: Mechanism: Joint policy candidate 25 with q-alpha=0.25, Gc-floor=0.7, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 90.79)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=90.79 (delta +2.41); earlier=70.24 (delta -12.10); worst_delta=-12.10

#### 1.26: Mechanism: Joint policy candidate 26 with q-alpha=0.25, Gc-floor=0.7, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 87.77)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=87.77 (delta -0.61); earlier=73.98 (delta -8.36); worst_delta=-8.36

#### 1.27: Mechanism: Joint policy candidate 27 with q-alpha=0.25, Gc-floor=0.7, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 80.11)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=80.11 (delta -8.27); earlier=75.99 (delta -6.35); worst_delta=-8.27

#### 1.28: Mechanism: Joint policy candidate 28 with q-alpha=0.25, Gc-floor=0.7, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 62.36)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=62.36 (delta -26.02); earlier=65.73 (delta -16.61); worst_delta=-26.02

#### 1.29: Mechanism: Joint policy candidate 29 with q-alpha=0.25, Gc-floor=0.75, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 95.53)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=95.53 (delta +7.15); earlier=75.38 (delta -6.96); worst_delta=-6.96

#### 1.30: Mechanism: Joint policy candidate 30 with q-alpha=0.25, Gc-floor=0.75, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.66)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.66 (delta +0.28); earlier=75.99 (delta -6.35); worst_delta=-6.35

#### 1.31: Mechanism: Joint policy candidate 31 with q-alpha=0.25, Gc-floor=0.75, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 81.09)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=81.09 (delta -7.29); earlier=75.68 (delta -6.66); worst_delta=-7.29

#### 1.32: Mechanism: Joint policy candidate 32 with q-alpha=0.25, Gc-floor=0.75, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 63.56)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=63.56 (delta -24.82); earlier=74.28 (delta -8.06); worst_delta=-24.82

#### 1.33: Mechanism: Joint policy candidate 33 with q-alpha=0.25, Gc-floor=0.8, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 97.91)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=97.91 (delta +9.53); earlier=79.57 (delta -2.77); worst_delta=-2.77

#### 1.34: Mechanism: Joint policy candidate 34 with q-alpha=0.25, Gc-floor=0.8, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.85)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.85 (delta +1.47); earlier=87.29 (delta +4.95); worst_delta=+1.47

#### 1.35: Mechanism: Joint policy candidate 35 with q-alpha=0.25, Gc-floor=0.8, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 85.78)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=85.78 (delta -2.60); earlier=84.27 (delta +1.93); worst_delta=-2.60

#### 1.36: Mechanism: Joint policy candidate 36 with q-alpha=0.25, Gc-floor=0.8, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 57.45)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=57.45 (delta -30.93); earlier=77.32 (delta -5.02); worst_delta=-30.93

#### 1.37: Mechanism: Joint policy candidate 37 with q-alpha=0.25, Gc-floor=0.85, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 98.1)

**Insight**: Robust winner across both folds.

**Result**: main=98.10 (delta +9.72); earlier=86.51 (delta +4.17); worst_delta=+4.17

#### 1.38: Mechanism: Joint policy candidate 38 with q-alpha=0.25, Gc-floor=0.85, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 86.67)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=86.67 (delta -1.71); earlier=86.28 (delta +3.94); worst_delta=-1.71

#### 1.39: Mechanism: Joint policy candidate 39 with q-alpha=0.25, Gc-floor=0.85, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 68.13)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=68.13 (delta -20.25); earlier=77.53 (delta -4.81); worst_delta=-20.25

#### 1.40: Mechanism: Joint policy candidate 40 with q-alpha=0.25, Gc-floor=0.85, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 46.3)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=46.30 (delta -42.08); earlier=71.40 (delta -10.94); worst_delta=-42.08

#### 1.41: Mechanism: Joint policy candidate 41 with q-alpha=0.5, Gc-floor=0.65, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 76.15)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=76.15 (delta -12.23); earlier=70.34 (delta -12.00); worst_delta=-12.23

#### 1.42: Mechanism: Joint policy candidate 42 with q-alpha=0.5, Gc-floor=0.65, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 92.75)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=92.75 (delta +4.37); earlier=72.42 (delta -9.92); worst_delta=-9.92

#### 1.43: Mechanism: Joint policy candidate 43 with q-alpha=0.5, Gc-floor=0.65, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.99)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.99 (delta -4.39); earlier=65.51 (delta -16.83); worst_delta=-16.83

#### 1.44: Mechanism: Joint policy candidate 44 with q-alpha=0.5, Gc-floor=0.65, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 73.2)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=73.20 (delta -15.18); earlier=58.17 (delta -24.17); worst_delta=-24.17

#### 1.45: Mechanism: Joint policy candidate 45 with q-alpha=0.5, Gc-floor=0.7, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 82.26)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=82.26 (delta -6.12); earlier=70.02 (delta -12.32); worst_delta=-12.32

#### 1.46: Mechanism: Joint policy candidate 46 with q-alpha=0.5, Gc-floor=0.7, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 94.32)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=94.32 (delta +5.94); earlier=66.83 (delta -15.51); worst_delta=-15.51

#### 1.47: Mechanism: Joint policy candidate 47 with q-alpha=0.5, Gc-floor=0.7, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.85)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.85 (delta -4.53); earlier=63.21 (delta -19.13); worst_delta=-19.13

#### 1.48: Mechanism: Joint policy candidate 48 with q-alpha=0.5, Gc-floor=0.7, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 74.62)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=74.62 (delta -13.76); earlier=58.20 (delta -24.14); worst_delta=-24.14

#### 1.49: Mechanism: Joint policy candidate 49 with q-alpha=0.5, Gc-floor=0.75, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.37)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.37 (delta -0.01); earlier=72.27 (delta -10.07); worst_delta=-10.07

#### 1.50: Mechanism: Joint policy candidate 50 with q-alpha=0.5, Gc-floor=0.75, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 94.68)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=94.68 (delta +6.30); earlier=74.89 (delta -7.45); worst_delta=-7.45

#### 1.51: Mechanism: Joint policy candidate 51 with q-alpha=0.5, Gc-floor=0.75, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 84.3)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=84.30 (delta -4.08); earlier=76.57 (delta -5.77); worst_delta=-5.77

#### 1.52: Mechanism: Joint policy candidate 52 with q-alpha=0.5, Gc-floor=0.75, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 76.46)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=76.46 (delta -11.92); earlier=61.74 (delta -20.60); worst_delta=-20.60

#### 1.53: Mechanism: Joint policy candidate 53 with q-alpha=0.5, Gc-floor=0.8, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 100)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=100.02 (delta +11.64); earlier=80.47 (delta -1.87); worst_delta=-1.87

#### 1.54: Mechanism: Joint policy candidate 54 with q-alpha=0.5, Gc-floor=0.8, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 96.15)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=96.15 (delta +7.77); earlier=79.27 (delta -3.07); worst_delta=-3.07

#### 1.55: Mechanism: Joint policy candidate 55 with q-alpha=0.5, Gc-floor=0.8, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 90.28)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=90.28 (delta +1.90); earlier=72.55 (delta -9.79); worst_delta=-9.79

#### 1.56: Mechanism: Joint policy candidate 56 with q-alpha=0.5, Gc-floor=0.8, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 75.92)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=75.92 (delta -12.46); earlier=57.24 (delta -25.10); worst_delta=-25.10

#### 1.57: Mechanism: Joint policy candidate 57 with q-alpha=0.5, Gc-floor=0.85, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 100)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=100.02 (delta +11.64); earlier=80.55 (delta -1.79); worst_delta=-1.79

#### 1.58: Mechanism: Joint policy candidate 58 with q-alpha=0.5, Gc-floor=0.85, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 92.05)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=92.05 (delta +3.67); earlier=77.60 (delta -4.74); worst_delta=-4.74

#### 1.59: Mechanism: Joint policy candidate 59 with q-alpha=0.5, Gc-floor=0.85, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 85.74)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=85.74 (delta -2.64); earlier=63.60 (delta -18.74); worst_delta=-18.74

#### 1.60: Mechanism: Joint policy candidate 60 with q-alpha=0.5, Gc-floor=0.85, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 65.42)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=65.42 (delta -22.96); earlier=61.69 (delta -20.65); worst_delta=-22.96

#### 1.61: Mechanism: Joint policy candidate 61 with q-alpha=0.75, Gc-floor=0.65, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 68.91)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=68.91 (delta -19.47); earlier=69.51 (delta -12.83); worst_delta=-19.47

#### 1.62: Mechanism: Joint policy candidate 62 with q-alpha=0.75, Gc-floor=0.65, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 81.16)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=81.16 (delta -7.22); earlier=61.89 (delta -20.45); worst_delta=-20.45

#### 1.63: Mechanism: Joint policy candidate 63 with q-alpha=0.75, Gc-floor=0.65, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 86.4)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=86.40 (delta -1.98); earlier=57.89 (delta -24.45); worst_delta=-24.45

#### 1.64: Mechanism: Joint policy candidate 64 with q-alpha=0.75, Gc-floor=0.65, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.09)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.09 (delta -5.29); earlier=54.72 (delta -27.62); worst_delta=-27.62

#### 1.65: Mechanism: Joint policy candidate 65 with q-alpha=0.75, Gc-floor=0.7, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 71.2)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=71.20 (delta -17.18); earlier=67.34 (delta -15.00); worst_delta=-17.18

#### 1.66: Mechanism: Joint policy candidate 66 with q-alpha=0.75, Gc-floor=0.7, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 84.99)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=84.99 (delta -3.39); earlier=61.14 (delta -21.20); worst_delta=-21.20

#### 1.67: Mechanism: Joint policy candidate 67 with q-alpha=0.75, Gc-floor=0.7, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.4)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.40 (delta +1.02); earlier=56.48 (delta -25.86); worst_delta=-25.86

#### 1.68: Mechanism: Joint policy candidate 68 with q-alpha=0.75, Gc-floor=0.7, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.1)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.10 (delta -5.28); earlier=54.21 (delta -28.13); worst_delta=-28.13

#### 1.69: Mechanism: Joint policy candidate 69 with q-alpha=0.75, Gc-floor=0.75, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 77.33)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=77.33 (delta -11.05); earlier=73.59 (delta -8.75); worst_delta=-11.05

#### 1.70: Mechanism: Joint policy candidate 70 with q-alpha=0.75, Gc-floor=0.75, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.97)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.97 (delta +0.59); earlier=64.26 (delta -18.08); worst_delta=-18.08

#### 1.71: Mechanism: Joint policy candidate 71 with q-alpha=0.75, Gc-floor=0.75, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.3)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.30 (delta +0.92); earlier=62.16 (delta -20.18); worst_delta=-20.18

#### 1.72: Mechanism: Joint policy candidate 72 with q-alpha=0.75, Gc-floor=0.75, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 84.6)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=84.60 (delta -3.78); earlier=55.73 (delta -26.61); worst_delta=-26.61

#### 1.73: Mechanism: Joint policy candidate 73 with q-alpha=0.75, Gc-floor=0.8, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 84.66)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=84.66 (delta -3.72); earlier=74.52 (delta -7.82); worst_delta=-7.82

#### 1.74: Mechanism: Joint policy candidate 74 with q-alpha=0.75, Gc-floor=0.8, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 96.39)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=96.39 (delta +8.01); earlier=63.23 (delta -19.11); worst_delta=-19.11

#### 1.75: Mechanism: Joint policy candidate 75 with q-alpha=0.75, Gc-floor=0.8, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 98.11)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=98.11 (delta +9.73); earlier=59.24 (delta -23.10); worst_delta=-23.10

#### 1.76: Mechanism: Joint policy candidate 76 with q-alpha=0.75, Gc-floor=0.8, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.05)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.05 (delta -0.33); earlier=55.51 (delta -26.83); worst_delta=-26.83

#### 1.77: Mechanism: Joint policy candidate 77 with q-alpha=0.75, Gc-floor=0.85, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 90.09)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=90.09 (delta +1.71); earlier=67.15 (delta -15.19); worst_delta=-15.19

#### 1.78: Mechanism: Joint policy candidate 78 with q-alpha=0.75, Gc-floor=0.85, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 94.3)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=94.30 (delta +5.92); earlier=64.87 (delta -17.47); worst_delta=-17.47

#### 1.79: Mechanism: Joint policy candidate 79 with q-alpha=0.75, Gc-floor=0.85, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 90.47)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=90.47 (delta +2.09); earlier=57.40 (delta -24.94); worst_delta=-24.94

#### 1.80: Mechanism: Joint policy candidate 80 with q-alpha=0.75, Gc-floor=0.85, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 80.14)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=80.14 (delta -8.24); earlier=54.95 (delta -27.39); worst_delta=-27.39

#### 1.81: Mechanism: Joint policy candidate 81 with q-alpha=1, Gc-floor=0.65, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 61.77)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=61.77 (delta -26.61); earlier=63.15 (delta -19.19); worst_delta=-26.61

#### 1.82: Mechanism: Joint policy candidate 82 with q-alpha=1, Gc-floor=0.65, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 67.3)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=67.30 (delta -21.08); earlier=54.65 (delta -27.69); worst_delta=-27.69

#### 1.83: Mechanism: Joint policy candidate 83 with q-alpha=1, Gc-floor=0.65, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 81.13)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=81.13 (delta -7.25); earlier=46.54 (delta -35.80); worst_delta=-35.80

#### 1.84: Mechanism: Joint policy candidate 84 with q-alpha=1, Gc-floor=0.65, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 87.47)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=87.47 (delta -0.91); earlier=50.30 (delta -32.04); worst_delta=-32.04

#### 1.85: Mechanism: Joint policy candidate 85 with q-alpha=1, Gc-floor=0.7, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 64.75)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=64.75 (delta -23.63); earlier=60.18 (delta -22.16); worst_delta=-23.63

#### 1.86: Mechanism: Joint policy candidate 86 with q-alpha=1, Gc-floor=0.7, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 70.07)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=70.07 (delta -18.31); earlier=51.53 (delta -30.81); worst_delta=-30.81

#### 1.87: Mechanism: Joint policy candidate 87 with q-alpha=1, Gc-floor=0.7, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 82.32)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=82.32 (delta -6.06); earlier=47.04 (delta -35.30); worst_delta=-35.30

#### 1.88: Mechanism: Joint policy candidate 88 with q-alpha=1, Gc-floor=0.7, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 88.37)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=88.37 (delta -0.01); earlier=50.04 (delta -32.30); worst_delta=-32.30

#### 1.89: Mechanism: Joint policy candidate 89 with q-alpha=1, Gc-floor=0.75, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 66.87)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=66.87 (delta -21.51); earlier=59.54 (delta -22.80); worst_delta=-22.80

#### 1.90: Mechanism: Joint policy candidate 90 with q-alpha=1, Gc-floor=0.75, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 72.41)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=72.41 (delta -15.97); earlier=55.16 (delta -27.18); worst_delta=-27.18

#### 1.91: Mechanism: Joint policy candidate 91 with q-alpha=1, Gc-floor=0.75, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.26)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.26 (delta -5.12); earlier=53.73 (delta -28.61); worst_delta=-28.61

#### 1.92: Mechanism: Joint policy candidate 92 with q-alpha=1, Gc-floor=0.75, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 89.26)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=89.26 (delta +0.88); earlier=53.02 (delta -29.32); worst_delta=-29.32

#### 1.93: Mechanism: Joint policy candidate 93 with q-alpha=1, Gc-floor=0.8, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 72.09)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=72.09 (delta -16.29); earlier=60.98 (delta -21.36); worst_delta=-21.36

#### 1.94: Mechanism: Joint policy candidate 94 with q-alpha=1, Gc-floor=0.8, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 83.81)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=83.81 (delta -4.57); earlier=52.60 (delta -29.74); worst_delta=-29.74

#### 1.95: Mechanism: Joint policy candidate 95 with q-alpha=1, Gc-floor=0.8, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 91.8)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=91.80 (delta +3.42); earlier=56.33 (delta -26.01); worst_delta=-26.01

#### 1.96: Mechanism: Joint policy candidate 96 with q-alpha=1, Gc-floor=0.8, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 93.35)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=93.35 (delta +4.97); earlier=52.47 (delta -29.87); worst_delta=-29.87

#### 1.97: Mechanism: Joint policy candidate 97 with q-alpha=1, Gc-floor=0.85, min-EV=0
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 82.44)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=82.44 (delta -5.94); earlier=58.76 (delta -23.58); worst_delta=-23.58

#### 1.98: Mechanism: Joint policy candidate 98 with q-alpha=1, Gc-floor=0.85, min-EV=0.01
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 86.58)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=86.58 (delta -1.80); earlier=57.73 (delta -24.61); worst_delta=-24.61

#### 1.99: Mechanism: Joint policy candidate 99 with q-alpha=1, Gc-floor=0.85, min-EV=0.02
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 98.12)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=98.12 (delta +9.74); earlier=52.74 (delta -29.60); worst_delta=-29.60

#### 1.100: Mechanism: Joint policy candidate 100 with q-alpha=1, Gc-floor=0.85, min-EV=0.03
Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.
Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.
Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence. [DONE] (score: 87.32)

**Insight**: Recorded as part of the complete two-fold interaction search.

**Result**: main=87.32 (delta -1.06); earlier=49.84 (delta -32.50); worst_delta=-32.50
