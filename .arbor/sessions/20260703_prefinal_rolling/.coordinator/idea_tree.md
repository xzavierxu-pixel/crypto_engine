# Idea Tree

**Baseline**: 27.44 | **Trunk**: 27.44

## ROOT: Deep pre-B_test rolling study to maximize frozen final-month sum_pnl toward 100; train/evaluate chronological weekly folds across all available pre-test history; optimize correctness, Gc, and policy without B_test iteration; explicit sample-weight leakage ban [DONE]

**Insight**: Children findings: [1, done, score=221.8] Six expanding weekly folds exposed stable pre-test uplift. The selected q/tree policy scored 221.78 across tune weeks and 127.07 on untouched weeks, but its frozen B_test milestone reached only 42.43. | [2, done, score=218.5] Ensemble-disagreement penalties did not improve the pre-test selection objective; the optimum penalty was zero. | [3, done, score=221.8] 14/21/28-day recency windows did not beat expanding-history correctness models; the selected recency was all available history.

### 1: Mechanism: expanding-window weekly cross-fitting of Gc, correctness probability, and policy selection across all pre-test history
Hypothesis: selecting on worst-week and lower-tail PnL will expose the temporal instability hidden by two aggregated B_dev blocks
Observable: positive aggregate and lower-tail PnL across at least five untouched chronological weeks, with stable calibration gaps
Conflicts: prior two-fold LGBM q reached 129.97 B_dev but only 9.99 B_test; this counters via finer rolling falsification [DONE] (score: 221.8)

**Insight**: Six expanding weekly folds exposed stable pre-test uplift. The selected q/tree policy scored 221.78 across tune weeks and 127.07 on untouched weeks, but its frozen B_test milestone reached only 42.43.

**Result**: B_test improved 27.44 to 42.43, below target 100; no promotion.

### 2: Mechanism: conservative distributionally robust bid policy using empirical lower confidence bounds for q and Gc
Hypothesis: penalizing fold-to-fold uncertainty should reduce expensive wrong forced fills under monthly regime shift
Observable: improved worst-week PnL and lower submitted fill calibration gap versus plug-in EV
Conflicts: prior point-estimate EV policies overfit; this replaces point forecasts with uncertainty-aware decisions [DONE] (score: 218.5)

**Insight**: Ensemble-disagreement penalties did not improve the pre-test selection objective; the optimum penalty was zero.

**Result**: Uncertainty penalty rejected; point blend remained better.

### 3: Mechanism: temporal mixture-of-experts with recency weighting and regime-distance gating
Hypothesis: conditioning q and Gc on recent regimes can adapt when pooled historical relationships drift across months
Observable: recent weekly folds improve without materially degrading earlier folds or calibration
Conflicts: prior global LGBM correctness model failed B_test; this counters through explicit temporal adaptation [DONE] (score: 221.8)

**Insight**: 14/21/28-day recency windows did not beat expanding-history correctness models; the selected recency was all available history.

**Result**: Explicit recency adaptation rejected on rolling B_dev.
