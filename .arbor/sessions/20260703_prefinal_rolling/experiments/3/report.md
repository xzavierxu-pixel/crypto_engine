# Experiment 3

**Hypothesis**: Mechanism: temporal mixture-of-experts with recency weighting and regime-distance gating
Hypothesis: conditioning q and Gc on recent regimes can adapt when pooled historical relationships drift across months
Observable: recent weekly folds improve without materially degrading earlier folds or calibration
Conflicts: prior global LGBM correctness model failed B_test; this counters through explicit temporal adaptation

**Score**: 221.78

**Insight**: 14/21/28-day recency windows did not beat expanding-history correctness models; the selected recency was all available history.

**Result**: Explicit recency adaptation rejected on rolling B_dev.
