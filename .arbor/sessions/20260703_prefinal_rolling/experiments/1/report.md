# Experiment 1

**Hypothesis**: Mechanism: expanding-window weekly cross-fitting of Gc, correctness probability, and policy selection across all pre-test history
Hypothesis: selecting on worst-week and lower-tail PnL will expose the temporal instability hidden by two aggregated B_dev blocks
Observable: positive aggregate and lower-tail PnL across at least five untouched chronological weeks, with stable calibration gaps
Conflicts: prior two-fold LGBM q reached 129.97 B_dev but only 9.99 B_test; this counters via finer rolling falsification

**Score**: 221.78

**Insight**: Six expanding weekly folds exposed stable pre-test uplift. The selected q/tree policy scored 221.78 across tune weeks and 127.07 on untouched weeks, but its frozen B_test milestone reached only 42.43.

**Result**: B_test improved 27.44 to 42.43, below target 100; no promotion.
