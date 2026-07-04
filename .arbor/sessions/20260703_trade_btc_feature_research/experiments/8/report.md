# Experiment 8

**Hypothesis**: Mechanism: prequential daily online adaptation that refits q and empirical Gc using only markets settled before each decision day
Hypothesis: every static candidate collapses at the month boundary while prior validation outcomes become observable after five minutes; legal walk-forward updates should track the new regime without future-label leakage
Observable: improve daily/weekly B_dev PnL and q/Gc calibration after the first adaptation day, then remain positive on untouched w5-w6 under one frozen update protocol
Conflicts: all prior nodes trained once before validation; this changes the control flow from static inference to causally ordered online learning

**Score**: 173.2

**Insight**: Prequential updates improved q calibration on early folds but did not translate to enough PnL; tune 173.20 and holdout 95.65, below static policies.

**Result**: Rejected on B_dev; causal protocol verified; no B_test used.
