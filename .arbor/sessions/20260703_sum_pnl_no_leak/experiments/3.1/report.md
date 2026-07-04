# Experiment 3.1

**Hypothesis**: Mechanism: Earlier blocked-fold falsification of the 0.025-bin blend
Hypothesis: A real Gc calibration improvement should preserve positive uplift when the entire fit/calibration/dev sequence is shifted back two weeks.
Observable: The blend beats its fold-specific H14 baseline on 2026-03-14 through 2026-03-27 without accessing B_test.
Conflicts: node 1.5 won one B_dev split; this tests whether that gain is temporal overfit.

**Score**: 65.96

**Insight**: The 0.025-bin blend reduced the earlier-fold calibration gap from 0.1564 to 0.1284 but lowered sum_pnl from 68.16 to 65.96; calibration improvement did not translate into robust PnL uplift.

**Result**: Earlier fold delta -2.20. Node 1.5 is rejected for B_test/promotion because uplift failed chronological falsification.
