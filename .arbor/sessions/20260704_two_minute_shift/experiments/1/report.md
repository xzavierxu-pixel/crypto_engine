# Idea

Shift the complete accepted pipeline from `market_t0 + 1m` to `market_t0 + 2m`, price against the selected-side low on `[decision_time, endDate]`, and retain the frozen 20260703 policy.

# Changes

- Built two-minute features in monthly warm-start chunks to control peak memory.
- Retrained the accepted 569-feature CatBoost direction model.
- Reused the accepted UTC day/session threshold table without validation retuning.
- Retrained six chronological H2/Gc folds and one full H2 model.
- Kept `raw_tree_blend`, shrink `0`, Gc floor `0.85`, and minimum EV `0.02`.

# Baseline vs Result

- Original validation sum_pnl: `42.43`.
- Two-minute validation sum_pnl: `-0.75`.
- Delta: `-43.18`.
- Pretest w1-w4 sum_pnl: `59.90`.
- Untouched pretest w5-w6 sum_pnl: `30.38`; both folds positive.

# Score

`59.90` absolute B_dev tune-fold sum_pnl; final B_test/validation sum_pnl was `-0.75`.

# Analysis

The time shift improved accepted direction accuracy to `0.73652`, but the frozen price policy submitted only 962 orders and realized 704 fills. Win PnL (`155.79`) and forced-loss PnL (`-156.54`) nearly cancelled. The submitted Gc calibration gap widened to `0.22652`, so the pricing/fill model did not transfer to the shifted three-minute window.

# Insights

The two-minute direction signal is stronger, but the unchanged H2/Gc policy is not economically calibrated for the shortened pricing window. This experiment does not improve validation sum_pnl and must not be promoted.
