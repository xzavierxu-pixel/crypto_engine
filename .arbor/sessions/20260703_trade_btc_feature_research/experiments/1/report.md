# Experiment 1

**Hypothesis**: Mechanism: decision-time cross-market feature bank combining Polymarket pre-decision trade microstructure with expanded BTC second-level regime signals
Hypothesis: conditional order value decays across months because q and Gc omit contemporaneous liquidity/regime state; legal trailing features should reduce q/Gc calibration drift
Observable: improve w1-w4 robust sum_pnl and q/Gc Brier, then remain positive and beat baseline on untouched w5-w6
Conflicts: none - prior nodes varied temporal weighting and policy uncertainty but did not add new market-state observations

**Score**: 266.34

**Insight**: Strictly pre-decision Polymarket trade features improved q Brier/logloss on all six folds and raised w1-w4 sum_pnl from 144.14 to 266.34; untouched w5-w6 improved by 42.78. Broad BTC expansion was inconsistent and not selected.

**Result**: Winner oldnew_trade_lgbm_blend with Gc floor 0.80 and min_ev 0.02: weekly pnl 62.48,60.81,63.07,79.98,58.74,69.75; no B_test used.
