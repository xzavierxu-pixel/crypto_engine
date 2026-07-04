# Experiment 7

**Hypothesis**: Mechanism: full-universe joint direction-and-bid policy using reconstructed UP/DOWN lows and EV gating instead of inheriting the legacy 70% direction acceptance mask
Hypothesis: the fixed legacy mask discards 30% of markets before the PnL model can evaluate them; 100% direction coverage plus explicit EV no-order gating should expose additional profitable orders without changing fill semantics
Observable: materially increase w1-w4 and untouched w5-w6 sum_pnl while reporting direction accuracy at coverage 1.0 and stable order coverage
Conflicts: prior joint-direction node retained threshold_accepted, so its 29.48 B_test result did not test the target redesign's full-universe action space

**Score**: 283.59

**Insight**: Full-universe direction coverage unlocked substantial additional value: tune 283.59 and untouched 173.10, with both holdout weeks above 85 PnL; the legacy 70% mask was constraining the action space.

**Result**: Selected tree direction, min_q 0.50, hazard floor 0.80, min_ev 0; coverage 1.0. Qualified for a new-action-space B_test milestone.
