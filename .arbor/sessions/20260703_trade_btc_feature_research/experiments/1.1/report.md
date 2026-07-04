# Experiment 1.1

**Hypothesis**: Mechanism: monotone residual Gc calibrator trained on bid-expanded correct orders using base hazard CDF plus pre-decision trade state
Hypothesis: the remaining month drift is miscalibrated conditional fill probability; conditioning the hazard CDF on observed liquidity should improve bid ranking without changing fill semantics
Observable: lower Gc Brier on every rolling fold and improve robust w1-w4 sum_pnl while preserving positive w5-w6 uplift
Conflicts: none - node 1 improved q while holding Gc fixed, so this attacks the remaining probability component

**Score**: 290.31

**Insight**: Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved.

**Result**: Winner gc_new, floor 0.90, min_ev 0.05; weekly pnl 70.35,68.82,68.96,82.18,68.55,70.03. B_test not used during selection.
