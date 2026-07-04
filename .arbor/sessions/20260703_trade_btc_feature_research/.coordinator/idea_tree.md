# Idea Tree

**Baseline**: 221.8 | **Trunk**: 221.8

## ROOT: Maximize no-leak chronological validation sum_pnl beyond 200 using decision-time Polymarket trade/L2 and BTC feature engineering; iterate only on rolling B_dev, protect final B_test, preserve direction universe and fill semantics. [DONE]

**Insight**: Children findings: [1, done, score=266.3] Children findings: [1.1, done, score=290.3] Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved. | [2, done, score=241.4] Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2. | [3, done, score=166.1] Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side. | [4, done, score=68.22] Direct PnL regression lowered bids but failed to learn stable action ranking: tune sum 68.22 with a negative week, far below probability-factorized policies. | [5, done, score=272.9] Low-variance empirical Gc stratified by p_side beat the contextual hazard on untouched PnL (148.22 vs 128.49); all-history p-bin CDF improved Gc Brier on all six folds, validating selection-overconfidence as the bottleneck. | [6, done, score=199.5] Feature minimization reduced som...

### 1: Mechanism: decision-time cross-market feature bank combining Polymarket pre-decision trade microstructure with expanded BTC second-level regime signals
Hypothesis: conditional order value decays across months because q and Gc omit contemporaneous liquidity/regime state; legal trailing features should reduce q/Gc calibration drift
Observable: improve w1-w4 robust sum_pnl and q/Gc Brier, then remain positive and beat baseline on untouched w5-w6
Conflicts: none - prior nodes varied temporal weighting and policy uncertainty but did not add new market-state observations [DONE] (score: 266.3)

**Insight**: Children findings: [1.1, done, score=290.3] Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved.

**Result**: Winner oldnew_trade_lgbm_blend with Gc floor 0.80 and min_ev 0.02: weekly pnl 62.48,60.81,63.07,79.98,58.74,69.75; no B_test used.

**Branch**: session-only-experiment

#### 1.1: Mechanism: monotone residual Gc calibrator trained on bid-expanded correct orders using base hazard CDF plus pre-decision trade state
Hypothesis: the remaining month drift is miscalibrated conditional fill probability; conditioning the hazard CDF on observed liquidity should improve bid ranking without changing fill semantics
Observable: lower Gc Brier on every rolling fold and improve robust w1-w4 sum_pnl while preserving positive w5-w6 uplift
Conflicts: none - node 1 improved q while holding Gc fixed, so this attacks the remaining probability component [DONE] (score: 290.3)

**Insight**: Trade-conditioned monotone Gc increased PnL on all six weeks (tune 290.31, holdout 138.58), but Gc Brier improved only on w1-w2 and degraded on w3-w6; the PnL mechanism is robust while calibration drift remains unresolved.

**Result**: Winner gc_new, floor 0.90, min_ev 0.05; weekly pnl 70.35,68.82,68.96,82.18,68.55,70.03. B_test not used during selection.

**Branch**: session-only-experiment

### 2: Mechanism: bid-expanded joint win-and-fill classifier r(b,X)=P(correct and winner_low<=b|X) replacing the q times Gc independence decomposition
Hypothesis: Gc calibration drift shows correctness and fill depth are conditionally coupled; directly estimating their joint event should rank bids by realized EV more stably across months
Observable: improve robust w1-w4 sum_pnl and remain above the node-1 base-Gc policy on both untouched w5-w6 weeks without using B_test
Conflicts: node 1.1 improved PnL but degraded Gc Brier after w2; joint-event supervision removes the unstable conditional division [DONE] (score: 241.4)

**Insight**: Direct joint win-fill modeling improved early calibration only; it scored 241.36 on tune and 96.56 on untouched weeks, below node 1's 266.34/128.49, with worsening Brier after w2.

**Result**: Rejected on B_dev; no B_test used.

**Branch**: session-only-experiment

### 3: Mechanism: contemporaneous market-implied correctness anchor derived from selected and opposite pre-decision trade prices with complement normalization
Hypothesis: learned q models retain stale month-specific relationships, while the live market price aggregates current regime information and should reduce cross-month q drift
Observable: market-price or blended q improves Brier and robust PnL on w1-w4 and beats node 1 on both untouched w5-w6 weeks
Conflicts: prior L2 market-mid experiments used a shorter common window; this uses trade-derived anchors across the full rolling period and explicit complement normalization [DONE] (score: 166.1)

**Insight**: Contemporaneous trade price is not a calibrated correctness probability here: every market-anchor variant worsened Brier/logloss, and the winner reverted to raw p_side.

**Result**: Rejected on B_dev: tune 166.13, untouched 65.09; no B_test used.

**Branch**: session-only-experiment

### 4: Mechanism: bid-expanded direct realized-PnL regression with conservative quantile objective over legal pre-decision market state
Hypothesis: q and Gc probability errors compound under distribution shift; directly learning the asymmetric payoff surface should favor lower bids that retain upside while limiting forced wrong-fill losses
Observable: exceed node 1 on robust w1-w4 sum_pnl and both untouched w5-w6 weeks with materially lower mean submitted bid
Conflicts: node 2 modeled a joint probability then reconstructed EV; direct payoff supervision removes probability-calibration dependence entirely [DONE] (score: 68.22)

**Insight**: Direct PnL regression lowered bids but failed to learn stable action ranking: tune sum 68.22 with a negative week, far below probability-factorized policies.

**Result**: Rejected on B_dev; no B_test used.

**Branch**: session-only-experiment

### 5: Mechanism: temporally pooled empirical winner-low CDF with optional p_side stratification replacing the overconfident contextual hazard
Hypothesis: winner-low quantiles are stable while contextual Gc selection is overconfident; a low-variance empirical CDF should choose lower, better-calibrated bids across months
Observable: improve Gc Brier and robust PnL on w1-w4, remain positive on w5-w6, and reduce submitted fill calibration gap
Conflicts: node 1.1 added more contextual Gc capacity and worsened late calibration; this deliberately removes capacity and pools the stable target distribution [DONE] (score: 272.9)

**Insight**: Low-variance empirical Gc stratified by p_side beat the contextual hazard on untouched PnL (148.22 vs 128.49); all-history p-bin CDF improved Gc Brier on all six folds, validating selection-overconfidence as the bottleneck.

**Result**: Selected 28-day p-bin empirical Gc, floor 0.80, min_ev 0.005: tune 272.94, holdout 148.22; B_test reserved for milestone.

**Branch**: session-only-experiment

### 6: Mechanism: covariate-shift-resistant minimal q model using only p_side and pre-decision trade state, fitted before a temporal calibration tail
Hypothesis: the 576-feature q model and trade augmentation degrade on the final month because stale BTC relationships dominate; removing shifting features and calibrating on the latest train tail should preserve only transportable error signals
Observable: improve q Brier and PnL over raw p_side on w1-w4 and every untouched w5-w6 fold using both hazard and empirical Gc
Conflicts: node 1's broad trade model improved rolling folds but final-month q Brier worsened to 0.2088; this counters by feature minimization and temporal calibration [DONE] (score: 199.5)

**Insight**: Feature minimization reduced some q drift, but improvements were not uniform; the best 25% minimal-q blend scored 199.52 tune and 78.79 holdout, below broader trade q.

**Result**: Rejected on B_dev; no new B_test evaluation.

**Branch**: session-only-experiment

### 7: Mechanism: full-universe joint direction-and-bid policy using reconstructed UP/DOWN lows and EV gating instead of inheriting the legacy 70% direction acceptance mask
Hypothesis: the fixed legacy mask discards 30% of markets before the PnL model can evaluate them; 100% direction coverage plus explicit EV no-order gating should expose additional profitable orders without changing fill semantics
Observable: materially increase w1-w4 and untouched w5-w6 sum_pnl while reporting direction accuracy at coverage 1.0 and stable order coverage
Conflicts: prior joint-direction node retained threshold_accepted, so its 29.48 B_test result did not test the target redesign's full-universe action space [DONE] (score: 283.6)

**Insight**: Children findings: [7.1, done, score=260.5] Daily online direction adaptation modestly improved accuracy but reduced robust tune PnL versus static full-universe direction; tune 260.53, holdout 165.61.

**Result**: Selected tree direction, min_q 0.50, hazard floor 0.80, min_ev 0; coverage 1.0. Qualified for a new-action-space B_test milestone.

**Branch**: session-only-experiment

#### 7.1: Mechanism: prequential full-universe direction model retrained daily on all markets settled before the current UTC day
Hypothesis: full-universe value is real on rolling folds but static direction accuracy drops at the final boundary; online direction updates should preserve the enlarged action space while adapting side selection
Observable: exceed node 7 on w1-w4 robust sum_pnl and both untouched weeks, with improved daily direction accuracy after day one
Conflicts: node 8 adapted q only inside the legacy universe; this adapts the upstream side decision that determines both correctness and which low distribution applies [DONE] (score: 260.5)

**Insight**: Daily online direction adaptation modestly improved accuracy but reduced robust tune PnL versus static full-universe direction; tune 260.53, holdout 165.61.

**Result**: Rejected on B_dev; no additional B_test used.

**Branch**: session-only-experiment

### 8: Mechanism: prequential daily online adaptation that refits q and empirical Gc using only markets settled before each decision day
Hypothesis: every static candidate collapses at the month boundary while prior validation outcomes become observable after five minutes; legal walk-forward updates should track the new regime without future-label leakage
Observable: improve daily/weekly B_dev PnL and q/Gc calibration after the first adaptation day, then remain positive on untouched w5-w6 under one frozen update protocol
Conflicts: all prior nodes trained once before validation; this changes the control flow from static inference to causally ordered online learning [DONE] (score: 173.2)

**Insight**: Prequential updates improved q calibration on early folds but did not translate to enough PnL; tune 173.20 and holdout 95.65, below static policies.

**Result**: Rejected on B_dev; causal protocol verified; no B_test used.

**Branch**: session-only-experiment
