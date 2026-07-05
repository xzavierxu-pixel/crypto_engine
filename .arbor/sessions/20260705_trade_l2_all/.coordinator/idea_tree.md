# Idea Tree

**Baseline**: 42.43 | **Trunk**: 42.43

## ROOT: Implement and evaluate all T0-T9 experiments from docs/trade l2 optimization-0705.md; maximize B_test sum_pnl versus 42.43 with required artifacts and no deployment promotion. [DONE]

**Insight**: All T0-T9 completed. Only T8 and T9 preserved the 42.43 B_test anchor by selecting no auxiliary/no abstention; every active challenger degraded B_test PnL. No promotion is justified.

**Result**: Ten experiments produced B_dev, rolling, B_test, predictions, manifests, leakage checks, reports, and ledger rows; best B_test remained 42.43.

### 1: Mechanism: reproducible trade/L2 baseline and submitted-action audit pipeline
Hypothesis: month collapse cannot be corrected until the exact action-level calibration and loss decomposition are reproduced under one immutable evaluator
Observable: T0 reproduces rolling evidence and T1 attributes the B_test gap to q, fill calibration, or loss exposure
Conflicts: none - prior sessions reported milestones but did not emit the complete mandated ledger and artifacts [DONE]

**Insight**: T0/T1 reproduced and diagnosed severe B_test degradation despite strong rolling PnL.

#### 1.1: Mechanism: T0 frozen reproduction of node 1 and 1.1 feature-policy variants
Hypothesis: exact code and data reuse will distinguish genuine rolling reproducibility from historical implementation drift
Observable: B_dev and w1-w6 match prior results within tolerance and a first complete B_test artifact row is produced
Conflicts: none - this is the required falsifiable baseline [DONE] (score: 290.3)

**Insight**: T0 exactly reproduced rolling node1/node1.1, but B_test fell to 40.00/14.78; trade-conditioned Gc increased wrong submissions and loss exposure under month drift.

**Result**: T0 complete: node1 w1-w4 266.34 and B_test 40.00; node1.1 w1-w4 290.31, w5-w6 138.58, B_test 14.78; exactly-once guard and leakage checks passed.

**Branch**: bb31a1a

#### 1.2: Mechanism: T1 submitted-action calibration audit stratified by bid, q, liquidity, and correctness
Hypothesis: aggregate fill metrics hide whether month degradation comes from correct-fill overprediction or forced-wrong loss concentration
Observable: calibration gaps and B_test PnL decomposition identify a dominant failure bucket without changing policy
Conflicts: none - diagnostic-only experiment [DONE] (score: 290.3)

**Insight**: Frozen trade-Gc candidate is overconfident on submitted correct fills: predicted 0.91659 versus realized 0.67142; bid 0.60-0.70 and q 0.70-0.80 dominate B_test losses.

**Result**: T1 diagnostic completed: B_test sum_pnl 14.78, delta -27.65, 809 wrong submissions, calibration gap -0.24517; 4 tests passed.

**Branch**: 0aad794

### 2: Mechanism: chronological confidence-bound segment abstention learned from each experiment's own development distribution
Hypothesis: forced wrong-fill losses cluster in stable price-confidence cells that EV overconfidence fails to reject
Observable: T9 reduces B_test loss_pnl_sum and increases sum_pnl while quantifying sacrificed wins and rolling stability
Conflicts: none - prior q gate used one fixed threshold rather than learned multi-fold segment evidence [DONE]

**Insight**: T9 found no stable negative segments and retained the anchor.

#### 2.1: Mechanism: T9 lower-confidence-bound abstention tables over p_side, bid, and joint cells
Hypothesis: cells with consistently negative development PnL can be removed without hard-coding the observed B_test month
Observable: three aggressiveness levels report B_test delta, saved loss, killed wins, coverage change, and rolling stability
Conflicts: prior node 2.2 used fixed q>=0.55; this learns cell decisions chronologically [DONE] (score: 330.8)

**Insight**: Chronological LCB segment abstention found no stable negative cells after freezing all six folds; the B_test anchor was preserved with zero abstentions, showing fixed low-confidence bans are unsupported by the available development evidence.

**Result**: T9 completed: w1-w6 sum 330.82, B_test 42.43, delta 0.00, all 3x3 frozen variants reported; no orders rejected.

**Branch**: 1b9b53f

### 3: Mechanism: bid-conditioned joint profitable-fill models with trade/L2 relative-state representation and submitted-action calibration
Hypothesis: directly modeling correct-and-filled avoids multiplicative q-times-Gc error and exposes market-path selection information
Observable: T2-T5 and T8 improve joint-event calibration and B_test sum_pnl over the reproduced T0 baseline
Conflicts: prior node 1.1 improved rolling PnL but B_test fell to 14.78; joint credit assignment replaces its separability assumption [DONE]

**Insight**: T2-T5/T8 did not beat the 42.43 anchor; calibration improved Brier but not PnL.

#### 3.1: Mechanism: T2 monotone bid-conditioned model of P(correct and winner_low<=bid given X,bid)
Hypothesis: joint profitable-fill prediction removes independence error between direction correctness and winner fill
Observable: joint-event Brier and B_test sum_pnl improve over T0 with complete win-loss decomposition
Conflicts: prior joint-winfill attempt lacked the full trade/L2 representation and mandated submitted calibration [DONE] (score: 237.7)

**Insight**: Joint profitable-fill worsened submitted-action calibration and B_test despite positive rolling folds.

**Result**: B_dev w1-w4 sum 237.70; holdout w5-w6 127.40; B_test 12.90 versus 42.43 anchor.

**Branch**: 62b720a

#### 3.2: Mechanism: T3 submitted-subset calibration maps for the T2 joint score
Hypothesis: calibrating only actions the frozen policy would submit corrects selection-induced overconfidence
Observable: raw-versus-calibrated B_test comparison lowers calibration gap or loss without excessive order collapse
Conflicts: none - calibration is fit only on the chronological calibration window [DONE] (score: 233.1)

**Insight**: Submitted-action Platt calibration improved joint Brier and reduced loss exposure, but removed profitable orders and lowered B_test PnL.

**Result**: B_dev w1-w4 233.10, w5-w6 118.90; B_test 4.20, delta -38.23 vs anchor and -8.70 vs raw T2.

**Branch**: 55584a1

#### 3.3: Mechanism: T4 trade/L2 bucket shrinkage and caps applied to hazard Gc ranking
Hypothesis: preserving row ranking while shrinking overconfident conditional fill probabilities improves action selection
Observable: empirical, hazard-only, shrink-blend, and upper-cap variants are compared on rolling and B_test PnL plus Gc Brier
Conflicts: prior fully empirical Gc lost row-level ranking; this explicitly retains it [DONE] (score: 281)

**Insight**: Empirical Gc shrink variants underperformed the frozen hazard-only policy; selected hazard-only by w1-w4.

**Result**: B_dev w1-w4 280.96, w5-w6 129.10; B_test 4.65 versus 42.43 anchor.

**Branch**: 05f104d

#### 3.4: Mechanism: T5 explicit UP/DOWN path and relative liquidity-pressure features
Hypothesis: selected-only features miss opposite-token pressure and complement deviations that predict adverse selection
Observable: ablations attribute B_test gain to q quality versus fill selection and report side-specific diagnostics
Conflicts: prior path signatures improved Brier but not PnL; relative side state attacks omitted representation [DONE] (score: 276.8)

**Insight**: Side-specific feature ablations did not beat the legacy trade/L2 feature set; the frozen selector retained legacy.

**Result**: B_dev robust score 276.8364; frozen legacy candidate; B_test 8.47 versus 42.43 anchor.

**Branch**: 5ccd357

#### 3.5: Mechanism: T8 two-minute trade/L2 score used only as first-minute auxiliary feature or no-order gate
Hypothesis: later-horizon market path can identify first-minute loser exposure without replacing the production horizon
Observable: B_test loss_pnl_sum falls while first-minute direction and accepted universe remain fixed
Conflicts: standalone two-minute replacement is explicitly excluded [DONE] (score: 218.5)

**Insight**: Two-minute auxiliary vetoes did not identify first-minute loser exposure robustly; B_dev selected no auxiliary.

**Result**: B_dev w1-w4 218.50, w5-w6 112.32; B_test retained anchor exactly at 42.43 with zero vetoes.

**Branch**: cf93649

### 4: Mechanism: explicit loss-exposure constraints and posterior-safe anchor replacement
Hypothesis: challenger gains survive month drift only when loser bid exposure is capped and replacements require positive historical lower bounds
Observable: T6 lowers wrong-submitted loss and T7 preserves or exceeds anchor B_test with auditable replacement counts
Conflicts: prior node 2.2 fixed q gate raised B_test only from 28.82 to 29.27; this controls action-specific loss and replacement uncertainty [DONE]

**Insight**: T6 constraints and T7 safe replacement did not preserve anchor PnL.

#### 4.1: Mechanism: T6 loser-exposure constrained optimizer with bid cap, q floor, and liquidity disagreement gate
Hypothesis: optimizing expected reward without an explicit loser-cost proxy submits expensive wrong orders under drift
Observable: wrong-submitted rate and average loser bid fall with a favorable B_test delta versus the unconstrained challenger
Conflicts: prior fixed q gate barely changed B_test; multi-signal loss constraints target actual exposure [DONE] (score: 279.1)

**Insight**: Loss constraints selected only q>=0.60; stricter caps/gates sacrificed more winning PnL than loss exposure they rescued.

**Result**: B_dev robust 279.134; B_test 5.51, delta -36.92 vs anchor and -2.96 vs unconstrained.

**Branch**: 5846e86

#### 4.2: Mechanism: T7 posterior-lower-bound gate choosing between anchor and best trade/L2 challenger per segment
Hypothesis: selective replacement captures stable challenger gains while reverting uncertain segments to the 42.43 anchor
Observable: safe-gated B_test beats or preserves anchor and reports replacement count plus replaced win/loss PnL
Conflicts: none - this is the required promotion-safe composition layer [DONE] (score: 250.3)

**Insight**: Posterior safe gate collapsed toward the T4 challenger under month shift and failed to preserve anchor PnL.

**Result**: B_dev w1-w4 250.31, w5-w6 129.10; B_test anchor 42.43, challenger 4.65, safe 4.65 with 2018 replacements.

**Branch**: e1626f3
