# Idea Tree

**Baseline**: 218.5 | **Trunk**: 218.5

## ROOT: Complete all experiments in docs/requirement0706.md under the frozen B_test and no-promotion protocol. [DONE]

**Insight**: All mandatory G/M/S nodes and conditional gates completed. Market orders are the only strong result under distinct semantics; hybrid modestly beat the limit anchor. No promotion was performed. G1-G3 accidental B_test reads are quarantined.

### 1: Mechanism: Constrained high-Gc EV bidding with chronological Gc recalibration
Hypothesis: Restricting bids to genuinely calibrated high-fill regions can reduce adverse winner non-fill without increasing forced-wrong losses beyond the added winner value
Observable: G0 reproduces 42.43; G1-G3 pass w1-w6 gates and report B_test sum_pnl plus realized submitted fill calibration
Conflicts: none - tests the plan's fill-threshold axis on the frozen accepted universe [DONE]

**Insight**: High-Gc G1-G3 all failed the required >=3 positive w1-w4 delta gate. Recalibration reduced Gc error but did not improve tune PnL; no G candidate is promotable.

#### 1.1: Mechanism: Exact frozen anchor replay using raw_tree_blend, Gc floor 0.85, and min_ev 0.02
Hypothesis: Reusing the accepted preparation and bid functions on the frozen train/test files should reproduce 42.43 and validate the environment before any search
Observable: B_test sum_pnl matches 42.43 within floating-point tolerance and all required accounting fields are emitted
Conflicts: none - mandatory preflight before changing policy mechanics [DONE] (score: 42.43)

**Insight**: Exact raw_tree_blend/0.85/0.02 replay reproduced 42.43; submitted correct-fill realized 0.7176 versus modeled 0.8625, confirming a 0.1449 optimism gap.

**Result**: G0 passed with all required artifacts and one ledger row.

**Branch**: session-artifact:G0_anchor_reproduction

#### 1.2: Mechanism: High-fill feasible-set scan over Gc floors 0.90 to 0.95 with EV abstention
Hypothesis: Constraining the action set rather than merely ranking all bids can increase realized winner fills, but must be selected by rolling robust PnL to control higher forced-loser cost
Observable: At least three of w1-w4 beat their anchor, w5 and w6 remain positive, and submitted realized correct-fill approaches 0.85
Conflicts: G0 found a 0.1449 optimistic fill gap; this tests whether a nominally higher floor overcomes it [DONE] (score: 202.9)

**Insight**: Nominal Gc>=0.90 raised realized submitted correct-fill to the quarantined diagnostic value 0.801 but only one of four tune weeks beat anchor; selection gate failed.

**Result**: G1 failed w1-w4 and its accidental B_test read was quarantined.

**Branch**: session-artifact:G1_high_fill_scan

#### 1.3: Mechanism: Joint high-fill action constraint and shrinkage of selected-side correctness toward one half
Hypothesis: Conservative q can reject expensive high-floor bids whose apparent EV depends on overconfident correctness, reducing forced-wrong loss while preserving filled winners
Observable: The w1-w4 robust winner passes positive w5-w6 and improves win/loss decomposition relative to G1
Conflicts: G0 showed Gc optimism rather than q-only error; this combines orthogonal q conservatism with the mandated fill constraint [DONE] (score: 202.9)

**Insight**: q shrinkage did not win; shrink=0 remained best and only one of four tune weeks beat anchor.

**Result**: G2 failed w1-w4 and its accidental B_test read was quarantined.

**Branch**: session-artifact:G2_high_fill_conservative_q

#### 1.4: Mechanism: Chronological 0.025-width p_side-bin empirical winner-low CDF blended 15 percent into hazard Gc
Hypothesis: A training-only empirical reliability correction can reduce the submitted Gc Brier and optimism gap so a nominal 0.90 floor represents actual fills more faithfully
Observable: Recalibrated Gc lowers rolling Brier/gap and the frozen candidate passes w5-w6 before one B_test evaluation
Conflicts: G0 showed nominal Gc overstates submitted fills; this directly recalibrates that forecast instead of raising bids further [DONE] (score: 198.9)

**Insight**: 15% empirical p_side-bin CDF blend improved grid Brier but zero of four tune weeks beat anchor; calibration improvement did not convert to PnL.

**Result**: G3 failed w1-w4 and its accidental B_test read was quarantined.

**Branch**: session-artifact:G3_recalibrated_gc

### 2: Mechanism: Timestamp-bounded market-order policy using the last selected-token trade by market_t0 plus 68 seconds
Hypothesis: Removing winner-fill adverse selection can improve PnL when q minus executable market price is selected chronologically and missing-price rows abstain
Observable: M0 proves timestamp integrity and coverage; M1-M2 pass rolling gates and beat the same-universe limit anchor on B_test
Conflicts: none - changes order semantics only inside a separately reported track [DONE]

**Insight**: M0 timestamp QA passed. M1 market orders achieved 122.63; paired M2 beat the legal-m limit anchor 45.41 by 77.22; M3 hybrid achieved 46.30 versus 42.43.

#### 2.1: Mechanism: Point-in-time market-price join using the last selected-token trade no later than market_t0 plus 68 seconds
Hypothesis: A condition-and-side keyed as-of join can establish an executable market-price universe without leaking later path trades
Observable: M0 reports coverage and distributions and proves every m_trade_time is at or before its row cutoff
Conflicts: none - mandatory timestamp-integrity prerequisite [DONE] (score: 0)

**Insight**: M0 found legal market prices for the majority of accepted rows with zero post-cutoff joins; B_test m coverage is recorded in QA.

**Result**: M0 timestamp QA passed.

**Branch**: session-artifact:M0_market_price_qa

#### 2.2: Mechanism: Market-order EV threshold policy over raw_tree_blend, isotonic, and raw p_side q forecasts
Hypothesis: Always filling at observed m removes winner non-fill adverse selection when q-minus-m is large enough, though calibration drift may concentrate wrong orders
Observable: The w1-w4 robust candidate has positive w5-w6 and reports B_test order coverage, wrong-order share, and loser cost once
Conflicts: none - separately changes entry order semantics with explicit timestamps [DONE] (score: 390.5)

**Insight**: Market EV selected raw_tree_blend with tau=0; w5-w6 were positive and B_test sum_pnl was 122.63 under market-order semantics.

**Result**: M1 passed and was evaluated once on B_test.

**Branch**: session-artifact:M1_market_ev_scan

#### 2.3: Mechanism: Same-universe counterfactual ledger pairing the frozen M1 market candidate with the exact limit anchor
Hypothesis: Restricting both policies to rows with legal m isolates order-type value from market-price data coverage
Observable: M2 reports paired B_test PnL and win/loss decomposition on identical rows, with promotion judged only against that paired anchor
Conflicts: none - prevents mixed-universe comparison to 42.43 [DONE] (score: 390.5)

**Insight**: On 5,164 rows with legal m, market PnL 122.63 beat the paired limit anchor 45.41 by 77.22.

**Result**: M2 fair same-universe comparison passed and triggered M3.

**Branch**: session-artifact:M2_same_universe_comparison

#### 2.4: Mechanism: Conditional order router using market entry only for large q-minus-m edge when anchor Gc is low, otherwise retaining the frozen limit action
Hypothesis: A hybrid can capture market-order removal of adverse winner non-fill selectively while preserving cheaper limit entries where modeled fill is already strong
Observable: The w1-w4 robust router has positive w5-w6 and one B_test result against the exact full-universe anchor
Conflicts: M2 showed broad market orders already dominate on the legal-m subset; this tests whether selective routing adds value rather than assuming it [DONE] (score: 236.9)

**Insight**: Selective hybrid passed w5-w6 and improved B_test from 42.43 to 46.30, but broad M1 market routing remained much stronger at 122.63 on its distinct semantics.

**Result**: M3 completed one B_test evaluation and beat the full-universe limit anchor by 3.87.

**Branch**: session-artifact:M3_limit_market_hybrid

### 3: Mechanism: Post-fill path reconstruction with stop-loss and take-profit state transitions
Hypothesis: Capping forced-wrong losses after actual limit entry can outweigh false stops on eventual winners under conservative execution prices
Observable: S0 reconstructs entry and post-entry paths; S1-S2 improve B_test PnL in primary and conservative variants without slippage sign reversal
Conflicts: none - changes only post-fill holding management on the anchor order set [DONE]

**Insight**: S0 reduced loser loss but false-stopped too many winners, yielding 20.91. S1/S2 failed tune gates and S3 was not triggered.

#### 3.1: Mechanism: Exact anchor entries plus deterministic stop s=min(0.30,0.50*bid) reconstructed from selected-token trades
Hypothesis: Post-fill path state can cap wrong-side losses while exposing how many eventual winners cross the same stop before settlement
Observable: S0 reports entry reconstruction coverage, stopped losers, false-stopped winners, and primary/conservative/pressure PnL against identical anchor orders
Conflicts: none - mandatory path-reconstruction prerequisite for holding-policy experiments [DONE] (score: 150.1)

**Insight**: Fixed stop cut B_test losses from -316.44 to -176.83 but false-stopped 227 winners, reducing wins enough for net PnL 20.91; conservative/pressure variants were worse.

**Result**: S0 path reconstruction passed technically but stop mechanism was net negative.

**Branch**: session-artifact:S0_fixed_stop_preflight

#### 3.2: Mechanism: Rolling search over proportional-capped and fixed stop levels with three execution-price models
Hypothesis: A stop surface tied to entry price can retain more winner convexity than a single fixed stop while materially compressing forced-wrong losses
Observable: At least three tune weeks improve, w5-w6 stay positive, and primary plus conservative B_test exceed the same-order anchor without pressure sign reversal
Conflicts: none - changes post-fill exits only, leaving G0 selection and fills fixed [DONE] (score: 170.7)

**Insight**: Best stop was fixed 0.20, yet zero tune weeks beat the identical-order anchor; S1 failed before holdout/B_test.

**Result**: S1 selection gate failed; B_test was not read.

**Branch**: session-artifact:S1_stop_grid

#### 3.3: Mechanism: First-trigger take-profit overlay at 0.90 or 0.95 on the frozen best stop policy
Hypothesis: Early profit-taking is useful only if reduced exposure to later stop-outs exceeds foregone settlement payoff, which chronological first-trigger simulation can test
Observable: Rolling robust PnL improves over S1 and the holdout gate passes before one B_test evaluation
Conflicts: requirement expects take-profit may be harmful; this directly measures rather than assumes its value [DONE] (score: 132.2)

**Insight**: Take-profit overlays further reduced rolling PnL and the S1 prerequisite had already failed.

**Result**: S2 was correctly skipped on B_test.

**Branch**: session-artifact:S2_take_profit_overlay

#### 3.4: Mechanism: Conditional composition of the best validated stop with a winning high-fill or market-entry policy
Hypothesis: Composition is justified only if S1 independently shows stable positive holding-management value, preventing a failed stop from contaminating a strong entry policy
Observable: Launch only when S1 passes its rolling and B_test criteria; otherwise persist an explicit not-triggered artifact with zero B_test reads
Conflicts: S1 failed all four tune-week deltas, so the prerequisite is false and this node must be pruned [PRUNED]

**Insight**: S1 improved zero tune weeks, so the explicit S3 prerequisite was false; no B_test read.

**Result**: Conditional experiment not triggered; complete skipped artifact set saved.

**Branch**: session-artifact:S3_combination_not_triggered
