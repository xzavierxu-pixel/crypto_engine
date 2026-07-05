# Research Report: Implement and evaluate all T0-T9 experiments from docs/trade l2 optimization-0705.md; maximize B_...

## Results

- B_dev baseline: `42.43`
- B_dev final trunk: `42.43`
- B_test baseline: `42.43`
- B_test final trunk: `42.43`

## Exploration

- Nodes total: `14`
- Scored nodes: `10`
- Merged nodes: `0`

### Top Ideas By Score

- **2.1** `330.8` _done_: Mechanism: T9 lower-confidence-bound abstention tables over p_side, bid, and joint cells Hypothesis: cells with consi...
- **1.1** `290.3` _done_: Mechanism: T0 frozen reproduction of node 1 and 1.1 feature-policy variants Hypothesis: exact code and data reuse wil...
- **1.2** `290.3` _done_: Mechanism: T1 submitted-action calibration audit stratified by bid, q, liquidity, and correctness Hypothesis: aggrega...
- **3.3** `281` _done_: Mechanism: T4 trade/L2 bucket shrinkage and caps applied to hazard Gc ranking Hypothesis: preserving row ranking whil...
- **4.1** `279.1` _done_: Mechanism: T6 loser-exposure constrained optimizer with bid cap, q floor, and liquidity disagreement gate Hypothesis:...
- **3.4** `276.8` _done_: Mechanism: T5 explicit UP/DOWN path and relative liquidity-pressure features Hypothesis: selected-only features miss ...
- **4.2** `250.3` _done_: Mechanism: T7 posterior-lower-bound gate choosing between anchor and best trade/L2 challenger per segment Hypothesis:...
- **3.1** `237.7` _done_: Mechanism: T2 monotone bid-conditioned model of P(correct and winner_low<=bid given X,bid) Hypothesis: joint profitab...
- **3.2** `233.1` _done_: Mechanism: T3 submitted-subset calibration maps for the T2 joint score Hypothesis: calibrating only actions the froze...
- **3.5** `218.5` _done_: Mechanism: T8 two-minute trade/L2 score used only as first-minute auxiliary feature or no-order gate Hypothesis: late...

## Global Insight

All T0-T9 completed. Only T8 and T9 preserved the 42.43 B_test anchor by selecting no auxiliary/no abstention; every active challenger degraded B_test PnL. No promotion is justified.

## Artifacts

- Idea tree JSON: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260705_trade_l2_all\.coordinator\idea_tree.json`
- Idea tree Markdown: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260705_trade_l2_all\.coordinator\idea_tree.md`
- Experiments: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260705_trade_l2_all\experiments`

## Mandatory B_test Results

| Experiment | B_test sum_pnl | Delta vs 42.43 | Decision |
|---|---:|---:|---|
| T0 frozen trade/L2 reproduction | 14.78 | -27.65 | Reject |
| T1 submitted-action audit | 14.78 | -27.65 | Diagnostic only |
| T2 joint profitable-fill | 12.90 | -29.53 | Reject |
| T3 submitted calibration | 4.20 | -38.23 | Reject |
| T4 empirical Gc shrink | 4.65 | -37.78 | Reject |
| T5 side-specific path | 8.47 | -33.96 | Reject |
| T6 loss-exposure constraints | 5.51 | -36.92 | Reject |
| T7 anchor-safe challenger | 4.65 | -37.78 | Reject |
| T8 two-minute auxiliary | 42.43 | 0.00 | Preserve anchor; no auxiliary selected |
| T9 segment abstention | 42.43 | 0.00 | Preserve anchor; no segment rejected |

## Conclusion

No tested active challenger improved the mandatory B_test anchor. T3 improved submitted joint-event Brier and reduced loss exposure, but removed too many profitable orders. T6 similarly rescued less loss PnL than the win PnL it sacrificed. T7's development posterior gate collapsed to the challenger under month shift. T8 and T9 matched the anchor only because chronological development selection rejected their added actions.

No deploy artifact, default execution config, live behavior, or user branch was changed. The isolated research trunk is `arbor/trunk/20260705_trade_l2_all`; promotion requires a separate explicit approval and is not recommended by these results.

## Protocol Notes

- Every experiment recorded B_dev/rolling/B_test as explicitly required by the design document.
- T5's first B_test attempt was interrupted after its one-shot guard was written, then failed during artifact output because the breakdown directory was absent; reruns used the same frozen policy and did not tune on B_test.
- T8 and T7 preflight runs exposed an incorrect reconstruction seed (`20260705` instead of the authoritative `20260703`). They were rerun solely to reproduce the fixed 42.43 anchor; no threshold or model selection used B_test.
