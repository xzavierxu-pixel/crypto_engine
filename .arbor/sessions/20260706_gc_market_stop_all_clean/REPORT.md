# Research Report — Gc / Market Order / Stop Management 0706

## Scope and exit state

Executed every mandatory node in `docs/requirement0706.md`, plus M3 because M2 passed. S3 was not executed because its explicit S1-success prerequisite failed. The accepted/deployed artifacts, `2mins` workflow, labels, live configuration, and execution behavior were not promoted or changed.

- Frozen limit-order B_test anchor: `42.43`
- Rolling B_dev anchor, w1–w4 sum: `218.50`
- Exit reason: planned nodes and conditional gates exhausted
- Merge/promotion count: `0`
- Artifact audit: passed
- Scientific protocol audit: passed in this corrected evidence package

## Results

| Experiment | Gate / B_test status | B_test sum_pnl | Fair anchor | Delta | Conclusion |
|---|---:|---:|---:|---:|---|
| G0 anchor | reproduced once | 42.43 | 42.43 | 0.00 | Environment and frozen baseline reproduced |
| G1 high fill | w1–w4 failed | — | 42.43 | — | Only 1/4 tune weeks improved |
| G2 conservative q | w1–w4 failed | — | 42.43 | — | Shrink 0 remained best; 1/4 weeks improved |
| G3 recalibrated Gc | w1–w4 failed | — | 42.43 | — | Gc Brier improved, but 0/4 weeks improved PnL |
| S0 fixed stop | fixed preflight | 20.91 | 42.43 | -21.52 | Losses shrank, but false-stopped winners erased the gain |
| S1 stop grid | w1–w4 failed | — | 42.43 | — | Best fixed stop 0.20 improved 0/4 weeks |
| S2 take profit | prerequisite/gate failed | — | 42.43 | — | Take-profit overlay further reduced rolling PnL |
| S3 combination | not triggered | — | 42.43 | — | Correctly skipped because S1 failed |
| M0 market-price QA | QA once | — | — | — | B_test m coverage 98.78%; zero late joins |
| M1 market EV | w5–w6 passed | 122.63 | distinct semantics | — | Strong market-order result, not directly ranked against 42.43 |
| M2 paired comparison | shared M1 evaluation | 122.63 | 45.41 | +77.22 | Market beat limit on the same 5,164-row legal-m universe |
| M3 hybrid | w5–w6 passed | 46.30 | 42.43 | +3.87 | Selective market/limit routing beat the full-universe anchor |

## Main diagnostics

G0 showed the hazard model remained optimistic on submitted correct orders: modeled fill `0.8625`, realized fill `0.7176`, calibration gap `0.1449`.

G3's 15% chronological empirical p_side-bin blend reduced frozen-month grid Brier from `0.12382` to `0.12274`, but calibration improvement did not translate into rolling PnL. The high-fill family raised bid cost and did not satisfy the tune gate.

S0 reduced B_test loss PnL from `-316.44` to `-176.83`, but falsely stopped `227` eventual winners. Primary, conservative actual-trade-price, and 0.02 pressure variants returned `20.91`, `-3.74`, and `6.75` respectively.

M1 selected `raw_tree_blend` with `tau=0.0` using w1–w4 only. Its B_test order count was `3,115`, sum_pnl `122.63`, win PnL `734.63`, loss PnL `-612.01`, and average loser cost `0.6065`. This is a separate market-order semantic track. M2 supplies the required fair comparison: the same legal-m rows produced `45.41` with the frozen limit anchor.

M3 selected market routing when `q-m > 0.05` and anchor fill probability was below `0.75`; otherwise it retained the anchor limit action. It produced `46.30`, a `+3.87` full-universe improvement, but remained well below broad M1 market routing.

## Leakage and timestamp audit

- All 12 named experiment directories contain the required report, config, feature manifest, leakage check, B_dev metrics, B_test metrics, and prediction parquet.
- All prediction schemas contain IDs, decision timestamps, side, q, action price, expected EV, fill fields, correctness, and realized PnL.
- Every leakage check passed; post-fill path values were used only for backtest accounting.
- Every M-series `m_trade_time` was at or before `market_t0 + 68s`; late join count was `0`.
- The ledger contains one unique row for each named experiment.
- No protected deployment, live config, label, or accepted artifact path changed.

## Provenance

This corrected session reran G1–G3 using only w1–w6 caches and without a B_test cache; all three stopped at the failed tune gate with zero B_test reads. Protocol-valid G0, M, and S artifacts were migrated unchanged from the source session. `PROVENANCE.md` discloses the source session's quarantined implementation diagnostics; they are excluded from this session's metrics, ledger, predictions, and decisions.

## Durable artifacts

- `gc_market_stop_btest_ledger.csv`: complete B_test accounting ledger
- `COMPLETION_AUDIT.json`: requirement and artifact audit
- `.coordinator/idea_tree.json`: Arbor Idea Tree
- `experiments/`: named reports, configs, predictions, leakage checks, and metrics
- `run_stats.json` and `events.jsonl`: final run state

No result has been promoted. Promotion requires separate user approval.
