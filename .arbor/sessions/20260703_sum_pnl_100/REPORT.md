# Research Report: Run exactly 100 no-leak B_dev policy experiments; maximize robust chronological sum_pnl; target f...

## Results

- B_dev baseline: `88.38`
- B_dev final trunk: `88.38`
- B_test baseline: `27.44`
- B_test final trunk: `27.44`

Exactly 100 candidates were evaluated on two chronological B_dev folds. The
grid crossed five isotonic-q blend weights, five strict `candidate_gc` floors,
and four `min_ev` values. Candidate selection maximized the worst improvement
over the fold-specific baseline, then mean improvement; it did not rank by the
single main-fold score shown in the generic Arbor top-ideas list below.

The selected candidate was experiment `37` / node `1.37`:

- `q = 0.75 * raw_p_side + 0.25 * isotonic_q`
- strict `candidate_gc > 0.85`
- `min_ev = 0.00`
- main B_dev: `88.38 -> 98.10` (`+9.72`)
- earlier B_dev: `82.34 -> 86.51` (`+4.17`)

It was then evaluated once on frozen B_test and scored `5.48`, below the valid
baseline `27.44` and far below the requested target `100`. The candidate was
rejected, no node was merged, and deploy/live artifacts remain unchanged.

All evaluated checkpoints contain 576 features. The run aborts if any feature
name contains `sample_weight`; the leakage guard passed, including the explicit
ban on `stage1_sample_weight`.

## Exploration

- Nodes total: `101`
- Scored nodes: `101`
- Merged nodes: `0`

### Top Ideas By Score

- **1.53** `100` _done_: Mechanism: Joint policy candidate 53 with q-alpha=0.5, Gc-floor=0.8, min-EV=0 Hypothesis: This interaction may contro...
- **1.57** `100` _done_: Mechanism: Joint policy candidate 57 with q-alpha=0.5, Gc-floor=0.85, min-EV=0 Hypothesis: This interaction may contr...
- **1.99** `98.12` _done_: Mechanism: Joint policy candidate 99 with q-alpha=1, Gc-floor=0.85, min-EV=0.02 Hypothesis: This interaction may cont...
- **1.75** `98.11` _done_: Mechanism: Joint policy candidate 75 with q-alpha=0.75, Gc-floor=0.8, min-EV=0.02 Hypothesis: This interaction may co...
- **1** `98.1` _done_: Mechanism: Robust two-fold joint search over q shrinkage, Gc floor, and EV admission Hypothesis: Jointly controlling ...
- **1.37** `98.1` _done_: Mechanism: Joint policy candidate 37 with q-alpha=0.25, Gc-floor=0.85, min-EV=0 Hypothesis: This interaction may cont...
- **1.33** `97.91` _done_: Mechanism: Joint policy candidate 33 with q-alpha=0.25, Gc-floor=0.8, min-EV=0 Hypothesis: This interaction may contr...
- **1.74** `96.39` _done_: Mechanism: Joint policy candidate 74 with q-alpha=0.75, Gc-floor=0.8, min-EV=0.01 Hypothesis: This interaction may co...
- **1.54** `96.15` _done_: Mechanism: Joint policy candidate 54 with q-alpha=0.5, Gc-floor=0.8, min-EV=0.01 Hypothesis: This interaction may con...
- **1.29** `95.53` _done_: Mechanism: Joint policy candidate 29 with q-alpha=0.25, Gc-floor=0.75, min-EV=0 Hypothesis: This interaction may cont...

## Global Insight

Children findings: [1, pending] Children findings: [1.1, done, score=88.9] Recorded as part of the complete two-fold interaction search. | [1.2, done, score=78.56] Recorded as part of the complete two-fold interaction search. | [1.3, done, score=64.83] Recorded as part of the complete two-fold interaction search. | [1.4, done, score=36.69] Recorded as part of the complete two-fold interaction search. | [1.5, done, score=89.4] Recorded as part of the complete two-fold interaction search. | [1.6, done, score=81.24] Recorded as part of the complete two-fold interaction search. | [1.7, done, score=66.26] Recorded as part of the complete two-fold interaction search. | [1.8, done, score=37.29] Recorded as part of the complete two-fold interaction search. | [1.9, done, score=88.38] Recorded as part of the complete two-fold interaction search. | [1.10, done, score=80.01] Recorded as part of the complete two-fold interaction search. | [1.11, done, score=68.76] Recorded as part of the complete two-fold interaction search. | [1.12, done, score=37.07] Recorded as part of the complete two-fold interaction search. | [1.13, done, score=89.99] Recorded as part of the complete two-fold interacti...

## Artifacts

- Idea tree JSON: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_100\.coordinator\idea_tree.json`
- Idea tree Markdown: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_100\.coordinator\idea_tree.md`
- Experiments: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_100\experiments`
