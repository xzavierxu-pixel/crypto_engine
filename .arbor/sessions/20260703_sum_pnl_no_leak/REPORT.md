# Research Report: Maximize B_dev sum_pnl, preserve frozen final-month B_test, beat valid baseline 27.44 without lea...

## Decision

- Exit reason: completed the authorized 10-cycle budget.
- Promotion: rejected. No experiment was merged and no deploy/live files were changed.
- Best single-split node: `1.5`, a 25% empirical-CDF blend with 0.025-wide `p_side` bins, improved B_dev `sum_pnl` from `88.38` to `95.84` and reduced submitted-fill calibration gap from `0.17334` to `0.12958`.
- Chronological falsification: on 2026-03-14 through 2026-03-27, the same mechanism reduced the calibration gap from `0.14707` to `0.12835` but reduced `sum_pnl` from `82.34` to `65.96`. The apparent PnL gain is not temporally robust.
- B_test discipline: `27.44` is the reproduced unchanged H14 baseline, not a score for node `1.5`. No candidate was evaluated on B_test after the earlier-fold failure.
- Leakage audit: the reproduced baseline checkpoint has 576 features; `stage1_sample_weight` and all other sample-weight-like columns are absent. The B_dev configs explicitly forbid `stage1_sample_weight`.

## Main Findings

- Blending a small empirical correction into H2 Gc retained useful row-level ranking; replacing H2 with a fully empirical CDF substantially reduced PnL.
- Lower aggregate fill-calibration error did not guarantee higher PnL. Direct empirical Gc reached a `0.09282` gap but only `54.03` PnL.
- The current H14 runner ignores the checkpoint's q calibrator and reads raw `p_side`; q calibration cannot affect this path until the downstream probability interface is made explicit.
- The 7-day calibration tail is too unstable to justify fine-bin policy promotion from one split.

## Results

- B_dev baseline: `88.38`
- B_dev final trunk: `88.38`
- B_test baseline: `27.44`
- B_test final trunk: `27.44`

## Exploration

- Nodes total: `14`
- Scored nodes: `10`
- Merged nodes: `0`

### Top Ideas By Score

- **1.5** `95.84` _done_: Mechanism: Fine 0.025 p_side-bin empirical-CDF blend Hypothesis: Finer conditioning may capture heterogeneous fill cu...
- **1.1** `94.18` _done_: Mechanism: Cross-fitted p_side-bin empirical-CDF blend Hypothesis: Blending H2 Gc with calibration-only empirical CDF...
- **1.2** `89.79` _done_: Mechanism: Beta-posterior upper cap on p_side-bin Gc Hypothesis: Capping overconfident H2 curves with calibration-onl...
- **4.1** `88.38` _done_: Mechanism: Calibration-tail isotonic q with retrained H2 Gc Hypothesis: Monotone correctness calibration should impro...
- **1.4** `86.66` _done_: Mechanism: Pooled 0.20 p_side-bin empirical-CDF blend Hypothesis: Aggressive pooling should expose whether p_side con...
- **1.3** `80.24` _done_: Mechanism: Coarse p_side-bin empirical-CDF blend Hypothesis: Wider 0.10 bins should lower calibration variance while ...
- **3.1** `65.96` _done_: Mechanism: Earlier blocked-fold falsification of the 0.025-bin blend Hypothesis: A real Gc calibration improvement sh...
- **2.2** `54.03` _done_: Mechanism: Fit-window empirical p_side CDF policy Hypothesis: A larger historical fit window should reduce empirical-...
- **2.1** `51.87` _done_: Mechanism: Calibration-tail empirical p_side CDF policy Hypothesis: A fully empirical Gc eliminates neural calibratio...
- **2.3** `29.23` _done_: Mechanism: Coarse-bin fit-window empirical CDF policy Hypothesis: Combining the large fit window with 0.10 bins shoul...

## Global Insight

Children findings: [1, pending] Children findings: [1.1, done, score=94.18] A 25% calibration-only empirical-CDF blend reduced submitted fill calibration gap from 0.1733 to 0.1310 and improved B_dev sum_pnl from 88.38 to 94.18; gains came with higher order coverage 0.834. | [1.2, done, score=89.79] The beta upper-cap correction reduced the fill calibration gap to 0.1397 but improved B_dev only 1.41, underperforming the smoother empirical blend. | [1.3, done, score=80.24] 0.10 bins over-pooled and selected min_ev 0.01; B_dev fell to 80.24 and calibration gap returned to 0.1737. | [1.4, done, score=86.66] 0.20 bins reduced the calibration gap to 0.1622 but lost PnL; p_side conditioning is useful and aggressive pooling removes signal. | [1.5, done, score=95.84] Fine 0.025 bins with 25% empirical blend improved B_dev to 95.84 and reduced the calibration gap to 0.1296; this is the best current node. | [2, pending] Children findings: [2.1, done, score=51.87] Calibration-tail empirical Gc discarded useful H2 row-level ranking and cut B_dev PnL nearly in half despite high order coverage. | [2.2, done, score=54.03] Fit-window empirical Gc achieved a low calibration gap 0.0928 but poor Pn...

## Artifacts

- Idea tree JSON: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_no_leak\.coordinator\idea_tree.json`
- Idea tree Markdown: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_no_leak\.coordinator\idea_tree.md`
- Experiments: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_no_leak\experiments`
