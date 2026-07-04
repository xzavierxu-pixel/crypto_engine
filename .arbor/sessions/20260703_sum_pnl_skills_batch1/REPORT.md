# Research Report: Continue no-leak sum_pnl research using five disabled skills as mechanisms; 10 evaluations across...

## Results

- B_dev baseline: `88.38`
- B_dev final trunk: `88.38`
- B_test baseline: `27.44`
- B_test final trunk: `27.44`

The only mechanism that improved both chronological B_dev folds was node 2,
a 50/50 blend of raw `p_side` and calibration-only isotonic `q`. Policy
selection jointly searched `candidate_gc` floor and `min_ev` on the second
half of the calibration tail; neither B_dev fold participated in fitting or
selection.

| Mechanism | Main B_dev | Earlier B_dev |
|---|---:|---:|
| Fold-specific H14 baseline | 88.38 | 82.34 |
| isotonic q | 80.60 | 59.68 |
| raw/isotonic shrink | **97.21** | **90.25** |
| bounded q | 69.62 | 54.92 |
| UTC-session residual q | 80.01 | 62.32 |
| p_side-bin residual q | 79.50 | 47.46 |

Node 2 was therefore evaluated once on frozen B_test. It selected
`candidate_gc > 0.80` and `min_ev = 0.02`, but produced `sum_pnl = -12.14`
versus the valid baseline `27.44`. It is rejected; `test_trunk_score` remains
`27.44`, and no deploy/live artifacts were changed.

The five sampled disabled skills were adversarial validation, bootstrapped
residual prediction intervals, confidence probability clipping, cyclical
feature encoding, and declarative groupby aggregation. They were adapted as
policy/calibration mechanisms rather than copied as label-aware feature code.
All three evaluated H2 checkpoints contain 576 features and no
`stage1_sample_weight` or other sample-weight-named feature.

## Exploration

- Nodes total: `5`
- Scored nodes: `5`
- Merged nodes: `0`

### Top Ideas By Score

- **2** `97.21` _done_: Mechanism: Raw/isotonic shrinkage ensemble for q Hypothesis: Blending calibrated and raw correctness probabilities sh...
- **1** `80.6` _done_: Mechanism: Chronological isotonic selected-side q calibration Hypothesis: Replacing raw p_side with calibration-only ...
- **4** `80.01` _done_: Mechanism: Shrunk UTC-session residual calibration Hypothesis: Six-hour cyclical session offsets fitted before policy...
- **5** `79.5` _done_: Mechanism: Hierarchically shrunk p_side-bin residual calibration Hypothesis: Grouped reliability residuals over p_sid...
- **3** `69.62` _done_: Mechanism: Conservative bounded q probabilities Hypothesis: Clipping calibrated q to a defensible interval should lim...

## Global Insight

Children findings: [1, done, score=80.6] Explicit q integration works mechanically but monotone calibration alone is temporally weak. | [2, done, score=97.21] Shrinkage is the only robust B_dev mechanism in this batch, but it failed the frozen-month verification and is rejected. | [3, done, score=69.62] Fixed clipping destroyed useful q dispersion and underperformed. | [4, done, score=80.01] Intraday residual offsets did not stabilize policy PnL. | [5, done, score=79.5] Grouped residual calibration overfit and was worst on the earlier fold.

## Artifacts

- Idea tree JSON: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_skills_batch1\.coordinator\idea_tree.json`
- Idea tree Markdown: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_skills_batch1\.coordinator\idea_tree.md`
- Experiments: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_sum_pnl_skills_batch1\experiments`
