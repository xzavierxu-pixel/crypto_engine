# Optimization Log

## 20260508_codex_iter01_htf_time

- Hypothesis: replacing completed-candle HTF context with trailing 1m rolling context ending at `t-1`, and replacing raw `minute_bucket` with cyclical minute encoding, should improve timestamp freshness without introducing lookahead.
- Changed files: `src/features/htf_context.py`, `src/features/time_features.py`, `src/core/constants.py`, `tests/test_htf_context.py`, `tests/test_features.py`.
- Config: `experiments/configs/20260508_codex_iter01_htf_time.yaml`.
- Baseline command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_full_start_cached_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter00_baseline_rerun --config config/settings.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Baseline report: `artifacts/data_v2/experiments/20260508_codex_iter00_baseline_rerun/metrics.json`.
- Evaluation command: `rtk python scripts/model/train_model.py --input artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet --output-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --config experiments/configs/20260508_codex_iter01_htf_time.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter01_htf_time/metrics.json`.
- Score before: `0.15945443699911072`.
- Score after: `0.14804920471517846`.
- Utility before / after: `0.07429695637079047` / `0.06518921721099015`.
- Accepted accuracy before / after: `0.5730541647701412` / `0.5719599427753934`.
- Accepted count before / after: `4394` / `3495`.
- Coverage before / after: `0.5085059599583381` / `0.45295489891135304`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m pytest -q tests/test_htf_context.py tests/test_features.py tests/test_train_live_feature_parity_with_15m.py` passed.
- Interpretation: the logic fix is valid and leakage-safe, but reduced validation selection_score under the current LightGBM setup, mostly through lower accepted count and coverage.
- Next step: run a feature availability/nullness and model-input audit on the rebuilt split, then target feature selection or model settings rather than further threshold changes.

## 20260508_codex_iter02_catboost_default

- Hypothesis: CatBoost default settings may handle the repaired HTF/time feature distribution better than the current LightGBM baseline.
- Changed files: `experiments/configs/20260508_codex_iter02_catboost_default.yaml`.
- Config: `experiments/configs/20260508_codex_iter02_catboost_default.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter02_catboost_default --config experiments/configs/20260508_codex_iter02_catboost_default.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter02_catboost_default/metrics.json`.
- Score before: `0.14804920471517846`.
- Score after: `0.1539530578291256`.
- Utility before / after: `0.06518921721099015` / `0.06389320891653702`.
- Accepted accuracy before / after: `0.5719599427753934` / `0.5782291336083782`.
- Accepted count before / after: `3495` / `3151`.
- Coverage before / after: `0.45295489891135304` / `0.4083722135821669`.
- Coverage constraint satisfied: yes.
- Tests: reused `20260508_codex_iter01_htf_time` cached split; no code changes in this iteration.
- Interpretation: CatBoost improves accepted accuracy and selection_score over the repaired LightGBM run, but coverage is close to the 0.40 floor and score remains below the original cached baseline.
- Next step: reduce noisy/redundant model inputs while retaining HTF/time features, then compare LightGBM and CatBoost on the same repaired split.

## 20260508_codex_iter03_top250_lgbm

- Hypothesis: reducing model inputs from 1732 features to the top 250 LightGBM gain features, while forcing all HTF/time features to remain, may reduce noise and improve validation selection_score.
- Changed files: `experiments/configs/20260508_codex_iter03_top250_lgbm.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter03_top250_split`.
- Config: `experiments/configs/20260508_codex_iter03_top250_lgbm.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter03_top250_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter03_top250_lgbm --config experiments/configs/20260508_codex_iter03_top250_lgbm.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter03_top250_lgbm/metrics.json`.
- Score before: `0.14804920471517846`.
- Score after: `0.14743848182308736`.
- Utility before / after: `0.06518921721099015` / `0.0800933125972006`.
- Accepted accuracy before / after: `0.5719599427753934` / `0.5597447795823666`.
- Accepted count before / after: `3495` / `5172`.
- Coverage before / after: `0.45295489891135304` / `0.6702954898911353`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `1732` / `275`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: aggressive top-gain filtering increased coverage but reduced accepted accuracy enough to lower selection_score; single-run gain is too lossy for this feature set.
- Next step: try a less aggressive family-level data-processing change or tune model regularization on the full repaired split.

## 20260508_codex_iter04_lgbm_regularized

- Hypothesis: stronger LightGBM regularization on the full repaired split may reduce overfit and improve validation selection_score without dropping feature families.
- Changed files: `experiments/configs/20260508_codex_iter04_lgbm_regularized.yaml`.
- Config: `experiments/configs/20260508_codex_iter04_lgbm_regularized.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter04_lgbm_regularized --config experiments/configs/20260508_codex_iter04_lgbm_regularized.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter04_lgbm_regularized/metrics.json`.
- Score before: `0.14804920471517846`.
- Score after: `0.15440666188914284`.
- Utility before / after: `0.06518921721099015` / `0.07698289269051319`.
- Accepted accuracy before / after: `0.5719599427753934` / `0.5670428893905192`.
- Accepted count before / after: `3495` / `4430`.
- Coverage before / after: `0.45295489891135304` / `0.5741316744427164`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: regularization improved selection_score mainly by restoring accepted count and coverage while keeping accepted accuracy above 0.56; best repaired-split score so far, but still below the original cached baseline.
- Next step: test sample weighting because current weights may be overemphasizing small return regimes after the HTF semantic change.

## 20260508_codex_iter05_lgbm_regularized_unweighted

- Hypothesis: disabling abs-return sample weights may improve validation selection_score if the weighting scheme overfits return magnitude after the HTF semantic change.
- Changed files: `experiments/configs/20260508_codex_iter05_lgbm_regularized_unweighted.yaml`.
- Config: `experiments/configs/20260508_codex_iter05_lgbm_regularized_unweighted.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter05_lgbm_regularized_unweighted --config experiments/configs/20260508_codex_iter05_lgbm_regularized_unweighted.yaml --horizon 5m --train-window-days 183 --validation-window-days 30 --unweighted`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter05_lgbm_regularized_unweighted/metrics.json`.
- Score before: `0.15440666188914284`.
- Score after: `0.15152842409805947`.
- Utility before / after: `0.07698289269051319` / `0.07840850181441164`.
- Accepted accuracy before / after: `0.5670428893905192` / `0.5638589824783619`.
- Accepted count before / after: `4430` / `4737`.
- Coverage before / after: `0.5741316744427164` / `0.6139191290824261`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: unweighted training increases accepted count and coverage, but accepted accuracy drops enough to reduce selection_score; keep weighted training for now.
- Next step: tune weighted LightGBM around the regularized configuration, especially depth, leaves, and column sampling.

## 20260508_codex_iter06_lgbm_mid_regularized

- Hypothesis: an intermediate LightGBM configuration between the original and strongly regularized setup may recover accepted accuracy while maintaining enough coverage.
- Changed files: `experiments/configs/20260508_codex_iter06_lgbm_mid_regularized.yaml`.
- Config: `experiments/configs/20260508_codex_iter06_lgbm_mid_regularized.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter06_lgbm_mid_regularized --config experiments/configs/20260508_codex_iter06_lgbm_mid_regularized.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter06_lgbm_mid_regularized/metrics.json`.
- Score before: `0.15440666188914284`.
- Score after: `0.14327132997552222`.
- Utility before / after: `0.07698289269051319` / `0.07840850181441154`.
- Accepted accuracy before / after: `0.5670428893905192` / `0.5578725846565907`.
- Accepted count before / after: `4430` / `5227`.
- Coverage before / after: `0.5741316744427164` / `0.6774235355106273`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: intermediate complexity over-accepts lower-quality predictions; the useful direction is closer to stronger regularization and lower column sampling.
- Next step: try a high-precision CatBoost variant with reduced depth and stronger regularization.

## 20260508_codex_iter07_catboost_regularized

- Hypothesis: reducing CatBoost depth and increasing L2 regularization may retain CatBoost's validation accepted accuracy while reducing the severe train/validation gap from the default CatBoost run.
- Changed files: `experiments/configs/20260508_codex_iter07_catboost_regularized.yaml`.
- Config: `experiments/configs/20260508_codex_iter07_catboost_regularized.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter07_catboost_regularized --config experiments/configs/20260508_codex_iter07_catboost_regularized.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter07_catboost_regularized/metrics.json`.
- Score before: `0.1539530578291256`.
- Score after: `0.15779432383682843`.
- Utility before / after: `0.06389320891653702` / `0.07153965785381029`.
- Accepted accuracy before / after: `0.5782291336083782` / `0.5741138560687433`.
- Accepted count before / after: `3151` / `3724`.
- Coverage before / after: `0.4083722135821669` / `0.48263348885432866`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: regularized CatBoost is the best repaired-split run so far, improving score by increasing accepted count while keeping accepted accuracy above 0.574.
- Next step: run a neighboring CatBoost configuration with slightly higher depth or lower L2 to see if score can clear the original cached baseline.

## 20260508_codex_iter08_catboost_depth7_l2_15

- Hypothesis: slightly higher CatBoost capacity than iteration 7 may improve accepted accuracy enough to lift selection_score.
- Changed files: `experiments/configs/20260508_codex_iter08_catboost_depth7_l2_15.yaml`.
- Config: `experiments/configs/20260508_codex_iter08_catboost_depth7_l2_15.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter01_htf_time --output-dir artifacts/data_v2/experiments/20260508_codex_iter08_catboost_depth7_l2_15 --config experiments/configs/20260508_codex_iter08_catboost_depth7_l2_15.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter08_catboost_depth7_l2_15/metrics.json`.
- Score before: `0.15779432383682843`.
- Score after: `0.1532816563036069`.
- Utility before / after: `0.07153965785381029` / `0.06454121306376356`.
- Accepted accuracy before / after: `0.5741138560687433` / `0.5769944341372912`.
- Accepted count before / after: `3724` / `3234`.
- Coverage before / after: `0.48263348885432866` / `0.4191290824261275`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: higher capacity increases accepted accuracy slightly but drops coverage and accepted count too much; iteration 7 remains the best repaired-split CatBoost setup.
- Next step: try feature family pruning that removes only low-signal generated interaction families while retaining raw HTF/time/second-level signals.

## 20260508_codex_iter09_top500_catboost

- Hypothesis: a moderate top-500 gain feature subset, with all HTF/time features forced in, may reduce noise without losing as many weak useful features as the top-250 subset.
- Changed files: `experiments/configs/20260508_codex_iter09_top500_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter09_top500_split`.
- Config: `experiments/configs/20260508_codex_iter09_top500_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_catboost --config experiments/configs/20260508_codex_iter09_top500_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter09_top500_catboost/metrics.json`.
- Score before: `0.15779432383682843`.
- Score after: `0.1660762617203513`.
- Utility before / after: `0.07153965785381029` / `0.07879730430274755`.
- Accepted accuracy before / after: `0.5741138560687433` / `0.5744732974032337`.
- Accepted count before / after: `3724` / `4082`.
- Coverage before / after: `0.48263348885432866` / `0.5290305857957491`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `1732` / `518`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is the best run so far and beats the original cached baseline; the improvement comes from higher accepted count at roughly the same accepted accuracy.
- Next step: tune CatBoost regularization on the top-500 split to try to increase accepted accuracy without losing coverage.

## 20260508_codex_iter10_top500_catboost_stronger_reg

- Hypothesis: stronger CatBoost regularization on the top-500 split may push accepted accuracy toward 0.59 while preserving coverage above 0.40.
- Changed files: `experiments/configs/20260508_codex_iter10_top500_catboost_stronger_reg.yaml`.
- Config: `experiments/configs/20260508_codex_iter10_top500_catboost_stronger_reg.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter10_top500_catboost_stronger_reg --config experiments/configs/20260508_codex_iter10_top500_catboost_stronger_reg.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter10_top500_catboost_stronger_reg/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15995031563702308`.
- Utility before / after: `0.07879730430274755` / `0.06700362882322443`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5801550387596899`.
- Accepted count before / after: `4082` / `3225`.
- Coverage before / after: `0.5290305857957491` / `0.4179626749611198`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger regularization raises accepted accuracy but loses too much coverage and utility; iteration 9 remains the best result.
- Next step: stop under the 10-iteration stopping condition and summarize bottlenecks before further changes.

## Stop-condition audit and bottleneck summary

- Objective target: validation `selection_score >= 0.24` with `coverage >= 0.40`.
- Stop condition used: 10 focused iterations completed without reaching `selection_score >= 0.24`.
- Official baseline rerun: `20260508_codex_iter00_baseline_rerun`, validation `selection_score=0.15945443699911072`, `coverage=0.5085059599583381`, `accepted_sample_accuracy=0.5730541647701412`.
- Best completed run: `20260508_codex_iter09_top500_catboost`, validation `selection_score=0.1660762617203513`, `coverage=0.5290305857957491`, `accepted_sample_accuracy=0.5744732974032337`, `accepted_count=4082`.
- Best config: `experiments/configs/20260508_codex_iter09_top500_catboost.yaml`.
- Best report: `artifacts/data_v2/experiments/20260508_codex_iter09_top500_catboost/metrics.json`.
- Best experiment commit: `d4a5bc4`.
- Coverage constraint: satisfied for all 10 completed iterations.
- HTF/time requirement: retained in all feature subsets; top-500 subset forced all `htf_`, `hour_`, and `minute_` features.
- Threshold policy: no manual threshold retuning was used as an optimization lever; runs used the project training/evaluation command and recorded selected thresholds.

Iteration score summary:

| Iteration | Primary change | Validation selection_score | Coverage | Accepted accuracy |
| --- | --- | ---: | ---: | ---: |
| 00 | baseline rerun | 0.15945443699911072 | 0.5085059599583381 | 0.5730541647701412 |
| 01 | trailing HTF + cyclic minute | 0.14804920471517846 | 0.45295489891135304 | 0.5719599427753934 |
| 02 | CatBoost default | 0.1539530578291256 | 0.4083722135821669 | 0.5782291336083782 |
| 03 | top-250 + HTF/time LightGBM | 0.14743848182308736 | 0.6702954898911353 | 0.5597447795823666 |
| 04 | strongly regularized LightGBM | 0.15440666188914284 | 0.5741316744427164 | 0.5670428893905192 |
| 05 | unweighted regularized LightGBM | 0.15152842409805947 | 0.6139191290824261 | 0.5638589824783619 |
| 06 | mid-regularized LightGBM | 0.14327132997552222 | 0.6774235355106273 | 0.5578725846565907 |
| 07 | regularized CatBoost | 0.15779432383682843 | 0.48263348885432866 | 0.5741138560687433 |
| 08 | higher-capacity CatBoost | 0.1532816563036069 | 0.4191290824261275 | 0.5769944341372912 |
| 09 | top-500 + HTF/time CatBoost | 0.1660762617203513 | 0.5290305857957491 | 0.5744732974032337 |
| 10 | stronger-reg top-500 CatBoost | 0.15995031563702308 | 0.4179626749611198 | 0.5801550387596899 |

Bottlenecks:

- The current feature/model stack can trade coverage for accepted accuracy, but cannot yet lift both at the same time. The strongest-regularized top-500 CatBoost run reached the highest accepted accuracy, `0.5801550387596899`, but coverage dropped to `0.4179626749611198`, leaving utility too low.
- The best score run, `20260508_codex_iter09_top500_catboost`, improved accepted count and coverage at roughly the same accepted accuracy, but `accepted_sample_accuracy=0.5744732974032337` is still below the accuracy level likely needed for `selection_score >= 0.24` at comparable coverage.
- Aggressive top-gain filtering hurt accepted accuracy, while full-feature models had weaker score/coverage tradeoffs. A moderate top-500 subset was the only feature-selection setting that clearly improved the baseline.
- LightGBM regularization improved the repaired split but plateaued below CatBoost. CatBoost capacity increases raised accepted accuracy only by sacrificing too much coverage.
- The HTF/time semantic fix is leakage-safe and retained, but it reduced the original LightGBM score before downstream tuning, so future work should treat the repaired split as the new clean baseline rather than comparing only to the old completed-candle HTF behavior.

Recommended next work after this stop condition:

- Add a proper fixed-threshold evaluation path separate from training-time threshold search, then rerun the best candidates under a stable threshold protocol.
- Run adversarial validation and null-importance selection on the repaired split to identify unstable feature families before another model-tuning cycle.
- Try a top-500/top700 feature subset selected from multiple seeds or folds instead of one LightGBM gain run.
- Tune CatBoost around iteration 9 with smaller one-variable steps that target coverage retention, not only accepted accuracy.

## 20260508_codex_iter11_top500_catboost_importance

- Hypothesis: CatBoost is the best current model family, but feature selection was still driven by LightGBM gain. Recording native CatBoost importances should enable better downstream feature subsets without changing labels, thresholds, or feature semantics.
- Changed files: `src/model/train.py`; `experiments/configs/20260508_codex_iter11_top500_catboost_importance.yaml`.
- Config: `experiments/configs/20260508_codex_iter11_top500_catboost_importance.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter11_top500_catboost_importance --config experiments/configs/20260508_codex_iter11_top500_catboost_importance.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter11_top500_catboost_importance/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.1660762617203513`.
- Utility before / after: `0.07879730430274755` / `0.07879730430274755`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5744732974032337`.
- Accepted count before / after: `4082` / `4082`.
- Coverage before / after: `0.5290305857957491` / `0.5290305857957491`.
- Coverage constraint satisfied: yes.
- Feature count: `518`.
- Tests: `rtk python -m pytest -q tests/test_model_pipeline.py::test_train_model_pipeline_and_roundtrip` still fails because the current project config filters the unit-test fixture dates out of the training frame; this matches the pre-existing full-suite failure class from the previous run.
- Interpretation: score is unchanged, as expected, but `feature_importance.csv` is now populated for CatBoost. Top features include `sl_agg_buy_trade_cluster_score_1s`, `htf_range_pos_15m`, `htf_close_z_15m_lag3`, `sl_taker_count_imbalance_5s`, and `htf_close_z_15m`, supporting the requirement to retain HTF/time and second-level context.
- Next step: build a CatBoost-importance feature subset from iteration 11, forcing HTF/time features to remain, and evaluate top-N variants.

## 20260508_codex_iter12_cb_top300_catboost

- Hypothesis: a CatBoost-ranked top-300 subset, with all HTF/time features forced in, may remove weak/noisy features better than the earlier LightGBM-ranked top-250 subset.
- Changed files: `experiments/configs/20260508_codex_iter12_cb_top300_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter12_cb_top300_split`.
- Config: `experiments/configs/20260508_codex_iter12_cb_top300_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter12_cb_top300_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter12_cb_top300_catboost --config experiments/configs/20260508_codex_iter12_cb_top300_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter12_cb_top300_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15375940258602763`.
- Utility before / after: `0.07879730430274755` / `0.06544841886988073`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5764919721296576`.
- Accepted count before / after: `4082` / `3301`.
- Coverage before / after: `0.5290305857957491` / `0.4278123379989632`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `315`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: accepted accuracy improved modestly, but the narrower subset lost too much coverage and utility. CatBoost ranking is useful, but top-300 is too aggressive for this objective.
- Next step: evaluate a less aggressive CatBoost-ranked top-400 subset with HTF/time forced in.

## 20260508_codex_iter13_cb_top400_catboost

- Hypothesis: a CatBoost-ranked top-400 subset may preserve more coverage than top-300 while still dropping the lowest-ranked noisy columns.
- Changed files: `experiments/configs/20260508_codex_iter13_cb_top400_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter13_cb_top400_split`.
- Config: `experiments/configs/20260508_codex_iter13_cb_top400_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter13_cb_top400_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter13_cb_top400_catboost --config experiments/configs/20260508_codex_iter13_cb_top400_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter13_cb_top400_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15259826085449175`.
- Utility before / after: `0.07879730430274755` / `0.06505961638154481`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5758700906344411`.
- Accepted count before / after: `4082` / `3308`.
- Coverage before / after: `0.5290305857957491` / `0.42871954380508034`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `409`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: top-400 remains too narrow; accuracy is slightly higher than the best run, but accepted count and coverage collapse.
- Next step: return to the top-500 split and tune CatBoost regularization/capacity for coverage retention.

## 20260508_codex_iter14_multiscale_catboost

- Hypothesis: adding leakage-safe multi-scale rolling return, absolute move, variation, positive-share, range, and close-position features may give CatBoost a cleaner short/medium-term context without changing HTF/time semantics.
- Changed files: `src/core/config.py`; `src/features/multi_scale_rolling.py`; `src/features/registry.py`; `experiments/configs/20260508_codex_iter14_multiscale_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter14_multiscale_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --input artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet --output-dir artifacts/data_v2/experiments/20260508_codex_iter14_multiscale_catboost --config experiments/configs/20260508_codex_iter14_multiscale_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter14_multiscale_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15826411642324503`.
- Utility before / after: `0.07879730430274755` / `0.0740020736132711`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5723700887198986`.
- Accepted count before / after: `4082` / `3945`.
- Coverage before / after: `0.5290305857957491` / `0.5112752721617418`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m pytest -q tests/test_features.py tests/test_train_live_feature_parity_with_15m.py` passed.
- Interpretation: the feature pack is valid but does not improve the objective; it slightly weakens both accepted accuracy and coverage versus the best top-500 CatBoost run.
- Next step: avoid more broad feature additions and use diagnostics/selection or model training changes on the existing best feature subset.

## 20260508_codex_iter15_drop_shifted20_catboost

- Hypothesis: adversarial validation shows severe train/validation feature shift (`AUC=0.9958715543946381`); dropping the top 20 shifted non-HTF/non-time features may reduce temporal overfit while preserving the required HTF/time context.
- Changed files: `experiments/configs/20260508_codex_iter15_drop_shifted20_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter15_drop_shifted20_split`.
- Diagnostic output: `artifacts/data_v2/experiments/20260508_codex_adversarial_top500/summary.json`; `artifacts/data_v2/experiments/20260508_codex_adversarial_top500/adversarial_feature_importance.csv`.
- Dropped features: `sl_vwap_30s`, `sl_vwap_10s`, `low_volume_flag_share_20_mean_gap_6`, `legal_prev_trade_count_sum_20`, `sl_agg_median_trade_size_300s`, `sl_agg_large_trade_volume_share_300s`, `stale_trade_share_5_rolling_z_12`, `sl_mirror_relative_volume_30s_lag10s`, `legal_prev_trade_count_sum_3`, `sl_trade_count_300s`, `sl_range_10s`, `sl_agg_trade_cluster_score_300s`, `sl_agg_median_trade_size_60s`, `sl_range_3s`, `legal_prev_taker_buy_base_volume_sum_20`, `low_volume_flag_share_20_rolling_z_6`, `sl_agg_intrasecond_flow_concentration_300s`, `abs_ret_mean_20`, `prev_bar_trade_count_lag12`, `prev_bar_avg_quote_per_trade`.
- Config: `experiments/configs/20260508_codex_iter15_drop_shifted20_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter15_drop_shifted20_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter15_drop_shifted20_catboost --config experiments/configs/20260508_codex_iter15_drop_shifted20_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter15_drop_shifted20_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.16543238270404207`.
- Utility before / after: `0.07879730430274755` / `0.08035251425505443`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5727272727272728`.
- Accepted count before / after: `4082` / `4268`.
- Coverage before / after: `0.5290305857957491` / `0.5523597719025402`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: shift pruning increased coverage, accepted count, and utility, but reduced accepted accuracy enough to miss the best score by a small margin.
- Next step: try a smaller shifted-feature drop set or tune CatBoost on this pruned split to recover accuracy.

## 20260508_codex_iter16_drop_shifted10_catboost

- Hypothesis: dropping only the top 10 shifted non-HTF/non-time features may preserve more accepted accuracy than the top-20 drop while still reducing temporal shift.
- Changed files: `experiments/configs/20260508_codex_iter16_drop_shifted10_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter16_drop_shifted10_split`.
- Config: `experiments/configs/20260508_codex_iter16_drop_shifted10_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter16_drop_shifted10_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter16_drop_shifted10_catboost --config experiments/configs/20260508_codex_iter16_drop_shifted10_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter16_drop_shifted10_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15081018755574904`.
- Utility before / after: `0.07879730430274755` / `0.0765593571809228`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5646142237223254`.
- Accepted count before / after: `4082` / `4570`.
- Coverage before / after: `0.5290305857957491` / `0.5922768273716952`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: dropping only the strongest shift drivers over-accepts lower-quality predictions; the top-20 drop was better balanced.
- Next step: tune CatBoost regularization on the top-20 shifted-pruned split to recover accepted accuracy while keeping the utility gain.

## 20260508_codex_iter17_drop_shifted20_stronger_catboost

- Hypothesis: stronger CatBoost regularization on the top-20 shifted-pruned split may recover accepted accuracy while keeping the higher coverage/utility from shift pruning.
- Changed files: `experiments/configs/20260508_codex_iter17_drop_shifted20_stronger_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter17_drop_shifted20_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter15_drop_shifted20_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter17_drop_shifted20_stronger_catboost --config experiments/configs/20260508_codex_iter17_drop_shifted20_stronger_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter17_drop_shifted20_stronger_catboost/metrics.json`.
- Score before: `0.16543238270404207`.
- Score after: `0.1573470734005547`.
- Utility before / after: `0.08035251425505443` / `0.08001036806635562`.
- Accepted accuracy before / after: `0.5727272727272728` / `0.5670526726368251`.
- Accepted count before / after: `4268` / `4603`.
- Coverage before / after: `0.5523597719025402` / `0.5965536547433904`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this regularization setting further increased coverage but reduced accepted accuracy too much; it does not solve the pruned split's score tradeoff.
- Next step: return to the original top-500 split and test training-stability changes.

## 20260508_codex_iter18_top500_catboost_ensemble3

- Hypothesis: averaging three CatBoost seeds may reduce stochastic variance and improve the validation score/coverage tradeoff on the best top-500 feature subset.
- Changed files: `src/model/catboost_ensemble_plugin.py`; `src/model/registry.py`; `src/model/train.py`; `experiments/configs/20260508_codex_iter18_top500_catboost_ensemble3.yaml`.
- Config: `experiments/configs/20260508_codex_iter18_top500_catboost_ensemble3.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter18_top500_catboost_ensemble3 --config experiments/configs/20260508_codex_iter18_top500_catboost_ensemble3.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter18_top500_catboost_ensemble3/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15864128427178014`.
- Utility before / after: `0.07879730430274755` / `0.08071539761501684`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5673652694610778`.
- Accepted count before / after: `4082` / `4623`.
- Coverage before / after: `0.5290305857957491` / `0.5991446345256609`.
- Coverage constraint satisfied: yes.
- Tests: config/plugin smoke check passed with `CatBoostSeedEnsemblePlugin [42, 43, 44]`; DQC ran during training.
- Interpretation: seed averaging smooths probabilities and increases accepted count/utility, but it reduces accepted accuracy too much for `selection_score`.
- Next step: test single-model CatBoost settings that are less smoothing-heavy and target higher accepted accuracy at moderate coverage.

## 20260508_codex_iter19_top500_catboost_bernoulli

- Hypothesis: CatBoost Bernoulli row sampling may reduce temporal overfit on the top-500 split and improve accepted accuracy without feature changes.
- Changed files: `experiments/configs/20260508_codex_iter19_top500_catboost_bernoulli.yaml`.
- Config: `experiments/configs/20260508_codex_iter19_top500_catboost_bernoulli.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter19_top500_catboost_bernoulli --config experiments/configs/20260508_codex_iter19_top500_catboost_bernoulli.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter19_top500_catboost_bernoulli/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.1527359111423785`.
- Utility before / after: `0.07879730430274755` / `0.06441109383100054`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5766387889435337`.
- Accepted count before / after: `4082` / `3243`.
- Coverage before / after: `0.5290305857957491` / `0.42029549092794194`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: row sampling increased accepted accuracy slightly but lost too much coverage and utility.
- Next step: avoid Bernoulli sampling and test smaller changes around learning rate/depth/L2.

## 20260508_codex_iter20_top500_catboost_l2_12_rs_05

- Hypothesis: slightly lower CatBoost L2 and random strength may increase accepted accuracy without the larger coverage loss seen from depth increases.
- Changed files: `experiments/configs/20260508_codex_iter20_top500_catboost_l2_12_rs_05.yaml`.
- Config: `experiments/configs/20260508_codex_iter20_top500_catboost_l2_12_rs_05.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter20_top500_catboost_l2_12_rs_05 --config experiments/configs/20260508_codex_iter20_top500_catboost_l2_12_rs_05.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter20_top500_catboost_l2_12_rs_05/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.1621357461619461`.
- Utility before / after: `0.07879730430274755` / `0.06743131156039416`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5815556954396483`.
- Accepted count before / after: `4082` / `3188`.
- Coverage before / after: `0.5290305857957491` / `0.41316744427164336`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this setting gives the highest accepted accuracy so far, but coverage is too low to beat the best score.
- Next step: interpolate between this high-accuracy setting and the best coverage setting.

## 20260508_codex_iter21_top500_catboost_l2_16_rs_075

- Hypothesis: intermediate CatBoost L2/random-strength settings may retain some of iteration 20's accuracy gain while restoring coverage.
- Changed files: `experiments/configs/20260508_codex_iter21_top500_catboost_l2_16_rs_075.yaml`.
- Config: `experiments/configs/20260508_codex_iter21_top500_catboost_l2_16_rs_075.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter21_top500_catboost_l2_16_rs_075 --config experiments/configs/20260508_codex_iter21_top500_catboost_l2_16_rs_075.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter21_top500_catboost_l2_16_rs_075/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.16078907544354405`.
- Utility before / after: `0.07879730430274755` / `0.08567910834629349`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5655913978494623`.
- Accepted count before / after: `4082` / `5038`.
- Coverage before / after: `0.5290305857957491` / `0.6529294971498185`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this setting restores coverage too aggressively and loses too much accepted accuracy.
- Next step: test a narrower feature subset around the current best rather than further smoothing/regularization.

## 20260508_codex_iter22_top500_catboost_unweighted

- Hypothesis: disabling abs-return sample weights for CatBoost may improve probability ranking if the current weights overfit return magnitude.
- Changed files: `experiments/configs/20260508_codex_iter22_top500_catboost_unweighted.yaml`.
- Config: `experiments/configs/20260508_codex_iter22_top500_catboost_unweighted.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter22_top500_catboost_unweighted --config experiments/configs/20260508_codex_iter22_top500_catboost_unweighted.yaml --horizon 5m --train-window-days 183 --validation-window-days 30 --unweighted`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter22_top500_catboost_unweighted/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.14850907805746492`.
- Utility before / after: `0.07879730430274755` / `0.08009331259720064`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5605214152700186`.
- Accepted count before / after: `4082` / `5102`.
- Coverage before / after: `0.5290305857957491` / `0.6612244686365992`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: unweighted CatBoost over-accepts and loses too much accepted accuracy; weighted training remains preferred.
- Next step: adjust sample-weight shape rather than turning weights off entirely.

## 20260508_codex_iter23_top500_catboost_stronger_weights

- Hypothesis: stronger abs-return sample weighting may focus CatBoost on clearer moves and increase accepted accuracy without changing features or thresholds.
- Changed files: `experiments/configs/20260508_codex_iter23_top500_catboost_stronger_weights.yaml`; regenerated weighted split `artifacts/data_v2/experiments/20260508_codex_iter23_top500_stronger_weights_split`.
- Config: `experiments/configs/20260508_codex_iter23_top500_catboost_stronger_weights.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter23_top500_stronger_weights_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter23_top500_catboost_stronger_weights --config experiments/configs/20260508_codex_iter23_top500_catboost_stronger_weights.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter23_top500_catboost_stronger_weights/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15831786842218392`.
- Utility before / after: `0.07879730430274755` / `0.06635562467651632`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5794906832298136`.
- Accepted count before / after: `4082` / `3220`.
- Coverage before / after: `0.5290305857957491` / `0.4173146708138932`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger weighting improves accepted accuracy but loses too much coverage and utility.
- Next step: test a milder sample-weight adjustment if revisiting weights; otherwise keep the original weighting.

## 20260508_codex_iter24_top500_catboost_mild_weights

- Hypothesis: a milder sample-weight adjustment may retain more coverage than iteration 23 while improving accepted accuracy over the original weights.
- Changed files: `experiments/configs/20260508_codex_iter24_top500_catboost_mild_weights.yaml`; regenerated weighted split `artifacts/data_v2/experiments/20260508_codex_iter24_top500_mild_weights_split`.
- Config: `experiments/configs/20260508_codex_iter24_top500_catboost_mild_weights.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter24_top500_mild_weights_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter24_top500_catboost_mild_weights --config experiments/configs/20260508_codex_iter24_top500_catboost_mild_weights.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter24_top500_catboost_mild_weights/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15516344073503724`.
- Utility before / after: `0.07879730430274755` / `0.06635562467651632`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5768175582990398`.
- Accepted count before / after: `4082` / `3332`.
- Coverage before / after: `0.5290305857957491` / `0.43182996371176775`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: even mild stronger weighting loses too much coverage; keep the original sample-weight shape.
- Next step: stop changing weights and investigate alternative feature subsets/model families.

## 20260508_codex_iter25_top500_xgboost

- Hypothesis: XGBoost with conservative histogram-tree settings may produce a better coverage/accuracy tradeoff than LightGBM and CatBoost on the top-500 feature subset.
- Changed files: `src/model/xgboost_plugin.py`; `src/model/registry.py`; `experiments/configs/20260508_codex_iter25_top500_xgboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter25_top500_xgboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter09_top500_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter25_top500_xgboost --config experiments/configs/20260508_codex_iter25_top500_xgboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter25_top500_xgboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.13719392956385527`.
- Utility before / after: `0.07879730430274755` / `0.07283566614826332`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.557179983726607`.
- Accepted count before / after: `4082` / `4918`.
- Coverage before / after: `0.5290305857957491` / `0.6373768792118704`.
- Coverage constraint satisfied: yes.
- Tests: config/plugin smoke check passed with `XGBoostClassifierPlugin`; DQC ran during training.
- Interpretation: this XGBoost baseline over-accepts and has materially weaker accepted accuracy than CatBoost.
- Next step: keep CatBoost as the primary model and use XGBoost only if a later narrow tuning reason emerges.

## 20260508_codex_iter26_top700_catboost

- Hypothesis: a broader LightGBM-ranked top-700 subset, with HTF/time forced in, may recover useful weak features excluded from the best top-500 subset.
- Changed files: `experiments/configs/20260508_codex_iter26_top700_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter26_top700_split`.
- Config: `experiments/configs/20260508_codex_iter26_top700_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter26_top700_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter26_top700_catboost --config experiments/configs/20260508_codex_iter26_top700_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter26_top700_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.14928366702871515`.
- Utility before / after: `0.07879730430274755` / `0.0684292379471229`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5700079625298495`.
- Accepted count before / after: `4082` / `3768`.
- Coverage before / after: `0.5290305857957491` / `0.48833592534992224`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `710`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: adding weaker LightGBM-ranked features hurts both accepted accuracy and score; top-500 remains the best feature-size anchor.
- Next step: investigate targeted family removals rather than broad top-N expansion.

## 20260508_codex_iter27_drop_interactions_catboost

- Hypothesis: generated interaction-bank columns may overfit temporal quirks; dropping them from the top-500 subset may improve accepted accuracy while retaining HTF/time and core features.
- Changed files: `experiments/configs/20260508_codex_iter27_drop_interactions_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter27_drop_interactions_split`.
- Config: `experiments/configs/20260508_codex_iter27_drop_interactions_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter27_drop_interactions_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter27_drop_interactions_catboost --config experiments/configs/20260508_codex_iter27_drop_interactions_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter27_drop_interactions_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.15467651929973518`.
- Utility before / after: `0.07879730430274755` / `0.06621772939346815`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.576497005988024`.
- Accepted count before / after: `4082` / `3340`.
- Coverage before / after: `0.5290305857957491` / `0.4328667703473302`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `407`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: interactions likely add noise, but dropping them removes too much coverage; keep them in the current best subset.
- Next step: try narrower family-level drops among shifted second-level features instead.

## 20260508_codex_iter28_drop_sl_vwap_catboost

- Hypothesis: the two strongest adversarial-shift features, `sl_vwap_30s` and `sl_vwap_10s`, may be hurting validation stability; dropping only these may improve score without the broader pruning damage.
- Changed files: `experiments/configs/20260508_codex_iter28_drop_sl_vwap_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter28_drop_sl_vwap_split`.
- Config: `experiments/configs/20260508_codex_iter28_drop_sl_vwap_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter28_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter28_drop_sl_vwap_catboost --config experiments/configs/20260508_codex_iter28_drop_sl_vwap_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter28_drop_sl_vwap_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.16356339207951035`.
- Utility before / after: `0.07879730430274755` / `0.08527734577501298`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5680314465408805`.
- Accepted count before / after: `4082` / `4836`.
- Coverage before / after: `0.5290305857957491` / `0.6267496111975117`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `516`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: removing the shifted VWAP features improves coverage and utility but hurts accepted accuracy; it is close but not better than the best score.
- Next step: pair this surgical drop with a higher-accuracy CatBoost setting.

## 20260508_codex_iter29_drop_sl_vwap_l2_12_catboost

- Hypothesis: combining the surgical VWAP drop's higher coverage with the high-accuracy lower-L2 CatBoost setting may improve the score tradeoff.
- Changed files: `experiments/configs/20260508_codex_iter29_drop_sl_vwap_l2_12_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter29_drop_sl_vwap_l2_12_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter28_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter29_drop_sl_vwap_l2_12_catboost --config experiments/configs/20260508_codex_iter29_drop_sl_vwap_l2_12_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter29_drop_sl_vwap_l2_12_catboost/metrics.json`.
- Score before: `0.16356339207951035`.
- Score after: `0.15438031097322155`.
- Utility before / after: `0.08527734577501298` / `0.06350440642820009`.
- Accepted accuracy before / after: `0.5680314465408805` / `0.5790254756126482`.
- Accepted count before / after: `4836` / `3100`.
- Coverage before / after: `0.6267496111975117` / `0.4017625712804562`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the high-accuracy setting again collapses coverage too close to the floor; this combination is worse than both parents.
- Next step: avoid low-L2 settings unless paired with a mechanism that preserves coverage.

## 20260508_codex_iter30_top500_train90_catboost

- Hypothesis: adversarial validation indicates strong temporal shift, so a shorter 90-day recent training window may better match the validation period than the full 183-day development window.
- Changed files: `experiments/configs/20260508_codex_iter30_top500_train90_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter30_top500_train90_split`.
- Config: `experiments/configs/20260508_codex_iter30_top500_train90_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter30_top500_train90_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter30_top500_train90_catboost --config experiments/configs/20260508_codex_iter30_top500_train90_catboost.yaml --horizon 5m --train-window-days 90 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter30_top500_train90_catboost/metrics.json`.
- Score before: `0.1660762617203513`.
- Score after: `0.17068437537379072`.
- Utility before / after: `0.07879730430274755` / `0.07465007776049767`.
- Accepted accuracy before / after: `0.5744732974032337` / `0.5816742209631728`.
- Accepted count before / after: `4082` / `3527`.
- Coverage before / after: `0.5290305857957491` / `0.45710212545360395`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is the new best score. Recent-window training improves accepted accuracy enough to offset lower coverage.
- Next step: sweep nearby recent training windows to see whether score can rise further while staying above coverage 0.40.

## 20260508_codex_iter31_top500_train120_catboost

- Hypothesis: a 120-day recent training window may retain the recency benefit while recovering coverage versus the 90-day run.
- Changed files: `experiments/configs/20260508_codex_iter31_top500_train120_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter31_top500_train120_split`.
- Config: `experiments/configs/20260508_codex_iter31_top500_train120_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter31_top500_train120_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter31_top500_train120_catboost --config experiments/configs/20260508_codex_iter31_top500_train120_catboost.yaml --horizon 5m --train-window-days 120 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter31_top500_train120_catboost/metrics.json`.
- Score before: `0.17068437537379072`.
- Score after: `0.1619470658932365`.
- Utility before / after: `0.07465007776049767` / `0.06933644375324002`.
- Accepted accuracy before / after: `0.5816742209631728` / `0.5793768545994065`.
- Accepted count before / after: `3527` / `3368`.
- Coverage before / after: `0.45710212545360395` / `0.4364955935717999`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: 120 days does not recover enough coverage and scores below 90 days.
- Next step: test a shorter 60-day recent training window.

## 20260508_codex_iter32_top500_train60_catboost

- Hypothesis: a 60-day recent training window may further reduce temporal shift versus validation.
- Changed files: `experiments/configs/20260508_codex_iter32_top500_train60_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter32_top500_train60_split`.
- Config: `experiments/configs/20260508_codex_iter32_top500_train60_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter32_top500_train60_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter32_top500_train60_catboost --config experiments/configs/20260508_codex_iter32_top500_train60_catboost.yaml --horizon 5m --train-window-days 60 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter32_top500_train60_catboost/metrics.json`.
- Score before: `0.17068437537379072`.
- Score after: `0.15493733317107526`.
- Utility before / after: `0.07465007776049767` / `0.06609642301710731`.
- Accepted accuracy before / after: `0.5816742209631728` / `0.5767550567737166`.
- Accepted count before / after: `3527` / `3327`.
- Coverage before / after: `0.45710212545360395` / `0.43118299637117675`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: 60 days is too little training data for this feature/model setup; it lowers both coverage and score.
- Next step: bracket the current best with 75-day and 105-day windows.

## 20260508_codex_iter33_top500_train75_catboost

- Hypothesis: a 75-day recent training window may improve on 90 days by using fresher data while retaining more samples than the weak 60-day run.
- Changed files: `experiments/configs/20260508_codex_iter33_top500_train75_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split`.
- Config: `experiments/configs/20260508_codex_iter33_top500_train75_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_catboost --config experiments/configs/20260508_codex_iter33_top500_train75_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_catboost/metrics.json`.
- Score before: `0.17068437537379072`.
- Score after: `0.17101863059180616`.
- Utility before / after: `0.07465007776049767` / `0.07490927941938828`.
- Accepted accuracy before / after: `0.5816742209631728` / `0.5816145766416624`.
- Accepted count before / after: `3527` / `3541`.
- Coverage before / after: `0.45710212545360395` / `0.4589165370658372`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is the new best score by a small margin. The useful training window is around 75-90 recent days.
- Next step: test 80-day and 85-day windows.

## 20260508_codex_iter34_top500_train80_catboost

- Hypothesis: an 80-day recent training window may refine the best 75-90 day recency range.
- Changed files: `experiments/configs/20260508_codex_iter34_top500_train80_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter34_top500_train80_split`.
- Config: `experiments/configs/20260508_codex_iter34_top500_train80_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter34_top500_train80_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter34_top500_train80_catboost --config experiments/configs/20260508_codex_iter34_top500_train80_catboost.yaml --horizon 5m --train-window-days 80 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter34_top500_train80_catboost/metrics.json`.
- Score before: `0.17101863059180616`.
- Score after: `0.1480181947347743`.
- Utility before / after: `0.07490927941938828` / `0.07309486780715302`.
- Accepted accuracy before / after: `0.5816145766416624` / `0.5651596121883656`.
- Accepted count before / after: `3541` / `4328`.
- Coverage before / after: `0.4589165370658372` / `0.560912389839294`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the 80-day window over-accepts and loses too much accepted accuracy; window effects are not monotonic.
- Next step: test 85 days and keep 75 days as current best.

## 20260508_codex_iter35_top500_train85_catboost

- Hypothesis: an 85-day recent training window may improve on the 75/90-day region.
- Changed files: `experiments/configs/20260508_codex_iter35_top500_train85_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter35_top500_train85_split`.
- Config: `experiments/configs/20260508_codex_iter35_top500_train85_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter35_top500_train85_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter35_top500_train85_catboost --config experiments/configs/20260508_codex_iter35_top500_train85_catboost.yaml --horizon 5m --train-window-days 85 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter35_top500_train85_catboost/metrics.json`.
- Score before: `0.17101863059180616`.
- Score after: `0.15809358977537023`.
- Utility before / after: `0.07490927941938828` / `0.07413167444271635`.
- Accepted accuracy before / after: `0.5816145766416624` / `0.5721616161616162`.
- Accepted count before / after: `3541` / `3960`.
- Coverage before / after: `0.4589165370658372` / `0.5132192846034215`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: 85 days is weaker than both 75 and 90 days; 75 days remains the best.
- Next step: combine the 75-day window with CatBoost hyperparameter variants.

## 20260508_codex_iter36_train75_l2_12_catboost

- Hypothesis: the high-accuracy lower-L2 CatBoost setting may work better with the best 75-day recent training window.
- Changed files: `experiments/configs/20260508_codex_iter36_train75_l2_12_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter36_train75_l2_12_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter36_train75_l2_12_catboost --config experiments/configs/20260508_codex_iter36_train75_l2_12_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter36_train75_l2_12_catboost/metrics.json`.
- Score before: `0.17101863059180616`.
- Score after: `0.14477851012860884`.
- Utility before / after: `0.07490927941938828` / `0.0777604976671851`.
- Accepted accuracy before / after: `0.5816145766416624` / `0.5590294966571156`.
- Accepted count before / after: `3541` / `5083`.
- Coverage before / after: `0.4589165370658372` / `0.6586314152410576`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: lower regularization over-accepts badly on the 75-day split and loses accepted accuracy.
- Next step: test a more regularized 75-day CatBoost variant.

## 20260508_codex_iter37_train75_stronger_catboost

- Hypothesis: stronger CatBoost regularization on the best 75-day window may raise accepted accuracy while keeping coverage above the 0.40 floor.
- Changed files: `experiments/configs/20260508_codex_iter37_train75_stronger_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter37_train75_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter37_train75_stronger_catboost --config experiments/configs/20260508_codex_iter37_train75_stronger_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter37_train75_stronger_catboost/metrics.json`.
- Score before: `0.17101863059180616`.
- Score after: `0.17464090309426274`.
- Utility before / after: `0.07490927941938828` / `0.07439087610160703`.
- Accepted accuracy before / after: `0.5816145766416624` / `0.5850148367952522`.
- Accepted count before / after: `3541` / `3370`.
- Coverage before / after: `0.4589165370658372` / `0.43675505443234836`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is the new best score. The stronger model sacrifices some coverage but improves accepted accuracy enough to win.
- Next step: tune around this regularized 75-day setting.

## 20260508_codex_iter38_train75_depth5_l2_25_catboost

- Hypothesis: slightly lower L2 than the best stronger CatBoost run may recover coverage while preserving most accepted accuracy.
- Changed files: `experiments/configs/20260508_codex_iter38_train75_depth5_l2_25_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter38_train75_depth5_l2_25_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter38_train75_depth5_l2_25_catboost --config experiments/configs/20260508_codex_iter38_train75_depth5_l2_25_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter38_train75_depth5_l2_25_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.16601226380469362`.
- Utility before / after: `0.07439087610160703` / `0.07192846034214623`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5804403479276252`.
- Accepted count before / after: `3370` / `3454`.
- Coverage before / after: `0.43675505443234836` / `0.44764126594090204`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: l2=25 recovers slight coverage but loses too much accepted accuracy; l2=30 remains better.
- Next step: test more regularization or learning-rate/iteration adjustments around the best.

## 20260508_codex_iter39_train75_depth5_l2_35_catboost

- Hypothesis: slightly stronger L2 than the best run may increase accepted accuracy while keeping coverage valid.
- Changed files: `experiments/configs/20260508_codex_iter39_train75_depth5_l2_35_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter39_train75_depth5_l2_35_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter39_train75_depth5_l2_35_catboost --config experiments/configs/20260508_codex_iter39_train75_depth5_l2_35_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter39_train75_depth5_l2_35_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.1625254115969243`.
- Utility before / after: `0.07439087610160703` / `0.06985484681285636`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5795357160483498`.
- Accepted count before / after: `3370` / `3389`.
- Coverage before / after: `0.43675505443234836` / `0.43921721099015035`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: l2=35 loses accepted accuracy without a meaningful coverage gain; keep l2=30.
- Next step: tune learning rate/iterations around the best l2=30 setup.

## 20260508_codex_iter40_train75_lr001_iter1600_catboost

- Hypothesis: lower learning rate with more iterations may smooth CatBoost training and improve validation ranking on the best 75-day setup.
- Changed files: `experiments/configs/20260508_codex_iter40_train75_lr001_iter1600_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter40_train75_lr001_iter1600_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter40_train75_lr001_iter1600_catboost --config experiments/configs/20260508_codex_iter40_train75_lr001_iter1600_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter40_train75_lr001_iter1600_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.16515125134975292`.
- Utility before / after: `0.07439087610160703` / `0.06933644375324002`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5821806554027356`.
- Accepted count before / after: `3370` / `3255`.
- Coverage before / after: `0.43675505443234836` / `0.42198444790046654`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: lower learning rate loses coverage and does not improve score.
- Next step: test a slightly higher learning rate/fewer iterations around the best.

## 20260508_codex_iter41_train75_lr002_iter800_depth5_catboost

- Hypothesis: higher learning rate with fewer iterations may sharpen CatBoost probabilities on the best 75-day setup.
- Changed files: `experiments/configs/20260508_codex_iter41_train75_lr002_iter800_depth5_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter41_train75_lr002_iter800_depth5_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter41_train75_lr002_iter800_depth5_catboost --config experiments/configs/20260508_codex_iter41_train75_lr002_iter800_depth5_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter41_train75_lr002_iter800_depth5_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.1722299384791155`.
- Utility before / after: `0.07439087610160703` / `0.0710212545355106`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5863749605806246`.
- Accepted count before / after: `3370` / `3168`.
- Coverage before / after: `0.43675505443234836` / `0.41057542768273715`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this setting has higher accepted accuracy but too little coverage; the best remains iteration 37.
- Next step: test an intermediate learning-rate/iteration pair.

## 20260508_codex_iter42_train75_lr0175_iter1000_catboost

- Hypothesis: an intermediate learning rate/iteration pair may balance iteration 37 coverage and iteration 41 accepted accuracy.
- Changed files: `experiments/configs/20260508_codex_iter42_train75_lr0175_iter1000_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter42_train75_lr0175_iter1000_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter42_train75_lr0175_iter1000_catboost --config experiments/configs/20260508_codex_iter42_train75_lr0175_iter1000_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter42_train75_lr0175_iter1000_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.1605592798676145`.
- Utility before / after: `0.07439087610160703` / `0.06829963711767756`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5793973286808349`.
- Accepted count before / after: `3370` / `3316`.
- Coverage before / after: `0.43675505443234836` / `0.4297563504406428`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this intermediate pair loses accepted accuracy and coverage versus iteration 37.
- Next step: combine the best 75-day regularized model with the surgical VWAP feature drop.

## 20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost

- Hypothesis: the surgical `sl_vwap_10s`/`sl_vwap_30s` drop may improve temporal stability when paired with the best recent-window regularized CatBoost setup.
- Changed files: `experiments/configs/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Config: `experiments/configs/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost --config experiments/configs/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost/metrics.json`.
- Score before: `0.17464090309426274`.
- Score after: `0.1809240380968129`.
- Utility before / after: `0.07439087610160703` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5850148367952522` / `0.5893814907872698`.
- Accepted count before / after: `3370` / `3245`.
- Coverage before / after: `0.43675505443234836` / `0.4205546915500259`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `518` / `516`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is the new best score. Dropping the two shifted VWAP features improves accepted accuracy enough to offset lower coverage.
- Next step: tune coverage/accuracy around this best combined setup.

## 20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost

- Hypothesis: a 70-day recent window with the best VWAP-drop/regularized CatBoost setup may improve coverage or accuracy around the 75-day optimum.
- Changed files: `experiments/configs/20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split`.
- Config: `experiments/configs/20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost --config experiments/configs/20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost.yaml --horizon 5m --train-window-days 70 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_stronger_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16678035266271873`.
- Utility before / after: `0.0751684810782789` / `0.06842923794712289`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5844765342960282`.
- Accepted count before / after: `3245` / `3125`.
- Coverage before / after: `0.4205546915500259` / `0.4050025920165889`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: 70 days is too close to the coverage floor and scores below the 75-day best.
- Next step: test 78 or 90 days with the VWAP-drop/regularized setup.

## 20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost

- Hypothesis: the VWAP-drop/regularized setup may benefit from the 90-day window, which was strong before the VWAP drop.
- Changed files: `experiments/configs/20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_split`.
- Config: `experiments/configs/20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost --config experiments/configs/20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost.yaml --horizon 5m --train-window-days 90 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_stronger_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16931283044835744`.
- Utility before / after: `0.0751684810782789` / `0.07788906298600313`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5777133243606999`.
- Accepted count before / after: `3245` / `3866`.
- Coverage before / after: `0.4205546915500259` / `0.5010368066355624`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: 90 days recovers coverage but loses the high accepted accuracy that made iteration 43 best.
- Next step: keep 75 days as best and tune feature drops/model regularization around it.

## 20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost

- Hypothesis: dropping only `sl_vwap_30s` may retain more coverage than dropping both shifted VWAP features while removing the worst adversarial-shift feature.
- Changed files: `experiments/configs/20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split`.
- Config: `experiments/configs/20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost --config experiments/configs/20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_stronger_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17589539124508847`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5853372434017596`.
- Accepted count before / after: `3245` / `3394`.
- Coverage before / after: `0.4205546915500259` / `0.4398652151373769`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `516` / `517`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: dropping only `sl_vwap_30s` is better than no drop, but dropping both VWAP features remains best.
- Next step: test dropping only `sl_vwap_10s`.

## 20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost

- Hypothesis: dropping only `sl_vwap_10s` may isolate the useful part of the two-feature VWAP drop.
- Changed files: `experiments/configs/20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split`.
- Config: `experiments/configs/20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost --config experiments/configs/20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_stronger_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1658763696132359`.
- Utility before / after: `0.0751684810782789` / `0.07050285121824782`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5816353887399464`.
- Accepted count before / after: `3245` / `3326`.
- Coverage before / after: `0.4205546915500259` / `0.43105339554173147`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `516` / `517`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: dropping only `sl_vwap_10s` is worse than dropping both or only `sl_vwap_30s`; the best remains dropping both.
- Next step: test whether adding one more shifted feature drop helps the best two-VWAP-drop setup.

## 20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost

- Hypothesis: adding the next strongest adversarial-shift feature, `low_volume_flag_share_20_mean_gap_6`, to the two-VWAP drop may further improve stability.
- Changed files: `experiments/configs/20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost.yaml`; generated split `artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_split`.
- Config: `experiments/configs/20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost --config experiments/configs/20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_stronger_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15708125435718116`.
- Utility before / after: `0.0751684810782789` / `0.06557802073509596`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5791979949874687`.
- Accepted count before / after: `3245` / `3192`.
- Coverage before / after: `0.4205546915500259` / `0.41368636599274236`.
- Coverage constraint satisfied: yes.
- Feature count before / after: `516` / `515`.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the third drop removes useful signal; the best feature-pruned set remains the two VWAP columns only.
- Next step: test seed sensitivity on the current best.

## 20260508_codex_iter49_best_seed43_catboost

- Hypothesis: the current best setup may be seed-sensitive; a neighboring CatBoost seed could improve selection_score.
- Changed files: `experiments/configs/20260508_codex_iter49_best_seed43_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter49_best_seed43_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter49_best_seed43_catboost --config experiments/configs/20260508_codex_iter49_best_seed43_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter49_best_seed43_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1750182045063192`.
- Utility before / after: `0.0751684810782789` / `0.08035251425505444`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5800232288037166`.
- Accepted count before / after: `3245` / `3874`.
- Coverage before / after: `0.4205546915500259` / `0.5020736132711259`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: seed 43 improves coverage/utility but loses too much accepted accuracy; seed 42 remains best.
- Next step: test another seed once, then stop at the 50-iteration condition if the target is not reached.

## 20260508_codex_iter50_best_seed44_catboost

- Hypothesis: another neighboring CatBoost seed may improve the current best seed-42 result.
- Changed files: `experiments/configs/20260508_codex_iter50_best_seed44_catboost.yaml`.
- Config: `experiments/configs/20260508_codex_iter50_best_seed44_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter50_best_seed44_catboost --config experiments/configs/20260508_codex_iter50_best_seed44_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter50_best_seed44_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16811928538055864`.
- Utility before / after: `0.0751684810782789` / `0.0705028512182478`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5834615384615385`.
- Accepted count before / after: `3245` / `3250`.
- Coverage before / after: `0.4205546915500259` / `0.4212026967330223`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: seed 44 is below the best seed-42 result. Seed 42 remains the final best run.
- Next step: stop under the requested 50-iteration limit and summarize bottlenecks.

## Extended stop-condition summary

- Requested target: validation `selection_score >= 0.24` with `coverage >= 0.40`.
- Stopping condition reached: 50 completed iterations without reaching target.
- Best completed run: `20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost`.
- Best validation score: `0.1809240380968129`.
- Best validation utility: `0.0751684810782789`.
- Best accepted accuracy: `0.5893814907872698`.
- Best accepted count: `3245`.
- Best coverage: `0.4205546915500259`.
- Best config: `experiments/configs/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost.yaml`.
- Best report: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_stronger_catboost/metrics.json`.
- Best commit: `44ab69c`.
- Improvement versus official rerun baseline score `0.15945443699911072`: `+0.02146960109770218`.
- Improvement versus previous best before extension `0.1660762617203513`: `+0.0148477763764616`.
- Coverage constraint: satisfied by the best run and by all logged accepted-result runs in the extended cycle.
- Threshold policy: no threshold-range or manual-threshold optimization was used as an optimization lever; runs used the official training/evaluation threshold search and reported selected thresholds.
- HTF/time requirement: retained throughout; all generated feature subsets forced HTF/time columns or started from subsets that already retained them.

Main bottlenecks:

- The target score likely requires accepted accuracy around or above ~0.60 at coverage near 0.42-0.50, or a meaningful coverage increase while preserving the best run's `0.589` accepted accuracy. The best run improved accuracy but remains below that level.
- Recent-window training is the most useful data-processing change found. The best window was 75 days; 60, 70, 80, 85, 90, and 120 day variants were worse for the best feature/model combinations.
- Surgical removal of `sl_vwap_10s` and `sl_vwap_30s` helped only when combined with the 75-day regularized CatBoost setup. Broader adversarial pruning, interaction removal, and top-N changes reduced score.
- CatBoost remains the best model family. XGBoost, LightGBM variants, and CatBoost seed averaging either over-accepted lower-quality predictions or collapsed coverage.
- Stronger CatBoost regularization helped in the 75-day window, but nearby L2 and learning-rate variants did not improve the score.

## 20260508_codex_iter51_collinear_pruned_catboost

- Skill used: `tabular-collinear-feature-removal`.
- Hypothesis: removing only near-duplicate features (`abs(corr) >= 0.995`) while protecting HTF and cyclical time features may reduce redundant split noise and improve validation `selection_score`.
- Changed files: `experiments/configs/20260508_codex_iter51_collinear_pruned_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter51_collinear_pruned_split`.
- Feature set: 505 selected features, down from 516; HTF/time features retained.
- Config: `experiments/configs/20260508_codex_iter51_collinear_pruned_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter51_collinear_pruned_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter51_collinear_pruned_catboost --config experiments/configs/20260508_codex_iter51_collinear_pruned_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter51_collinear_pruned_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16250720004851474`.
- Utility before / after: `0.0751684810782789` / `0.0725764644893727`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5769653655854865`.
- Accepted count before / after: `3245` / `3638`.
- Coverage before / after: `0.4205546915500259` / `0.4714878175220321`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: conservative near-duplicate pruning increased coverage but reduced accepted accuracy enough to lower the objective. The removed redundancy appears to have included useful confidence-separation signal for CatBoost.
- Next step: try a recency-weighted cached split that keeps more history than the 75-day best run while emphasizing the recent regime.

## 20260508_codex_iter52_recency_decay_fullhist_catboost

- Hypothesis: using the full VWAP-pruned top-500 development history with exponential recency-decayed sample weights may combine the 75-day model's recent-regime advantage with additional older examples for coverage.
- Changed files: `experiments/configs/20260508_codex_iter52_recency_decay_fullhist_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter52_recency_decay_fullhist_split`.
- Data processing: multiplied existing `stage1_sample_weight` by `0.20 + 0.80 * exp(-age_days / 45)`.
- Feature set: VWAP-pruned top-500 feature set; HTF/time features retained.
- Config: `experiments/configs/20260508_codex_iter52_recency_decay_fullhist_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter52_recency_decay_fullhist_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter52_recency_decay_fullhist_catboost --config experiments/configs/20260508_codex_iter52_recency_decay_fullhist_catboost.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter52_recency_decay_fullhist_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1675144439455203`.
- Utility before / after: `0.0751684810782789` / `0.06920684292379468`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.584280303030303`.
- Accepted count before / after: `3245` / `3168`.
- Coverage before / after: `0.4205546915500259` / `0.4105754276827372`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: soft full-history recency weighting did not beat a hard 75-day window. Older regimes remain a net drag even with substantial downweighting.
- Next step: run null-importance feature selection on the current best 75-day VWAP-pruned split, protecting HTF and time features.

## 20260508_codex_iter53_null_importance_pruned_catboost

- Skill used: `tabular-null-importance-feature-selection`.
- Hypothesis: removing features whose LightGBM RF gain fails a shuffled-target null distribution may reduce noisy feature competition and improve accepted accuracy for the best CatBoost setup.
- Changed files: `experiments/configs/20260508_codex_iter53_null_importance_pruned_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter53_null_importance_split`.
- Feature set: 436 selected features, down from 516; HTF/time features retained.
- Selection details: 20 shuffled-target null runs on the development split only; dropped unprotected features with `null_score < 35` and `actual_gain/null_p75_gain < 0.60`.
- Config: `experiments/configs/20260508_codex_iter53_null_importance_pruned_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter53_null_importance_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter53_null_importance_pruned_catboost --config experiments/configs/20260508_codex_iter53_null_importance_pruned_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter53_null_importance_pruned_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16497435165296886`.
- Utility before / after: `0.0751684810782789` / `0.07102125453602905`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5803990610328639`.
- Accepted count before / after: `3245` / `3408`.
- Coverage before / after: `0.4205546915500259` / `0.4416796267496112`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: aggressive null-importance pruning again increased coverage but reduced accepted accuracy. The null model is over-penalizing features that CatBoost uses for selective confidence.
- Next step: evaluate a smaller bottom-tail null-importance removal instead of the 80-feature drop.

## 20260508_codex_iter54_null_tail20_catboost

- Skill used: `tabular-null-importance-feature-selection`.
- Hypothesis: dropping only the weakest 20 unprotected null-importance features may remove noise without damaging CatBoost's selective confidence separation.
- Changed files: `experiments/configs/20260508_codex_iter54_null_tail20_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_split`.
- Feature set: 496 selected features, down from 516; HTF/time features retained.
- Config: `experiments/configs/20260508_codex_iter54_null_tail20_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_catboost --config experiments/configs/20260508_codex_iter54_null_tail20_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16545486377312257`.
- Utility before / after: `0.0751684810782789` / `0.06765163297045106`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5841392649903289`.
- Accepted count before / after: `3245` / `3102`.
- Coverage before / after: `0.4205546915500259` / `0.4020217729393468`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: even a small null-importance tail cut reduces the objective. Feature pruning has repeatedly reduced validation confidence quality.
- Next step: try a model-training ensemble variant based on rank averaging, using the best feature/data split unchanged.

## 20260508_codex_iter55_catboost_rank_ensemble

- Skill used: `tabular-rank-averaging-ensemble`.
- Hypothesis: averaging empirical-rank-transformed predictions from nearby CatBoost seeds may reduce seed-specific probability-scale noise while preserving online-safe inference through stored training-score reference distributions.
- Changed files: `src/model/catboost_ensemble_plugin.py`, `src/model/registry.py`, `experiments/configs/20260508_codex_iter55_catboost_rank_ensemble.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_rank_ensemble`, seeds `42,43,44`, same CatBoost hyperparameters as the best seed-42 run.
- Config: `experiments/configs/20260508_codex_iter55_catboost_rank_ensemble.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter55_catboost_rank_ensemble --config experiments/configs/20260508_codex_iter55_catboost_rank_ensemble.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter55_catboost_rank_ensemble/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15483669489117222`.
- Utility before / after: `0.0751684810782789` / `0.0758164852255054`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5682613768961493`.
- Accepted count before / after: `3245` / `4285`.
- Coverage before / after: `0.4205546915500259` / `0.5553395541731467`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m compileall -q src/model`; DQC ran during training.
- Interpretation: rank averaging over-expanded the acceptance set and reduced accepted accuracy. The raw seed-42 CatBoost remains the best model-training result.
- Next step: continue with simpler CatBoost regularization/iteration ablations on the best split rather than ensemble scaling.

## 20260508_codex_iter56_depth4_catboost

- Hypothesis: reducing CatBoost tree depth from 5 to 4 may reduce confidence overfit and improve validation accepted accuracy on the current best data/feature split.
- Changed files: `experiments/configs/20260508_codex_iter56_depth4_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost depth `4`, otherwise same as the current best CatBoost settings.
- Config: `experiments/configs/20260508_codex_iter56_depth4_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter56_depth4_catboost --config experiments/configs/20260508_codex_iter56_depth4_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter56_depth4_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16538464391392693`.
- Utility before / after: `0.0751684810782789` / `0.0688180404354588`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5828910396503278`.
- Accepted count before / after: `3245` / `3203`.
- Coverage before / after: `0.4205546915500259` / `0.41511145671332295`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: shallower trees reduced both confidence quality and score. Depth 5 remains better.
- Next step: test stronger CatBoost stochastic regularization at depth 5.

## 20260508_codex_iter57_stochastic_catboost

- Hypothesis: increasing CatBoost stochastic regularization may improve validation generalization while keeping the current best data/feature split unchanged.
- Changed files: `experiments/configs/20260508_codex_iter57_stochastic_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `random_strength=4.0`, `bagging_temperature=1.0`, otherwise same as the current best settings.
- Config: `experiments/configs/20260508_codex_iter57_stochastic_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter57_stochastic_catboost --config experiments/configs/20260508_codex_iter57_stochastic_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter57_stochastic_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16470796966717988`.
- Utility before / after: `0.0751684810782789` / `0.06881804043545879`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5823255813953488`.
- Accepted count before / after: `3245` / `3225`.
- Coverage before / after: `0.4205546915500259` / `0.4179626749611198`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger stochastic regularization reduced the validation objective. The current best `random_strength=2.0`, `bagging_temperature=0.5` remains preferable.
- Next step: evaluate a slightly less stochastic depth-5 variant.

## 20260508_codex_iter58_less_stochastic_catboost

- Hypothesis: reducing CatBoost stochastic regularization may sharpen the selective probability tails and improve `selection_score` without touching threshold policy.
- Changed files: `experiments/configs/20260508_codex_iter58_less_stochastic_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `random_strength=1.0`, `bagging_temperature=0.25`, otherwise same as the current best settings.
- Config: `experiments/configs/20260508_codex_iter58_less_stochastic_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter58_less_stochastic_catboost --config experiments/configs/20260508_codex_iter58_less_stochastic_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter58_less_stochastic_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16895658701097926`.
- Utility before / after: `0.0751684810782789` / `0.07503888024883362`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5799061551200663`.
- Accepted count before / after: `3245` / `3623`.
- Coverage before / after: `0.4205546915500259` / `0.46954380508035254`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: lower stochasticity increased coverage but reduced accepted accuracy. The existing stochastic settings remain best.
- Next step: return to feature/data processing with a focused time-regime feature variant.

## 20260508_codex_iter59_weekday_time_catboost

- Skill used: `tabular-season-phase-labeling` adapted to crypto calendar-regime features.
- Hypothesis: adding leak-free weekday cyclic features may capture weekly crypto liquidity/regime effects and improve accepted accuracy while preserving existing HTF/hour/minute features.
- Changed files: `src/features/time_features.py`, `src/core/constants.py`, `tests/test_features.py`, `experiments/configs/20260508_codex_iter59_weekday_time_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter59_weekday_time_split`.
- Feature set: current best VWAP-pruned top-500 split plus `weekday_sin`, `weekday_cos`; HTF/hour/minute features retained.
- Config: `experiments/configs/20260508_codex_iter59_weekday_time_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter59_weekday_time_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter59_weekday_time_catboost --config experiments/configs/20260508_codex_iter59_weekday_time_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter59_weekday_time_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17121192784720896`.
- Utility before / after: `0.0751684810782789` / `0.07011404872991185`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5864493448386066`.
- Accepted count before / after: `3245` / `3129`.
- Coverage before / after: `0.4205546915500259` / `0.40552099533437014`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m pytest -q tests/test_features.py tests/test_train_live_feature_parity_with_15m.py` passed.
- Interpretation: weekday cyclic context is valid and modestly competitive, but it does not beat the current best; it mainly reduces coverage while only partially preserving accepted accuracy.
- Next step: test a weekend flag as a simpler weekly regime feature.

## 20260508_codex_iter60_weekend_time_catboost

- Skill used: `tabular-season-phase-labeling` adapted to a crypto weekend regime flag.
- Hypothesis: adding an explicit `is_weekend` feature may provide a simpler weekly-regime split than cyclic weekday coordinates alone.
- Changed files: `src/features/time_features.py`, `src/core/constants.py`, `tests/test_features.py`, `experiments/configs/20260508_codex_iter60_weekend_time_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter60_weekend_time_split`.
- Feature set: current best VWAP-pruned top-500 split plus `weekday_sin`, `weekday_cos`, `is_weekend`; HTF/hour/minute features retained.
- Config: `experiments/configs/20260508_codex_iter60_weekend_time_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter60_weekend_time_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter60_weekend_time_catboost --config experiments/configs/20260508_codex_iter60_weekend_time_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter60_weekend_time_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.161647137836776`.
- Utility before / after: `0.0751684810782789` / `0.0672628304821151`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.581322469445315`.
- Accepted count before / after: `3245` / `3191`.
- Coverage before / after: `0.4205546915500259` / `0.41355624675997926`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m pytest -q tests/test_features.py tests/test_train_live_feature_parity_with_15m.py` passed.
- Interpretation: the explicit weekend flag hurts selection quality. If weekly time context is retained, the cyclic weekday-only version is preferable, but neither beats the current best.
- Next step: test CatBoost class weighting on the current best split.

## 20260508_codex_iter61_sqrt_class_weight_catboost

- Hypothesis: CatBoost `auto_class_weights: SqrtBalanced` may improve directional class balance and selective accuracy on the current best split without changing thresholds.
- Changed files: `experiments/configs/20260508_codex_iter61_sqrt_class_weight_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost best settings plus `auto_class_weights: SqrtBalanced`.
- Config: `experiments/configs/20260508_codex_iter61_sqrt_class_weight_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260508_codex_iter61_sqrt_class_weight_catboost --config experiments/configs/20260508_codex_iter61_sqrt_class_weight_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260508_codex_iter61_sqrt_class_weight_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1660647536537468`.
- Utility before / after: `0.0751684810782789` / `0.07076205287713838`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5815412186379928`.
- Accepted count before / after: `3245` / `3348`.
- Coverage before / after: `0.4205546915500259` / `0.43390357698289267`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: class weighting again increases acceptance volume at lower accepted accuracy. The current best unweighted CatBoost setup remains strongest.
- Next step: preserve the current best run as the active benchmark and avoid adopting the class-weighted variant.

## 20260509_codex_iter62_session_catboost

- Skill used: `tabular-per-type-model-training`.
- Hypothesis: training separate CatBoost models by UTC session may capture session-specific market behavior that the global model only partially learns from `hour_sin/hour_cos`.
- Changed files: `src/model/catboost_session_plugin.py`, `src/model/registry.py`, `experiments/configs/20260509_codex_iter62_session_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained. Session routing is derived from existing `hour_sin/hour_cos`.
- Model settings: `catboost_session`, one global fallback model plus per-session CatBoost models with the same hyperparameters as the current best run.
- Config: `experiments/configs/20260509_codex_iter62_session_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter62_session_catboost --config experiments/configs/20260509_codex_iter62_session_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter62_session_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15068218469943065`.
- Utility before / after: `0.0751684810782789` / `0.0648004147226542`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5745378652355396`.
- Accepted count before / after: `3245` / `3354`.
- Coverage before / after: `0.4205546915500259` / `0.43468118195956457`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m compileall -q src/model`; DQC ran during training.
- Interpretation: per-session models overfit development data and reduce validation accepted accuracy. The global CatBoost remains better.
- Next step: use session information only as lightweight engineered features or interactions, not separate models.

## 20260509_codex_iter63_session_flags_catboost

- Skill used: `tabular-season-phase-labeling` adapted to UTC session flags.
- Hypothesis: simple session indicator features may give the global CatBoost cleaner regime splits than cyclic hour features alone, without the overfitting risk of separate per-session models.
- Changed files: `experiments/configs/20260509_codex_iter63_session_flags_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_split`.
- Feature set: current best VWAP-pruned top-500 split plus `session_asia`, `session_europe`, `session_us`; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter63_session_flags_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_catboost --config experiments/configs/20260509_codex_iter63_session_flags_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16511443510357593`.
- Utility before / after: `0.0751684810782789` / `0.07607568688439612`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5759772197773751`.
- Accepted count before / after: `3245` / `3863`.
- Coverage before / after: `0.4205546915500259` / `0.5006480041472265`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: session flags increase accepted volume and utility slightly but reduce accepted accuracy, lowering the objective. Session-specific handling is not the current bottleneck.
- Next step: move back to data-window or feature-family selection rather than further session features.

## 20260509_codex_iter64_regime_flags_catboost

- Skill used: `tabular-season-phase-labeling` adapted to train-fitted volatility and volume regimes.
- Hypothesis: adding development-fitted `rv_5` and `volume` tercile flags may expose the regime-slice structure seen in the current best report without training separate models.
- Changed files: `experiments/configs/20260509_codex_iter64_regime_flags_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_split`.
- Feature set: current best VWAP-pruned top-500 split plus six regime flags; HTF/time features retained.
- Cutpoints: `rv_5` q33 `0.0003847713141339769`, q67 `0.0006854816179791311`; `volume` q33 `5.857366666666667`, q67 `14.411116666666665`.
- Config: `experiments/configs/20260509_codex_iter64_regime_flags_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_catboost --config experiments/configs/20260509_codex_iter64_regime_flags_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1698404939306286`.
- Utility before / after: `0.0751684810782789` / `0.07153965785381025`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5838905775075988`.
- Accepted count before / after: `3245` / `3290`.
- Coverage before / after: `0.4205546915500259` / `0.4263867288750648`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: regime flags are directionally reasonable but still reduce accepted accuracy relative to the best model. The underlying continuous features are likely sufficient for CatBoost.
- Next step: test stricter recency weighting within the best 75-day split.

## 20260509_codex_iter65_recent75_decay_catboost

- Hypothesis: applying additional recency decay inside the best 75-day development window may emphasize the regime closest to validation and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter65_recent75_decay_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_split`.
- Data processing: multiplied existing `stage1_sample_weight` by `0.40 + 0.60 * exp(-age_days / 20)`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter65_recent75_decay_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_catboost --config experiments/configs/20260509_codex_iter65_recent75_decay_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1584630212684989`.
- Utility before / after: `0.0751684810782789` / `0.07374287195438055`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5727435438506776`.
- Accepted count before / after: `3245` / `3911`.
- Coverage before / after: `0.4205546915500259` / `0.5068688439606014`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: extra recency decay inside the 75-day window over-accepts and lowers accepted accuracy. The hard 75-day window with original sample weights remains best.
- Next step: avoid broad recency reweighting and inspect prediction-error slices for a more targeted feature or family ablation.
