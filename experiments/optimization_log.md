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
