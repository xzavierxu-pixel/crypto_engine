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

## 20260509_codex_iter66_session_relative_catboost

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: train-fitted session-relative deviations for volatility and volume context may help CatBoost distinguish unusually active regimes where false signals concentrate.
- Changed files: `experiments/configs/20260509_codex_iter66_session_relative_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split`.
- Feature set: current best VWAP-pruned top-500 split plus session-relative diff/ratio/z features for `rv_5`, `volume`, `relative_volume_20`, `htf_rv_15m`, and `dollar_vol_mean_20`; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter66_session_relative_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_catboost --config experiments/configs/20260509_codex_iter66_session_relative_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.177574444268134`.
- Utility before / after: `0.0751684810782789` / `0.07361327112493518`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5881987577639751`.
- Accepted count before / after: `3245` / `3220`.
- Coverage before / after: `0.4205546915500259` / `0.41731467081389323`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: session-relative features are the strongest new feature-family result in this continuation batch, but still narrowly below the current best because coverage and utility slip.
- Next step: refine this near-miss by using fewer relative features or combining only the most relevant volume/volatility deviations.

## 20260509_codex_iter67_session_z2_catboost

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: a narrower session-relative feature set using only `rv_5_session_z` and `volume_session_z` may retain the useful signal from iteration 66 with less noise.
- Changed files: `experiments/configs/20260509_codex_iter67_session_z2_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_split`.
- Feature set: current best VWAP-pruned top-500 split plus two train-fitted session z-score features; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter67_session_z2_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_catboost --config experiments/configs/20260509_codex_iter67_session_z2_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1646700524546782`.
- Utility before / after: `0.0751684810782789` / `0.06933644375324001`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5817792723937634`.
- Accepted count before / after: `3245` / `3271`.
- Coverage before / after: `0.4205546915500259` / `0.4239243131156039`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: narrowing to two z-score features loses the useful context from iteration 66 and hurts accepted accuracy. The broader session-relative feature set remains the stronger relative-deviation variant.
- Next step: reassess feature-family candidates before another ablation.

## 20260509_codex_iter68_volume_session_relative_catboost

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: iteration 66's importance profile suggests the useful relative-deviation signal is concentrated in `volume_session_diff`, `volume_session_ratio`, and `volume_session_z`; keeping only those may improve the score.
- Changed files: `experiments/configs/20260509_codex_iter68_volume_session_relative_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_split`.
- Feature set: current best VWAP-pruned top-500 split plus three train-fitted session-relative volume features; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter68_volume_session_relative_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_catboost --config experiments/configs/20260509_codex_iter68_volume_session_relative_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17271620686122222`.
- Utility before / after: `0.0751684810782789` / `0.07141005702436497`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5863907180934462`.
- Accepted count before / after: `3245` / `3189`.
- Coverage before / after: `0.4205546915500259` / `0.41329704510108867`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: volume-only relative deviation keeps accepted accuracy fairly high, but coverage and score are still below the best. Iteration 66 remains the stronger relative-deviation variant.
- Next step: continue with another feature-family candidate rather than further narrowing this one.

## 20260509_codex_iter69_catboost_od

- Hypothesis: enabling CatBoost's overfitting detector and `use_best_model` may reduce confidence overfit while keeping the best feature/data split unchanged.
- Changed files: `experiments/configs/20260509_codex_iter69_catboost_od.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost plus `use_best_model: true`, `od_type: Iter`, `od_wait: 100`.
- Config: `experiments/configs/20260509_codex_iter69_catboost_od.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter69_catboost_od --config experiments/configs/20260509_codex_iter69_catboost_od.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter69_catboost_od/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the overfitting detector reproduces the current best within rounding/count differences and does not materially improve the objective.
- Next step: keep the current best as benchmark; do not rely on OD as a scoring improvement.

## 20260509_codex_iter70_session_relative_od_catboost

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: combining the strongest new feature-family result from iteration 66 with CatBoost's overfitting detector may close the small score gap to the best benchmark.
- Changed files: `experiments/configs/20260509_codex_iter70_session_relative_od_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split`.
- Feature set: iteration 66 session-relative volume/volatility feature set; HTF/time features retained.
- Model settings: iteration 66 CatBoost settings plus `use_best_model: true`, `od_type: Iter`, `od_wait: 100`.
- Config: `experiments/configs/20260509_codex_iter70_session_relative_od_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter70_session_relative_od_catboost --config experiments/configs/20260509_codex_iter70_session_relative_od_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter70_session_relative_od_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1735426588189493`.
- Utility before / after: `0.0751684810782789` / `0.07115085536547439`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.587336939230035`.
- Accepted count before / after: `3245` / `3143`.
- Coverage before / after: `0.4205546915500259` / `0.4073354069466045`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: OD reduces coverage and utility for the session-relative feature set. Iteration 66 remains the better relative-deviation result; iteration 43 remains best overall.
- Next step: continue with a different feature-construction path rather than combining OD with relative features.

## 20260509_codex_iter71_top_importance_rowagg_catboost

- Skill used: `tabular-row-aggregate-features`.
- Hypothesis: row-wise aggregates over the top 40 CatBoost importance features, standardized using development statistics only, may expose compact cross-feature extremes that improve confidence separation.
- Changed files: `experiments/configs/20260509_codex_iter71_top_importance_rowagg_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_split`.
- Feature set: current best VWAP-pruned top-500 split plus `top40_imp_z_mean`, `top40_imp_z_std`, `top40_imp_z_min`, and `top40_imp_z_max`; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter71_top_importance_rowagg_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_catboost --config experiments/configs/20260509_codex_iter71_top_importance_rowagg_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16271912674863467`.
- Utility before / after: `0.0751684810782789` / `0.06933644375323998`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5801618219958046`.
- Accepted count before / after: `3245` / `3337`.
- Coverage before / after: `0.4205546915500259` / `0.4324779678589943`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: row aggregates over high-importance features add noise for this CatBoost setup and reduce accepted accuracy.
- Next step: do not pursue row-aggregate feature expansion further unless a narrower homogeneous block is identified.

## 20260509_codex_iter72_relative_drop_bases_catboost

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: session-relative features may work better if redundant raw base columns are removed, forcing the model to use normalized context rather than raw scale.
- Changed files: `experiments/configs/20260509_codex_iter72_relative_drop_bases_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_split`.
- Feature set: iteration 66 session-relative split with raw `rv_5`, `relative_volume_20`, `htf_rv_15m`, and `dollar_vol_mean_20` removed; HTF/time features otherwise retained.
- Config: `experiments/configs/20260509_codex_iter72_relative_drop_bases_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_catboost --config experiments/configs/20260509_codex_iter72_relative_drop_bases_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16770962736249925`.
- Utility before / after: `0.0751684810782789` / `0.07257646448937277`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5811594202898551`.
- Accepted count before / after: `3245` / `3450`.
- Coverage before / after: `0.4205546915500259` / `0.44712286158631415`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: raw volatility/volume context remains useful alongside relative deviations. Removing it increases coverage but lowers accepted accuracy and score.
- Next step: keep raw context when using relative-deviation features.

## 20260509_codex_iter73_lgbm_dart

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: LightGBM DART with aggressive subsampling may reduce overfitting on the wide feature set and provide a stronger model-family alternative to CatBoost.
- Changed files: `experiments/configs/20260509_codex_iter73_lgbm_dart.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `active_plugin: lightgbm`, `boosting_type: dart`, `n_estimators: 1600`, `learning_rate: 0.01`, `colsample_bytree: 0.35`, `subsample: 0.6`.
- Config: `experiments/configs/20260509_codex_iter73_lgbm_dart.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter73_lgbm_dart --config experiments/configs/20260509_codex_iter73_lgbm_dart.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter73_lgbm_dart/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15579683529138663`.
- Utility before / after: `0.0751684810782789` / `0.0640228097459824`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5796774193548387`.
- Accepted count before / after: `3245` / `3100`.
- Coverage before / after: `0.4205546915500259` / `0.40176257128045617`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: DART overfits development and barely clears validation coverage. CatBoost remains the best model family.
- Next step: keep CatBoost as the primary model for subsequent iterations.

## 20260509_codex_iter74_catboost_lgbm_logit_blend

- Skill used: `tabular-log-odds-fold-averaging`.
- Hypothesis: a heterogeneous CatBoost/LightGBM blend in logit space may add complementary ranking signal while preserving CatBoost as the dominant model.
- Changed files: `src/model/logit_blend_plugin.py`, `src/model/registry.py`, `experiments/configs/20260509_codex_iter74_catboost_lgbm_logit_blend.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, CatBoost weight `0.85`, best CatBoost settings plus regularized LightGBM settings.
- Config: `experiments/configs/20260509_codex_iter74_catboost_lgbm_logit_blend.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter74_catboost_lgbm_logit_blend --config experiments/configs/20260509_codex_iter74_catboost_lgbm_logit_blend.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter74_catboost_lgbm_logit_blend/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1682325759784555`.
- Utility before / after: `0.0751684810782789` / `0.0740020736132711`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5802642676412707`.
- Accepted count before / after: `3245` / `3557`.
- Coverage before / after: `0.4205546915500259` / `0.46099015033696217`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m compileall -q src/model`; DQC ran during training.
- Interpretation: LightGBM adds acceptance volume but dilutes accepted accuracy. CatBoost alone remains better.
- Next step: if blending is revisited, use a much smaller non-CatBoost weight; otherwise continue with CatBoost-only changes.

## 20260509_codex_iter75_catboost_lgbm_logit_blend95

- Skill used: `tabular-log-odds-fold-averaging`.
- Hypothesis: reducing LightGBM's contribution to 5% in the logit blend may capture a small complementary ranking benefit without over-expanding lower-quality accepted predictions.
- Changed files: `experiments/configs/20260509_codex_iter75_catboost_lgbm_logit_blend95.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, CatBoost weight `0.95`, same base model settings as iteration 74.
- Config: `experiments/configs/20260509_codex_iter75_catboost_lgbm_logit_blend95.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter75_catboost_lgbm_logit_blend95 --config experiments/configs/20260509_codex_iter75_catboost_lgbm_logit_blend95.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter75_catboost_lgbm_logit_blend95/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17252509801406793`.
- Utility before / after: `0.0751684810782789` / `0.07853810264385687`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5796529968454258`.
- Accepted count before / after: `3245` / `3804`.
- Coverage before / after: `0.4205546915500259` / `0.49300155520995337`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the smaller blend improves utility and coverage but still lowers accepted accuracy enough to trail the CatBoost-only benchmark.
- Next step: stop using LightGBM as a blend component unless a more orthogonal feature/data setup is found.

## 20260509_codex_iter76_catboost_lgbm_logit_blend99

- Skill used: `tabular-log-odds-fold-averaging`.
- Hypothesis: a 1% LightGBM component may provide a tiny complementary ranking adjustment without materially diluting CatBoost's confidence quality.
- Changed files: `experiments/configs/20260509_codex_iter76_catboost_lgbm_logit_blend99.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, CatBoost weight `0.99`, same base model settings as iteration 74.
- Config: `experiments/configs/20260509_codex_iter76_catboost_lgbm_logit_blend99.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter76_catboost_lgbm_logit_blend99 --config experiments/configs/20260509_codex_iter76_catboost_lgbm_logit_blend99.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter76_catboost_lgbm_logit_blend99/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1796445308845688`.
- Utility before / after: `0.0751684810782789` / `0.07477967858994297`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5887419255613657`.
- Accepted count before / after: `3245` / `3251`.
- Coverage before / after: `0.4205546915500259` / `0.4213322965266978`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the 1% blend is close but still worse than CatBoost alone. LightGBM blending does not improve the objective.
- Next step: abandon LightGBM blending for this split and keep CatBoost-only as the best model family.

## 20260509_codex_iter77_high_regime_downweight_catboost

- Hypothesis: false-signal diagnostics concentrate in high `rv_5` and high `volume` regimes; downweighting those training rows may reduce noisy confidence tails while leaving validation unchanged.
- Changed files: `experiments/configs/20260509_codex_iter77_high_regime_downweight_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_split`.
- Data processing: multiplied existing `stage1_sample_weight` by `0.75` where development `rv_5` or `volume` exceeded its development q67 cutoff.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter77_high_regime_downweight_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_catboost --config experiments/configs/20260509_codex_iter77_high_regime_downweight_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17338252935206183`.
- Utility before / after: `0.0751684810782789` / `0.07348367029548988`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5849056603773585`.
- Accepted count before / after: `3245` / `3339`.
- Coverage before / after: `0.4205546915500259` / `0.43273716951788493`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: high-regime downweighting is valid but does not improve enough; it trades a little accepted accuracy for extra coverage and remains below the best.
- Next step: test mid-regime upweighting, since mid volatility/volume had the strongest regime-slice scores.

## 20260509_codex_iter78_mid_regime_upweight_catboost

- Hypothesis: the current best validation report shows mid-volatility and mid-volume slices have the strongest selection scores; upweighting matching development rows may improve the model's accepted-set quality.
- Changed files: `experiments/configs/20260509_codex_iter78_mid_regime_upweight_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_split`.
- Data processing: multiplied existing `stage1_sample_weight` by `1.20` for development rows inside both `rv_5` and `volume` development tercile middle buckets.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter78_mid_regime_upweight_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_catboost --config experiments/configs/20260509_codex_iter78_mid_regime_upweight_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16480445950815406`.
- Utility before / after: `0.0751684810782789` / `0.0695956454121306`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.581635755548799`.
- Accepted count before / after: `3245` / `3289`.
- Coverage before / after: `0.4205546915500259` / `0.4262571280456195`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: mid-regime upweighting does not transfer as a useful training-weight rule and reduces accepted accuracy.
- Next step: avoid regime sample-weighting unless combined with a more robust time split.

## 20260509_codex_iter79_depth6_l260_catboost

- Hypothesis: depth 6 with stronger L2 regularization may capture useful interactions missed by the best depth-5 model without overfitting excessively.
- Changed files: `experiments/configs/20260509_codex_iter79_depth6_l260_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `depth=6`, `l2_leaf_reg=60.0`, otherwise same as the best CatBoost settings.
- Config: `experiments/configs/20260509_codex_iter79_depth6_l260_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter79_depth6_l260_catboost --config experiments/configs/20260509_codex_iter79_depth6_l260_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter79_depth6_l260_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1708701494439188`.
- Utility before / after: `0.0751684810782789` / `0.07283566614826333`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5834818775995246`.
- Accepted count before / after: `3245` / `3366`.
- Coverage before / after: `0.4205546915500259` / `0.43623639191290825`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: extra tree depth does not improve validation confidence quality even with stronger L2. Depth 5 remains best.
- Next step: continue with depth-5 CatBoost or data/feature changes rather than deeper models.

## 20260509_codex_iter80_ordered_catboost

- Hypothesis: CatBoost ordered boosting may reduce overfitting in the current small 75-day train window and improve validation accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter80_ordered_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `boosting_type: Ordered`.
- Config: `experiments/configs/20260509_codex_iter80_ordered_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter80_ordered_catboost --config experiments/configs/20260509_codex_iter80_ordered_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter80_ordered_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17262590564826172`.
- Utility before / after: `0.0751684810782789` / `0.07244686365992739`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5852914250839182`.
- Accepted count before / after: `3245` / `3277`.
- Coverage before / after: `0.4205546915500259` / `0.4247019180922758`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: ordered boosting is slower and still below the best score. Plain depth-5 CatBoost remains better.
- Next step: avoid ordered boosting for this setup.

## 20260509_codex_iter81_unweighted_best_catboost

- Skill used: `tabular-balanced-log-loss` / `tabular-prior-rebalancing-oversampling` as data-weighting guidance.
- Hypothesis: the current sample-weight ramp may be over-shaping probability tails; setting development weights to `1.0` tests whether the best 75-day split performs better unweighted.
- Changed files: `experiments/configs/20260509_codex_iter81_unweighted_best_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_split`.
- Data processing: set all development `stage1_sample_weight` values to `1.0`; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter81_unweighted_best_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_catboost --config experiments/configs/20260509_codex_iter81_unweighted_best_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16093642981380082`.
- Utility before / after: `0.0751684810782789` / `0.07231726283048208`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5759390310288514`.
- Accepted count before / after: `3245` / `3674`.
- Coverage before / after: `0.4205546915500259` / `0.4761534473820632`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: unweighted training over-accepts lower-quality predictions. The existing weight ramp is important for accepted accuracy.
- Next step: test a milder weight ramp rather than removing weights completely.

## 20260509_codex_iter82_softened_weights_catboost

- Skill used: `tabular-balanced-log-loss` / data-weighting guidance.
- Hypothesis: partially softening the existing sample-weight ramp may retain useful abs-return emphasis while reducing overfit from low-weight samples.
- Changed files: `experiments/configs/20260509_codex_iter82_softened_weights_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_split`.
- Data processing: replaced development weight `w` with `0.5 * w + 0.5`; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter82_softened_weights_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_catboost --config experiments/configs/20260509_codex_iter82_softened_weights_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16213329573999713`.
- Utility before / after: `0.0751684810782789` / `0.06726283048211511`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.581732283464567`.
- Accepted count before / after: `3245` / `3175`.
- Coverage before / after: `0.4205546915500259` / `0.41148263348885433`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: softening the weights loses accepted-set quality. The original stronger weight ramp remains best.
- Next step: keep the original sample weights for subsequent CatBoost experiments.

## 20260509_codex_iter83_squared_weights_catboost

- Skill used: `tabular-balanced-log-loss` / data-weighting guidance.
- Hypothesis: strengthening the existing sample-weight ramp by squaring weights may further emphasize higher-absolute-return examples and improve selective confidence.
- Changed files: `experiments/configs/20260509_codex_iter83_squared_weights_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_split`.
- Data processing: replaced development weight `w` with `clip(w ** 2, 0.15, 1.0)`; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter83_squared_weights_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_catboost --config experiments/configs/20260509_codex_iter83_squared_weights_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16971684191691397`.
- Utility before / after: `0.0751684810782789` / `0.07400207361327117`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5814550641940086`.
- Accepted count before / after: `3245` / `3505`.
- Coverage before / after: `0.4205546915500259` / `0.45425090720580613`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: strengthening the weight ramp increases coverage but reduces accepted accuracy. The original weight ramp remains best.
- Next step: keep original sample weights.

## 20260509_codex_iter84_htf_micro_interactions_catboost

- Skill used: `tabular-polynomial-interaction-features`.
- Hypothesis: direct products between repaired trailing HTF context and the strongest second-level microstructure features may capture conditional microstructure signal under different higher-timeframe regimes.
- Changed files: `experiments/configs/20260509_codex_iter84_htf_micro_interactions_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_split`.
- Feature set: current best VWAP-pruned top-500 split plus 32 HTF x microstructure product features; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter84_htf_micro_interactions_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_catboost --config experiments/configs/20260509_codex_iter84_htf_micro_interactions_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16198013391301166`.
- Utility before / after: `0.0751684810782789` / `0.06661482633488856`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5822663252240717`.
- Accepted count before / after: `3245` / `3124`.
- Coverage before / after: `0.4205546915500259` / `0.4048729911871436`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: explicit HTF x microstructure products add noise and reduce score. CatBoost already handles enough of this interaction structure from the base features.
- Next step: continue with conservative feature selection rather than interaction expansion.

## 20260509_codex_iter85_drop_bottom20_importance_catboost

- Skill used: `tabular-recursive-feature-elimination` as conservative importance-tail pruning.
- Hypothesis: dropping the 20 lowest CatBoost-importance unprotected features from the current best split may reduce noise while keeping HTF/time context intact.
- Changed files: `experiments/configs/20260509_codex_iter85_drop_bottom20_importance_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split`.
- Feature set: 496 selected features, down from 516; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter85_drop_bottom20_importance_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_catboost --config experiments/configs/20260509_codex_iter85_drop_bottom20_importance_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17951039054804577`.
- Utility before / after: `0.0751684810782789` / `0.08048211508553654`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5834004834810637`.
- Accepted count before / after: `3245` / `3723`.
- Coverage before / after: `0.4205546915500259` / `0.48250388802488337`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: bottom-tail importance pruning improves utility and coverage but loses enough accepted accuracy to remain just below the best score.
- Next step: test a smaller bottom-10 importance drop to preserve more accepted accuracy.

## 20260509_codex_iter86_drop_bottom10_importance_catboost

- Skill used: `tabular-recursive-feature-elimination` as conservative importance-tail pruning.
- Hypothesis: dropping only the 10 lowest CatBoost-importance unprotected features may preserve accepted accuracy better than the bottom-20 pruning while reducing some noise.
- Changed files: `experiments/configs/20260509_codex_iter86_drop_bottom10_importance_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_split`.
- Feature set: 506 selected features, down from 516; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter86_drop_bottom10_importance_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_catboost --config experiments/configs/20260509_codex_iter86_drop_bottom10_importance_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1751908729974561`.
- Utility before / after: `0.0751684810782789` / `0.07141005702436491`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5884430176565008`.
- Accepted count before / after: `3245` / `3115`.
- Coverage before / after: `0.4205546915500259` / `0.4037065837221358`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: bottom-10 pruning preserves accepted accuracy but gives up too much coverage/utility. Bottom-20 remains the better pruning variant, but still trails the best.
- Next step: test an intermediate bottom-15 prune.

## 20260509_codex_iter87_drop_bottom15_importance_catboost

- Skill used: `tabular-recursive-feature-elimination` as conservative importance-tail pruning.
- Hypothesis: dropping the bottom 15 unprotected CatBoost-importance features may balance the higher utility of bottom-20 pruning with the better accuracy retention of bottom-10 pruning.
- Changed files: `experiments/configs/20260509_codex_iter87_drop_bottom15_importance_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter87_drop_bottom15_importance_split`.
- Feature set: 501 selected features, down from 516; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter87_drop_bottom15_importance_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter87_drop_bottom15_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter87_drop_bottom15_importance_catboost --config experiments/configs/20260509_codex_iter87_drop_bottom15_importance_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter87_drop_bottom15_importance_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17087458305233366`.
- Utility before / after: `0.0751684810782789` / `0.0724468636599274`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5838583858385838`.
- Accepted count before / after: `3245` / `3333`.
- Coverage before / after: `0.4205546915500259` / `0.43195956454121304`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: bottom-15 pruning is worse than both the bottom-20 near-miss and the best benchmark. Importance-tail pruning does not beat iteration 43.
- Next step: keep bottom-20 as a useful near-miss but continue with other approaches.

## 20260509_codex_iter88_bottom20_l240_catboost

- Hypothesis: the bottom-20 feature-pruned near-miss may benefit from slightly stronger CatBoost L2 regularization, recovering accepted accuracy while preserving higher coverage.
- Changed files: `experiments/configs/20260509_codex_iter88_bottom20_l240_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split`.
- Feature set: iteration 85 bottom-20 pruned feature set; HTF/time features retained.
- Model settings: CatBoost `l2_leaf_reg=40.0`, otherwise iteration 85 settings.
- Config: `experiments/configs/20260509_codex_iter88_bottom20_l240_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter88_bottom20_l240_catboost --config experiments/configs/20260509_codex_iter88_bottom20_l240_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter88_bottom20_l240_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16273784821681872`.
- Utility before / after: `0.0751684810782789` / `0.07568688439606013`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5744518103008669`.
- Accepted count before / after: `3245` / `3922`.
- Coverage before / after: `0.4205546915500259` / `0.5082944530844997`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger L2 on the bottom-20 subset over-accepts and sharply lowers accepted accuracy. Iteration 85 remains the best pruning variant.
- Next step: test a lower L2 on the bottom-20 subset only if pursuing this branch further.

## 20260509_codex_iter89_bottom20_l220_catboost

- Hypothesis: the bottom-20 feature-pruned near-miss may need slightly lower L2 regularization to recover accepted accuracy without losing too much coverage.
- Changed files: `experiments/configs/20260509_codex_iter89_bottom20_l220_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split`.
- Feature set: iteration 85 bottom-20 pruned feature set; HTF/time features retained.
- Model settings: CatBoost `l2_leaf_reg=20.0`, otherwise iteration 85 settings.
- Config: `experiments/configs/20260509_codex_iter89_bottom20_l220_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter89_bottom20_l220_catboost --config experiments/configs/20260509_codex_iter89_bottom20_l220_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter89_bottom20_l220_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17229454732887595`.
- Utility before / after: `0.0751684810782789` / `0.07050285121824784`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5869565217391305`.
- Accepted count before / after: `3245` / `3128`.
- Coverage before / after: `0.4205546915500259` / `0.40539139450492484`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: lower L2 recovers some accepted accuracy but loses too much coverage and utility. Iteration 85 remains the best bottom-20 pruning setup.
- Next step: leave the bottom-20 branch and continue elsewhere.

## 20260509_codex_iter90_catboost_rsm08

- Hypothesis: CatBoost feature subsampling (`rsm=0.8`) may reduce overfitting across the wide feature set and improve validation accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter90_catboost_rsm08.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `rsm: 0.8`.
- Config: `experiments/configs/20260509_codex_iter90_catboost_rsm08.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter90_catboost_rsm08 --config experiments/configs/20260509_codex_iter90_catboost_rsm08.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter90_catboost_rsm08/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16118940613674487`.
- Utility before / after: `0.0751684810782789` / `0.07205806117159146`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5763736263736263`.
- Accepted count before / after: `3245` / `3640`.
- Coverage before / after: `0.4205546915500259` / `0.47174701918092277`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: feature subsampling over-expands lower-quality acceptance and hurts the objective. Full feature availability remains better for CatBoost.
- Next step: do not pursue stronger `rsm` unless revisiting with a different feature subset.

## 20260509_codex_iter91_return_consistency_catboost

- Skill used: `tabular-last-diff-lag-features` and `tabular-leak-free-loop-features` principles.
- Hypothesis: leak-free return sign-consistency and lag-summary features may capture short-term persistence/reversal state better than individual lag columns.
- Changed files: `experiments/configs/20260509_codex_iter91_return_consistency_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_split`.
- Feature set: current best VWAP-pruned top-500 split plus 16 return/HTF-return lag summary features; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter91_return_consistency_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_catboost --config experiments/configs/20260509_codex_iter91_return_consistency_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17139599764919622`.
- Utility before / after: `0.0751684810782789` / `0.07024364955935719`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5864709636247607`.
- Accepted count before / after: `3245` / `3134`.
- Coverage before / after: `0.4205546915500259` / `0.4061689994815967`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the return-consistency features preserve reasonable accepted accuracy but reduce coverage and utility. They do not improve the objective.
- Next step: continue with other feature/model approaches.

## 20260509_codex_iter92_bottom20_seed43_catboost

- Hypothesis: the bottom-20 feature-pruned near-miss may be seed-sensitive; seed 43 may improve the selection-score tradeoff.
- Changed files: `experiments/configs/20260509_codex_iter92_bottom20_seed43_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split`.
- Feature set: iteration 85 bottom-20 pruned feature set; HTF/time features retained.
- Model settings: CatBoost `random_seed=43`, otherwise iteration 85 settings.
- Config: `experiments/configs/20260509_codex_iter92_bottom20_seed43_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter92_bottom20_seed43_catboost --config experiments/configs/20260509_codex_iter92_bottom20_seed43_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter92_bottom20_seed43_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17118341139584678`.
- Utility before / after: `0.0751684810782789` / `0.07244686365992745`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5841107433042432`.
- Accepted count before / after: `3245` / `3323`.
- Coverage before / after: `0.4205546915500259` / `0.43066355624676`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: seed 43 does not rescue the bottom-20 feature-pruning near-miss. Seed 42 remains better for that branch and iteration 43 remains best overall.
- Next step: leave the bottom-20 pruning branch.

## 20260509_codex_iter93_catboost_bernoulli08

- Hypothesis: Bernoulli bootstrap row subsampling may regularize the best CatBoost model better than Bayesian bagging and improve validation accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter93_catboost_bernoulli08.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `bootstrap_type: Bernoulli`, `subsample: 0.8` replacing `bagging_temperature`.
- Config: `experiments/configs/20260509_codex_iter93_catboost_bernoulli08.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter93_catboost_bernoulli08 --config experiments/configs/20260509_codex_iter93_catboost_bernoulli08.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter93_catboost_bernoulli08/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1696886067484292`.
- Utility before / after: `0.0751684810782789` / `0.07348367029548988`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5819127419820861`.
- Accepted count before / after: `3245` / `3461`.
- Coverage before / after: `0.4205546915500259` / `0.44854847071021253`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: Bernoulli bootstrap over-accepts and lowers accepted accuracy. The best Bayesian bootstrap CatBoost settings remain preferable.
- Next step: keep the current best as benchmark.

## 20260509_codex_iter94_drop_low_abs_return_train_catboost

- Skill used: data-weighting guidance from `tabular-balanced-log-loss` / prior-focused training.
- Hypothesis: very low-absolute-return development rows are noisy for a 5m direction task; removing them may improve selective confidence more than simply downweighting them.
- Changed files: `experiments/configs/20260509_codex_iter94_drop_low_abs_return_train_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split`.
- Data processing: removed development rows with `abs_return < 0.0001`, keeping 18,428 of 19,922 rows; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter94_drop_low_abs_return_train_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_catboost --config experiments/configs/20260509_codex_iter94_drop_low_abs_return_train_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1771564762476538`.
- Utility before / after: `0.0751684810782789` / `0.07257646448937274`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5888888888888889`.
- Accepted count before / after: `3245` / `3150`.
- Coverage before / after: `0.4205546915500259` / `0.4082426127527216`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: filtering low-absolute-return training rows is a valid near-miss, preserving accepted accuracy but losing too much coverage and utility.
- Next step: test a milder low-absolute-return filter.

## 20260509_codex_iter95_drop_tiny_abs_return_train_catboost

- Skill used: data-weighting guidance from `tabular-balanced-log-loss` / prior-focused training.
- Hypothesis: a milder low-absolute-return training filter may remove only the noisiest examples while preserving more coverage than iteration 94.
- Changed files: `experiments/configs/20260509_codex_iter95_drop_tiny_abs_return_train_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_split`.
- Data processing: removed development rows with `abs_return < 0.00005`, keeping 19,194 of 19,922 rows; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter95_drop_tiny_abs_return_train_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_catboost --config experiments/configs/20260509_codex_iter95_drop_tiny_abs_return_train_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16896951322359216`.
- Utility before / after: `0.0751684810782789` / `0.06959564541213063`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5851030110935024`.
- Accepted count before / after: `3245` / `3155`.
- Coverage before / after: `0.4205546915500259` / `0.40889061689994816`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the milder filter is worse than the stricter low-return filter and worse than the benchmark. Low-return row filtering is not enough.
- Next step: stop this filtering branch.

## 20260509_codex_iter96_bottom20_drop_low_return_catboost

- Hypothesis: combining the bottom-20 importance-pruned feature set with low-absolute-return row filtering may preserve accepted accuracy while reducing noisy examples and features.
- Changed files: `experiments/configs/20260509_codex_iter96_bottom20_drop_low_return_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter96_bottom20_drop_low_return_split`.
- Data processing: removed development rows with `abs_return < 0.0001` from the bottom-20 feature-pruned split, keeping 18,428 rows; validation unchanged.
- Feature set: 496 selected features; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter96_bottom20_drop_low_return_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter96_bottom20_drop_low_return_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter96_bottom20_drop_low_return_catboost --config experiments/configs/20260509_codex_iter96_bottom20_drop_low_return_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter96_bottom20_drop_low_return_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16973275974618138`.
- Utility before / after: `0.0751684810782789` / `0.06907724209434937`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5862738750404661`.
- Accepted count before / after: `3245` / `3089`.
- Coverage before / after: `0.4205546915500259` / `0.4003369621565578`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stacking the two near-miss branches lowers coverage to the floor and does not improve score. Do not combine these branches further.
- Next step: continue with a different approach.

## 20260509_codex_iter97_train72_drop_sl_vwap_catboost

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`; feature-selection discipline from `tabular-recursive-feature-elimination`.
- Hypothesis: a slightly shorter 72-day recent development window around the current best 75-day setup may improve validation regime match while retaining enough coverage.
- Changed files: `experiments/configs/20260509_codex_iter97_train72_drop_sl_vwap_catboost.yaml`, `experiments/optimization_log.md`.
- Input data: local raw 1m parquet `artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet`; no download.
- Feature set: current best VWAP-pruned top-500 recipe; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter97_train72_drop_sl_vwap_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --input artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet --output-dir artifacts/data_v2/experiments/20260509_codex_iter97_train72_drop_sl_vwap_catboost --config experiments/configs/20260509_codex_iter97_train72_drop_sl_vwap_catboost.yaml --horizon 5m --train-window-days 72 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter97_train72_drop_sl_vwap_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15651017565652897`.
- Utility before / after: `0.0751684810782789` / `0.06493001555209954`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5793474817865062`.
- Accepted count before / after: `3245` / `3157`.
- Coverage before / after: `0.4205546915500259` / `0.40914981855883875`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the 72-day fresh rebuild underperforms the current best; shortening the train window around 75 days does not improve the objective.
- Next step: return to the current best cached split and test narrowly-scoped CatBoost regularization knobs.

## 20260509_codex_iter98_catboost_model_size_reg2

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: adding CatBoost `model_size_reg: 2.0` may reduce overfit splits and improve selective validation accuracy without changing thresholds or feature semantics.
- Changed files: `experiments/configs/20260509_codex_iter98_catboost_model_size_reg2.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `model_size_reg: 2.0`.
- Config: `experiments/configs/20260509_codex_iter98_catboost_model_size_reg2.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter98_catboost_model_size_reg2 --config experiments/configs/20260509_codex_iter98_catboost_model_size_reg2.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter98_catboost_model_size_reg2/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: model-size regularization is effectively neutral and does not beat the incumbent after the accepted-count/coverage tie-breakers.
- Next step: test CatBoost quantization regularization with `border_count`.

## 20260509_codex_iter99_catboost_border_count128

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: reducing continuous feature quantization to `border_count: 128` may regularize noisy numeric splits and improve validation selection_score.
- Changed files: `experiments/configs/20260509_codex_iter99_catboost_border_count128.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `border_count: 128`.
- Config: `experiments/configs/20260509_codex_iter99_catboost_border_count128.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter99_catboost_border_count128 --config experiments/configs/20260509_codex_iter99_catboost_border_count128.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter99_catboost_border_count128/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17450864950997969`.
- Utility before / after: `0.0751684810782789` / `0.0736132711249352`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5856970428485214`.
- Accepted count before / after: `3245` / `3314`.
- Coverage before / after: `0.4205546915500259` / `0.4294971487817522`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: coarser quantization increases coverage but lowers accepted accuracy enough to reduce the objective.
- Next step: try a stronger `border_count: 64` only if it changes the accuracy/coverage tradeoff.

## 20260509_codex_iter100_catboost_border_count64

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: stronger numeric quantization regularization with `border_count: 64` may improve selective confidence if 128 still overfits noisy cut points.
- Changed files: `experiments/configs/20260509_codex_iter100_catboost_border_count64.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `border_count: 64`.
- Config: `experiments/configs/20260509_codex_iter100_catboost_border_count64.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter100_catboost_border_count64 --config experiments/configs/20260509_codex_iter100_catboost_border_count64.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter100_catboost_border_count64/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17066523243914425`.
- Utility before / after: `0.0751684810782789` / `0.07529808190772418`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5810320781032078`.
- Accepted count before / after: `3245` / `3585`.
- Coverage before / after: `0.4205546915500259` / `0.4646189735614308`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger quantization increases accepted count and utility slightly but sacrifices accepted accuracy and downside-risk enough to lower selection_score. Stop this branch.
- Next step: test CatBoost leaf-estimation regularization rather than split quantization.

## 20260509_codex_iter101_catboost_leaf_estimation5

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: reducing CatBoost `leaf_estimation_iterations` to 5 may shrink per-tree leaf values and improve validation selective accuracy without touching thresholds.
- Changed files: `experiments/configs/20260509_codex_iter101_catboost_leaf_estimation5.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `leaf_estimation_iterations: 5`.
- Config: `experiments/configs/20260509_codex_iter101_catboost_leaf_estimation5.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter101_catboost_leaf_estimation5 --config experiments/configs/20260509_codex_iter101_catboost_leaf_estimation5.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter101_catboost_leaf_estimation5/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17027308905605648`.
- Utility before / after: `0.0751684810782789` / `0.07037325038880246`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5854042151620006`.
- Accepted count before / after: `3245` / `3179`.
- Coverage before / after: `0.4205546915500259` / `0.4120010368066356`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: leaf-estimation shrinkage reduces coverage and does not improve accuracy enough; it is worse than the incumbent.
- Next step: try `leaf_estimation_iterations: 1` once, then stop this branch if it also underperforms.

## 20260509_codex_iter102_catboost_leaf_estimation1

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: a stronger `leaf_estimation_iterations: 1` setting may reduce overfit leaf values enough to improve validation selection_score.
- Changed files: `experiments/configs/20260509_codex_iter102_catboost_leaf_estimation1.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `leaf_estimation_iterations: 1`.
- Config: `experiments/configs/20260509_codex_iter102_catboost_leaf_estimation1.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter102_catboost_leaf_estimation1 --config experiments/configs/20260509_codex_iter102_catboost_leaf_estimation1.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter102_catboost_leaf_estimation1/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17043634959221762`.
- Utility before / after: `0.0751684810782789` / `0.06933644375324004`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5865976044027194`.
- Accepted count before / after: `3245` / `3089`.
- Coverage before / after: `0.4205546915500259` / `0.4003369621565578`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger leaf-estimation shrinkage leaves coverage barely above the floor and still does not beat the incumbent. Stop this branch.
- Next step: return to feature selection, using existing importance evidence for smaller low-importance pruning steps.

## 20260509_codex_iter103_drop_bottom25_non_htf_time_catboost

- Skill used: feature-selection discipline from `tabular-recursive-feature-elimination`.
- Hypothesis: removing the 25 lowest CatBoost-importance features while protecting all HTF/time features may reduce noisy covariates and improve validation selection_score.
- Changed files: `experiments/configs/20260509_codex_iter103_drop_bottom25_non_htf_time_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter103_drop_bottom25_non_htf_time_split`.
- Data processing: removed 25 non-HTF/time low-importance columns from the incumbent split; development and validation went from 540 to 515 columns.
- Dropped features: `sl_total_volume_10s`, `sl_agg_mean_interarrival_ms_5s`, `signed_volume_1_lag1`, `sl_agg_median_trade_size_3s`, `ret_3_rolling_z_12`, `ret_term_ratio__ret_1__ret_10`, `sl_agg_sell_trade_cluster_score_60s`, `abs_ret_mean_20`, `upside_rv_20`, `ret_vol_product__ret_10__rv_5`, `efficiency_10_delta_3`, `sl_mirror_true_range_pct_1s`, `wick_pressure_1_lag3`, `sl_agg_large_sell_trade_count_5s`, `sl_agg_max_trade_notional_10s`, `sl_post_jump_followthrough_ratio`, `efficiency_10_mean_gap_24`, `sl_taker_buy_count_60s`, `sl_mirror_close_z_60s`, `signed_dollar_volume_1`, `sl_interaction__sl_mirror_ret_60s__x__sl_mirror_rv_60s`, `efficiency_10_rolling_z_12`, `compression_score_rolling_z_24`, `ret_range_pos_product__ret_15__range_pos_10`, `low_volume_flag_share_5_rolling_z_12`.
- Feature set: HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter103_drop_bottom25_non_htf_time_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter103_drop_bottom25_non_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter103_drop_bottom25_non_htf_time_catboost --config experiments/configs/20260509_codex_iter103_drop_bottom25_non_htf_time_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter103_drop_bottom25_non_htf_time_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16561879591697629`.
- Utility before / after: `0.0751684810782789` / `0.078149300155521`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5746471898984897`.
- Accepted count before / after: `3245` / `4039`.
- Coverage before / after: `0.4205546915500259` / `0.5234577501296008`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: the pruned model accepts many more samples but loses too much accepted accuracy. This low-importance feature set is not a clean path to the target.
- Next step: use probability distribution diagnostics or a narrower feature perturbation rather than broader pruning.

## 20260509_codex_iter104_train_abs_return_ge5bp_catboost

- Skill used: data-processing discipline from `tabular-balanced-log-loss` / prior-focused training.
- Hypothesis: because validation diagnostics show much stronger score on `abs_return_ge_5bp`, training only on higher-magnitude development examples may make the model prefer cleaner directional moves.
- Changed files: `experiments/configs/20260509_codex_iter104_train_abs_return_ge5bp_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split`.
- Data processing: kept development rows with `abs(abs_return) >= 0.0005`, reducing development from 19,922 to 12,683 rows; validation unchanged. `abs_return` was not used as a feature.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter104_train_abs_return_ge5bp_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_catboost --config experiments/configs/20260509_codex_iter104_train_abs_return_ge5bp_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17400432702731852`.
- Utility before / after: `0.0751684810782789` / `0.07218766200103677`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5866791160908807`.
- Accepted count before / after: `3245` / `3213`.
- Coverage before / after: `0.4205546915500259` / `0.41640746500777603`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: high-magnitude-only training improves in-sample quality but does not transfer enough to validation; the incumbent still has better score and utility.
- Next step: stop target-magnitude filtering and inspect feature-family effects.

## 20260509_codex_iter105_catboost_depthwise

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: `grow_policy: Depthwise` may capture asymmetric feature interactions better than symmetric trees while preserving the same feature set and thresholds.
- Changed files: `experiments/configs/20260509_codex_iter105_catboost_depthwise.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `grow_policy: Depthwise`.
- Config: `experiments/configs/20260509_codex_iter105_catboost_depthwise.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter105_catboost_depthwise --config experiments/configs/20260509_codex_iter105_catboost_depthwise.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter105_catboost_depthwise/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16479724716812935`.
- Utility before / after: `0.0751684810782789` / `0.06946604458268531`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5817571690054911`.
- Accepted count before / after: `3245` / `3278`.
- Coverage before / after: `0.4205546915500259` / `0.4248315189217211`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: Depthwise trees overfit train and degrade validation accepted accuracy. Stop this tree-shape branch.
- Next step: return to symmetric CatBoost and test smaller, monotonic parameter changes only.

## 20260509_codex_iter106_catboost_lr012_iter1600

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: lower learning rate with more iterations (`learning_rate: 0.012`, `iterations: 1600`) may smooth probability ranking and improve selection_score.
- Changed files: `experiments/configs/20260509_codex_iter106_catboost_lr012_iter1600.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `iterations: 1600`, `learning_rate: 0.012`.
- Config: `experiments/configs/20260509_codex_iter106_catboost_lr012_iter1600.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter106_catboost_lr012_iter1600 --config experiments/configs/20260509_codex_iter106_catboost_lr012_iter1600.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter106_catboost_lr012_iter1600/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.17120761753786404`.
- Utility before / after: `0.0751684810782789` / `0.06959564541213067`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.586977648202138`.
- Accepted count before / after: `3245` / `3087`.
- Coverage before / after: `0.4205546915500259` / `0.4000777604976672`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: smoother longer training lowers coverage to the floor and does not improve score. The incumbent remains better.
- Next step: avoid more same-family CatBoost smoothing unless paired with a clear feature/data change.

## 20260509_codex_iter107_catboost_temp025

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: lowering Bayesian bootstrap randomness to `bagging_temperature: 0.25` while keeping `random_strength: 2.0` may preserve the current best structure but reduce variance in noisy splits.
- Changed files: `experiments/configs/20260509_codex_iter107_catboost_temp025.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `bagging_temperature: 0.25`.
- Config: `experiments/configs/20260509_codex_iter107_catboost_temp025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter107_catboost_temp025 --config experiments/configs/20260509_codex_iter107_catboost_temp025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter107_catboost_temp025/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is effectively neutral and does not beat the incumbent after accepted-count/coverage tie-breakers.
- Next step: test the opposite side with `bagging_temperature: 0.75`.

## 20260509_codex_iter108_catboost_temp075

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: increasing Bayesian bootstrap randomness to `bagging_temperature: 0.75` may improve validation robustness relative to the incumbent `0.5`.
- Changed files: `experiments/configs/20260509_codex_iter108_catboost_temp075.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `bagging_temperature: 0.75`.
- Config: `experiments/configs/20260509_codex_iter108_catboost_temp075.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter108_catboost_temp075 --config experiments/configs/20260509_codex_iter108_catboost_temp075.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter108_catboost_temp075/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: this is also neutral. Do not spend more iterations on Bayesian bootstrap temperature.
- Next step: switch to feature input side rather than CatBoost temperature.

## 20260509_codex_iter109_drop_bottom5_non_htf_time_catboost

- Skill used: feature-selection discipline from `tabular-recursive-feature-elimination`.
- Hypothesis: removing only the 5 lowest CatBoost-importance non-HTF/time features may reduce noise without the over-pruning seen in the bottom25 experiment.
- Changed files: `experiments/configs/20260509_codex_iter109_drop_bottom5_non_htf_time_catboost.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_split`.
- Data processing: removed `sl_total_volume_10s`, `sl_agg_mean_interarrival_ms_5s`, `signed_volume_1_lag1`, `sl_agg_median_trade_size_3s`, `ret_3_rolling_z_12`; development and validation went from 540 to 535 columns.
- Feature set: HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter109_drop_bottom5_non_htf_time_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_catboost --config experiments/configs/20260509_codex_iter109_drop_bottom5_non_htf_time_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.170999728660344`.
- Utility before / after: `0.0751684810782789` / `0.07102125453602902`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5853582554517134`.
- Accepted count before / after: `3245` / `3210`.
- Coverage before / after: `0.4205546915500259` / `0.41601866251944014`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: even minimal low-importance pruning hurts validation objective. Stop this pruning branch.
- Next step: inspect model/report artifacts for another feature/data direction.

## 20260509_codex_iter110_catboost_langevin

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: CatBoost Langevin regularization may reduce overfit to noisy microstructure patterns and improve validation selection_score.
- Changed files: `experiments/configs/20260509_codex_iter110_catboost_langevin.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `langevin: true`, `diffusion_temperature: 10000`.
- Config: `experiments/configs/20260509_codex_iter110_catboost_langevin.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter110_catboost_langevin --config experiments/configs/20260509_codex_iter110_catboost_langevin.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter110_catboost_langevin/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.16565921536408432`.
- Utility before / after: `0.0751684810782789` / `0.06985484707102127`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.582089552238806`.
- Accepted count before / after: `3245` / `3283`.
- Coverage before / after: `0.4205546915500259` / `0.42547952306894765`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: Langevin increases train score but lowers validation accepted accuracy; it is not useful here.
- Next step: continue with a different, non-temperature/non-pruning direction.

## 20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost

- Skill used: `tabular-adversarial-validation`.
- Hypothesis: adversarial validation shows strong train/validation drift; removing the highest-drift non-HTF/time features may reduce overfit to the development period.
- Changed files: `experiments/configs/20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost.yaml`, `experiments/optimization_log.md`.
- Diagnostic: train-vs-validation adversarial AUC `0.9844297893979366`; drift importances saved at `artifacts/data_v2/experiments/20260509_codex_iter111_adversarial_validation/drift_feature_importance.csv`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_split`.
- Data processing: removed `low_volume_flag_share_20_mean_gap_6`, `sl_range_10s`, `sl_mirror_true_range_pct_1s`, `sl_range_3s`, `low_volume_flag_share_20_rolling_z_6`; development and validation went from 540 to 535 columns.
- Feature set: HTF/time features retained.
- Config: `experiments/configs/20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost --config experiments/configs/20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_catboost/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1678702462932051`.
- Utility before / after: `0.0751684810782789` / `0.07050285121824777`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5832823025107164`.
- Accepted count before / after: `3245` / `3266`.
- Coverage before / after: `0.4205546915500259` / `0.4232763089683774`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: drift-driven feature removal lowers validation objective; these shifted features still carry useful directional signal.
- Next step: use adversarial validation as diagnostic only, not as a blind removal rule.

## 20260509_codex_iter112_catboost_rv5_regime

- Skill used: `tabular-per-type-model-training`.
- Hypothesis: validation diagnostics differ by `rv_5` volatility regime, so routing predictions through regime-specific CatBoost models may improve selective accuracy.
- Changed files: `src/model/catboost_regime_plugin.py`, `src/model/registry.py`, `experiments/configs/20260509_codex_iter112_catboost_rv5_regime.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: new `catboost_regime` plugin with a global fallback model plus low/mid/high `rv_5` regime models using training-set tertiles and current best CatBoost parameters.
- Config: `experiments/configs/20260509_codex_iter112_catboost_rv5_regime.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter112_catboost_rv5_regime --config experiments/configs/20260509_codex_iter112_catboost_rv5_regime.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter112_catboost_rv5_regime/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.12618394358065715`.
- Utility before / after: `0.0751684810782789` / `0.05572835666148263`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5625`.
- Accepted count before / after: `3245` / `3440`.
- Coverage before / after: `0.4205546915500259` / `0.44582685329186106`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no separate unit test was added because this branch underperformed and should not become active.
- Interpretation: regime routing overfits the development data and sharply reduces validation accepted accuracy. Do not expand this plugin direction without a better gating design.
- Next step: keep the plugin available but inactive; return to config-only experiments.

## 20260509_codex_iter113_catboost_mvs_bootstrap

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: CatBoost `bootstrap_type: MVS` with `subsample: 0.8` may regularize sample usage differently from Bayesian and Bernoulli bootstrap while preserving the incumbent feature set.
- Changed files: `experiments/configs/20260509_codex_iter113_catboost_mvs_bootstrap.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `bootstrap_type: MVS`, `subsample: 0.8`, and no `bagging_temperature`.
- Config: `experiments/configs/20260509_codex_iter113_catboost_mvs_bootstrap.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter113_catboost_mvs_bootstrap --config experiments/configs/20260509_codex_iter113_catboost_mvs_bootstrap.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter113_catboost_mvs_bootstrap/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: MVS produces the same acceptance metrics as the incumbent and does not beat it on accepted-count/coverage tie-breakers.
- Next step: move to another model-training parameter family.

## 20260509_codex_iter114_catboost_balanced_class_weights

- Skill used: class-balance guidance from `tabular-balanced-log-loss`.
- Hypothesis: CatBoost `auto_class_weights: Balanced` may improve directional class treatment and accepted-sample accuracy beyond the earlier `SqrtBalanced` test.
- Changed files: `experiments/configs/20260509_codex_iter114_catboost_balanced_class_weights.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `auto_class_weights: Balanced`.
- Config: `experiments/configs/20260509_codex_iter114_catboost_balanced_class_weights.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter114_catboost_balanced_class_weights --config experiments/configs/20260509_codex_iter114_catboost_balanced_class_weights.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter114_catboost_balanced_class_weights/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1765719188387535`.
- Utility before / after: `0.0751684810782789` / `0.07646448937273198`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5846727898966705`.
- Accepted count before / after: `3245` / `3484`.
- Coverage before / after: `0.4205546915500259` / `0.45152928978745466`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: stronger class weighting raises accepted count and utility slightly but lowers accepted accuracy enough to reduce selection_score.
- Next step: do not continue CatBoost class-weight escalation.

## 20260509_codex_iter115_lgbm_goss

- Skill used: LightGBM model-family guidance from `tabular-lgbm-dart-boosting` and related model tuning discipline.
- Hypothesis: LightGBM GOSS may focus on harder gradients and improve selective validation performance on the current best feature split.
- Changed files: `experiments/configs/20260509_codex_iter115_lgbm_goss.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `lightgbm` active plugin with `boosting_type: goss`, `n_estimators: 900`, `learning_rate: 0.025`, `num_leaves: 16`, `min_child_samples: 350`, `max_depth: 5`, `top_rate: 0.2`, `other_rate: 0.1`.
- Config: `experiments/configs/20260509_codex_iter115_lgbm_goss.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter115_lgbm_goss --config experiments/configs/20260509_codex_iter115_lgbm_goss.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter115_lgbm_goss/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.15329534943382508`.
- Utility before / after: `0.0751684810782789` / `0.06311560393986526`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5784724460199807`.
- Accepted count before / after: `3245` / `3103`.
- Coverage before / after: `0.4205546915500259` / `0.4021513737687921`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: LightGBM GOSS underperforms the CatBoost incumbent and leaves coverage close to the floor. Do not pursue this model branch.
- Next step: return to CatBoost/data feature experiments.

## 20260509_codex_iter116_catboost_crossentropy

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: CatBoost `CrossEntropy` may produce a different probability surface than `Logloss` and improve selective accuracy under the existing threshold grid.
- Changed files: `experiments/configs/20260509_codex_iter116_catboost_crossentropy.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `loss_function: CrossEntropy`, `eval_metric: CrossEntropy`.
- Config: `experiments/configs/20260509_codex_iter116_catboost_crossentropy.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter116_catboost_crossentropy --config experiments/configs/20260509_codex_iter116_catboost_crossentropy.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter116_catboost_crossentropy/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.1809171641645443`.
- Utility before / after: `0.0751684810782789` / `0.0751684810782789`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.5893958076448829`.
- Accepted count before / after: `3245` / `3244`.
- Coverage before / after: `0.4205546915500259` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; no code changes in this iteration.
- Interpretation: CrossEntropy is effectively identical to the incumbent for this binary-label setup and does not beat it after tie-breakers.
- Next step: stop CatBoost loss-function variants unless a materially different objective is introduced.

## 20260509_codex_iter117_catboost_platt_calibration

- Skill used: probability calibration workflow using the existing `platt` calibration plugin.
- Hypothesis: the binary selective path was ignoring configured calibration; fitting Platt scaling on development predictions only may improve probability scale and threshold selection without using validation labels for calibration.
- Changed files: `src/model/train.py`, `experiments/configs/20260509_codex_iter117_catboost_platt_calibration.yaml`, `experiments/optimization_log.md`.
- Code change: `train_binary_selective_model_from_split` now creates `settings.calibration`'s binary calibrator, fits it on development raw probabilities and labels, and transforms both development and validation probabilities.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`.
- Config: `experiments/configs/20260509_codex_iter117_catboost_platt_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter117_catboost_platt_calibration --config experiments/configs/20260509_codex_iter117_catboost_platt_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter117_catboost_platt_calibration/metrics.json`.
- Score before: `0.1809240380968129`.
- Score after: `0.18451203496023655`.
- Utility before / after: `0.0751684810782789` / `0.07465007776049763`.
- Accepted accuracy before / after: `0.5893814907872698` / `0.59284332688588`.
- Accepted count before / after: `3245` / `3102`.
- Coverage before / after: `0.4205546915500259` / `0.4020217729393468`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: Platt calibration is the new best selection_score so far, but it buys accuracy by dropping coverage close to the floor. It is still below the 0.24 target.
- Next step: test the existing isotonic calibrator on the same split.

## 20260509_codex_iter118_catboost_isotonic_calibration

- Skill used: probability calibration workflow using the existing `isotonic` calibration plugin.
- Hypothesis: non-parametric isotonic calibration may improve the probability scale more flexibly than Platt while still fitting only on development predictions.
- Changed files: `experiments/configs/20260509_codex_iter118_catboost_isotonic_calibration.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: isotonic`.
- Config: `experiments/configs/20260509_codex_iter118_catboost_isotonic_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter118_catboost_isotonic_calibration --config experiments/configs/20260509_codex_iter118_catboost_isotonic_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter118_catboost_isotonic_calibration/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.17148532779167233`.
- Utility before / after: `0.07465007776049763` / `0.07750129600829445`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5797333333333333`.
- Accepted count before / after: `3102` / `3750`.
- Coverage before / after: `0.4020217729393468` / `0.4860031104199067`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: isotonic calibration expands coverage but lowers accepted accuracy too much. Platt remains the best calibration result.
- Next step: keep Platt as the benchmark and test whether it combines with a near-miss model/data branch.

## 20260509_codex_iter119_bottom20_platt_calibration

- Skill used: probability calibration workflow plus feature-selection discipline from `tabular-recursive-feature-elimination`.
- Hypothesis: the bottom20 feature-pruned branch had higher coverage and near-best utility; Platt calibration may recover accepted accuracy enough to beat the calibrated incumbent.
- Changed files: `experiments/configs/20260509_codex_iter119_bottom20_platt_calibration.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split`.
- Feature set: bottom20 low-importance pruned feature set; HTF/time features retained.
- Model settings: bottom20 CatBoost settings plus `calibration.active_plugin: platt`.
- Config: `experiments/configs/20260509_codex_iter119_bottom20_platt_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter85_drop_bottom20_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter119_bottom20_platt_calibration --config experiments/configs/20260509_codex_iter119_bottom20_platt_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter119_bottom20_platt_calibration/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.1816880856947416`.
- Utility before / after: `0.07465007776049763` / `0.0756868843960601`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5895156345800122`.
- Accepted count before / after: `3102` / `3262`.
- Coverage before / after: `0.4020217729393468` / `0.4227579056505962`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: this combination improves utility and coverage but loses enough accepted accuracy to trail Platt on the full current-best feature set.
- Next step: keep full feature set + Platt as benchmark.

## 20260509_codex_iter120_balanced_platt_calibration

- Skill used: probability calibration workflow plus class-balance guidance from `tabular-balanced-log-loss`.
- Hypothesis: the Balanced class-weight branch had higher utility/coverage; Platt calibration may recover enough accepted accuracy to beat the full-feature Platt benchmark.
- Changed files: `experiments/configs/20260509_codex_iter120_balanced_platt_calibration.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `auto_class_weights: Balanced` plus `calibration.active_plugin: platt`.
- Config: `experiments/configs/20260509_codex_iter120_balanced_platt_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter120_balanced_platt_calibration --config experiments/configs/20260509_codex_iter120_balanced_platt_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter120_balanced_platt_calibration/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.1799434037197668`.
- Utility before / after: `0.07465007776049763` / `0.07711249351995852`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5867599883347915`.
- Accepted count before / after: `3102` / `3429`.
- Coverage before / after: `0.4020217729393468` / `0.4444012441679627`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: the combination improves utility but gives up too much accepted accuracy. Full-feature CatBoost + Platt remains the best result.
- Next step: keep Platt benchmark and look for a model/data change that raises accuracy without crushing coverage.

## 20260509_codex_iter121_low_abs_return_platt_calibration

- Skill used: probability calibration workflow plus data-filtering guidance from `tabular-balanced-log-loss`.
- Hypothesis: the low-absolute-return training filter preserved high accepted accuracy but lost coverage; Platt calibration may improve the probability scale enough to raise selection_score.
- Changed files: `experiments/configs/20260509_codex_iter121_low_abs_return_platt_calibration.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split`.
- Data processing: development rows with `abs_return < 0.0001` removed; validation unchanged.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`.
- Config: `experiments/configs/20260509_codex_iter121_low_abs_return_platt_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter121_low_abs_return_platt_calibration --config experiments/configs/20260509_codex_iter121_low_abs_return_platt_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter121_low_abs_return_platt_calibration/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.17756830898375978`.
- Utility before / after: `0.07465007776049763` / `0.07257646448937272`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5892288081580624`.
- Accepted count before / after: `3102` / `3138`.
- Coverage before / after: `0.4020217729393468` / `0.40668740279937793`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: Platt does not rescue the low-return filter branch. Full-feature CatBoost + Platt remains best.
- Next step: avoid further target-magnitude filter combinations.

## 20260509_codex_iter122_border64_platt_calibration

- Skill used: probability calibration workflow plus CatBoost parameter discipline.
- Hypothesis: `border_count: 64` had higher coverage/utility but low accepted accuracy; Platt calibration may improve the probability scale and recover selection_score.
- Changed files: `experiments/configs/20260509_codex_iter122_border64_platt_calibration.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `border_count: 64` plus `calibration.active_plugin: platt`.
- Config: `experiments/configs/20260509_codex_iter122_border64_platt_calibration.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter122_border64_platt_calibration --config experiments/configs/20260509_codex_iter122_border64_platt_calibration.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter122_border64_platt_calibration/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.17086816420181444`.
- Utility before / after: `0.07465007776049763` / `0.07490927941938831`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5815462753950339`.
- Accepted count before / after: `3102` / `3544`.
- Coverage before / after: `0.4020217729393468` / `0.45930533955417313`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: Platt does not fix the accuracy loss from coarse quantization. Full-feature CatBoost + Platt remains best.
- Next step: keep calibration code, but avoid broad combinations that mainly increase coverage at lower precision.

## 20260509_codex_iter123_catboost_platt_c025

- Skill used: probability calibration workflow inspired by `tabular-confidence-probability-clipping`, using calibrated probability scaling rather than hard clipping.
- Hypothesis: stronger Platt regularization (`C: 0.25`) may soften the calibrated probability scale and improve the coverage/accuracy balance versus default Platt.
- Changed files: `src/calibration/registry.py`, `src/calibration/platt.py`, `experiments/configs/20260509_codex_iter123_catboost_platt_c025.yaml`, `experiments/optimization_log.md`.
- Code change: calibration plugin creation now passes per-plugin config parameters; Platt scaling accepts `C` and `max_iter`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`, `calibration.plugins.platt.C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter123_catboost_platt_c025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter123_catboost_platt_c025 --config experiments/configs/20260509_codex_iter123_catboost_platt_c025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter123_catboost_platt_c025/metrics.json`.
- Score before: `0.18451203496023655`.
- Score after: `0.18462759471376494`.
- Utility before / after: `0.07465007776049763` / `0.07516848107827886`.
- Accepted accuracy before / after: `0.59284332688588` / `0.5924155513065646`.
- Accepted count before / after: `3102` / `3138`.
- Coverage before / after: `0.4020217729393468` / `0.40668740279937793`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: `C: 0.25` is a small new best by improving coverage and utility while retaining nearly the same accepted accuracy. It remains well below the 0.24 target.
- Next step: probe nearby Platt regularization values.

## 20260509_codex_iter124_catboost_platt_c01

- Skill used: probability calibration workflow.
- Hypothesis: stronger Platt regularization (`C: 0.1`) may further improve coverage/utility while preserving accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter124_catboost_platt_c01.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`, `calibration.plugins.platt.C: 0.1`.
- Config: `experiments/configs/20260509_codex_iter124_catboost_platt_c01.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter124_catboost_platt_c01 --config experiments/configs/20260509_codex_iter124_catboost_platt_c01.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter124_catboost_platt_c01/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.1823274584883904`.
- Utility before / after: `0.07516848107827886` / `0.07529808190772425`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5904139433551199`.
- Accepted count before / after: `3138` / `3213`.
- Coverage before / after: `0.40668740279937793` / `0.41640746500777603`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: `C: 0.1` increases coverage but loses too much accepted accuracy. `C: 0.25` remains best.
- Next step: test the other side with `C: 0.5`.

## 20260509_codex_iter125_catboost_platt_c05

- Skill used: probability calibration workflow.
- Hypothesis: milder Platt regularization (`C: 0.5`) may preserve more accepted accuracy than `C: 0.1` while improving over default Platt.
- Changed files: `experiments/configs/20260509_codex_iter125_catboost_platt_c05.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`, `calibration.plugins.platt.C: 0.5`.
- Config: `experiments/configs/20260509_codex_iter125_catboost_platt_c05.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter125_catboost_platt_c05 --config experiments/configs/20260509_codex_iter125_catboost_platt_c05.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter125_catboost_platt_c05/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.18176268376131016`.
- Utility before / after: `0.07516848107827886` / `0.07400207361327116`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5912432086928732`.
- Accepted count before / after: `3138` / `3129`.
- Coverage before / after: `0.40668740279937793` / `0.40552099533437014`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: `C: 0.5` is worse than `C: 0.25`; stop this immediate Platt-C sweep.
- Next step: keep `C: 0.25` as the calibration benchmark.

## 20260509_codex_iter126_catboost_platt_c02

- Skill used: probability calibration workflow.
- Hypothesis: a nearby Platt regularization value (`C: 0.2`) may improve the small coverage/accuracy tradeoff seen at `C: 0.25`.
- Changed files: `experiments/configs/20260509_codex_iter126_catboost_platt_c02.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`, `calibration.plugins.platt.C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter126_catboost_platt_c02.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter126_catboost_platt_c02 --config experiments/configs/20260509_codex_iter126_catboost_platt_c02.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter126_catboost_platt_c02/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.18253714386992323`.
- Utility before / after: `0.07516848107827886` / `0.07452047693105238`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5913568477915475`.
- Accepted count before / after: `3138` / `3147`.
- Coverage before / after: `0.40668740279937793` / `0.4078538102643857`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: `C: 0.2` does not improve on `C: 0.25`; stop this local Platt sweep.
- Next step: keep `C: 0.25` as best.

## 20260509_codex_iter127_catboost_platt_c025_balanced

- Skill used: probability calibration workflow plus class-balance guidance from `tabular-balanced-log-loss`.
- Hypothesis: adding `class_weight: balanced` to the best Platt calibrator may improve asymmetric UP/DOWN calibration without changing the base model.
- Changed files: `src/calibration/platt.py`, `experiments/configs/20260509_codex_iter127_catboost_platt_c025_balanced.yaml`, `experiments/optimization_log.md`.
- Code change: Platt scaling now accepts `class_weight`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt`, `C: 0.25`, `class_weight: balanced`.
- Config: `experiments/configs/20260509_codex_iter127_catboost_platt_c025_balanced.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter127_catboost_platt_c025_balanced --config experiments/configs/20260509_codex_iter127_catboost_platt_c025_balanced.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter127_catboost_platt_c025_balanced/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.18234190152224153`.
- Utility before / after: `0.07516848107827886` / `0.07374287195438049`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5919818946007113`.
- Accepted count before / after: `3138` / `3093`.
- Coverage before / after: `0.40668740279937793` / `0.40085536547433903`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: balanced Platt hurts coverage and score. The best calibration remains unweighted Platt `C: 0.25`.
- Next step: test a different calibration family rather than more Platt class weighting.

## 20260509_codex_iter128_catboost_temperature125

- Skill used: probability calibration workflow inspired by `tabular-confidence-probability-clipping`, using logit temperature scaling rather than hard clipping.
- Hypothesis: softening probabilities with fixed `temperature: 1.25` may improve coverage/accuracy balance without fitting calibration to validation.
- Changed files: `src/calibration/temperature.py`, `src/calibration/registry.py`, `experiments/configs/20260509_codex_iter128_catboost_temperature125.yaml`, `experiments/optimization_log.md`.
- Code change: added `temperature` calibration plugin that scales logits by a fixed configured temperature and fits no validation-dependent parameters.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: temperature`, `temperature: 1.25`.
- Config: `experiments/configs/20260509_codex_iter128_catboost_temperature125.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter128_catboost_temperature125 --config experiments/configs/20260509_codex_iter128_catboost_temperature125.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter128_catboost_temperature125/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.18082709172769737`.
- Utility before / after: `0.07516848107827886` / `0.07465007776049767`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5898315658140986`.
- Accepted count before / after: `3138` / `3206`.
- Coverage before / after: `0.40668740279937793` / `0.4155002592016589`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; fixed-temperature calibration fits no validation labels.
- Interpretation: temperature softening is below Platt `C: 0.25`; it improves coverage but loses too much accepted accuracy.
- Next step: test the sharpening side with `temperature: 0.8`.

## 20260509_codex_iter129_catboost_temperature08

- Skill used: probability calibration workflow inspired by `tabular-confidence-probability-clipping`.
- Hypothesis: sharpening probabilities with fixed `temperature: 0.8` may improve selective confidence and selection_score.
- Changed files: `experiments/configs/20260509_codex_iter129_catboost_temperature08.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: temperature`, `temperature: 0.8`.
- Config: `experiments/configs/20260509_codex_iter129_catboost_temperature08.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter129_catboost_temperature08 --config experiments/configs/20260509_codex_iter129_catboost_temperature08.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter129_catboost_temperature08/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.1773129128752469`.
- Utility before / after: `0.07516848107827886` / `0.07400207361327109`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5876035593740411`.
- Accepted count before / after: `3138` / `3259`.
- Coverage before / after: `0.40668740279937793` / `0.42236910316226023`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; fixed-temperature calibration fits no validation labels.
- Interpretation: probability sharpening also trails Platt `C: 0.25`; stop fixed-temperature calibration.
- Next step: return to model/data experiments around the calibrated benchmark.

## 20260509_codex_iter130_catboost_auc_metric_platt_c025

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: using CatBoost `eval_metric: AUC` while keeping `loss_function: Logloss` may improve ranking quality for selective prediction, especially after Platt calibration.
- Changed files: `experiments/configs/20260509_codex_iter130_catboost_auc_metric_platt_c025.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: CatBoost `eval_metric: AUC`, otherwise current best settings plus Platt `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter130_catboost_auc_metric_platt_c025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter130_catboost_auc_metric_platt_c025 --config experiments/configs/20260509_codex_iter130_catboost_auc_metric_platt_c025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter130_catboost_auc_metric_platt_c025/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.17544972990282762`.
- Utility before / after: `0.07516848107827886` / `0.0732244686365993`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5868429142330157`.
- Accepted count before / after: `3138` / `3253`.
- Coverage before / after: `0.40668740279937793` / `0.4215914981855884`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: AUC eval metric improves neither accepted accuracy nor objective; keep Logloss eval metric.
- Next step: continue with a different model/data direction.

## 20260509_codex_iter131_seed43_platt_c025

- Skill used: probability calibration workflow plus `tabular-multi-seed-fold-averaging` diagnostics.
- Hypothesis: seed43 had higher utility/coverage but lower accepted accuracy; Platt `C: 0.25` may recover enough calibration quality to beat the seed42 calibrated benchmark.
- Changed files: `experiments/configs/20260509_codex_iter131_seed43_platt_c025.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `random_seed: 43` plus Platt `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter131_seed43_platt_c025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter131_seed43_platt_c025 --config experiments/configs/20260509_codex_iter131_seed43_platt_c025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter131_seed43_platt_c025/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.17560279500873954`.
- Utility before / after: `0.07516848107827886` / `0.08113011923276307`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.5798469387755102`.
- Accepted count before / after: `3138` / `3920`.
- Coverage before / after: `0.40668740279937793` / `0.5080352514256091`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Interpretation: seed43 plus Platt increases coverage and utility but drops accepted accuracy too far. Seed42 Platt `C: 0.25` remains best.
- Next step: do not pursue seed43 as a calibrated single-model replacement.

## 20260509_codex_iter132_catboost_platt_logit_c025

- Skill used: `tabular-logit-transform-stacking`, applied as a single-model logit-space Platt calibrator.
- Hypothesis: fitting the calibration logistic regression on `logit(p_up)` instead of raw `p_up` should better preserve odds-space separation and may improve selective accepted accuracy near the coverage boundary.
- Changed files: `src/calibration/platt_logit.py`, `src/calibration/registry.py`, `experiments/configs/20260509_codex_iter132_catboost_platt_logit_c025.yaml`, `experiments/optimization_log.md`.
- Code change: added `platt_logit` calibration plugin; existing `platt` behavior unchanged for reproducibility.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt_logit`, `C: 0.25`, `max_iter: 1000`.
- Config: `experiments/configs/20260509_codex_iter132_catboost_platt_logit_c025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter132_catboost_platt_logit_c025 --config experiments/configs/20260509_codex_iter132_catboost_platt_logit_c025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter132_catboost_platt_logit_c025/metrics.json`.
- Score before: `0.18462759471376494`.
- Score after: `0.1846861980124185`.
- Utility before / after: `0.07516848107827886` / `0.07477967858994294`.
- Accepted accuracy before / after: `0.5924155513065646` / `0.592854843900869`.
- Accepted count before / after: `3138` / `3107`.
- Coverage before / after: `0.40668740279937793` / `0.40266977708657337`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m py_compile src\calibration\platt_logit.py src\calibration\registry.py`; DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f5cecee`.
- Interpretation: logit-space calibration is a tiny but valid new best under the coverage constraint by improving accepted accuracy while staying above `coverage >= 0.40`.
- Next step: use this as the new calibration benchmark and test nearby regularization or data/model changes against it.

## 20260509_codex_iter133_catboost_platt_logit_c030

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: slightly weaker regularization (`C: 0.3`) for the new logit-space calibrator may recover a little utility while preserving the accepted-accuracy gain.
- Changed files: `experiments/configs/20260509_codex_iter133_catboost_platt_logit_c030.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt_logit`, `C: 0.3`, `max_iter: 1000`.
- Config: `experiments/configs/20260509_codex_iter133_catboost_platt_logit_c030.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter133_catboost_platt_logit_c030 --config experiments/configs/20260509_codex_iter133_catboost_platt_logit_c030.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter133_catboost_platt_logit_c030/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.18268522318286035`.
- Utility before / after: `0.07477967858994294` / `0.07426127527216173`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5917387127761767`.
- Accepted count before / after: `3107` / `3123`.
- Coverage before / after: `0.40266977708657337` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: weaker regularization is worse; keep `platt_logit C: 0.25`.
- Next step: if continuing calibration, test only the lower side; otherwise return to feature/model changes.

## 20260509_codex_iter134_catboost_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: slightly stronger regularization (`C: 0.2`) for the logit-space calibrator may improve accepted accuracy at the lower coverage boundary.
- Changed files: `experiments/configs/20260509_codex_iter134_catboost_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt_logit`, `C: 0.2`, `max_iter: 1000`.
- Config: `experiments/configs/20260509_codex_iter134_catboost_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter134_catboost_platt_logit_c020 --config experiments/configs/20260509_codex_iter134_catboost_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter134_catboost_platt_logit_c020/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.18312798896889548`.
- Utility before / after: `0.07477967858994294` / `0.07400207361327112`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5923649304432222`.
- Accepted count before / after: `3107` / `3091`.
- Coverage before / after: `0.40266977708657337` / `0.40059616381544844`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: stronger regularization also trails `C: 0.25`; stop local logit-Platt sweep.
- Next step: retain iteration 132 as best and shift to feature/data or model training changes.

## 20260509_codex_iter135_catboost_l220_platt_logit

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: reducing CatBoost `l2_leaf_reg` from `30.0` to `20.0` may sharpen learned interactions while logit-space calibration controls probability scale.
- Changed files: `experiments/configs/20260509_codex_iter135_catboost_l220_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `l2_leaf_reg: 20.0`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter135_catboost_l220_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter135_catboost_l220_platt_logit --config experiments/configs/20260509_codex_iter135_catboost_l220_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter135_catboost_l220_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.17428654505234928`.
- Utility before / after: `0.07477967858994294` / `0.07257646448937276`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5865265760197775`.
- Accepted count before / after: `3107` / `3236`.
- Coverage before / after: `0.40266977708657337` / `0.41938828408501816`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: weaker L2 hurts accepted accuracy and score; do not lower L2 on the current split.
- Next step: test the stronger L2 side once, then move away from local CatBoost L2 tuning if it fails.

## 20260509_codex_iter136_catboost_l240_platt_logit

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: increasing CatBoost `l2_leaf_reg` from `30.0` to `40.0` may reduce overfit and improve accepted accuracy after logit-space calibration.
- Changed files: `experiments/configs/20260509_codex_iter136_catboost_l240_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `l2_leaf_reg: 40.0`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter136_catboost_l240_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter136_catboost_l240_platt_logit --config experiments/configs/20260509_codex_iter136_catboost_l240_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter136_catboost_l240_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.1600604577681406`.
- Utility before / after: `0.07477967858994294` / `0.06791083462934161`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5793458509993943`.
- Accepted count before / after: `3107` / `3302`.
- Coverage before / after: `0.40266977708657337` / `0.4279419388284085`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: stronger L2 materially hurts accepted accuracy; keep `l2_leaf_reg: 30.0`.
- Next step: stop local L2 tuning and return to data/feature generation or a different model-family lever.

## 20260509_codex_iter137_multiscale_platt_logit

- Skill used: `timeseries-multi-gap-lag-diff-features`, adapted through the existing leak-free `multi_scale_rolling` pack.
- Hypothesis: adding multi-resolution rolling return/range/position features may capture medium-scale context not covered by the base momentum and HTF packs, improving selective accepted accuracy after logit calibration.
- Changed files: `experiments/configs/20260509_codex_iter137_multiscale_platt_logit.yaml`, `experiments/optimization_log.md`.
- Input data: `artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet`; no download.
- Feature set: current best feature set plus `multi_scale_rolling` with windows `[5, 15, 30, 60]`; HTF/time features retained.
- Model settings: current best CatBoost settings plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter137_multiscale_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --input artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet --output-dir artifacts/data_v2/experiments/20260509_codex_iter137_multiscale_platt_logit --config experiments/configs/20260509_codex_iter137_multiscale_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter137_multiscale_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.17010495689245902`.
- Utility before / after: `0.07477967858994294` / `0.07465007776049763`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5811724915445321`.
- Accepted count before / after: `3107` / `3548`.
- Coverage before / after: `0.40266977708657337` / `0.4598237428719544`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during full local-data rebuild and training.
- Git commit: $h.
- Interpretation: multiscale features increase coverage but reduce accepted accuracy too much. Do not add this pack to the best profile.
- Next step: return to compact, more targeted features or model-family alternatives.

## 20260509_codex_iter138_catboost_iter900_platt_logit

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: reducing CatBoost `iterations` from `1200` to `900` may reduce overfit while preserving the calibrated probability shape.
- Changed files: `experiments/configs/20260509_codex_iter138_catboost_iter900_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `iterations: 900`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter138_catboost_iter900_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter138_catboost_iter900_platt_logit --config experiments/configs/20260509_codex_iter138_catboost_iter900_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter138_catboost_iter900_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.1846861980124185`.
- Utility before / after: `0.07477967858994294` / `0.07477967858994294`.
- Accepted accuracy before / after: `0.592854843900869` / `0.592854843900869`.
- Accepted count before / after: `3107` / `3107`.
- Coverage before / after: `0.40266977708657337` / `0.40266977708657337`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `iterations: 900` is metric-neutral on this setup, likely because the fitted tree sequence is effectively unchanged by the training path. It does not improve the objective.
- Next step: do not pursue simple iteration reduction further.

## 20260509_codex_iter139_session_relative_platt_logit

- Skill used: `tabular-relative-deviation-features` plus the current best logit-space calibration path.
- Hypothesis: the near-miss session-relative feature split from iteration 66 may improve once probabilities are calibrated in logit space.
- Changed files: `experiments/configs/20260509_codex_iter139_session_relative_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split`.
- Feature set: current best VWAP-pruned top-500 split plus session-relative diff/ratio/z features for `rv_5`, `volume`, `relative_volume_20`, `htf_rv_15m`, and `dollar_vol_mean_20`; HTF/time features retained.
- Model settings: iteration 66 CatBoost settings plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter139_session_relative_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter139_session_relative_platt_logit --config experiments/configs/20260509_codex_iter139_session_relative_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter139_session_relative_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.1770827791259692`.
- Utility before / after: `0.07477967858994294` / `0.07335406946604461`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5880522713130056`.
- Accepted count before / after: `3107` / `3214`.
- Coverage before / after: `0.40266977708657337` / `0.41653706583722133`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: logit calibration does not rescue session-relative features; the best calibrated base split remains stronger.
- Next step: continue with a different small model/data lever.

## 20260509_codex_iter140_catboost_no_bootstrap_platt_logit

- Skill used: CatBoost parameter discipline from `tabular-catboost-multirmse`.
- Hypothesis: disabling bootstrap sampling may reduce probability noise on the 75-day window and improve calibrated selective precision.
- Changed files: `experiments/configs/20260509_codex_iter140_catboost_no_bootstrap_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best CatBoost settings with `bootstrap_type: "No"` replacing `bagging_temperature`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter140_catboost_no_bootstrap_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter140_catboost_no_bootstrap_platt_logit --config experiments/configs/20260509_codex_iter140_catboost_no_bootstrap_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter140_catboost_no_bootstrap_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.175327687444702`.
- Utility before / after: `0.07477967858994294` / `0.07141005702436498`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5885567341690775`.
- Accepted count before / after: `3107` / `3111`.
- Coverage before / after: `0.40266977708657337` / `0.4031881804043546`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: deterministic no-bootstrap sampling reduces accepted accuracy and score; keep Bayesian bootstrap with `bagging_temperature: 0.5`.
- Next step: avoid more bootstrap variants unless paired with a distinct model/data change.

## 20260509_codex_iter141_xgboost_platt_logit

- Skill used: model-family check informed by `tabular-xgb-gpu-batch-iterator`/XGBoost guidance, using CPU hist locally.
- Hypothesis: XGBoost may produce a different ranking/coverage tradeoff on the current 75-day split when paired with the best logit-space calibration.
- Changed files: `experiments/configs/20260509_codex_iter141_xgboost_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `active_plugin: xgboost`, conservative hist-tree parameters from the earlier XGBoost baseline, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter141_xgboost_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter141_xgboost_platt_logit --config experiments/configs/20260509_codex_iter141_xgboost_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter141_xgboost_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.1324412169914792`.
- Utility before / after: `0.07477967858994294` / `0.06350440642820114`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5606736007924715`.
- Accepted count before / after: `3107` / `4038`.
- Coverage before / after: `0.40266977708657337` / `0.5233281493001555`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: XGBoost over-accepts lower-quality predictions and is not competitive with calibrated CatBoost on this split.
- Next step: keep CatBoost as the model family and use future XGBoost only for explicit ensemble diagnostics.

## 20260509_codex_iter142_blend99_platt_logit

- Skill used: `tabular-logit-transform-stacking`, applied to the existing CatBoost/LightGBM logit-blend plugin plus logit-space calibration.
- Hypothesis: a tiny LightGBM component (`1%`) may perturb CatBoost's probability ranking enough to improve accepted count/utility while logit-space calibration preserves accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter142_blend99_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `active_plugin: catboost_lgbm_logit_blend`, `catboost_weight: 0.99`, current best CatBoost settings, regularized LightGBM settings, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter142_blend99_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter142_blend99_platt_logit --config experiments/configs/20260509_codex_iter142_blend99_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter142_blend99_platt_logit/metrics.json`.
- Score before: `0.1846861980124185`.
- Score after: `0.18511956039735958`.
- Utility before / after: `0.07477967858994294` / `0.0754276827371695`.
- Accepted accuracy before / after: `0.592854843900869` / `0.5925572519083969`.
- Accepted count before / after: `3107` / `3144`.
- Coverage before / after: `0.40266977708657337` / `0.40746500777604977`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: calibrated 99/1 logit blending is a small valid new best under the coverage constraint, mainly by increasing utility and accepted count while keeping accepted accuracy close to the CatBoost-only benchmark.
- Next step: sweep only very nearby blend weights; wider LightGBM weights had already underperformed.

## 20260509_codex_iter143_blend995_platt_logit

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: a smaller LightGBM contribution (`0.5%`) may preserve CatBoost accepted accuracy better than the 99/1 blend while retaining some ranking perturbation.
- Changed files: `experiments/configs/20260509_codex_iter143_blend995_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.995`, current best CatBoost settings, regularized LightGBM settings, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter143_blend995_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter143_blend995_platt_logit --config experiments/configs/20260509_codex_iter143_blend995_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter143_blend995_platt_logit/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.18293232145740404`.
- Utility before / after: `0.0754276827371695` / `0.07439087610160704`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.5918106206014075`.
- Accepted count before / after: `3144` / `3126`.
- Coverage before / after: `0.40746500777604977` / `0.4051321928460342`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.995` is worse than `0.99`; the 1% LightGBM perturbation is better than 0.5%.
- Next step: test the other nearby side (`0.985`) before stopping the local blend sweep.

## 20260509_codex_iter144_blend985_platt_logit

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: a slightly larger LightGBM contribution (`1.5%`) may improve accepted count/utility beyond the 99/1 blend while calibration maintains precision.
- Changed files: `experiments/configs/20260509_codex_iter144_blend985_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.985`, current best CatBoost settings, regularized LightGBM settings, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter144_blend985_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter144_blend985_platt_logit --config experiments/configs/20260509_codex_iter144_blend985_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter144_blend985_platt_logit/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.18382818887408964`.
- Utility before / after: `0.0754276827371695` / `0.07452047693105233`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.5924140147862423`.
- Accepted count before / after: `3144` / `3111`.
- Coverage before / after: `0.40746500777604977` / `0.4031881804043546`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.985` is also worse than `0.99`; stop the immediate CatBoost-heavy blend sweep.
- Next step: keep iteration 142 as best and test a different small lever.

## 20260509_codex_iter145_blend99_platt_logit_c030

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: the 99/1 blended probability may benefit from slightly weaker logit-calibrator regularization than CatBoost-only.
- Changed files: `experiments/configs/20260509_codex_iter145_blend99_platt_logit_c030.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.99`, plus `calibration.active_plugin: platt_logit`, `C: 0.3`.
- Config: `experiments/configs/20260509_codex_iter145_blend99_platt_logit_c030.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter145_blend99_platt_logit_c030 --config experiments/configs/20260509_codex_iter145_blend99_platt_logit_c030.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter145_blend99_platt_logit_c030/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.18493304210334233`.
- Utility before / after: `0.0754276827371695` / `0.0755572835666148`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.5922760367204811`.
- Accepted count before / after: `3144` / `3159`.
- Coverage before / after: `0.40746500777604977` / `0.4094090202177294`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `C: 0.3` raises utility/coverage slightly but loses enough accepted accuracy to trail the best score.
- Next step: keep `C: 0.25`; optionally check the stronger side once.

## 20260509_codex_iter146_blend99_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: stronger logit-calibrator regularization (`C: 0.2`) may improve accepted accuracy for the 99/1 blend near the coverage boundary.
- Changed files: `experiments/configs/20260509_codex_iter146_blend99_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.99`, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter146_blend99_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter146_blend99_platt_logit_c020 --config experiments/configs/20260509_codex_iter146_blend99_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter146_blend99_platt_logit_c020/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.1842071111539714`.
- Utility before / after: `0.0754276827371695` / `0.0749092794193883`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.592332268370607`.
- Accepted count before / after: `3144` / `3130`.
- Coverage before / after: `0.40746500777604977` / `0.40565059616381544`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `C: 0.2` also trails `C: 0.25`; stop the local blend-calibration sweep.
- Next step: keep iteration 142 as best.

## 20260509_codex_iter147_blend99_lgbm_seed43_platt_logit

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: changing the LightGBM/random seed inside the 99/1 blend may provide a slightly better probability-rank perturbation while keeping CatBoost fixed.
- Changed files: `experiments/configs/20260509_codex_iter147_blend99_lgbm_seed43_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.99`, LightGBM/logistic `random_state: 43`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter147_blend99_lgbm_seed43_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter147_blend99_lgbm_seed43_platt_logit --config experiments/configs/20260509_codex_iter147_blend99_lgbm_seed43_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter147_blend99_lgbm_seed43_platt_logit/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.1833860834661099`.
- Utility before / after: `0.0754276827371695` / `0.07477967858994296`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.5917912822144448`.
- Accepted count before / after: `3144` / `3143`.
- Coverage before / after: `0.40746500777604977` / `0.4073354069466045`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: seed 43 is worse; keep the original seed-42 blend.
- Next step: keep iteration 142 as best and avoid seed-only blend variants unless a stronger model change motivates them.

## 20260509_codex_iter148_blend99_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: with only a 1% LightGBM contribution, a DART LightGBM component may add a different ranking perturbation without overpowering CatBoost; logit calibration should preserve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter148_blend99_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.99`, current best CatBoost settings, DART LightGBM component (`n_estimators: 1600`, `learning_rate: 0.01`, `boosting_type: dart`, `colsample_bytree: 0.35`), plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter148_blend99_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter148_blend99_dart_platt_logit --config experiments/configs/20260509_codex_iter148_blend99_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter148_blend99_dart_platt_logit/metrics.json`.
- Score before: `0.18511956039735958`.
- Score after: `0.18680416480395254`.
- Utility before / after: `0.0754276827371695` / `0.07542768273716952`.
- Accepted accuracy before / after: `0.5925572519083969` / `0.5939315687540349`.
- Accepted count before / after: `3144` / `3098`.
- Coverage before / after: `0.40746500777604977` / `0.4015033696215656`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: DART as a tiny blend component is a valid new best, improving accepted accuracy and downside-risk denominator while staying above the coverage floor.
- Next step: keep this as the new benchmark; because coverage is close to `0.40`, prefer changes that recover coverage without sacrificing accepted accuracy.

## 20260509_codex_iter149_blend985_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: a slightly larger DART LightGBM perturbation (`1.5%`) may recover coverage/utility while keeping accepted accuracy close to the DART 99/1 blend.
- Changed files: `experiments/configs/20260509_codex_iter149_blend985_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.985`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter149_blend985_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter149_blend985_dart_platt_logit --config experiments/configs/20260509_codex_iter149_blend985_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter149_blend985_dart_platt_logit/metrics.json`.
- Score before: `0.18680416480395254`.
- Score after: `0.18682834549642668`.
- Utility before / after: `0.07542768273716952` / `0.0755572835666148`.
- Accepted accuracy before / after: `0.5939315687540349` / `0.5938204055358867`.
- Accepted count before / after: `3098` / `3107`.
- Coverage before / after: `0.4015033696215656` / `0.40266977708657337`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: DART `0.985` is a tiny valid new best, improving utility and coverage enough to offset the small accepted-accuracy decrease.
- Next step: test the adjacent larger DART perturbation once.

## 20260509_codex_iter150_blend98_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: increasing the DART component to `2%` may further improve coverage/utility while staying close enough to CatBoost to preserve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter150_blend98_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.98`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter150_blend98_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter150_blend98_dart_platt_logit --config experiments/configs/20260509_codex_iter150_blend98_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter150_blend98_dart_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.1854819008505387`.
- Utility before / after: `0.0755572835666148` / `0.07542768273716954`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.5928525845564774`.
- Accepted count before / after: `3107` / `3134`.
- Coverage before / after: `0.40266977708657337` / `0.4061689994815967`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `2%` DART recovers coverage but loses too much accepted accuracy. Keep `catboost_weight: 0.985`.
- Next step: stop this coarse DART weight sweep or test only a fine midpoint around `0.985`.

## 20260509_codex_iter151_blend9875_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: a midpoint DART blend weight (`catboost_weight: 0.9875`) may improve the tradeoff between the 0.985 and 0.99 runs.
- Changed files: `experiments/configs/20260509_codex_iter151_blend9875_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.9875`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter151_blend9875_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter151_blend9875_dart_platt_logit --config experiments/configs/20260509_codex_iter151_blend9875_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter151_blend9875_dart_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.1865818229301139`.
- Utility before / after: `0.0755572835666148` / `0.07542768273716952`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.59375`.
- Accepted count before / after: `3107` / `3104`.
- Coverage before / after: `0.40266977708657337` / `0.4022809745982374`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: the midpoint is slightly worse; keep `catboost_weight: 0.985`.
- Next step: stop DART blend-weight tuning and test a different lever.

## 20260509_codex_iter152_blend985_dart_platt_logit_c030

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: the best DART blend may benefit from slightly weaker logit-calibrator regularization (`C: 0.3`) to recover coverage/utility.
- Changed files: `experiments/configs/20260509_codex_iter152_blend985_dart_platt_logit_c030.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.985`, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.3`.
- Config: `experiments/configs/20260509_codex_iter152_blend985_dart_platt_logit_c030.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter152_blend985_dart_platt_logit_c030 --config experiments/configs/20260509_codex_iter152_blend985_dart_platt_logit_c030.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter152_blend985_dart_platt_logit_c030/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.18545411201649561`.
- Utility before / after: `0.0755572835666148` / `0.07529808190772425`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.59296`.
- Accepted count before / after: `3107` / `3125`.
- Coverage before / after: `0.40266977708657337` / `0.4050025920165889`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `C: 0.3` adds coverage but loses accepted accuracy and score. Keep `C: 0.25`.
- Next step: avoid weakening calibration further on this branch.

## 20260509_codex_iter153_blend985_dart_aggressive_platt_logit

- Skill used: `tabular-lgbm-dart-boosting` plus `tabular-logit-transform-stacking`.
- Hypothesis: a more aggressively subsampled DART component may add a more diverse 1.5% ranking perturbation while CatBoost remains dominant.
- Changed files: `experiments/configs/20260509_codex_iter153_blend985_dart_aggressive_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.985`, DART LightGBM with `subsample: 0.5`, `colsample_bytree: 0.2`, `min_child_samples: 80`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter153_blend985_dart_aggressive_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter153_blend985_dart_aggressive_platt_logit --config experiments/configs/20260509_codex_iter153_blend985_dart_aggressive_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter153_blend985_dart_aggressive_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.18500627807829803`.
- Utility before / after: `0.0755572835666148` / `0.0749092794193883`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.592985842985843`.
- Accepted count before / after: `3107` / `3108`.
- Coverage before / after: `0.40266977708657337` / `0.40279937791601866`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: more aggressive DART subsampling lowers accepted accuracy and score. Keep the original DART component from iteration 149.
- Next step: avoid further DART component widening unless guided by new diagnostics.

## 20260509_codex_iter154_ge5bp_dart_blend_platt_logit

- Skill used: data-weighting/filtering guidance from `tabular-balanced-log-loss`, combined with the current best DART blend.
- Hypothesis: because validation diagnostics score much better on realized `abs_return >= 5bp`, training only on higher-magnitude development examples may improve selective confidence when paired with the DART blend.
- Changed files: `experiments/configs/20260509_codex_iter154_ge5bp_dart_blend_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split`.
- Data processing: reused existing split where development rows with `abs(abs_return) < 0.0005` were removed; validation unchanged and `abs_return` is not a feature.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend (`catboost_weight: 0.985`) plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter154_ge5bp_dart_blend_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter154_ge5bp_dart_blend_platt_logit --config experiments/configs/20260509_codex_iter154_ge5bp_dart_blend_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter154_ge5bp_dart_blend_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.17694788166734995`.
- Utility before / after: `0.0755572835666148` / `0.07296526697770866`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.5883275807969878`.
- Accepted count before / after: `3107` / `3187`.
- Coverage before / after: `0.40266977708657337` / `0.413037843442198`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: high-magnitude-only training still hurts accepted accuracy with the DART blend; keep full development data.
- Next step: avoid low-return row filtering on this branch.

## 20260509_codex_iter155_blend985_dart_catseed43_platt_logit

- Skill used: `tabular-multi-seed-fold-averaging` diagnostics applied as a single-seed sensitivity check.
- Hypothesis: the DART blend may make CatBoost seed 43 less harmful than in the CatBoost-only branch, possibly improving utility/coverage without a large accepted-accuracy loss.
- Changed files: `experiments/configs/20260509_codex_iter155_blend985_dart_catseed43_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with nested CatBoost `random_seed: 43`, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter155_blend985_dart_catseed43_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter155_blend985_dart_catseed43_platt_logit --config experiments/configs/20260509_codex_iter155_blend985_dart_catseed43_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter155_blend985_dart_catseed43_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.17604371498514013`.
- Utility before / after: `0.0755572835666148` / `0.08125972006220838`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.5800766283524904`.
- Accepted count before / after: `3107` / `3915`.
- Coverage before / after: `0.40266977708657337` / `0.5073872472783826`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: seed 43 again over-expands coverage and loses too much accepted accuracy. Keep CatBoost seed 42.
- Next step: do not pursue seed 43 in the best blend branch.

## 20260509_codex_iter156_blend9825_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: a fine lower-side DART blend weight (`catboost_weight: 0.9825`) may improve utility and accepted count over `0.985` while preserving accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter156_blend9825_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.9825`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter156_blend9825_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter156_blend9825_dart_platt_logit --config experiments/configs/20260509_codex_iter156_blend9825_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter156_blend9825_dart_platt_logit/metrics.json`.
- Score before: `0.18682834549642668`.
- Score after: `0.18741881994931217`.
- Utility before / after: `0.0755572835666148` / `0.07594608605495075`.
- Accepted accuracy before / after: `0.5938204055358867` / `0.5939102564102564`.
- Accepted count before / after: `3107` / `3120`.
- Coverage before / after: `0.40266977708657337` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.9825` is a valid new best, improving utility, coverage, accepted count, and accepted accuracy together.
- Next step: continue bracketing the DART blend optimum with a nearby lower weight.

## 20260509_codex_iter157_blend98125_dart_platt_logit

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: a lower fine DART blend weight (`catboost_weight: 0.98125`) may continue the improvement from `0.9825`.
- Changed files: `experiments/configs/20260509_codex_iter157_blend98125_dart_platt_logit.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.98125`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter157_blend98125_dart_platt_logit.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter157_blend98125_dart_platt_logit --config experiments/configs/20260509_codex_iter157_blend98125_dart_platt_logit.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter157_blend98125_dart_platt_logit/metrics.json`.
- Score before: `0.18741881994931217`.
- Score after: `0.18601940180283025`.
- Utility before / after: `0.07594608605495075` / `0.0755572835666148`.
- Accepted accuracy before / after: `0.5939102564102564` / `0.5931607542345797`.
- Accepted count before / after: `3120` / `3129`.
- Coverage before / after: `0.40435458786936235` / `0.40552099533437014`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: lower than `0.9825` loses accepted accuracy and score; keep `catboost_weight: 0.9825`.
- Next step: DART blend weight is locally bracketed; move to a different lever.

## 20260509_codex_iter158_blend9825_dart_platt_logit_c030

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: the best `0.9825` DART blend may benefit from slightly weaker logit-calibrator regularization (`C: 0.3`) to recover coverage without much accepted-accuracy loss.
- Changed files: `experiments/configs/20260509_codex_iter158_blend9825_dart_platt_logit_c030.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.9825`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.3`.
- Config: `experiments/configs/20260509_codex_iter158_blend9825_dart_platt_logit_c030.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter158_blend9825_dart_platt_logit_c030 --config experiments/configs/20260509_codex_iter158_blend9825_dart_platt_logit_c030.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter158_blend9825_dart_platt_logit_c030/metrics.json`.
- Score before: `0.18741881994931217`.
- Score after: `0.1854092625470479`.
- Utility before / after: `0.07594608605495075` / `0.07542768273716952`.
- Accepted accuracy before / after: `0.5939102564102564` / `0.5927933673469388`.
- Accepted count before / after: `3120` / `3136`.
- Coverage before / after: `0.40435458786936235` / `0.4064282011404873`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: weaker calibration increases coverage but loses accepted accuracy and score. Keep `C: 0.25`.
- Next step: optionally test the stronger calibration side once, then move to a different lever.

## 20260509_codex_iter159_blend9825_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: stronger logit-calibrator regularization (`C: 0.2`) may improve accepted accuracy on the best DART blend even if coverage tightens slightly.
- Changed files: `experiments/configs/20260509_codex_iter159_blend9825_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.9825`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter159_blend9825_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter159_blend9825_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter159_blend9825_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter159_blend9825_dart_platt_logit_c020/metrics.json`.
- Score before: `0.18741881994931217`.
- Score after: `0.1876924881310137`.
- Utility before / after: `0.07594608605495075` / `0.07581648522550545`.
- Accepted accuracy before / after: `0.5939102564102564` / `0.5942636158556236`.
- Accepted count before / after: `3120` / `3103`.
- Coverage before / after: `0.40435458786936235` / `0.4021513737687921`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `C: 0.2` is a valid new best by improving accepted accuracy enough to offset lower coverage and utility.
- Next step: keep this as the benchmark; coverage is close to the floor, so future changes should avoid reducing coverage further.

## 20260509_codex_iter160_blend98_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: with stronger logit calibration (`C: 0.2`), the `catboost_weight: 0.98` DART blend may regain enough accepted accuracy while recovering coverage/utility.
- Changed files: `experiments/configs/20260509_codex_iter160_blend98_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.98`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter160_blend98_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter160_blend98_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter160_blend98_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter160_blend98_dart_platt_logit_c020/metrics.json`.
- Score before: `0.1876924881310137`.
- Score after: `0.18778972634803784`.
- Utility before / after: `0.07581648522550545` / `0.07594608605495073`.
- Accepted accuracy before / after: `0.5942636158556236` / `0.5942122186495177`.
- Accepted count before / after: `3103` / `3110`.
- Coverage before / after: `0.4021513737687921` / `0.40305857957490926`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `catboost_weight: 0.98` with `C: 0.2` is a tiny new best, improving utility and accepted count while preserving accepted accuracy.
- Next step: test only one nearby lower DART weight to see where the accuracy starts to break.

## 20260509_codex_iter161_blend975_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: increasing the DART component to `2.5%` may recover more utility/coverage while the stronger calibrator keeps accepted accuracy high.
- Changed files: `experiments/configs/20260509_codex_iter161_blend975_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.975`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter161_blend975_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter161_blend975_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter161_blend975_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter161_blend975_dart_platt_logit_c020/metrics.json`.
- Score before: `0.18778972634803784`.
- Score after: `0.18761476545465552`.
- Utility before / after: `0.07594608605495073` / `0.07620528771384133`.
- Accepted accuracy before / after: `0.5942122186495177` / `0.5938098276962348`.
- Accepted count before / after: `3110` / `3134`.
- Coverage before / after: `0.40305857957490926` / `0.4061689994815967`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.975` improves utility/coverage but loses enough accepted accuracy to trail `0.98`. Keep `catboost_weight: 0.98`.
- Next step: bracket with a midpoint if continuing blend tuning.

## 20260509_codex_iter162_blend9775_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: the midpoint between `0.98` and `0.975` may capture the extra DART utility while preserving enough accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter162_blend9775_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.9775`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter162_blend9775_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter162_blend9775_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter162_blend9775_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter162_blend9775_dart_platt_logit_c020/metrics.json`.
- Score before: `0.18778972634803784`.
- Score after: `0.1884526862901693`.
- Utility before / after: `0.07594608605495073` / `0.07633488854328671`.
- Accepted accuracy before / after: `0.5942122186495177` / `0.594360781800705`.
- Accepted count before / after: `3110` / `3121`.
- Coverage before / after: `0.40305857957490926` / `0.40448418869880765`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.9775` is a valid new best, improving utility, coverage, accepted count, and accepted accuracy together.
- Next step: continue fine bracketing around this DART blend weight.

## 20260509_codex_iter163_blend97625_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: increasing the DART component slightly beyond iteration 162 may continue to improve utility without too much accepted-accuracy loss.
- Changed files: `experiments/configs/20260509_codex_iter163_blend97625_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.97625`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter163_blend97625_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter163_blend97625_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter163_blend97625_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter163_blend97625_dart_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18776231936299612`.
- Utility before / after: `0.07633488854328671` / `0.07620528771384134`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5939297124600639`.
- Accepted count before / after: `3121` / `3130`.
- Coverage before / after: `0.40448418869880765` / `0.40565059616381544`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.97625` loses accepted accuracy and score. Keep `catboost_weight: 0.9775`.
- Next step: local blend weight is bracketed; test a different lever.

## 20260509_codex_iter164_blend97875_dart_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` and `tabular-lgbm-dart-boosting`.
- Hypothesis: the upper midpoint between `0.9775` and `0.98` may improve the bracketed DART blend optimum.
- Changed files: `experiments/configs/20260509_codex_iter164_blend97875_dart_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_logit_blend`, `catboost_weight: 0.97875`, current best CatBoost settings, DART LightGBM component, plus `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter164_blend97875_dart_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter164_blend97875_dart_platt_logit_c020 --config experiments/configs/20260509_codex_iter164_blend97875_dart_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter164_blend97875_dart_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18749282540923462`.
- Utility before / after: `0.07633488854328671` / `0.07594608605495078`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5939704939063503`.
- Accepted count before / after: `3121` / `3118`.
- Coverage before / after: `0.40448418869880765` / `0.40409538621047175`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: `0.97875` is worse; keep `catboost_weight: 0.9775`.
- Next step: stop this local weight sweep.

## 20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: a slightly less aggressively subsampled DART component (`colsample_bytree: 0.45`) may provide a more stable perturbation than `0.35`.
- Changed files: `experiments/configs/20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `colsample_bytree: 0.45`.
- Config: `experiments/configs/20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020 --config experiments/configs/20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter165_blend9775_dart_col045_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18545411201649561`.
- Utility before / after: `0.07633488854328671` / `0.07529808190772425`.
- Accepted accuracy before / after: `0.594360781800705` / `0.59296`.
- Accepted count before / after: `3121` / `3125`.
- Coverage before / after: `0.40448418869880765` / `0.4050025920165889`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: larger DART column sampling hurts accepted accuracy. Keep `colsample_bytree: 0.35`.
- Next step: avoid widening DART column sampling.

## 20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: slightly more aggressive DART column subsampling (`colsample_bytree: 0.30`) may diversify the LightGBM perturbation without the over-regularization seen at `0.20`.
- Changed files: `experiments/configs/20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `colsample_bytree: 0.30`.
- Config: `experiments/configs/20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020 --config experiments/configs/20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter166_blend9775_dart_col030_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18424975896674967`.
- Utility before / after: `0.07633488854328671` / `0.07477967858994297`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5924975953831356`.
- Accepted count before / after: `3121` / `3119`.
- Coverage before / after: `0.40448418869880765` / `0.40422498703991705`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: lower DART column sampling hurts accepted accuracy. Keep `colsample_bytree: 0.35`.
- Next step: avoid DART column-sampling changes.

## 20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: increasing DART `min_child_samples` from `120` to `200` may smooth the LightGBM perturbation and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `min_child_samples: 200`.
- Config: `experiments/configs/20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020 --config experiments/configs/20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter167_blend9775_dart_minchild200_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18840901158061896`.
- Utility before / after: `0.07633488854328671` / `0.0760756868843961`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5945858846277796`.
- Accepted count before / after: `3121` / `3103`.
- Coverage before / after: `0.40448418869880765` / `0.4021513737687921`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: higher `min_child_samples` improves accepted accuracy but loses enough utility/coverage to narrowly trail the best.
- Next step: keep `min_child_samples: 120` unless combining with a coverage-recovering change.

## 20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: midpoint DART `min_child_samples: 160` may retain some smoothing benefit from `200` with less coverage loss.
- Changed files: `experiments/configs/20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `min_child_samples: 160`.
- Config: `experiments/configs/20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020 --config experiments/configs/20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter168_blend9775_dart_minchild160_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18739503550649042`.
- Utility before / after: `0.07633488854328671` / `0.07581648522550548`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5940212150433944`.
- Accepted count before / after: `3121` / `3111`.
- Coverage before / after: `0.40448418869880765` / `0.4031881804043546`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: midpoint smoothing is worse than the original `min_child_samples: 120`.
- Next step: stop DART min-child tuning.

## 20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: reducing DART `num_leaves` from `31` to `20` may make the small LightGBM component a cleaner low-variance perturbation.
- Changed files: `experiments/configs/20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `num_leaves: 20`.
- Config: `experiments/configs/20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020 --config experiments/configs/20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter169_blend9775_dart_leaves20_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.187199552589016`.
- Utility before / after: `0.07633488854328671` / `0.0755572835666148`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5941233451727478`.
- Accepted count before / after: `3121` / `3097`.
- Coverage before / after: `0.40448418869880765` / `0.4013737687921203`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: smaller DART trees reduce coverage/utility and do not improve score. Keep `num_leaves: 31`.
- Next step: avoid smaller DART capacity.

## 20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: modestly increasing DART `num_leaves` from `31` to `40` may improve the small LightGBM perturbation's ranking signal.
- Changed files: `experiments/configs/20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `num_leaves: 40`.
- Config: `experiments/configs/20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020 --config experiments/configs/20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter170_blend9775_dart_leaves40_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18665754726356037`.
- Utility before / after: `0.07633488854328671` / `0.07581648522550542`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5934206323858192`.
- Accepted count before / after: `3121` / `3131`.
- Coverage before / after: `0.40448418869880765` / `0.40578019699326073`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: higher DART leaf capacity reduces accepted accuracy and score. Keep `num_leaves: 31`.
- Next step: stop DART leaf-count tuning.

## 20260509_codex_iter171_train90_dart_blend_platt_logit_c020

- Skill used: recency/window discipline from prior adversarial-validation findings plus the current DART blend stack.
- Hypothesis: the 90-day VWAP-drop split may recover coverage/utility with the improved DART blend and stronger calibration, despite underperforming with earlier CatBoost-only models.
- Changed files: `experiments/configs/20260509_codex_iter171_train90_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_split`.
- Feature set: VWAP-drop top-500 split with 90-day development window; HTF/time features retained.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter171_train90_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter45_train90_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter171_train90_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter171_train90_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 90 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter171_train90_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.17288825028814156`.
- Utility before / after: `0.07633488854328671` / `0.08138932089165375`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5775691699604744`.
- Accepted count before / after: `3121` / `4048`.
- Coverage before / after: `0.40448418869880765` / `0.5246241575946086`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: 90-day training over-expands coverage and loses too much accepted accuracy. Keep the 75-day split.
- Next step: avoid longer-window variants unless paired with a precision-preserving change.

## 20260509_codex_iter172_train70_dart_blend_platt_logit_c020

- Skill used: recency/window discipline from prior adversarial-validation findings plus the current DART blend stack.
- Hypothesis: a shorter 70-day VWAP-drop split may increase accepted accuracy with the improved DART blend while preserving minimum coverage.
- Changed files: `experiments/configs/20260509_codex_iter172_train70_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split`.
- Feature set: VWAP-drop top-500 split with 70-day development window; HTF/time features retained.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter172_train70_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter172_train70_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter172_train70_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 70 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter172_train70_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.16771686161332752`.
- Utility before / after: `0.07633488854328671` / `0.06894764126490407`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5847133757961783`.
- Accepted count before / after: `3121` / `3140`.
- Coverage before / after: `0.40448418869880765` / `0.4069466044582685`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: 70-day training loses too much accepted accuracy. Keep the 75-day split.
- Next step: stop cached train-window retests around this branch.

## 20260509_codex_iter173_top700_dart_blend_platt_logit_c020

- Skill used: feature-width ablation discipline plus the current DART blend stack.
- Hypothesis: a wider top-700 cached feature set may give the small DART component more weak signals to perturb CatBoost rankings.
- Changed files: `experiments/configs/20260509_codex_iter173_top700_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter26_top700_split`.
- Feature set: top-700 split; HTF/time features retained by the original feature-selection workflow.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter173_top700_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter26_top700_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter173_top700_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter173_top700_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 183 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter173_top700_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.1537062721170869`.
- Utility before / after: `0.07633488854328671` / `0.07503888024883362`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5680056377730797`.
- Accepted count before / after: `3121` / `4257`.
- Coverage before / after: `0.40448418869880765` / `0.5517107309486781`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: the wider/full-history top-700 split badly lowers accepted accuracy. Keep the 75-day VWAP-drop split.
- Next step: avoid wider/full-history cached feature sets for this branch.

## 20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020

- Skill used: feature ablation discipline with the current DART blend stack.
- Hypothesis: dropping only `sl_vwap_30s` may retain more useful microstructure signal than dropping both VWAP features while still removing the noisier long VWAP feature.
- Changed files: `experiments/configs/20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split`.
- Feature set: 75-day split dropping only `sl_vwap_30s`; HTF/time features retained.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter174_drop_vwap30_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.17873368114876534`.
- Utility before / after: `0.07633488854328671` / `0.07426127527216175`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5885078776645042`.
- Accepted count before / after: `3121` / `3237`.
- Coverage before / after: `0.40448418869880765` / `0.41951788491446346`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: keeping `sl_vwap_10s` hurts accepted accuracy. The current drop-both VWAP split remains better.
- Next step: test the complementary `sl_vwap_10s`-drop split only if needed.

## 20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020

- Skill used: feature ablation discipline with the current DART blend stack.
- Hypothesis: dropping only `sl_vwap_10s` may retain useful longer VWAP context while removing short-horizon VWAP noise.
- Changed files: `experiments/configs/20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split`.
- Feature set: 75-day split dropping only `sl_vwap_10s`; HTF/time features retained.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter175_drop_vwap10_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.16691648826649505`.
- Utility before / after: `0.07633488854328671` / `0.07205806117159151`.
- Accepted accuracy before / after: `0.594360781800705` / `0.581002331002331`.
- Accepted count before / after: `3121` / `3432`.
- Coverage before / after: `0.40448418869880765` / `0.4447900466562986`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: keeping `sl_vwap_30s` is even worse; the current drop-both VWAP split remains best.
- Next step: stop VWAP-drop split variants.

## 20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020

- Skill used: feature ablation discipline with the current DART blend stack.
- Hypothesis: the cached low-volume VWAP-drop variant may remove VWAP noise more selectively than the current drop-both split when paired with the DART blend.
- Changed files: `experiments/configs/20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_split`.
- Feature set: 75-day low-volume VWAP-drop split; HTF/time features retained.
- Model settings: best DART blend stack with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter48_train75_drop_vwap_lowvol_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020 --config experiments/configs/20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter176_drop_vwap_lowvol_dart_blend_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.15776631448205672`.
- Utility before / after: `0.07633488854328671` / `0.06674442716433385`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5785779676533415`.
- Accepted count before / after: `3121` / `3277`.
- Coverage before / after: `0.40448418869880765` / `0.4247019180922758`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: low-volume VWAP-drop variant is much worse. Keep the current drop-both VWAP split.
- Next step: avoid further VWAP-drop variants.

## 20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: a lower-learning-rate, longer DART component may provide a smoother perturbation than `learning_rate: 0.01`, `n_estimators: 1600`.
- Changed files: `experiments/configs/20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `learning_rate: 0.005`, `n_estimators: 3200`.
- Config: `experiments/configs/20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020 --config experiments/configs/20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter177_blend9775_dart_lr005_iter3200_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.1747146637071683`.
- Utility before / after: `0.07633488854328671` / `0.07102125453602899`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5884441575209812`.
- Accepted count before / after: `3121` / `3098`.
- Coverage before / after: `0.40448418869880765` / `0.4015033696215656`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: lower-rate longer DART materially hurts accepted accuracy and score. Keep `learning_rate: 0.01`, `n_estimators: 1600`.
- Next step: test a shorter/faster DART component only if useful; otherwise shift levers.

## 20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: a higher-learning-rate, shorter DART component may improve the perturbation's coverage/utility while remaining small in the blend.
- Changed files: `experiments/configs/20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `learning_rate: 0.02`, `n_estimators: 800`.
- Config: `experiments/configs/20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020 --config experiments/configs/20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter178_blend9775_dart_lr02_iter800_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18345528573957598`.
- Utility before / after: `0.07633488854328671` / `0.07659409020217733`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5900640048765621`.
- Accepted count before / after: `3121` / `3281`.
- Coverage before / after: `0.40448418869880765` / `0.425220321410057`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: $h.
- Interpretation: higher-rate shorter DART over-expands coverage and loses accepted accuracy. Keep `learning_rate: 0.01`, `n_estimators: 1600`.
- Next step: stop DART learning-rate bracket.

## 20260509_codex_iter179_blend9775_dart_platt_raw_c020

- Skill used: `tabular-logit-transform-stacking` as a contrast against raw-probability Platt scaling.
- Hypothesis: the current DART blend may prefer raw-probability Platt calibration over logit-space Platt.
- Changed files: `experiments/configs/20260509_codex_iter179_blend9775_dart_platt_raw_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter179_blend9775_dart_platt_raw_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter179_blend9775_dart_platt_raw_c020 --config experiments/configs/20260509_codex_iter179_blend9775_dart_platt_raw_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter179_blend9775_dart_platt_raw_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.17940840488120582`.
- Utility before / after: `0.07633488854328671` / `0.07426127527216175`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5890581286913273`.
- Accepted count before / after: `3121` / `3217`.
- Coverage before / after: `0.40448418869880765` / `0.4169258683255573`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `a93dd71`.
- Interpretation: raw-probability Platt loses accepted accuracy. Keep `platt_logit`.
- Next step: avoid raw Platt on this branch.

## 20260509_codex_iter180_blend9775_dart_platt_logit_c015

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: stronger logit-space Platt regularization may sharpen accepted predictions near the 40% coverage boundary.
- Changed files: `experiments/configs/20260509_codex_iter180_blend9775_dart_platt_logit_c015.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.15`.
- Config: `experiments/configs/20260509_codex_iter180_blend9775_dart_platt_logit_c015.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter180_blend9775_dart_platt_logit_c015 --config experiments/configs/20260509_codex_iter180_blend9775_dart_platt_logit_c015.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter180_blend9775_dart_platt_logit_c015/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18616222609328612`.
- Utility before / after: `0.07633488854328671` / `0.07516848107827892`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5936692506459949`.
- Accepted count before / after: `3121` / `3096`.
- Coverage before / after: `0.40448418869880765` / `0.401244167962675`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `213e88e`.
- Interpretation: stronger Platt regularization loses both coverage and score. Keep `C: 0.2`.
- Next step: tune the DART side model regularization instead of lowering calibration `C`.

## 20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: slightly stronger L2 regularization on the nested DART component may reduce overfit while preserving the CatBoost-dominant blend.
- Changed files: `experiments/configs/20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `reg_lambda: 10.0`.
- Config: `experiments/configs/20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020 --config experiments/configs/20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter181_blend9775_dart_l2_10_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18756691860533328`.
- Utility before / after: `0.07633488854328671` / `0.07594608605495076`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5940308087291399`.
- Accepted count before / after: `3121` / `3116`.
- Coverage before / after: `0.40448418869880765` / `0.40383618455158116`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `6d8eda3`.
- Interpretation: `reg_lambda: 10.0` is slightly worse than `8.0`.
- Next step: test lighter DART L2 regularization.

## 20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: lighter L2 regularization on the nested DART component may recover useful variance while the CatBoost-dominant blend limits overfit.
- Changed files: `experiments/configs/20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: best DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, and nested DART `reg_lambda: 6.0`.
- Config: `experiments/configs/20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020 --config experiments/configs/20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter182_blend9775_dart_l2_6_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.1864850246734521`.
- Utility before / after: `0.07633488854328671` / `0.07568688439606017`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5934101087651952`.
- Accepted count before / after: `3121` / `3126`.
- Coverage before / after: `0.40448418869880765` / `0.4051321928460342`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `4b62f91`.
- Interpretation: lighter L2 raises coverage but loses accepted accuracy and score. Keep `reg_lambda: 8.0`.
- Next step: bracket DART L1 regularization.

## 20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: slightly stronger L1 regularization on the nested DART component may prune noisy side-model splits and improve accepted accuracy at valid coverage.
- Changed files: `experiments/configs/20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`, `reg_lambda: 8.0`.
- Config: `experiments/configs/20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter183_blend9775_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1884526862901693`.
- Score after: `0.18899670247663125`.
- Utility before / after: `0.07633488854328671` / `0.07646448937273198`.
- Accepted accuracy before / after: `0.594360781800705` / `0.5946726572528883`.
- Accepted count before / after: `3121` / `3116`.
- Coverage before / after: `0.40448418869880765` / `0.40383618455158116`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `ff0fce2`.
- Interpretation: stronger DART L1 gives a small valid improvement and becomes the new reference branch.
- Next step: bracket DART L1 around `1.2`.

## 20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: increasing nested DART L1 beyond `1.2` may further suppress noisy side-model splits and lift accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.5`, `reg_lambda: 8.0`.
- Config: `experiments/configs/20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020 --config experiments/configs/20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter184_blend9775_dart_l1_15_platt_logit_c020/metrics.json`.
- Score before: `0.18899670247663125`.
- Score after: `0.18781277903621266`.
- Utility before / after: `0.07646448937273198` / `0.07607568688439602`.
- Accepted accuracy before / after: `0.5946726572528883` / `0.594100673292722`.
- Accepted count before / after: `3116` / `3119`.
- Coverage before / after: `0.40383618455158116` / `0.40422498703991705`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `41679b6`.
- Interpretation: `reg_alpha: 1.5` is too strong and loses accuracy. Keep `1.2` as the reference.
- Next step: test a lower bracket at `reg_alpha: 1.0`.

## 20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: a lower nested DART L1 value than `1.2` may retain more useful side-model signal while still pruning more than the old `0.8`.
- Changed files: `experiments/configs/20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.0`, `reg_lambda: 8.0`.
- Config: `experiments/configs/20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020 --config experiments/configs/20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter185_blend9775_dart_l1_10_platt_logit_c020/metrics.json`.
- Score before: `0.18899670247663125`.
- Score after: `0.18717287178225642`.
- Utility before / after: `0.07646448937273198` / `0.07581648522550548`.
- Accepted accuracy before / after: `0.5946726572528883` / `0.5938402309913379`.
- Accepted count before / after: `3116` / `3117`.
- Coverage before / after: `0.40383618455158116` / `0.40396578538102645`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `3b773d4`.
- Interpretation: `reg_alpha: 1.0` is worse than `1.2`. Keep `1.2`.
- Next step: tune DART bagging around the `reg_alpha: 1.2` branch.

## 20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: a slightly higher nested DART row subsample may reduce training noise in the side model while preserving the CatBoost-dominant blend.
- Changed files: `experiments/configs/20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`, `subsample: 0.65`.
- Config: `experiments/configs/20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020 --config experiments/configs/20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter186_blend9775_dart_l1_12_sub65_platt_logit_c020/metrics.json`.
- Score before: `0.18899670247663125`.
- Score after: `0.1841049654206401`.
- Utility before / after: `0.07646448937273198` / `0.07477967858994303`.
- Accepted accuracy before / after: `0.5946726572528883` / `0.5923791226384887`.
- Accepted count before / after: `3116` / `3123`.
- Coverage before / after: `0.40383618455158116` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `6990704`.
- Interpretation: higher DART subsampling hurts accepted accuracy. Keep `subsample: 0.6`.
- Next step: test lower DART subsampling once.

## 20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: slightly lower nested DART row subsampling may regularize the side model and improve accepted accuracy after the `reg_alpha: 1.2` change.
- Changed files: `experiments/configs/20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9775`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`, `subsample: 0.55`.
- Config: `experiments/configs/20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020 --config experiments/configs/20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter187_blend9775_dart_l1_12_sub55_platt_logit_c020/metrics.json`.
- Score before: `0.18899670247663125`.
- Score after: `0.186165701001066`.
- Utility before / after: `0.07646448937273198` / `0.07555728356661484`.
- Accepted accuracy before / after: `0.5946726572528883` / `0.59328`.
- Accepted count before / after: `3116` / `3125`.
- Coverage before / after: `0.40383618455158116` / `0.4050025920165889`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f8830b3`.
- Interpretation: lower DART subsampling also hurts. Keep `subsample: 0.6`.
- Next step: retune CatBoost/DART blend weight around the `reg_alpha: 1.2` branch.

## 20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: after improving the DART side model with `reg_alpha: 1.2`, a slightly larger DART contribution may improve utility while preserving accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9765`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`.
- Config: `experiments/configs/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.18899670247663125`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07646448937273198` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946726572528883` / `0.5946205571565802`.
- Accepted count before / after: `3116` / `3123`.
- Coverage before / after: `0.40383618455158116` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `294c8aa`.
- Interpretation: a small increase in DART contribution improves utility enough to set a new best valid score.
- Next step: test the opposite blend-weight side.

## 20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: reducing the DART contribution after the `reg_alpha: 1.2` side-model improvement may lift accepted accuracy enough to offset lower coverage.
- Changed files: `experiments/configs/20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9785`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`.
- Config: `experiments/configs/20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter189_blend9785_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890715674835194`.
- Utility before / after: `0.07659409020217732` / `0.076464489372732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5947334617854849`.
- Accepted count before / after: `3123` / `3114`.
- Coverage before / after: `0.4047433903576983` / `0.4035769828926905`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `df2011d`.
- Interpretation: higher CatBoost weight is near-tied but loses utility and score. Keep `catboost_weight: 0.9765`.
- Next step: fine-tune the lower side of blend weight around `0.9765`.

## 20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: a slightly larger DART contribution than `0.9765` may improve utility after the DART L1 tuning.
- Changed files: `experiments/configs/20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9755`, `calibration.active_plugin: platt_logit`, `C: 0.2`, nested DART `reg_alpha: 1.2`.
- Config: `experiments/configs/20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter190_blend9755_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18894340804275994`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217726`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5944995203070035`.
- Accepted count before / after: `3123` / `3127`.
- Coverage before / after: `0.4047433903576983` / `0.40526179367547954`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `04ab7dc`.
- Interpretation: more DART weight keeps utility flat but lowers score through higher downside risk. Keep `catboost_weight: 0.9765`.
- Next step: tune calibration locally around the new best branch.

## 20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: slightly stronger logit-space Platt regularization than `C: 0.2` may improve accepted accuracy on the new blend branch while maintaining coverage.
- Changed files: `experiments/configs/20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.18`.
- Config: `experiments/configs/20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018 --config experiments/configs/20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter191_blend9765_dart_l1_12_platt_logit_c018/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.188505346877847`.
- Utility before / after: `0.07659409020217732` / `0.07620528771384134`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5945337620578778`.
- Accepted count before / after: `3123` / `3110`.
- Coverage before / after: `0.4047433903576983` / `0.40305857957490926`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `e7a645d`.
- Interpretation: lower calibration `C` loses coverage and score. Keep `C: 0.2` unless the upper side improves.
- Next step: test `C: 0.22`.

## 20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: slightly weaker logit-space Platt regularization than `C: 0.2` may improve score on the new blend branch.
- Changed files: `experiments/configs/20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.22`.
- Config: `experiments/configs/20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022 --config experiments/configs/20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter192_blend9765_dart_l1_12_platt_logit_c022/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18611929569882232`.
- Utility before / after: `0.07659409020217732` / `0.07568688439606015`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5931122448979592`.
- Accepted count before / after: `3123` / `3136`.
- Coverage before / after: `0.4047433903576983` / `0.4064282011404873`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `fccba95`.
- Interpretation: higher calibration `C` expands coverage but loses accepted accuracy and score. Keep `C: 0.2`.
- Next step: leave calibration and test another model-training lever.

## 20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-collinear-feature-removal`.
- Hypothesis: dropping only highly redundant features with absolute Pearson correlation above `0.98`, choosing the feature with higher development target correlation to keep, may reduce noise without changing label or feature semantics.
- Changed files: `experiments/configs/20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_split`; drop report: `artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_split/collinear_drop_report.json`.
- Feature set: current best VWAP-pruned split minus 20 collinear features; HTF/time packs retained, but some redundant HTF columns were dropped by the data-processing rule.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16972853470903534`.
- Utility before / after: `0.07659409020217732` / `0.06985484707102121`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5854741516016492`.
- Accepted count before / after: `3123` / `3153`.
- Coverage before / after: `0.4047433903576983` / `0.40863141524105756`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `ecdf16c`.
- Interpretation: correlation pruning damages accepted accuracy, likely because redundant HTF/flow variants still help tree splits and calibration. Do not use this collinear-pruned split.
- Next step: return to the current best full split and try model-training regularization.

## 20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: a slower CatBoost schedule may smooth the dominant CatBoost probabilities and improve accepted accuracy after calibration.
- Changed files: `experiments/configs/20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, CatBoost `iterations: 1600`, `learning_rate: 0.012`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter194_blend9765_cat_slow_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.17227089732634684`.
- Utility before / after: `0.07659409020217732` / `0.07503888024883357`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5825491873396065`.
- Accepted count before / after: `3123` / `3507`.
- Coverage before / after: `0.4047433903576983` / `0.4545101088646967`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f179ee0`.
- Interpretation: slower CatBoost training greatly expands accepted coverage but damages accepted accuracy and score. Keep the original CatBoost schedule.
- Next step: avoid broad CatBoost schedule changes; use narrower regularization or feature levers.

## 20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: slightly stronger CatBoost L2 regularization may smooth the dominant model without changing the training schedule.
- Changed files: `experiments/configs/20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, CatBoost `l2_leaf_reg: 35.0`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter195_blend9765_cat_l2_35_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18247679489010074`.
- Utility before / after: `0.07659409020217732` / `0.07581648522550546`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.590027700831025`.
- Accepted count before / after: `3123` / `3249`.
- Coverage before / after: `0.4047433903576983` / `0.42107309486780714`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `a355144`.
- Interpretation: stronger CatBoost L2 lowers accepted accuracy and score. Keep `l2_leaf_reg: 30.0` unless the lower side improves.
- Next step: test `l2_leaf_reg: 25.0`.

## 20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: slightly weaker CatBoost L2 regularization may recover useful signal in the dominant model and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, CatBoost `l2_leaf_reg: 25.0`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter196_blend9765_cat_l2_25_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16844822224789988`.
- Utility before / after: `0.07659409020217732` / `0.06868843960601345`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.585594315245478`.
- Accepted count before / after: `3123` / `3096`.
- Coverage before / after: `0.4047433903576983` / `0.401244167962675`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `220e72c`.
- Interpretation: weaker CatBoost L2 materially hurts accepted accuracy. Keep `l2_leaf_reg: 30.0`.
- Next step: leave CatBoost L2 bracket.

## 20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` and `tabular-logit-transform-stacking`.
- Hypothesis: stronger CatBoost split-score randomness may regularize the dominant model and improve validation selection score.
- Changed files: `experiments/configs/20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, CatBoost `random_strength: 3.0`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter197_blend9765_cat_rs3_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16741425107425806`.
- Utility before / after: `0.07659409020217732` / `0.06855883877656821`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5848572345203722`.
- Accepted count before / after: `3123` / `3117`.
- Coverage before / after: `0.4047433903576983` / `0.40396578538102645`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `1818571`.
- Interpretation: stronger CatBoost random strength sharply degrades accepted accuracy. Keep `random_strength: 2.0`.
- Next step: avoid CatBoost stochastic-strength increases.

## 20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020

- Skill used: `tabular-balanced-log-loss` for weighting discipline.
- Hypothesis: a stronger sample-weight ramp may reduce the influence of ambiguous tiny-return labels and improve accepted accuracy without filtering rows.
- Changed files: `experiments/configs/20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`, config sample-weight ramp `min_weight: 0.25`, `full_weight_abs_return: 0.0004`.
- Config: `experiments/configs/20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter198_blend9765_weight_ramp_stronger_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5946205571565802`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `6bbfaf9`.
- Interpretation: config-only sample-weighting changes do not affect cached split runs because the cached `stage1_sample_weight` column is already materialized. Need a derived cached split to test weighting.
- Next step: create a derived cached split with recomputed sample weights.

## 20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-balanced-log-loss` for weighting discipline.
- Hypothesis: materializing a stronger sample-weight ramp in the cached split may reduce low-return label noise and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter199_reweighted_split`; weight report: `artifacts/data_v2/experiments/20260509_codex_iter199_reweighted_split/sample_weight_report.json`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained; only `stage1_sample_weight` was rematerialized.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`, materialized weight ramp `min_weight: 0.25`, `full_weight_abs_return: 0.0004`.
- Config: `experiments/configs/20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter199_reweighted_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter199_reweighted_split_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.17683619665998612`.
- Utility before / after: `0.07659409020217732` / `0.07529808190772426`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5859721811186742`.
- Accepted count before / after: `3123` / `3379`.
- Coverage before / after: `0.4047433903576983` / `0.43792120269569723`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `c66df3e`.
- Interpretation: stronger materialized weighting expands accepted coverage and reduces accepted accuracy. Do not use this ramp.
- Next step: test a gentler materialized weighting ramp.

## 20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-balanced-log-loss` for weighting discipline.
- Hypothesis: a gentler materialized sample-weight ramp than the current cached split may avoid over-penalizing ambiguous samples and improve calibration.
- Changed files: `experiments/configs/20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter200_reweighted_gentle_split`; weight report: `artifacts/data_v2/experiments/20260509_codex_iter200_reweighted_gentle_split/sample_weight_report.json`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained; only `stage1_sample_weight` was rematerialized.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`, materialized weight ramp `min_weight: 0.45`, `full_weight_abs_return: 0.00025`.
- Config: `experiments/configs/20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter200_reweighted_gentle_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter200_reweighted_gentle_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16399769415815763`.
- Utility before / after: `0.07659409020217732` / `0.06752203214100568`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5830411220911699`.
- Accepted count before / after: `3123` / `3137`.
- Coverage before / after: `0.4047433903576983` / `0.40655780196993263`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `2711fa6`.
- Interpretation: gentler materialized weighting is worse than the current cached weights. Keep the original materialized `stage1_sample_weight`.
- Next step: leave sample weighting.

## 20260509_codex_iter201_blend9765_dart_l1_12_isotonic

- Skill used: `tabular-logit-transform-stacking` as a calibration contrast.
- Hypothesis: isotonic calibration may correct non-monotonic validation probability buckets and improve accepted accuracy without changing threshold policy.
- Changed files: `experiments/configs/20260509_codex_iter201_blend9765_dart_l1_12_isotonic.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: isotonic`.
- Config: `experiments/configs/20260509_codex_iter201_blend9765_dart_l1_12_isotonic.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter201_blend9765_dart_l1_12_isotonic --config experiments/configs/20260509_codex_iter201_blend9765_dart_l1_12_isotonic.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter201_blend9765_dart_l1_12_isotonic/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1723177340448186`.
- Utility before / after: `0.07659409020217732` / `0.07698289269051324`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5808383233532934`.
- Accepted count before / after: `3123` / `3674`.
- Coverage before / after: `0.4047433903576983` / `0.4761534473820632`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `a1a7efb`.
- Interpretation: isotonic over-expands accepted coverage and loses accepted accuracy. Keep `platt_logit`.
- Next step: continue with feature/model levers, not isotonic calibration.

## 20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-cyclical-feature-encoding`.
- Hypothesis: adding the existing online-consistent weekday cyclical features from `TimeFeaturePack` to the cached split may improve time-regime discrimination.
- Changed files: `experiments/configs/20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_split`; feature report: `artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_split/weekday_feature_report.json`.
- Feature set: current best VWAP-pruned top-500 split plus `weekday_sin`, `weekday_cos`, `is_weekend`; HTF/time features retained and semantics match the shared online builder.
- Model settings: current best DART blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16469253222951877`.
- Utility before / after: `0.07659409020217732` / `0.06817003628823225`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5829652996845426`.
- Accepted count before / after: `3123` / `3170`.
- Coverage before / after: `0.4047433903576983` / `0.4108346293416278`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `05bfe3f`.
- Interpretation: weekday/weekend features sharply hurt accepted accuracy. Keep the cached hour/minute time features only.
- Next step: avoid weekday feature variants.

## 20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: shallower nested DART trees may reduce side-model overfit while preserving the CatBoost-dominant blend.
- Changed files: `experiments/configs/20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `max_depth: 5`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020 --config experiments/configs/20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter203_blend9765_dart_l1_12_depth5_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18598916066644788`.
- Utility before / after: `0.07659409020217732` / `0.07503888024883361`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5936590100291168`.
- Accepted count before / after: `3123` / `3091`.
- Coverage before / after: `0.4047433903576983` / `0.40059616381544844`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f4e7e40`.
- Interpretation: shallower DART is worse and nearly violates coverage. Keep `max_depth: 6` unless deeper improves.
- Next step: test DART `max_depth: 7`.

## 20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: deeper nested DART trees may capture useful high-order interactions while the small blend weight limits overfit.
- Changed files: `experiments/configs/20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `max_depth: 7`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020 --config experiments/configs/20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter204_blend9765_dart_l1_12_depth7_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18472939602817065`.
- Utility before / after: `0.07659409020217732` / `0.07529808190772419`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5923688394276629`.
- Accepted count before / after: `3123` / `3145`.
- Coverage before / after: `0.4047433903576983` / `0.40759460860549507`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `936cba4`.
- Interpretation: deeper DART loses accepted accuracy and score. Keep `max_depth: 6`.
- Next step: leave DART depth.

## 20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: more frequent DART row resampling may reduce side-model variance and improve the blended validation frontier.
- Changed files: `experiments/configs/20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `subsample_freq: 5`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020 --config experiments/configs/20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter205_blend9765_dart_l1_12_subfreq5_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1820098679099266`.
- Utility before / after: `0.07659409020217732` / `0.07413167444271646`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5913154533844189`.
- Accepted count before / after: `3123` / `3132`.
- Coverage before / after: `0.4047433903576983` / `0.4059097978227061`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `3e20f4c`.
- Interpretation: more frequent DART subsampling hurts accepted accuracy. Keep `subsample_freq: 10`.
- Next step: avoid DART bagging-frequency changes.

## 20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020

- Skill used: `tabular-multi-seed-fold-averaging`.
- Hypothesis: changing only the nested DART random seed may find a better side-model perturbation for the CatBoost-dominant blend.
- Changed files: `experiments/configs/20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, nested DART `random_state: 7`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020 --config experiments/configs/20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter206_blend9765_dart_l1_12_seed7_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1842071111539714`.
- Utility before / after: `0.07659409020217732` / `0.0749092794193883`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.592332268370607`.
- Accepted count before / after: `3123` / `3130`.
- Coverage before / after: `0.4047433903576983` / `0.40565059616381544`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `6ef5902`.
- Interpretation: DART seed 7 is worse. Keep seed 42.
- Next step: do not pursue DART seed 7.

## 20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-rank-averaging-ensemble`.
- Hypothesis: rank-normalizing CatBoost and LightGBM predictions before blending may be more robust than logit blending when model probability scales differ.
- Changed files: `src/model/rank_blend_plugin.py`, `src/model/registry.py`, `experiments/configs/20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: new `catboost_lgbm_rank_blend` plugin with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter207_rank_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18808164303538216`.
- Utility before / after: `0.07659409020217732` / `0.07633488854328666`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.594059405940594`.
- Accepted count before / after: `3123` / `3131`.
- Coverage before / after: `0.4047433903576983` / `0.40578019699326073`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m py_compile src\model\rank_blend_plugin.py src\model\registry.py`; DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f88a4be`.
- Interpretation: rank blending is valid but slightly worse than logit blending. Keep the logit blend as the reference.
- Next step: only pursue rank-blend weights if a narrow bracket shows improvement.

## 20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020

- Skill used: `tabular-rank-averaging-ensemble`.
- Hypothesis: rank blending may need a slightly higher CatBoost weight than logit blending to preserve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `catboost_lgbm_rank_blend` plugin with active `catboost_weight: 0.9785`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter208_rank_blend9785_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18798430444863443`.
- Utility before / after: `0.07659409020217732` / `0.0762052877138414`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5941101152368758`.
- Accepted count before / after: `3123` / `3124`.
- Coverage before / after: `0.4047433903576983` / `0.4048729911871436`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f5db3b6`.
- Interpretation: higher CatBoost weight in rank blending still trails the logit-blend reference. Do not pursue rank blend further now.
- Next step: return to the logit-blend reference.

## 20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-last-diff-lag-features`.
- Hypothesis: online-safe trailing path features over completed bars through `t-1` may help separate larger-move regimes without using future `abs_return`.
- Changed files: `src/features/path_structure.py`, `experiments/configs/20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_split`; feature report: `artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_split/trailing_path_feature_report.json`.
- Feature set: current best VWAP-pruned top-500 split plus `trailing_return_{3,5,10}`, `trailing_abs_return_{3,5,10}`, `trailing_range_pct_{3,5,10}`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16489169305911755`.
- Utility before / after: `0.07659409020217732` / `0.06881804043545878`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5824790307548928`.
- Accepted count before / after: `3123` / `3219`.
- Coverage before / after: `0.4047433903576983` / `0.4171850699844479`.
- Coverage constraint satisfied: yes.
- Tests: `rtk python -m py_compile src\features\path_structure.py`; DQC ran during training; calibration was fit only on development predictions.
- Git commit: `b1583c2`.
- Interpretation: trailing path features materially hurt accepted accuracy. Revert this source feature addition and do not use the derived split.
- Next step: remove the harmful source change before continuing.

## 20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: shallower CatBoost trees may reduce overfit in the dominant model and improve accepted accuracy after calibration.
- Changed files: `experiments/configs/20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `depth: 4`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter210_blend9765_cat_depth4_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16812000816660205`.
- Utility before / after: `0.07659409020217732` / `0.07400207361327113`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5801741083965178`.
- Accepted count before / after: `3123` / `3561`.
- Coverage before / after: `0.4047433903576983` / `0.4615085536547434`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `f7236b9`.
- Interpretation: CatBoost depth 4 over-expands coverage and materially hurts accepted accuracy. Keep `depth: 5` unless depth 6 improves.
- Next step: test CatBoost `depth: 6`.

## 20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: deeper CatBoost trees may capture useful high-order interactions while the DART side model and Platt calibration control overfit.
- Changed files: `experiments/configs/20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `depth: 6`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter211_blend9765_cat_depth6_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1681628949795424`.
- Utility before / after: `0.07659409020217732` / `0.07322446863659927`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.580922371813234`.
- Accepted count before / after: `3123` / `3491`.
- Coverage before / after: `0.4047433903576983` / `0.4524364955935718`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `75f87a9`.
- Interpretation: CatBoost depth 6 also hurts accepted accuracy. Keep `depth: 5`.
- Next step: leave CatBoost depth.

## 20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: lower CatBoost bagging temperature may reduce dominant-model stochasticity and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `bagging_temperature: 0.2`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter212_blend9765_cat_bagtemp02_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5946205571565802`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `601066e`.
- Interpretation: CatBoost bagging temperature change is numerically inert under the current setup. Keep `0.5` and avoid this lever.
- Next step: move to another training lever.

## 20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: a slightly higher DART feature fraction may improve the side model after the `reg_alpha: 1.2` improvement.
- Changed files: `experiments/configs/20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `colsample_bytree: 0.38`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020 --config experiments/configs/20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter213_blend9765_dart_l1_12_col38_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18609250827128546`.
- Utility before / after: `0.07659409020217732` / `0.07555728356661481`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5932203389830508`.
- Accepted count before / after: `3123` / `3127`.
- Coverage before / after: `0.4047433903576983` / `0.40526179367547954`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `c6b5c1c`.
- Interpretation: higher DART feature fraction hurts accepted accuracy. Keep `colsample_bytree: 0.35`.
- Next step: avoid nearby higher DART feature fractions.

## 20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: a slightly lower DART feature fraction may regularize the side model and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `colsample_bytree: 0.33`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020 --config experiments/configs/20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter214_blend9765_dart_l1_12_col33_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18623898016194404`.
- Utility before / after: `0.07659409020217732` / `0.07555728356661483`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5933397374319564`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `d1b6464`.
- Interpretation: lower DART feature fraction also hurts accepted accuracy. Keep `colsample_bytree: 0.35`.
- Next step: close DART feature-fraction bracket.

## 20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: more DART trees at the same learning rate may improve the side model while the small blend weight limits overfit.
- Changed files: `experiments/configs/20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `n_estimators: 2000`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020 --config experiments/configs/20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter215_blend9765_dart_l1_12_n2000_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1861015296056356`.
- Utility before / after: `0.07659409020217732` / `0.0759460860549507`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5928390367553865`.
- Accepted count before / after: `3123` / `3156`.
- Coverage before / after: `0.4047433903576983` / `0.40902021772939345`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `08fce44`.
- Interpretation: more DART estimators expand coverage and lose accepted accuracy. Keep `n_estimators: 1600`.
- Next step: avoid larger DART estimator counts.

## 20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: fewer DART trees may reduce side-model overfit and improve accepted accuracy near the coverage floor.
- Changed files: `experiments/configs/20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `n_estimators: 1200`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020 --config experiments/configs/20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter216_blend9765_dart_l1_12_n1200_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18569370510155944`.
- Utility before / after: `0.07659409020217732` / `0.07503888024883355`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5934172313649564`.
- Accepted count before / after: `3123` / `3099`.
- Coverage before / after: `0.4047433903576983` / `0.4016329704510109`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `8cc96ee`.
- Interpretation: fewer DART estimators also hurts score and nearly hits the coverage floor. Keep `n_estimators: 1600`.
- Next step: close DART estimator-count bracket.

## 20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination`.
- Hypothesis: pruning the lowest-importance tail while force-preserving HTF and time features may reduce noise and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter217_top450_keep_htf_time_split`; feature-selection report: `artifacts/data_v2/experiments/20260509_codex_iter217_top450_keep_htf_time_split/feature_selection_report.json`.
- Feature set: current best split reduced from 516 to 459 features, preserving all `htf_*`, `hour_*`, and `minute_*` features.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter217_top450_keep_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter217_top450_keep_htf_time_blend9765_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16404357543310044`.
- Utility before / after: `0.07659409020217732` / `0.07089165370658375`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5797608632254302`.
- Accepted count before / after: `3123` / `3429`.
- Coverage before / after: `0.4047433903576983` / `0.4444012441679627`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `814452f`.
- Interpretation: even low-importance features are useful to the ensemble; aggressive top-450 pruning badly hurts accepted accuracy. Keep the full feature set.
- Next step: avoid aggressive feature-count pruning.

## 20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced

- Skill used: `tabular-balanced-log-loss`.
- Hypothesis: class-balanced Platt-logit calibration may improve calibrated decision quality while leaving threshold search unchanged.
- Changed files: `experiments/configs/20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`, `class_weight: balanced`.
- Config: `experiments/configs/20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced --config experiments/configs/20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter218_blend9765_dart_l1_12_platt_logit_c020_balanced/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18370391030053992`.
- Utility before / after: `0.07659409020217732` / `0.07490927941938827`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5919211195928753`.
- Accepted count before / after: `3123` / `3144`.
- Coverage before / after: `0.4047433903576983` / `0.40746500777604977`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `9b9f6ee`.
- Interpretation: balanced Platt-logit calibration hurts accepted accuracy. Keep unweighted `platt_logit`.
- Next step: avoid balanced calibration on this branch.

## 20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: fewer CatBoost iterations at the same learning rate may reduce dominant-model overfit.
- Changed files: `experiments/configs/20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `iterations: 1000`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter219_blend9765_cat_iter1000_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5946205571565802`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `4481bca`.
- Interpretation: CatBoost `iterations: 1000` is neutral. Keep the original `1200` for continuity.
- Next step: avoid CatBoost iteration-only reductions.

## 20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: a slightly lower CatBoost blend weight than `0.9765` may capture more useful DART side signal after the `reg_alpha: 1.2` improvement.
- Changed files: `experiments/configs/20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: logit blend with `catboost_weight: 0.9760`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter220_blend9760_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18862370684471802`.
- Utility before / after: `0.07659409020217732` / `0.07646448937273194`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5943698016634676`.
- Accepted count before / after: `3123` / `3126`.
- Coverage before / after: `0.4047433903576983` / `0.4051321928460342`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `2ca3c5d`.
- Interpretation: `catboost_weight: 0.9760` is worse than `0.9765`. Keep `0.9765`.
- Next step: close this fine blend-weight bracket.

## 20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: lower CatBoost random strength may reduce dominant-model split noise and improve accepted accuracy, after higher random strength was harmful.
- Changed files: `experiments/configs/20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `random_strength: 1.0`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter221_blend9765_cat_rs1_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1731609982358189`.
- Utility before / after: `0.07659409020217732` / `0.07153965785381028`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5866290018832392`.
- Accepted count before / after: `3123` / `3186`.
- Coverage before / after: `0.4047433903576983` / `0.4129082426127527`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `12d34d6`.
- Interpretation: lower CatBoost random strength is much worse. Keep `random_strength: 2.0`.
- Next step: close CatBoost random-strength bracket.

## 20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020

- Skill used: `tabular-balanced-log-loss`.
- Hypothesis: CatBoost class-balanced training may improve directional class treatment in the dominant model while preserving threshold search.
- Changed files: `experiments/configs/20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `auto_class_weights: Balanced`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter222_blend9765_cat_auto_balanced_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.178520689788613`.
- Utility before / after: `0.07659409020217732` / `0.07620528771384133`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5864705882352941`.
- Accepted count before / after: `3123` / `3400`.
- Coverage before / after: `0.4047433903576983` / `0.4406428201140487`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `3652a00`.
- Interpretation: CatBoost class balancing expands coverage and hurts accepted accuracy. Keep unweighted CatBoost training.
- Next step: avoid class-balanced CatBoost on this branch.

## 20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: LightGBM DART `extra_trees` may add useful side-model diversity while the CatBoost-dominant blend limits risk.
- Changed files: `experiments/configs/20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `extra_trees: true`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020 --config experiments/configs/20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter223_blend9765_dart_l1_12_extra_trees_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18279701864257217`.
- Utility before / after: `0.07659409020217732` / `0.07413167444271639`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5919614147909967`.
- Accepted count before / after: `3123` / `3110`.
- Coverage before / after: `0.4047433903576983` / `0.40305857957490926`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `8ae1ddf`.
- Interpretation: `extra_trees` hurts the DART side contribution. Keep it disabled.
- Next step: avoid DART extra-trees.

## 20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: a small DART `min_split_gain` may prune weak side-model splits and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `min_split_gain: 0.01`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020 --config experiments/configs/20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter224_blend9765_dart_l1_12_minsplit001_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1872710715464114`.
- Utility before / after: `0.07659409020217732` / `0.07594608605495072`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5937900128040973`.
- Accepted count before / after: `3123` / `3124`.
- Coverage before / after: `0.4047433903576983` / `0.4048729911871436`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `9839706`.
- Interpretation: DART `min_split_gain` lowers accepted accuracy and score. Keep no split-gain floor.
- Next step: avoid this DART split-gain regularizer on the current branch.

## 20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting` for model regularization discipline.
- Hypothesis: CatBoost Bernoulli bootstrap with `subsample: 0.8` may regularize the dominant model more effectively than the current bootstrap setup.
- Changed files: `experiments/configs/20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `bootstrap_type: Bernoulli`, `subsample: 0.8`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter225_blend9765_cat_bernoulli08_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.17432749100557504`.
- Utility before / after: `0.07659409020217732` / `0.07089165370658375`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5882542755727654`.
- Accepted count before / after: `3123` / `3099`.
- Coverage before / after: `0.4047433903576983` / `0.4016329704510109`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `893b427`.
- Interpretation: CatBoost Bernoulli bootstrap hurts accepted accuracy and nearly reaches the coverage floor. Keep the original bootstrap setup.
- Next step: avoid CatBoost bootstrap changes on this branch.

## 20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: lowering DART `drop_rate` to `0.05` may preserve more useful side-model signal while still regularizing enough to improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `drop_rate: 0.05`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020 --config experiments/configs/20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter226_blend9765_dart_l1_12_droprate005_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18549985592252746`.
- Utility before / after: `0.07659409020217732` / `0.07516848107827892`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5931278098908157`.
- Accepted count before / after: `3123` / `3114`.
- Coverage before / after: `0.4047433903576983` / `0.4035769828926905`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `906ccdb`.
- Interpretation: Lower DART dropout slightly reduces accepted accuracy and coverage, so the current best DART defaults remain preferable.
- Next step: avoid lower `drop_rate`; test orthogonal DART knobs only if they plausibly improve side-model diversity without shifting thresholds.

## 20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020

- Skill used: `tabular-collinear-feature-removal` as feature-redundancy motivation, applied here through model-side CatBoost feature subsampling rather than changing the cached feature set.
- Hypothesis: CatBoost `rsm: 0.8` may reduce reliance on redundant feature groups and improve accepted accuracy in the CatBoost-dominant blend.
- Changed files: `experiments/configs/20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `rsm: 0.8`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter227_blend9765_cat_rsm08_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5946205571565802`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `ca812d3`.
- Interpretation: `rsm: 0.8` is neutral on the validation objective for this blend and split.
- Next step: test CatBoost border resolution, which can alter numeric binning without changing feature semantics.

## 20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020

- Skill used: `tabular-squeeze-compact-local-search` as compact one-knob local search discipline.
- Hypothesis: CatBoost `border_count: 128` may regularize numeric binning and reduce over-specialized splits in the dominant model.
- Changed files: `experiments/configs/20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, CatBoost `border_count: 128`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter228_blend9765_cat_border128_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07659409020217732` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5946205571565802`.
- Accepted count before / after: `3123` / `3123`.
- Coverage before / after: `0.4047433903576983` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `67e017e`.
- Interpretation: `border_count: 128` is neutral on the official validation metrics. The implementation passes nested CatBoost params through, but this knob did not move the accepted slice.
- Next step: stop spending iterations on CatBoost-only schema knobs unless they are known to affect this plugin; return to data/feature/model changes with observed metric movement.

## 20260509_codex_iter229_blend9765_drop_low12_nonprotected_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination`.
- Hypothesis: dropping only the 12 weakest non-protected importance-tail features may reduce noise while keeping required HTF/time context.
- Changed files: `experiments/configs/20260509_codex_iter229_blend9765_drop_low12_nonprotected_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter229_drop_low12_nonprotected_split`.
- Feature set: 504 features; dropped `stale_trade_share_5_rolling_z_12`, `sl_agg_large_sell_trade_count_5s`, `ret_relative_volume_product__ret_15__relative_volume_3`, `ret_relative_volume_product__ret_10__relative_volume_5`, `ret_vol_ratio__ret_1__rv_30`, `sl_agg_signed_large_notional_1s`, `ret_efficiency_product__ret_15__efficiency_10`, `ret_1_rolling_z_24`, `price_location_product__range_pos_3__close_z_10`, `max_up_ret_5_delta_1`, `ret_1_rolling_z_12`, `ret_relative_volume_product__ret_1__relative_volume_5`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter229_blend9765_drop_low12_nonprotected_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter229_drop_low12_nonprotected_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter229_blend9765_drop_low12_nonprotected_platt_logit_c020 --config experiments/configs/20260509_codex_iter188_blend9765_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter229_blend9765_drop_low12_nonprotected_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.17832559840538859`.
- Utility before / after: `0.07659409020217732` / `0.07322446863659923`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5892011367224502`.
- Accepted count before / after: `3123` / `3167`.
- Coverage before / after: `0.4047433903576983` / `0.41044582685329184`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; feature filter summary saved at `artifacts/data_v2/experiments/20260509_codex_iter229_drop_low12_nonprotected_split/feature_filter_summary.json`.
- Git commit: `a040e38`.
- Interpretation: even a shallow low-importance prune increases coverage but lowers accepted accuracy enough to hurt selection_score. Keep the full 516-feature best split.
- Next step: avoid further simple tail pruning; consider model-side or split-side changes with no feature-family removal.

## 20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020

- Skill used: `tabular-row-aggregate-features`.
- Hypothesis: leak-free row aggregates over the top 20 second-level microstructure features may summarize broad tape pressure and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter230_sl_top20_rowagg_split`.
- Feature set: 520 features; added `sl_top20_z_mean`, `sl_top20_z_std`, `sl_top20_z_min`, `sl_top20_z_max`; standardization fit on development only; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter230_sl_top20_rowagg_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020 --config experiments/configs/20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter230_blend9765_sl_top20_rowagg_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.1684464240266692`.
- Utility before / after: `0.07659409020217732` / `0.0738724727838258`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5805539853024307`.
- Accepted count before / after: `3123` / `3538`.
- Coverage before / after: `0.4047433903576983` / `0.4585277345775013`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; row aggregate transform summary saved at `artifacts/data_v2/experiments/20260509_codex_iter230_sl_top20_rowagg_split/rowagg_summary.json`.
- Git commit: `af3539a`.
- Interpretation: broad row aggregates materially increase coverage but weaken accepted accuracy, reducing selection_score. Do not keep this aggregate pack.
- Next step: favor narrower model-side experiments or more targeted time/HTF interactions instead of broad second-level aggregates.

## 20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020

- Skill used: `tabular-polynomial-interaction-features`.
- Hypothesis: explicit interactions between important HTF context and cyclical time features may capture time-of-day-dependent HTF effects without changing base feature semantics.
- Changed files: `experiments/configs/20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter231_htf_time_interactions_split`.
- Feature set: 532 features; added 16 same-row interactions crossing the top 4 HTF features with `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`; HTF/time base features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9765`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter231_htf_time_interactions_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020 --config experiments/configs/20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter231_blend9765_htf_time_interactions_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.16960989197385432`.
- Utility before / after: `0.07659409020217732` / `0.07413167444271648`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.58125`.
- Accepted count before / after: `3123` / `3520`.
- Coverage before / after: `0.4047433903576983` / `0.45619491964748576`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; interaction transform summary saved at `artifacts/data_v2/experiments/20260509_codex_iter231_htf_time_interactions_split/interaction_summary.json`.
- Git commit: `1874d9f`.
- Interpretation: the interactions make the model accept many more samples but lower accepted accuracy, so they are harmful for selection_score. Keep base HTF/time features but not this interaction pack.
- Next step: avoid broad added-feature packs on the current split; return to model calibration/blend variants that preserve the best accepted boundary.

## 20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: pure CatBoost (`catboost_weight: 1.0`) may remove noisy LightGBM logit contribution and improve accepted accuracy near the selective boundary.
- Changed files: `experiments/configs/20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 1.0`, nested DART params unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter232_blend10000_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.18312798896889548`.
- Utility before / after: `0.07659409020217732` / `0.07400207361327112`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5923649304432222`.
- Accepted count before / after: `3123` / `3091`.
- Coverage before / after: `0.4047433903576983` / `0.40059616381544844`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed YAML attempt failed before training and was corrected before the recorded evaluation.
- Git commit: `9e70353`.
- Interpretation: the small DART contribution in the best blend improves both coverage and accepted accuracy. Keep `catboost_weight: 0.9765`.
- Next step: test a slightly larger DART contribution only if it remains close to the current accepted boundary.

## 20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: a slightly larger CatBoost logit weight (`0.9770`) may retain useful DART diversity while reducing side-model noise at the selective boundary.
- Changed files: `experiments/configs/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.1890925935441257`.
- Score after: `0.19027803605274402`.
- Utility before / after: `0.07659409020217732` / `0.07698289269051321`.
- Accepted accuracy before / after: `0.5946205571565802` / `0.5951923076923077`.
- Accepted count before / after: `3123` / `3120`.
- Coverage before / after: `0.4047433903576983` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed YAML attempt failed before training and was corrected before the recorded evaluation.
- Git commit: `2d8e1a1`.
- Interpretation: this is a new best under the coverage constraint. The gain comes from a small accepted-accuracy lift with nearly unchanged coverage and thresholds.
- Next step: continue fine blend-weight search just above `0.9770` while monitoring coverage floor.

## 20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: `catboost_weight: 0.9775` may improve on the new blend-weight best by slightly reducing DART side-model influence.
- Changed files: `experiments/configs/20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9775`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter234_blend9775_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18899670247663125`.
- Utility before / after: `0.07698289269051321` / `0.07646448937273198`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5946726572528883`.
- Accepted count before / after: `3120` / `3116`.
- Coverage before / after: `0.40435458786936235` / `0.40383618455158116`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `22a1a8a`.
- Interpretation: moving above `0.9770` reduces both coverage and accepted accuracy. Keep iteration 233 as current best.
- Next step: bracket the optimum around `0.9770` with smaller steps.

## 20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: `catboost_weight: 0.9769` may preserve the `0.9770` accepted-boundary gain while slightly improving coverage.
- Changed files: `experiments/configs/20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9769`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter235_blend9769_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1890925935441257`.
- Utility before / after: `0.07698289269051321` / `0.07659409020217732`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5946205571565802`.
- Accepted count before / after: `3120` / `3123`.
- Coverage before / after: `0.40435458786936235` / `0.4047433903576983`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `b019b83`.
- Interpretation: `0.9769` returns to the previous metric plateau and does not preserve the `0.9770` accuracy lift.
- Next step: test `0.9771` to bracket the other side of the narrow improvement.

## 20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking`.
- Hypothesis: `catboost_weight: 0.9771` may improve slightly above the current best while avoiding the degradation seen at `0.9775`.
- Changed files: `experiments/configs/20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9771`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter236_blend9771_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.19027803605274402`.
- Utility before / after: `0.07698289269051321` / `0.07698289269051321`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5951923076923077`.
- Accepted count before / after: `3120` / `3120`.
- Coverage before / after: `0.40435458786936235` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `e22f6b9`.
- Interpretation: `0.9771` ties the current best exactly. Treat `0.9770`/`0.9771` as the same accepted-boundary plateau.
- Next step: tune DART regularization while keeping the new blend plateau.

## 20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: slightly stronger DART L1 (`reg_alpha: 1.3`) may reduce side-model overfit while preserving the new `0.9770` blend-weight gain.
- Changed files: `experiments/configs/20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.3`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020 --config experiments/configs/20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter237_blend9770_dart_l1_13_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18759064675382858`.
- Utility before / after: `0.07698289269051321` / `0.07607568688439606`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.59392`.
- Accepted count before / after: `3120` / `3125`.
- Coverage before / after: `0.40435458786936235` / `0.4050025920165889`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed YAML attempt failed before training and was corrected before the recorded evaluation.
- Git commit: `60cddcb`.
- Interpretation: higher DART L1 adds coverage but lowers accepted accuracy enough to hurt selection_score. Keep `reg_alpha: 1.2`.
- Next step: avoid stronger DART L1; try weaker L1 or other regularization only if it can improve accepted accuracy.

## 20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: weaker DART L1 (`reg_alpha: 1.1`) may allow useful side-model signal while keeping the new `0.9770` blend-weight gain.
- Changed files: `experiments/configs/20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.1`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020 --config experiments/configs/20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter238_blend9770_dart_l1_11_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1872710715464114`.
- Utility before / after: `0.07698289269051321` / `0.07594608605495072`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5937900128040973`.
- Accepted count before / after: `3120` / `3124`.
- Coverage before / after: `0.40435458786936235` / `0.4048729911871436`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed YAML attempt failed before training and was corrected before the recorded evaluation.
- Git commit: `410149c`.
- Interpretation: weaker DART L1 also lowers accepted accuracy. Keep `reg_alpha: 1.2`.
- Next step: avoid DART L1 changes around the current blend plateau.

## 20260509_codex_iter239_blend9770_platt_logit_c015

- Skill used: `tabular-logit-transform-stacking` for logit-space calibration regularization.
- Hypothesis: stronger Platt-logit regularization (`C: 0.15`) may improve selective probability ordering while keeping the current best blend.
- Changed files: `experiments/configs/20260509_codex_iter239_blend9770_platt_logit_c015.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.15`.
- Config: `experiments/configs/20260509_codex_iter239_blend9770_platt_logit_c015.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter239_blend9770_platt_logit_c015 --config experiments/configs/20260509_codex_iter239_blend9770_platt_logit_c015.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter239_blend9770_platt_logit_c015/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18519931802728634`.
- Utility before / after: `0.07698289269051321` / `0.07477967858994299`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5932751374070482`.
- Accepted count before / after: `3120` / `3093`.
- Coverage before / after: `0.40435458786936235` / `0.40085536547433903`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed calibration nesting failed before training and was corrected before the recorded evaluation.
- Git commit: `6936ab4`.
- Interpretation: stronger calibration regularization hurts coverage and accepted accuracy. Keep `C: 0.2`.
- Next step: test weaker calibration regularization once to bracket the direction.

## 20260509_codex_iter240_blend9770_platt_logit_c025

- Skill used: `tabular-logit-transform-stacking` for logit-space calibration regularization.
- Hypothesis: weaker Platt-logit regularization (`C: 0.25`) may improve selective probability ordering while keeping the current best blend.
- Changed files: `experiments/configs/20260509_codex_iter240_blend9770_platt_logit_c025.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.25`.
- Config: `experiments/configs/20260509_codex_iter240_blend9770_platt_logit_c025.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter240_blend9770_platt_logit_c025 --config experiments/configs/20260509_codex_iter240_blend9770_platt_logit_c025.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter240_blend9770_platt_logit_c025/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18565515747096398`.
- Utility before / after: `0.07698289269051321` / `0.07555728356661484`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5928639694170118`.
- Accepted count before / after: `3120` / `3139`.
- Coverage before / after: `0.40435458786936235` / `0.4068170036288232`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration nesting was verified before the recorded evaluation.
- Git commit: `5aa602b`.
- Interpretation: weaker calibration regularization increases coverage but lowers accepted accuracy and score. Keep `C: 0.2`.
- Next step: stop calibration-C bracketing around this model.

## 20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020

- Skill used: `tabular-logit-transform-stacking` for preserving the current logit blend while making a narrow CatBoost capacity change.
- Hypothesis: a small CatBoost L2 decrease (`l2_leaf_reg: 28`) may improve the dominant model's accepted-boundary ranking without changing data or thresholds.
- Changed files: `experiments/configs/20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, CatBoost `l2_leaf_reg: 28.0`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020 --config experiments/configs/20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter241_blend9770_cat_l2_28_dart_l1_12_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.19027803605274402`.
- Utility before / after: `0.07698289269051321` / `0.07698289269051321`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5951923076923077`.
- Accepted count before / after: `3120` / `3120`.
- Coverage before / after: `0.40435458786936235` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; calibration was fit only on development predictions.
- Git commit: `6cf7e4b`.
- Interpretation: this L2 change is neutral on the accepted slice. Keep `l2_leaf_reg: 30.0` for the canonical best unless further combinations show a benefit.
- Next step: avoid very small CatBoost L2-only moves unless paired with another justified model change.

## 20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020

- Skill used: `tabular-multi-seed-fold-averaging`.
- Hypothesis: averaging three CatBoost seeds may reduce prediction variance and improve selective accepted accuracy versus a single CatBoost model.
- Changed files: `experiments/configs/20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: current best VWAP-pruned top-500 split; HTF/time features retained.
- Model settings: `model.active_plugin: catboost_ensemble`, `n_seeds: 3`, CatBoost base params matching the current best, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020 --config experiments/configs/20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter242_catboost_seed_ensemble3_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18284455810731287`.
- Utility before / after: `0.07698289269051321` / `0.07659409020217733`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5895725977568961`.
- Accepted count before / after: `3120` / `3299`.
- Coverage before / after: `0.40435458786936235` / `0.4275531363400726`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; initial malformed YAML attempt failed before training and was corrected before the recorded evaluation.
- Git commit: `b194374`.
- Interpretation: seed averaging increases coverage but lowers accepted accuracy. The CatBoost+DART logit blend remains better than CatBoost-only ensembling.
- Next step: avoid replacing the blend with CatBoost-only seed averages on this split.

## 20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020

- Skill used: `timeseries-multi-gap-lag-diff-features`, `timeseries-multi-scale-rolling-features`.
- Hypothesis: adding trailing multi-gap lag differences and rolling summaries over the top 8 microstructure features may expose short-term state persistence without lookahead.
- Changed files: `experiments/configs/20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter243_top8_trailing_lagdiff_roll_split`.
- Feature set: 628 features; added 112 trailing lagdiff/rolling/tv features over top second-level features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter243_top8_trailing_lagdiff_roll_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020 --config experiments/configs/20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter243_blend9770_top8_trailing_lagdiff_roll_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16830202757545706`.
- Utility before / after: `0.07698289269051321` / `0.07076205287713837`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5833842394624312`.
- Accepted count before / after: `3120` / `3274`.
- Coverage before / after: `0.40435458786936235` / `0.42431311560393986`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; transform summary saved at `artifacts/data_v2/experiments/20260509_codex_iter243_top8_trailing_lagdiff_roll_split/lagdiff_roll_summary.json`; validation features use chronological development+validation context with shift/lag only.
- Git commit: `4fb4081`.
- Interpretation: broad trailing microstructure expansion increases coverage and side balance but materially lowers accepted accuracy and selection_score. Do not keep this feature pack.
- Next step: use more local data via a longer training window, rather than adding broad derived features to the current split.

## 20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020

- Skill used: `timeseries-multi-scale-rolling-features` for preserving the full rebuilt feature set with multi-scale context from local data.
- Hypothesis: rebuilding from all local normalized 1m data with a 120-day training window and the materialized second-level store may improve generalization by using more history and all configured features.
- Changed files: `experiments/configs/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Input data: `artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet`; second-level store `artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020`.
- Feature set: 1735 rebuilt features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --input artifacts/data_v2/normalized/spot/klines/BTCUSDT-1m.parquet --output-dir artifacts/data_v2/experiments/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020 --config experiments/configs/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020.yaml --horizon 5m --train-window-days 120 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter244_blend9770_train120_rebuild_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16085485386795179`.
- Utility before / after: `0.07698289269051321` / `0.07011404872991182`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5778865534120357`.
- Accepted count before / after: `3120` / `3473`.
- Coverage before / after: `0.40435458786936235` / `0.45010368066355627`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; split was rebuilt from local data only and cached in the run directory.
- Git commit: `fadfafd`.
- Interpretation: using much more history and the full rebuilt feature set increases coverage but lowers accepted accuracy, so the current objective worsens. The narrower 516-feature split remains better.
- Next step: try a less aggressive rebuilt data window or constrain the rebuilt feature set, rather than using the full 120-day/1735-feature set as-is.

## 20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as controlled feature-set filtering, applied by retaining the current best 516-feature manifest on the longer rebuilt split.
- Hypothesis: using 120 days of local data with the proven 516-feature set may improve generalization while avoiding the noisy 1735-feature expansion from iter244.
- Changed files: `experiments/configs/20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter245_train120_best516_feature_filter_split`.
- Feature set: 516 features matching `artifacts/data_v2/experiments/20260509_codex_iter233_blend9770_dart_l1_12_platt_logit_c020/artifact_manifest.json`; no missing reference features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter245_train120_best516_feature_filter_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020 --config experiments/configs/20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020.yaml --horizon 5m --train-window-days 120 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter245_blend9770_train120_best516_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16988295531987718`.
- Utility before / after: `0.07698289269051321` / `0.07983411093831004`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5765407554671969`.
- Accepted count before / after: `3120` / `4024`.
- Coverage before / after: `0.40435458786936235` / `0.5215137376879212`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; feature-filter summary saved at `artifacts/data_v2/experiments/20260509_codex_iter245_train120_best516_feature_filter_split/feature_filter_summary.json`.
- Git commit: `9a8e816`.
- Interpretation: the extra history shifts the model toward much broader acceptance and lower accepted accuracy even with the best feature set. The recent 75-day window remains better.
- Next step: try a moderate window before abandoning extra-history splits.

## 20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as controlled feature-set filtering on a moderate extra-history split.
- Hypothesis: a 90-day development window with the proven 516-feature set may add useful history without the accepted-accuracy dilution seen at 120 days.
- Changed files: `experiments/configs/20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter246_train90_best516_feature_filter_split`.
- Feature set: 516 features matching the current best feature manifest; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter246_train90_best516_feature_filter_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020 --config experiments/configs/20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020.yaml --horizon 5m --train-window-days 90 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter246_blend9770_train90_best516_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16920230974904024`.
- Utility before / after: `0.07698289269051321` / `0.07814930015552096`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5774069319640565`.
- Accepted count before / after: `3120` / `3895`.
- Coverage before / after: `0.40435458786936235` / `0.5047952306894764`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; window-filter summary saved at `artifacts/data_v2/experiments/20260509_codex_iter246_train90_best516_feature_filter_split/window_filter_summary.json`.
- Git commit: `f8f91b7`.
- Interpretation: 90 days still over-broadens accepted predictions and reduces accepted accuracy. The 75-day split remains preferred for this validation period.
- Next step: focus on feature selection within the current recent-window data rather than adding older history.

## 20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as importance-based feature selection from the rebuilt 120-day full-feature branch.
- Hypothesis: selecting the top 700 rebuilt features, while forcing HTF/time retention, may keep useful extra feature coverage from the 1735-feature rebuild without as much noise.
- Changed files: `experiments/configs/20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter247_train120_top700_importance_split`.
- Feature set: 717 features after top-700 importance selection plus protected HTF/time features.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter247_train120_top700_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020 --config experiments/configs/20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020.yaml --horizon 5m --train-window-days 120 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter247_blend9770_train120_top700_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16159613759478425`.
- Utility before / after: `0.07698289269051321` / `0.06739243131156038`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5811485642946317`.
- Accepted count before / after: `3120` / `3204`.
- Coverage before / after: `0.40435458786936235` / `0.4152410575427683`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; feature-selection summary saved at `artifacts/data_v2/experiments/20260509_codex_iter247_train120_top700_importance_split/feature_select_summary.json`.
- Git commit: `69395d3`.
- Interpretation: the selected extra feature set still lowers accepted accuracy. The 120-day rebuilt branch is not competitive on the current validation slice.
- Next step: return to the 75-day recent-window branch and look for targeted feature additions rather than older-history expansion.

## 20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as existing top-700 feature-set evaluation.
- Hypothesis: the existing local top700 split may improve the current blend by adding more recent-window features without rebuilding raw data.
- Changed files: `experiments/configs/20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter26_top700_split`.
- Feature set: 710 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter26_top700_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020 --config experiments/configs/20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020.yaml --horizon 5m --train-window-days 180 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter248_blend9770_existing_top700_split_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.15319799697886222`.
- Utility before / after: `0.07698289269051321` / `0.06467081389320894`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5767928593413358`.
- Accepted count before / after: `3120` / `3249`.
- Coverage before / after: `0.40435458786936235` / `0.42107309486780714`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `2f0f9da`.
- Interpretation: adding the existing top700/full-history feature split lowers accepted accuracy substantially. Current best remains the 75-day 516-feature split.
- Next step: avoid broad top700/full-history splits; use smaller, targeted feature transforms on the best split only.

## 20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020

- Skill used: `timeseries-multi-gap-lag-diff-features` as targeted return-consistency feature evaluation.
- Hypothesis: the existing return-consistency feature split may improve the current CatBoost+DART blend even though its earlier CatBoost-only run was weak.
- Changed files: `experiments/configs/20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_split`.
- Feature set: 532 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter91_return_consistency_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020 --config experiments/configs/20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter249_blend9770_return_consistency_split_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17496376398996846`.
- Utility before / after: `0.07698289269051321` / `0.07153965785381024`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5881226053639846`.
- Accepted count before / after: `3120` / `3132`.
- Coverage before / after: `0.40435458786936235` / `0.4059097978227061`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `fe435c2`.
- Interpretation: return-consistency features lower accepted accuracy under the current blend as well. Keep the current best split.
- Next step: avoid this targeted feature branch.

## 20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020

- Skill used: `tabular-relative-deviation-features` as existing session-relative feature evaluation.
- Hypothesis: session-relative features may improve the current CatBoost+DART blend by normalizing microstructure behavior by intraday regime/session.
- Changed files: `experiments/configs/20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split`.
- Feature set: 531 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter66_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020 --config experiments/configs/20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter250_blend9770_session_relative_split_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17835303076410833`.
- Utility before / after: `0.07698289269051321` / `0.073094867807154`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5893536121673004`.
- Accepted count before / after: `3120` / `3156`.
- Coverage before / after: `0.40435458786936235` / `0.40902021772939345`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `105ebd2`.
- Interpretation: session-relative features lower accepted accuracy under the current blend. Keep the current best split.
- Next step: avoid this feature branch unless paired with a different model family.

## 20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020

- Skill used: `tabular-adversarial-validation`.
- Hypothesis: dropping the top 5 non-HTF/time adversarial-drift features may improve validation transfer under the current CatBoost+DART blend.
- Changed files: `experiments/configs/20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_split`.
- Feature set: 511 features; dropped `low_volume_flag_share_20_mean_gap_6`, `sl_range_10s`, `sl_mirror_true_range_pct_1s`, `sl_range_3s`, `low_volume_flag_share_20_rolling_z_6`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter111_drop_top5_drift_non_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020 --config experiments/configs/20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter251_blend9770_drop_top5_drift_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17269477312126522`.
- Utility before / after: `0.07698289269051321` / `0.071021254536029`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5867637745408486`.
- Accepted count before / after: `3120` / `3158`.
- Coverage before / after: `0.40435458786936235` / `0.4092794193882841`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; adversarial source report is `artifacts/data_v2/experiments/20260508_codex_adversarial_top500/summary.json`.
- Git commit: `843d5ae`.
- Interpretation: removing the top drift features worsens accepted accuracy and changes side balance. These shifted features still carry useful signal for the current objective.
- Next step: avoid further adversarial-drop pruning unless combined with a separate feature transform.

## 20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020

- Skill used: `tabular-anomaly-flag-imputation` as data filtering / noisy-sample processing discipline.
- Hypothesis: removing low-absolute-return training samples may reduce ambiguous labels and improve accepted accuracy under the current blend.
- Changed files: `experiments/configs/20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter94_drop_low_abs_return_train_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020 --config experiments/configs/20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter252_blend9770_drop_low_abs_return_train_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1800985512257311`.
- Utility before / after: `0.07698289269051321` / `0.07309486780715399`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5907920154539601`.
- Accepted count before / after: `3120` / `3106`.
- Coverage before / after: `0.40435458786936235` / `0.40254017625712807`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `68649dc`.
- Interpretation: filtering ambiguous low-return training samples reduces accepted accuracy and does not improve the objective.
- Next step: avoid low-return training filters on this branch.

## 20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020

- Skill used: `tabular-gmm-feature-augmentation`, adapted as unsupervised GMM regime feature augmentation rather than synthetic sampling.
- Hypothesis: development-fitted GMM cluster probabilities over top market/microstructure features may expose latent regimes and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter253_gmm_regime_top10_split`.
- Feature set: 524 features; added 5 GMM probabilities plus max-probability, entropy, and log-likelihood; GMM/imputer/scaler fit on development only; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter253_gmm_regime_top10_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020 --config experiments/configs/20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter253_blend9770_gmm_regime_top10_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16766327576169632`.
- Utility before / after: `0.07698289269051321` / `0.07555728356661487`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.578423459779392`.
- Accepted count before / after: `3120` / `3717`.
- Coverage before / after: `0.40435458786936235` / `0.4817262830482115`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; GMM transform summary saved at `artifacts/data_v2/experiments/20260509_codex_iter253_gmm_regime_top10_split/gmm_summary.json`.
- Git commit: `10e9828`.
- Interpretation: dense GMM regime probabilities over-broaden acceptance and dilute accepted accuracy. Do not keep this feature pack.
- Next step: try simpler discretized/bin features if adding regime context, not dense unsupervised probabilities.

## 20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020

- Skill used: `tabular-anomaly-flag-imputation`.
- Hypothesis: sparse quantile tail flags for the top 10 features may capture extreme-state regimes without the probability-scale distortion from dense GMM features.
- Changed files: `experiments/configs/20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter254_top10_tail_flags_split`.
- Feature set: 556 features; added 40 development-fitted 5/10/90/95% tail flags over top features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter254_top10_tail_flags_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020 --config experiments/configs/20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter254_blend9770_top10_tail_flags_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18413497208280713`.
- Utility before / after: `0.07698289269051321` / `0.07490927941938832`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5922733077905492`.
- Accepted count before / after: `3120` / `3132`.
- Coverage before / after: `0.40435458786936235` / `0.4059097978227061`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; quantile summary saved at `artifacts/data_v2/experiments/20260509_codex_iter254_top10_tail_flags_split/tail_flags_summary.json`.
- Git commit: `12239f7`.
- Interpretation: sparse tail flags are less harmful than broad dense features but still lower accepted accuracy. Do not keep this feature pack.
- Next step: stop adding top-feature regime flags unless they are side-specific or model-specific.

## 20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020

- Skill used: `tabular-null-importance-feature-selection`.
- Hypothesis: the existing null-importance tail20 split may remove noisy features and improve accepted accuracy under the current blend.
- Changed files: `experiments/configs/20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_split`.
- Feature set: 496 null-tail-selected features; HTF/time features retained by the existing split.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020 --config experiments/configs/20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter255_blend9770_null_tail20_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16538765086813575`.
- Utility before / after: `0.07698289269051321` / `0.0677812337998963`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5839486356340289`.
- Accepted count before / after: `3120` / `3115`.
- Coverage before / after: `0.40435458786936235` / `0.4037065837221358`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; null-tail source file is `artifacts/data_v2/experiments/20260508_codex_iter54_null_tail20_split/null_tail20_drops.csv`.
- Git commit: `6b3d651`.
- Interpretation: null-importance tail pruning materially lowers accepted accuracy. Keep the current 516-feature split.
- Next step: avoid aggressive null-importance pruning under the current blend.

## 20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020

- Skill used: `timeseries-burst-rle-detection`.
- Hypothesis: directional run-length features over key microstructure/HTF signals may capture persistent bursts without broad rolling-feature noise.
- Changed files: `experiments/configs/20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter256_directional_burst_rle_split`.
- Feature set: 564 features; added 48 positive/negative run-length and lagged rolling max features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter256_directional_burst_rle_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020 --config experiments/configs/20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter256_blend9770_directional_burst_rle_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18284980946198404`.
- Utility before / after: `0.07698289269051321` / `0.07542768273716956`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.59071072319202`.
- Accepted count before / after: `3120` / `3208`.
- Coverage before / after: `0.40435458786936235` / `0.4157594608605495`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; burst summary saved at `artifacts/data_v2/experiments/20260509_codex_iter256_directional_burst_rle_split/burst_rle_summary.json`.
- Git commit: `bc8522c`.
- Interpretation: burst features increase coverage but reduce accepted accuracy. Do not keep this feature pack.
- Next step: use smaller, side-targeted feature additions only.

## 20260509_codex_iter257_blend9770_session_flags_platt_logit_c020

- Skill used: `tabular-season-phase-labeling`.
- Hypothesis: coarse session phase flags (`session_asia`, `session_europe`, `session_us`) may add useful time-regime context beyond cyclical hour/minute encodings.
- Changed files: `experiments/configs/20260509_codex_iter257_blend9770_session_flags_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_split`.
- Feature set: 519 features; added three session flags; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter257_blend9770_session_flags_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter63_session_flags_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter257_blend9770_session_flags_platt_logit_c020 --config experiments/configs/20260509_codex_iter257_blend9770_session_flags_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter257_blend9770_session_flags_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1685522649912154`.
- Utility before / after: `0.07698289269051321` / `0.07711249351995848`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5777777777777777`.
- Accepted count before / after: `3120` / `3825`.
- Coverage before / after: `0.40435458786936235` / `0.49572317262830484`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `8ab2f6e`.
- Interpretation: session flags increase coverage but dilute accepted accuracy. Keep the existing cyclical time features without these flags.
- Next step: avoid coarse session flags on this validation slice.

## 20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020

- Skill used: `tabular-polynomial-interaction-features`.
- Hypothesis: a very small set of pairwise interactions among the top four microstructure features may add nonlinear signal without the over-broadening seen from larger feature packs.
- Changed files: `experiments/configs/20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter258_top4_micro_interactions_split`.
- Feature set: 522 features; added 6 pairwise same-row microstructure interactions; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter258_top4_micro_interactions_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020 --config experiments/configs/20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter258_blend9770_top4_micro_interactions_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16472046890899203`.
- Utility before / after: `0.07698289269051321` / `0.07011404872991185`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.581060833083608`.
- Accepted count before / after: `3120` / `3337`.
- Coverage before / after: `0.40435458786936235` / `0.4324779678589943`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; transform summary saved at `artifacts/data_v2/experiments/20260509_codex_iter258_top4_micro_interactions_split/micro_interactions_summary.json`.
- Git commit: `64a5dc0`.
- Interpretation: even compact explicit microstructure interactions over-broaden accepted predictions and lower accepted accuracy. Do not keep this feature pack.
- Next step: avoid interaction expansion on this split.

## 20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020

- Skill used: `tabular-group-shuffle-split` as split discipline context; evaluated an existing chronological 70-day recent-window split without changing label or thresholds.
- Hypothesis: a slightly shorter 70-day recent training window may improve accepted accuracy by reducing stale regime contamination versus the 75-day best split.
- Changed files: `experiments/configs/20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter44_train70_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020 --config experiments/configs/20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020.yaml --horizon 5m --train-window-days 70 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter259_blend9770_train70_drop_sl_vwap_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16724210380519422`.
- Utility before / after: `0.07698289269051321` / `0.06817003628823227`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5851132686084143`.
- Accepted count before / after: `3120` / `3090`.
- Coverage before / after: `0.40435458786936235` / `0.4004665629860031`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `db4f18d`.
- Interpretation: shortening to 70 days lowers accepted accuracy and nearly reaches the coverage floor. Keep the 75-day split.
- Next step: avoid further train-window bracketing around the current split.

## 20260509_codex_iter260_blend9770_time_te_platt_logit_c020

- Skill used: `tabular-inner-kfold-target-encoding`.
- Hypothesis: smoothed leak-free target encodings for hour and 5-minute buckets may add time-regime priors beyond cyclical encodings while fitting only on development data.
- Changed files: `experiments/configs/20260509_codex_iter260_blend9770_time_te_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter260_time_target_encoding_split`.
- Feature set: 522 features; added 6 OOF/full-development time target-encoding and count features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter260_blend9770_time_te_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter260_time_target_encoding_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter260_blend9770_time_te_platt_logit_c020 --config experiments/configs/20260509_codex_iter260_blend9770_time_te_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter260_blend9770_time_te_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16959291290406484`.
- Utility before / after: `0.07698289269051321` / `0.07153965785381029`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.583687083080655`.
- Accepted count before / after: `3120` / `3298`.
- Coverage before / after: `0.40435458786936235` / `0.4274235355106273`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; target encoding summary saved at `artifacts/data_v2/experiments/20260509_codex_iter260_time_target_encoding_split/time_target_encoding_summary.json`.
- Git commit: `14c0bb4`.
- Interpretation: time target encoding broadens accepted predictions and lowers accepted accuracy. Do not keep this feature pack.
- Next step: avoid target-encoded time buckets on this validation slice.

## 20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as feature-set ablation guidance; evaluated a nearby existing split that adds back the two VWAP features previously removed from the best split.
- Hypothesis: `sl_vwap_10s` and `sl_vwap_30s` may restore short-horizon microstructure context and improve accepted accuracy without changing labels, thresholds, or HTF/time features.
- Changed files: `experiments/configs/20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split`.
- Feature set: 518 features; current best 516 features plus `sl_vwap_10s` and `sl_vwap_30s`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter33_top500_train75_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020 --config experiments/configs/20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter261_blend9770_train75_top500_vwap_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18031147569562883`.
- Utility before / after: `0.07698289269051321` / `0.07309486780715395`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5909677419354838`.
- Accepted count before / after: `3120` / `3100`.
- Coverage before / after: `0.40435458786936235` / `0.40176257128045617`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `730dcc0`.
- Interpretation: adding both VWAP features improves AUC/Brier/logloss diagnostics slightly but lowers objective score through lower accepted accuracy and coverage. Keep the VWAP-dropped 516-feature split.
- Next step: test one-feature VWAP variants only if needed; otherwise continue with narrower model regularization or feature-selection checks.

## 20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as feature-set ablation guidance.
- Hypothesis: `sl_vwap_10s` alone may carry useful very-short-window price-location signal while avoiding the possible stale/noisy effect of `sl_vwap_30s`.
- Changed files: `experiments/configs/20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split`.
- Feature set: 517 features; current best 516 features plus `sl_vwap_10s`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter46_train75_drop_sl_vwap30_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020 --config experiments/configs/20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter262_blend9770_train75_vwap10_only_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1777100307587346`.
- Utility before / after: `0.07698289269051321` / `0.07400207361327112`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5879273175238682`.
- Accepted count before / after: `3120` / `3247`.
- Coverage before / after: `0.40435458786936235` / `0.42081389320891655`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `5e00fb9`.
- Interpretation: `sl_vwap_10s` increases coverage but lowers accepted accuracy enough to reduce selection_score. Keep it excluded.
- Next step: test the `sl_vwap_30s`-only variant, then stop this VWAP ablation branch if it also fails.

## 20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as feature-set ablation guidance.
- Hypothesis: `sl_vwap_30s` alone may add slightly smoother microstructure price-location signal than `sl_vwap_10s` while preserving the current feature set's HTF/time context.
- Changed files: `experiments/configs/20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split`.
- Feature set: 517 features; current best 516 features plus `sl_vwap_30s`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter47_train75_drop_sl_vwap10_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020 --config experiments/configs/20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter263_blend9770_train75_vwap30_only_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16721669777776554`.
- Utility before / after: `0.07698289269051321` / `0.07218766200103675`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5811243810078648`.
- Accepted count before / after: `3120` / `3433`.
- Coverage before / after: `0.40435458786936235` / `0.4449196474857439`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `e0e8f4a`.
- Interpretation: `sl_vwap_30s` over-broadens acceptance and reduces accepted accuracy more than the 10s-only variant. Keep both raw VWAP features excluded.
- Next step: move away from VWAP restoration and test model/regularization changes on the best 516-feature split.

## 20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: lowering nested LightGBM DART `colsample_bytree` from `0.35` to `0.25` may reduce high-dimensional overfit and sharpen accepted predictions in the CatBoost-dominant logit blend.
- Changed files: `experiments/configs/20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `colsample_bytree: 0.25`, `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020 --config experiments/configs/20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter264_blend9770_dart_colsample025_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.185034045094492`.
- Utility before / after: `0.07698289269051321` / `0.07503888024883364`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5928777670837344`.
- Accepted count before / after: `3120` / `3117`.
- Coverage before / after: `0.40435458786936235` / `0.40396578538102645`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `2849856`.
- Interpretation: stronger DART feature subsampling keeps the same threshold region but slightly lowers accepted accuracy and objective score. Keep `colsample_bytree: 0.35`.
- Next step: test the opposite DART direction only if warranted, or switch to CatBoost regularization around the current best.

## 20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020

- Skill used: `tabular-lgbm-dart-boosting`.
- Hypothesis: increasing nested LightGBM DART `colsample_bytree` from `0.35` to `0.45` may let the small LightGBM contribution capture useful cross-feature structure missing under stronger subsampling.
- Changed files: `experiments/configs/20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested DART `colsample_bytree: 0.45`, `reg_alpha: 1.2`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020 --config experiments/configs/20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter265_blend9770_dart_colsample045_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.18488840631152179`.
- Utility before / after: `0.07698289269051321` / `0.0750388802488336`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5927587311759052`.
- Accepted count before / after: `3120` / `3121`.
- Coverage before / after: `0.40435458786936235` / `0.40448418869880765`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `b4c3ec5`.
- Interpretation: increasing DART feature fraction does not help; both sides around `0.35` reduce accepted accuracy. Keep the current DART feature fraction.
- Next step: pivot to CatBoost regularization, which dominates the blend weight.

## 20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020

- Skill used: none specific; the local CatBoost skill is MultRMSE-focused and not applicable to this binary selection_score workflow.
- Hypothesis: increasing CatBoost `random_strength` from `2.0` to `3.0` may reduce split-selection overfit in the CatBoost-dominant blend and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `random_strength: 3.0`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020 --config experiments/configs/20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter266_blend9770_cat_random_strength3_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1674552378749784`.
- Utility before / after: `0.07698289269051321` / `0.06907724209434941`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5843621399176955`.
- Accepted count before / after: `3120` / `3159`.
- Coverage before / after: `0.40435458786936235` / `0.4094090202177294`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `52f985b`.
- Interpretation: stronger CatBoost randomization hurts accepted accuracy. Keep `random_strength: 2.0`.
- Next step: test lower CatBoost randomization (`random_strength: 1.0`) before leaving this branch.

## 20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020

- Skill used: none specific; the local CatBoost skill is MultRMSE-focused and not applicable to this binary selection_score workflow.
- Hypothesis: decreasing CatBoost `random_strength` from `2.0` to `1.0` may reduce underfitting from excessive split randomness and sharpen high-confidence accepted predictions.
- Changed files: `experiments/configs/20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `random_strength: 1.0`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020 --config experiments/configs/20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter267_blend9770_cat_random_strength1_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17354059301907032`.
- Utility before / after: `0.07698289269051321` / `0.07166925868325559`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5868131868131868`.
- Accepted count before / after: `3120` / `3185`.
- Coverage before / after: `0.40435458786936235` / `0.4127786417833074`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `169ed36`.
- Interpretation: lower CatBoost randomization also hurts accepted accuracy. Keep `random_strength: 2.0`.
- Next step: inspect probability and feature diagnostics before selecting the next focused iteration.

## 20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020

- Skill used: none specific; this is an existing sample-weighting configuration experiment guided by boundary-slice diagnostics.
- Hypothesis: lowering `sample_weighting.min_weight` from `0.35` to `0.20` may reduce the influence of ambiguous low-absolute-return rows, where slice diagnostics show much weaker accepted accuracy, without changing the fixed label rule.
- Changed files: `experiments/configs/20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, model params unchanged, `sample_weighting.min_weight: 0.20`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020 --config experiments/configs/20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter268_blend9770_sample_min_weight020_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.19027803605274402`.
- Utility before / after: `0.07698289269051321` / `0.07698289269051321`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5951923076923077`.
- Accepted count before / after: `3120` / `3120`.
- Coverage before / after: `0.40435458786936235` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `bf8e94e`.
- Interpretation: exact tie to the current best; either this change does not affect the active model path materially, or it preserves identical threshold-selected predictions. Keep the simpler current best config.
- Next step: inspect whether active training consumes sample weights before spending more iterations on sample-weighting brackets.

## 20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020

- Skill used: none specific; this is an existing sample-weighting/data-processing experiment guided by boundary-slice diagnostics.
- Hypothesis: recomputing cached split `stage1_sample_weight` with `sample_weighting.min_weight: 0.20` may make the config change from iter268 effective and improve accepted accuracy by downweighting ambiguous small-return rows.
- Changed files: `experiments/configs/20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter269_reweighted_min_weight020_split`.
- Feature set: 516 features; HTF/time features retained; `stage1_sample_weight` recomputed from existing `abs_return` with no label or feature changes.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, model params unchanged, `sample_weighting.min_weight: 0.20`, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter269_reweighted_min_weight020_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020 --config experiments/configs/20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter269_blend9770_reweighted_min_weight020_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16713162470307982`.
- Utility before / after: `0.07698289269051321` / `0.06933644375324002`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5838295205264807`.
- Accepted count before / after: `3120` / `3191`.
- Coverage before / after: `0.40435458786936235` / `0.41355624675997926`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; reweight summary saved at `artifacts/data_v2/experiments/20260509_codex_iter269_reweighted_min_weight020_split/reweight_summary.json`.
- Git commit: `0047ec7`.
- Interpretation: making the lower sample-weight floor effective reduces accepted accuracy and objective score. Keep the existing cached weights.
- Next step: stop sample-weight-floor bracketing and return to feature/model diagnostics.

## 20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020

- Skill used: CatBoost parameter discipline adapted from earlier CatBoost-focused experiments; no exact binary CatBoost skill is available locally.
- Hypothesis: switching the active nested CatBoost bootstrap from Bayesian `bagging_temperature: 0.5` to `bootstrap_type: MVS` with `subsample: 0.8` may regularize sample usage without changing features, labels, or thresholds.
- Changed files: `experiments/configs/20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `bootstrap_type: MVS`, `subsample: 0.8`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020 --config experiments/configs/20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter270_blend9770_cat_mvs_subsample08_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.19027803605274402`.
- Utility before / after: `0.07698289269051321` / `0.07698289269051321`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5951923076923077`.
- Accepted count before / after: `3120` / `3120`.
- Coverage before / after: `0.40435458786936235` / `0.40435458786936235`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `e544a57`.
- Interpretation: exact tie to the current best; MVS bootstrap does not change the selected validation predictions. Keep the simpler current best Bayesian bootstrap setting.
- Next step: look for changes that affect ranking enough to move the accepted set, not exact-tie CatBoost bootstrap variants.

## 20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020

- Skill used: none specific; CatBoost capacity bracket in the existing binary model path.
- Hypothesis: reducing nested CatBoost `depth` from `5` to `4` may lower interaction overfit and improve accepted accuracy on the validation accepted set.
- Changed files: `experiments/configs/20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `depth: 4`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020 --config experiments/configs/20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter271_blend9770_cat_depth4_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1683884155266234`.
- Utility before / after: `0.07698289269051321` / `0.07490927941938832`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5795704845814978`.
- Accepted count before / after: `3120` / `3632`.
- Coverage before / after: `0.40435458786936235` / `0.4707102125453603`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `cea29fa`.
- Interpretation: shallower CatBoost over-broadens the accepted set and reduces accepted accuracy. Keep `depth: 5`.
- Next step: test the paired higher-capacity side (`depth: 6`) only if it has not already been ruled out for the current blend.

## 20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020

- Skill used: none specific; CatBoost capacity bracket in the existing binary model path.
- Hypothesis: increasing nested CatBoost `depth` from `5` to `6` may capture higher-order interactions among HTF, time, and microstructure features and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `depth: 6`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020 --config experiments/configs/20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter272_blend9770_cat_depth6_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1679226611836875`.
- Utility before / after: `0.07698289269051321` / `0.07309486780715396`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.580848623853211`.
- Accepted count before / after: `3120` / `3488`.
- Coverage before / after: `0.40435458786936235` / `0.45204769310523585`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `fde54ed`.
- Interpretation: deeper CatBoost also broadens accepted predictions and lowers accepted accuracy. Keep `depth: 5`.
- Next step: avoid CatBoost depth changes around this blend; focus on feature/data diagnostics or a different model family knob.

## 20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020

- Skill used: none specific; CatBoost regularization bracket in the existing binary model path.
- Hypothesis: increasing nested CatBoost `l2_leaf_reg` from `30.0` to `32.0` may slightly regularize noisy microstructure splits and improve accepted accuracy while retaining the current feature set.
- Changed files: `experiments/configs/20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split`.
- Feature set: 516 features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost `l2_leaf_reg: 32.0`, nested DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260508_codex_iter43_train75_drop_sl_vwap_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020 --config experiments/configs/20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter273_blend9770_cat_l2_32_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1741944168646285`.
- Utility before / after: `0.07698289269051321` / `0.0728356661482633`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5861963190184049`.
- Accepted count before / after: `3120` / `3260`.
- Coverage before / after: `0.40435458786936235` / `0.4224987039917055`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `1943396`.
- Interpretation: stronger CatBoost L2 above 30 lowers accepted accuracy and score. Keep `l2_leaf_reg: 30.0` as the active setting.
- Next step: avoid further simple CatBoost capacity/regularization brackets unless a new diagnostic points to one.

## 20260509_codex_iter274_blend9770_session_z2_platt_logit_c020

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: session-normalized `rv_5` and `volume` z-score features may preserve time-of-day context while making unusually high/low activity easier for the model to split on.
- Changed files: `experiments/configs/20260509_codex_iter274_blend9770_session_z2_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_split`.
- Feature set: 518 features; current best 516 features plus `rv_5_session_z` and `volume_session_z`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter274_blend9770_session_z2_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter67_session_z2_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter274_blend9770_session_z2_platt_logit_c020 --config experiments/configs/20260509_codex_iter274_blend9770_session_z2_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter274_blend9770_session_z2_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16858359072530124`.
- Utility before / after: `0.07698289269051321` / `0.06959564541213066`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.584780549415851`.
- Accepted count before / after: `3120` / `3167`.
- Coverage before / after: `0.40435458786936235` / `0.41044582685329184`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `3c2458f`.
- Interpretation: session z-score features lower accepted accuracy and objective score. Do not keep this feature pack.
- Next step: test volume-session relative magnitude features, then avoid this session-normalization branch if it also fails.

## 20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: volume deviation from session norms (`diff`, `ratio`, `z`) may capture unusual activity regimes more directly than raw volume and cyclic time features.
- Changed files: `experiments/configs/20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_split`.
- Feature set: 519 features; current best 516 features plus `volume_session_diff`, `volume_session_ratio`, and `volume_session_z`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter68_volume_session_relative_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020 --config experiments/configs/20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter275_blend9770_volume_session_relative_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17453815206757506`.
- Utility before / after: `0.07698289269051321` / `0.07166925868325555`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5876386687797147`.
- Accepted count before / after: `3120` / `3155`.
- Coverage before / after: `0.40435458786936235` / `0.40889061689994816`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `ef5ede4`.
- Interpretation: volume-session relative features remain below the best; they broaden coverage slightly but lower accepted accuracy. Do not keep this feature pack.
- Next step: leave session-relative feature expansion and inspect other already-built richer splits or data windows.

## 20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020

- Skill used: `tabular-season-phase-labeling` style discretization applied to volatility/volume regimes.
- Hypothesis: compact low/mid/high regime flags for `rv_5` and volume may expose regime context to the global model without the overfit seen from separate regime-specific models.
- Changed files: `experiments/configs/20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_split`.
- Feature set: 522 features; current best 516 features plus six `rv5_regime_*` and `volume_regime_*` flags; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter64_regime_flags_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020 --config experiments/configs/20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter276_blend9770_regime_flags_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17453278350968812`.
- Utility before / after: `0.07698289269051321` / `0.07089165370658372`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5884254768832848`.
- Accepted count before / after: `3120` / `3093`.
- Coverage before / after: `0.40435458786936235` / `0.40085536547433903`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `0d96285`.
- Interpretation: compact regime flags do not recover the objective and leave coverage close to the floor. Do not keep this feature pack.
- Next step: inspect remaining untested cached richer splits before creating new feature data.

## 20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020

- Skill used: `tabular-polynomial-interaction-features`.
- Hypothesis: explicit interactions between HTF context and top second-level microstructure features may capture conditional short-horizon behavior that separate raw features miss.
- Changed files: `experiments/configs/20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_split`.
- Feature set: 548 features; current best 516 features plus 32 HTF-by-microstructure interaction features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter84_htf_micro_interactions_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020 --config experiments/configs/20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter277_blend9770_htf_micro_interactions_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16557707314813985`.
- Utility before / after: `0.07698289269051321` / `0.06920684292379474`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5826625386996904`.
- Accepted count before / after: `3120` / `3230`.
- Coverage before / after: `0.40435458786936235` / `0.4186106791083463`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `60d31f5`.
- Interpretation: broad HTF-micro interaction expansion lowers accepted accuracy and objective score. Do not keep this feature pack.
- Next step: use narrower feature selection or smaller targeted additions rather than broad interaction expansion.

## 20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020

- Skill used: `tabular-polynomial-interaction-features`.
- Hypothesis: a much smaller set of `htf_range_pos_15m` interactions with the top four microstructure features may capture useful conditional context without the noise of the 32-feature interaction pack.
- Changed files: `experiments/configs/20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter278_htf_range_top4_micro_interactions_split`.
- Feature set: 520 features; current best 516 features plus four `htf_range_pos_15m` by microstructure interaction features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter278_htf_range_top4_micro_interactions_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020 --config experiments/configs/20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter278_blend9770_htf_range_top4_micro_interactions_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17322282026027705`.
- Utility before / after: `0.07698289269051321` / `0.07944530844997404`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5794247214304223`.
- Accepted count before / after: `3120` / `3859`.
- Coverage before / after: `0.40435458786936235` / `0.5001296008294454`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; interaction summary saved at `artifacts/data_v2/experiments/20260509_codex_iter278_htf_range_top4_micro_interactions_split/interaction_summary.json`.
- Git commit: `9d7c5ce`.
- Interpretation: compact interactions increase utility and coverage but reduce accepted accuracy enough to lower selection_score. Do not keep this feature pack.
- Next step: avoid further direct interaction additions unless paired with feature selection or a better acceptance-sharpening model.

## 20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020

- Skill used: `tabular-row-aggregate-features`.
- Hypothesis: row-wise aggregate summaries across the top 40 importance-normalized features may capture broad market-state intensity that individual features miss.
- Changed files: `experiments/configs/20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_split`.
- Feature set: 520 features; current best 516 features plus `top40_imp_z_mean`, `top40_imp_z_std`, `top40_imp_z_min`, and `top40_imp_z_max`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter71_top_importance_rowagg_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020 --config experiments/configs/20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter279_blend9770_top40_rowagg_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1657742059251654`.
- Utility before / after: `0.07698289269051321` / `0.07413167444271646`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5781848004373975`.
- Accepted count before / after: `3120` / `3658`.
- Coverage before / after: `0.40435458786936235` / `0.4740798341109383`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `9be7bcd`.
- Interpretation: top-feature row aggregates broaden acceptance and reduce accepted accuracy. Do not keep this feature pack.
- Next step: compact additions continue to dilute precision; prioritize feature filtering or alternative data windows over more broad row aggregates.

## 20260509_codex_iter280_blend9770_collinear98_platt_logit_c020

- Skill used: `tabular-collinear-feature-removal`.
- Hypothesis: removing highly collinear features at a conservative threshold may reduce redundant noisy splits and improve accepted accuracy while keeping the core HTF/time feature families.
- Changed files: `experiments/configs/20260509_codex_iter280_blend9770_collinear98_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_split`.
- Feature set: 496 features; 20 redundant features removed from the current best split; HTF/time families retained though some derived HTF columns were dropped by the collinearity filter.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter280_blend9770_collinear98_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter193_collinear98_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter280_blend9770_collinear98_platt_logit_c020 --config experiments/configs/20260509_codex_iter280_blend9770_collinear98_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter280_blend9770_collinear98_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16979350257159054`.
- Utility before / after: `0.07698289269051321` / `0.0698548470710213`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.585528403681371`.
- Accepted count before / after: `3120` / `3151`.
- Coverage before / after: `0.40435458786936235` / `0.4083722135821669`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `b815328`.
- Interpretation: collinearity pruning removes signal that the blend still uses and lowers accepted accuracy. Do not keep this feature filter.
- Next step: test gentler importance-based pruning rather than correlation-only pruning.

## 20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020

- Skill used: `tabular-recursive-feature-elimination` as feature-selection discipline.
- Hypothesis: removing only five low-importance non-HTF/time features may reduce noise without damaging the protected HTF/time context.
- Changed files: `experiments/configs/20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_split`.
- Feature set: 511 features; five low-importance non-HTF/time features removed; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter109_drop_bottom5_non_htf_time_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020 --config experiments/configs/20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter281_blend9770_drop_bottom5_non_htf_time_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17227085050690963`.
- Utility before / after: `0.07698289269051321` / `0.07205806117159153`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5853808353808354`.
- Accepted count before / after: `3120` / `3256`.
- Coverage before / after: `0.40435458786936235` / `0.42198030067392434`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `d7daf21`.
- Interpretation: even gentle low-importance pruning lowers accepted accuracy. Keep the full current best 516-feature set.
- Next step: inspect remaining available more-data splits under the current blend.

## 20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020

- Skill used: `tabular-cyclical-feature-encoding`.
- Hypothesis: weekday cyclical encoding and weekend flag may add longer calendar context beyond hour/minute cyclic features while preserving online availability.
- Changed files: `experiments/configs/20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_split`.
- Feature set: 519 features; current best 516 features plus `weekday_sin`, `weekday_cos`, and `is_weekend`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter202_weekday_features_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020 --config experiments/configs/20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter282_blend9770_weekday_features_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16377147021625257`.
- Utility before / after: `0.07698289269051321` / `0.06829963711767756`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.582061663033323`.
- Accepted count before / after: `3120` / `3211`.
- Coverage before / after: `0.40435458786936235` / `0.41614826334888544`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `1010b7c`.
- Interpretation: weekday features lower accepted accuracy and score. Keep the existing hour/minute cyclic time feature set without weekday additions.
- Next step: avoid more calendar expansions unless a time-slice diagnostic shows a stable weekday effect.

## 20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020

- Skill used: `timeseries-multi-scale-rolling-features`.
- Hypothesis: past-only trailing return, absolute-return, and range features over 3/5/10 bars may add short path context aligned with the 5-minute forecast horizon.
- Changed files: `experiments/configs/20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_split`.
- Feature set: 525 features; current best 516 features plus nine trailing path features; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter209_trailing_path_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020 --config experiments/configs/20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter283_blend9770_trailing_path_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16458683811756675`.
- Utility before / after: `0.07698289269051321` / `0.07244686365992747`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5787545787545788`.
- Accepted count before / after: `3120` / `3549`.
- Coverage before / after: `0.40435458786936235` / `0.4599533437013997`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `c6ca109`.
- Interpretation: trailing path features broaden acceptance and lower accepted accuracy. Do not keep this feature pack.
- Next step: reassess remaining branches; added feature packs are consistently diluting precision under the current blend.

## 20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020

- Skill used: data-weighting/recency discipline; no exact local recency-weighting skill applies to this supervised classification path.
- Hypothesis: recency-decayed sample weights over the same 75-day window may improve adaptation to the current validation regime without adding noisy features or changing labels.
- Changed files: `experiments/configs/20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; sample weights are recency-decayed in the cached split.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter65_recent75_decay_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020 --config experiments/configs/20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter284_blend9770_recent75_decay_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1628970060693794`.
- Utility before / after: `0.07698289269051321` / `0.07516848107827885`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5750129332643559`.
- Accepted count before / after: `3120` / `3866`.
- Coverage before / after: `0.40435458786936235` / `0.5010368066355625`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `6dc9a61`.
- Interpretation: recency decay greatly broadens acceptance and lowers accepted accuracy. Keep the existing sample weights.
- Next step: stop recency-decay weighting; look for methods that improve ranking/precision instead of coverage.

## 20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020

- Skill used: data-weighting discipline; the available `tabular-time-varying-reward-shaping` skill was inspected but is RL-specific, so this uses the existing supervised regime-weighted split instead.
- Hypothesis: downweighting high-volatility regime training rows may reduce influence from the weakest validation slice and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights downweight high-volatility regime rows.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter77_high_regime_downweight_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020 --config experiments/configs/20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter285_blend9770_high_regime_downweight_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1759591382618237`.
- Utility before / after: `0.07698289269051321` / `0.0724468636599274`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5880314960629921`.
- Accepted count before / after: `3120` / `3175`.
- Coverage before / after: `0.40435458786936235` / `0.41148263348885433`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `47a3b53`.
- Interpretation: high-regime downweighting improves YES/NO balance but lowers accepted accuracy and selection_score. Do not keep this weighting.
- Next step: test the paired mid-regime upweight split.

## 20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020

- Skill used: data-weighting discipline; supervised regime weighting based on validation slice diagnostics.
- Hypothesis: upweighting mid-volatility regime rows may emphasize the slice with the strongest validation selection_score and improve the accepted region.
- Changed files: `experiments/configs/20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights upweight mid-volatility regime rows.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter78_mid_regime_upweight_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020 --config experiments/configs/20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter286_blend9770_mid_regime_upweight_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1676909097701595`.
- Utility before / after: `0.07698289269051321` / `0.06829963711767752`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5853579527048914`.
- Accepted count before / after: `3120` / `3087`.
- Coverage before / after: `0.40435458786936235` / `0.4000777604976672`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `91d671d`.
- Interpretation: mid-regime upweighting lowers both utility and accepted accuracy and nearly falls to the coverage floor. Do not keep this weighting.
- Next step: avoid regime-weighted sample variants under this blend.

## 20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020

- Skill used: sample-weighting/data-processing discipline.
- Hypothesis: softened sample weights may reduce overemphasis on larger-return rows and improve probability ranking across the accepted set.
- Changed files: `experiments/configs/20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights softened relative to the incumbent.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter82_softened_weights_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020 --config experiments/configs/20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter287_blend9770_softened_weights_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1651488294929241`.
- Utility before / after: `0.07698289269051321` / `0.06920684292379468`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5823057953144266`.
- Accepted count before / after: `3120` / `3244`.
- Coverage before / after: `0.40435458786936235` / `0.4204250907205806`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `2549d14`.
- Interpretation: softened weights reduce accepted accuracy and score. Do not keep this weighting.
- Next step: test the stronger squared-weight variant, then stop global weight-shape variants if it fails.

## 20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020

- Skill used: sample-weighting/data-processing discipline.
- Hypothesis: squared sample weights may emphasize cleaner larger-return examples and sharpen accepted predictions more than the incumbent linear weighting.
- Changed files: `experiments/configs/20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights square the return-ramp emphasis.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter83_squared_weights_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020 --config experiments/configs/20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter288_blend9770_squared_weights_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1730526276208296`.
- Utility before / after: `0.07698289269051321` / `0.07270606531881804`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5853881278538813`.
- Accepted count before / after: `3120` / `3285`.
- Coverage before / after: `0.40435458786936235` / `0.42573872472783825`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `68272ef`.
- Interpretation: squared weights still lower accepted accuracy and score. Keep the incumbent linear return-ramp sample weighting.
- Next step: move away from global sample-weight shape variants.

## 20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020

- Skill used: data-filtering discipline; anomaly/low-signal filtering inspired by `tabular-anomaly-flag-imputation` inspection but applied as an existing supervised split rather than sentinel imputation.
- Hypothesis: training only on clearer `abs_return >= 5bp` examples may improve accepted accuracy while keeping validation unchanged.
- Changed files: `experiments/configs/20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; development rows filtered to `abs_return >= 0.0005`.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter104_train_abs_return_ge5bp_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020 --config experiments/configs/20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter289_blend9770_train_abs_return_ge5bp_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17444922153409623`.
- Utility before / after: `0.07698289269051321` / `0.07763089683773977`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5819425444596443`.
- Accepted count before / after: `3120` / `3655`.
- Coverage before / after: `0.40435458786936235` / `0.4736910316226024`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `e750ddf`.
- Interpretation: filtering to clearer high-return training rows increases utility and coverage but lowers accepted accuracy. Do not keep this filtered training split.
- Next step: if continuing return-filter branches, test less aggressive existing low/tiny-return filters only once under the current blend.

## 20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020

- Skill used: data-filtering discipline for low-signal sample removal.
- Hypothesis: dropping only tiny sub-0.5bp training examples may remove noisy labels while preserving most development data and all current best features.
- Changed files: `experiments/configs/20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; development rows filtered to remove tiny absolute returns.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter95_drop_tiny_abs_return_train_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020 --config experiments/configs/20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter290_blend9770_drop_tiny_abs_return_train_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17364837860321397`.
- Utility before / after: `0.07698289269051321` / `0.07089165370658375`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5876883616543764`.
- Accepted count before / after: `3120` / `3119`.
- Coverage before / after: `0.40435458786936235` / `0.40422498703991705`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `0bc4214`.
- Interpretation: less aggressive tiny-return filtering still lowers accepted accuracy and score. Keep the full development set.
- Next step: avoid further simple return-filter branches unless paired with a materially different model.

## 20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020

- Skill used: `tabular-relative-deviation-features`.
- Hypothesis: replacing selected raw scale-sensitive features with session-relative diff/ratio/z variants may reduce intraday scale drift better than simply adding relative features.
- Changed files: `experiments/configs/20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_split`.
- Feature set: 527 features; added 15 session-relative features and dropped 4 raw base columns; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter72_relative_drop_bases_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020 --config experiments/configs/20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter291_blend9770_relative_drop_bases_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.1717958744062075`.
- Utility before / after: `0.07698289269051321` / `0.07270606531881803`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5843609022556391`.
- Accepted count before / after: `3120` / `3325`.
- Coverage before / after: `0.40435458786936235` / `0.4309227579056506`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `adfac87`.
- Interpretation: replacing raw bases with session-relative variants also lowers accepted accuracy. Do not keep this feature pack.
- Next step: avoid further session-relative feature variants under this split.

## 20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020

- Skill used: sample-weighting/data-processing control.
- Hypothesis: removing sample weights may reduce weighting-induced probability distortion and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights are all `1.0`.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter81_unweighted_best_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020 --config experiments/configs/20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter292_blend9770_unweighted_best_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16351971576373614`.
- Utility before / after: `0.07698289269051321` / `0.07257646448937267`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5777777777777777`.
- Accepted count before / after: `3120` / `3600`.
- Coverage before / after: `0.40435458786936235` / `0.4665629860031104`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `ea549a1`.
- Interpretation: unweighted training broadens acceptance and reduces accepted accuracy. Keep the incumbent return-ramp weighting.
- Next step: stop simple sample-weight variants.

## 20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020

- Skill used: `tabular-oof-meta-features`.
- Hypothesis: an expanding-window OOF logistic probability feature may add a leak-free learned linear summary that complements the CatBoost/LGBM blend.
- Changed files: `experiments/configs/20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter293_oof_logistic_meta_split`.
- Feature set: 517 features; current best 516 features plus `meta_oof_logistic_p_up_ts5`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter293_oof_logistic_meta_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020 --config experiments/configs/20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter293_blend9770_oof_logistic_meta_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17489598797968187`.
- Utility before / after: `0.07698289269051321` / `0.07153965785381024`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5880663688576898`.
- Accepted count before / after: `3120` / `3134`.
- Coverage before / after: `0.40435458786936235` / `0.4061689994815967`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; OOF generation summary saved at `artifacts/data_v2/experiments/20260509_codex_iter293_oof_logistic_meta_split/oof_logistic_meta_summary.json`.
- Git commit: `8466538`.
- Interpretation: the OOF logistic meta-feature does not improve ranking and lowers selection_score. Do not keep this meta-feature.
- Next step: avoid simple OOF logistic stacking unless paired with a materially different auxiliary target.

## 20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020

- Skill used: `tabular-oof-meta-features`.
- Hypothesis: a small expanding-window LightGBM OOF probability feature may provide a nonlinear learned summary that complements the final CatBoost/LGBM blend.
- Changed files: `experiments/configs/20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter294_oof_lgbm_meta_split`.
- Feature set: 517 features; current best 516 features plus `meta_oof_lgbm_p_up_ts5`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter294_oof_lgbm_meta_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020 --config experiments/configs/20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter294_blend9770_oof_lgbm_meta_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.16367834549920837`.
- Utility before / after: `0.07698289269051321` / `0.06881804043545879`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.581466707579012`.
- Accepted count before / after: `3120` / `3259`.
- Coverage before / after: `0.40435458786936235` / `0.42236910316226023`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; OOF generation summary saved at `artifacts/data_v2/experiments/20260509_codex_iter294_oof_lgbm_meta_split/oof_lgbm_meta_summary.json`.
- Git commit: `b6f6da9`.
- Interpretation: the nonlinear OOF LightGBM meta-feature lowers accepted accuracy and score. Do not keep this meta-feature.
- Next step: stop simple OOF meta-feature stacking.

## 20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020

- Skill used: `tabular-oof-meta-features`.
- Hypothesis: an expanding-window OOF LightGBM activity feature predicting `abs_return >= 5bp` may help the final direction model identify higher-quality movement regimes without using validation labels.
- Changed files: `experiments/configs/20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter295_oof_lgbm_activity_meta_split`.
- Feature set: 517 features; current best 516 features plus `meta_oof_lgbm_absret_ge5bp_ts5`; HTF/time features retained.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter295_oof_lgbm_activity_meta_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020 --config experiments/configs/20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter295_blend9770_oof_lgbm_activity_meta_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17335121335477854`.
- Utility before / after: `0.07698289269051321` / `0.07426127527216177`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5841409691629956`.
- Accepted count before / after: `3120` / `3405`.
- Coverage before / after: `0.40435458786936235` / `0.44129082426127525`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; OOF generation summary saved at `artifacts/data_v2/experiments/20260509_codex_iter295_oof_lgbm_activity_meta_split/oof_lgbm_activity_meta_summary.json`.
- Git commit: `d1a34e6`.
- Interpretation: predicted activity increases coverage and utility but lowers accepted accuracy enough to reduce selection_score. Do not keep this meta-feature.
- Next step: stop OOF meta-feature variants unless an auxiliary target can be implemented into the online artifact path.

## 20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020

- Skill used: targeted sample-weight data processing based on false-slice diagnostics.
- Hypothesis: conservatively downweighting training rows in weak validation regimes (high `rv_5`, low volume, Asia session) may reduce false accepted predictions and improve accepted accuracy.
- Changed files: `experiments/configs/20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter296_weak_regime_downweight_split`.
- Feature set: 516 features; same as current best; HTF/time features retained; cached sample weights apply `0.85` factors for high-volatility, low-volume, and Asia-session rows.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter296_weak_regime_downweight_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020 --config experiments/configs/20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter296_blend9770_weak_regime_downweight_platt_logit_c020/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17300964556562526`.
- Utility before / after: `0.07698289269051321` / `0.07620528771384133`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5820770519262981`.
- Accepted count before / after: `3120` / `3582`.
- Coverage before / after: `0.40435458786936235` / `0.4642301710730949`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training; weight summary saved at `artifacts/data_v2/experiments/20260509_codex_iter296_weak_regime_downweight_split/weak_regime_downweight_summary.json`.
- Git commit: `830157d`.
- Interpretation: combined weak-regime downweighting still broadens acceptance and lowers accepted accuracy. Do not keep this weighting.
- Next step: avoid further weak-regime weighting without a stronger model-side precision mechanism.

## 20260509_codex_iter297_drop_bottom10_importance_current_blend

- Skill used: `tabular-recursive-feature-elimination`.
- Hypothesis: removing the 10 lowest-importance non-protected features may reduce noise while retaining the HTF and cyclical time context required by the current objective.
- Changed files: `experiments/configs/20260509_codex_iter297_drop_bottom10_importance_current_blend.yaml`, `experiments/optimization_log.md`.
- Cached split: `artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_split`.
- Feature set: 506 features; HTF/time features retained; lowest-importance feature subset removed by the cached split.
- Model settings: current best logit blend with `catboost_weight: 0.9770`, nested CatBoost/DART unchanged, `calibration.active_plugin: platt_logit`, `C: 0.2`.
- Config: `experiments/configs/20260509_codex_iter297_drop_bottom10_importance_current_blend.yaml`.
- Evaluation command: `rtk python scripts/model/train_model.py --cached-split-dir artifacts/data_v2/experiments/20260509_codex_iter86_drop_bottom10_importance_split --output-dir artifacts/data_v2/experiments/20260509_codex_iter297_drop_bottom10_importance_current_blend --config experiments/configs/20260509_codex_iter297_drop_bottom10_importance_current_blend.yaml --horizon 5m --train-window-days 75 --validation-window-days 30`.
- Evaluation report: `artifacts/data_v2/experiments/20260509_codex_iter297_drop_bottom10_importance_current_blend/metrics.json`.
- Score before: `0.19027803605274402`.
- Score after: `0.17864711463061556`.
- Utility before / after: `0.07698289269051321` / `0.07400207361327114`.
- Accepted accuracy before / after: `0.5951923076923077` / `0.5886921404162784`.
- Accepted count before / after: `3120` / `3219`.
- Coverage before / after: `0.40435458786936235` / `0.4171850699844479`.
- Coverage constraint satisfied: yes.
- Tests: DQC ran during training.
- Git commit: `pending`.
- Interpretation: the smaller feature set lowers accepted accuracy and selection_score despite satisfying coverage. Do not keep this split.
- Next step: test neighboring bottom-importance pruning only if it may locate a narrower noise-removal band; otherwise pivot back to model regularization.
