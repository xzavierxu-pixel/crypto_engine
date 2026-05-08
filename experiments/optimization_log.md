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
