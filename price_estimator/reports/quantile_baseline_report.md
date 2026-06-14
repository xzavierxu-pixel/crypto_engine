# Price Estimator Quantile Baseline

experiment_id: `20260612_catboost_quantile_winner_price_baseline`
prediction_path: `C:\Users\ROG\Desktop\crypto_engine_version1\price_estimator\data\predictions_validation.parquet`

## Validation Metrics

| metric | value |
|---|---:|
| sample_count | 7432.00000000 |
| coverage_q70 | 0.65365985 |
| pinball_q70 | 0.05211771 |
| mean_pred_q70 | 0.50067811 |
| coverage_q80 | 0.74569429 |
| pinball_q80 | 0.03992610 |
| mean_pred_q80 | 0.53234015 |
| coverage_q90 | 0.83705597 |
| pinball_q90 | 0.02446991 |
| mean_pred_q90 | 0.57643604 |
| crossing_rate_raw | 0.01358988 |
| crossing_rate_postprocessed | 0.00000000 |

## Conditional Metrics

- selected_side=DOWN: n=3597, cov70=0.6564, cov80=0.7459, cov90=0.8440
- selected_side=UP: n=3835, cov70=0.6511, cov80=0.7455, cov90=0.8305
- p_bin=0.50_0.55: n=760, cov70=0.6671, cov80=0.7566, cov90=0.8237
- p_bin=0.55_0.60: n=708, cov70=0.6540, cov80=0.7415, cov90=0.8489
- p_bin=0.60_0.65: n=678, cov70=0.6475, cov80=0.7227, cov90=0.8407
- p_bin=0.65_0.70: n=657, cov70=0.6469, cov80=0.7397, cov90=0.8204
- p_bin=0.70_1.00: n=1762, cov70=0.5863, cov80=0.6833, cov90=0.7900
- p_bin=missing: n=2867, cov70=0.6945, cov80=0.7890, cov90=0.8696
- p_side_bucket=0.50_0.55: n=760, cov70=0.6671, cov80=0.7566, cov90=0.8237
- p_side_bucket=0.55_0.60: n=708, cov70=0.6540, cov80=0.7415, cov90=0.8489
- p_side_bucket=0.60_0.65: n=678, cov70=0.6475, cov80=0.7227, cov90=0.8407
- p_side_bucket=0.65_0.70: n=657, cov70=0.6469, cov80=0.7397, cov90=0.8204
- p_side_bucket=0.70_1.00: n=1762, cov70=0.5863, cov80=0.6833, cov90=0.7900
- p_side_bucket=missing: n=2867, cov70=0.6945, cov80=0.7890, cov90=0.8696
- time_to_lowest_trade_sec_bucket=000_060: n=4050, cov70=0.4553, cov80=0.5901, cov90=0.7311
- time_to_lowest_trade_sec_bucket=060_120: n=1322, cov70=0.8510, cov80=0.8971, cov90=0.9387
- time_to_lowest_trade_sec_bucket=120_180: n=918, cov70=0.8943, cov80=0.9412, cov90=0.9815
- time_to_lowest_trade_sec_bucket=180_240: n=1142, cov70=0.9352, cov80=0.9650, cov90=0.9790
- market_time_bucket=asia: n=2490, cov70=0.6566, cov80=0.7502, cov90=0.8305
- market_time_bucket=europe: n=2422, cov70=0.6462, cov80=0.7370, cov90=0.8386
- market_time_bucket=us: n=2520, cov70=0.6579, cov80=0.7496, cov90=0.8421
