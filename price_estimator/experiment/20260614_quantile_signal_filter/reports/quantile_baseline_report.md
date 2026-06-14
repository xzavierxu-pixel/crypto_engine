# Price Estimator Quantile Baseline

experiment_id: `20260614_quantile_signal_filter`
prediction_path: `C:\Users\ROG\Desktop\crypto_engine_version1\price_estimator\experiment\20260614_quantile_signal_filter\reports\predictions_validation.parquet`

## Validation Metrics

| metric | value |
|---|---:|
| sample_count | 5184.00000000 |
| coverage_q70 | 0.62249228 |
| pinball_q70 | 0.05237268 |
| mean_pred_q70 | 0.50872082 |
| mean_gap_q70 | 0.09053698 |
| median_gap_q70 | 0.05972749 |
| gap_p05_q70 | -0.12854250 |
| gap_p95_q70 | 0.39200871 |
| gap_p95_p05_range_q70 | 0.52055122 |
| p99_violation_q70 | 0.21199884 |
| max_violation_q70 | 0.37731877 |
| coverage_q80 | 0.71701389 |
| pinball_q80 | 0.03978334 |
| mean_pred_q80 | 0.54272513 |
| mean_gap_q80 | 0.12454129 |
| median_gap_q80 | 0.09394636 |
| gap_p05_q80 | -0.09108160 |
| gap_p95_q80 | 0.42730209 |
| gap_p95_p05_range_q80 | 0.51838369 |
| p99_violation_q80 | 0.17427872 |
| max_violation_q80 | 0.34179218 |
| coverage_q90 | 0.80922068 |
| pinball_q90 | 0.02463434 |
| mean_pred_q90 | 0.58517484 |
| mean_gap_q90 | 0.16699101 |
| median_gap_q90 | 0.14043745 |
| gap_p05_q90 | -0.05909452 |
| gap_p95_q90 | 0.46734911 |
| gap_p95_p05_range_q90 | 0.52644362 |
| p99_violation_q90 | 0.13345366 |
| max_violation_q90 | 0.30120924 |
| crossing_rate_raw | 0.02854938 |
| crossing_rate_postprocessed | 0.00000000 |

## Conditional Metrics

- selected_side=DOWN: n=2466, cov70=0.6480, cov80=0.7441, cov90=0.8378
- selected_side=UP: n=2718, cov70=0.6424, cov80=0.7454, cov90=0.8256
- p_bin=0.55_0.60: n=362, cov70=0.6851, cov80=0.7818, cov90=0.8508
- p_bin=0.60_0.65: n=678, cov70=0.6534, cov80=0.7478, cov90=0.8289
- p_bin=0.65_0.70: n=657, cov70=0.6438, cov80=0.7473, cov90=0.8234
- p_bin=0.70_1.00: n=1762, cov70=0.5823, cov80=0.6805, cov90=0.7838
- p_bin=missing: n=1725, cov70=0.6980, cov80=0.8006, cov90=0.8800
- p_side_bucket=0.55_0.60: n=362, cov70=0.6851, cov80=0.7818, cov90=0.8508
- p_side_bucket=0.60_0.65: n=678, cov70=0.6534, cov80=0.7478, cov90=0.8289
- p_side_bucket=0.65_0.70: n=657, cov70=0.6438, cov80=0.7473, cov90=0.8234
- p_side_bucket=0.70_1.00: n=1762, cov70=0.5823, cov80=0.6805, cov90=0.7838
- p_side_bucket=missing: n=1725, cov70=0.6980, cov80=0.8006, cov90=0.8800
- time_to_lowest_trade_sec_bucket=000_060: n=2793, cov70=0.4397, cov80=0.5893, cov90=0.7211
- time_to_lowest_trade_sec_bucket=060_120: n=912, cov70=0.8344, cov80=0.8882, cov90=0.9342
- time_to_lowest_trade_sec_bucket=120_180: n=669, cov70=0.8849, cov80=0.9402, cov90=0.9761
- time_to_lowest_trade_sec_bucket=180_240: n=810, cov70=0.9420, cov80=0.9580, cov90=0.9765
- market_time_bucket=asia: n=1730, cov70=0.6457, cov80=0.7445, cov90=0.8254
- market_time_bucket=europe: n=1664, cov70=0.6328, cov80=0.7302, cov90=0.8287
- market_time_bucket=us: n=1790, cov70=0.6559, cov80=0.7587, cov90=0.8397
