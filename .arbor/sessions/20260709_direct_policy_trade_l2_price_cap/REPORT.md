# Research Report - Direct Policy Trade/L2 Price Cap 20260709

Implemented the V3 isolated Arbor session for 1:08 market-order direct policy. Phase 1 tested baseline, trade-path, L2, lag-1m, and combined research features. Phase 2 ran post-model cap and price-band diagnostics on the Phase 1 B_dev winner. No deploy manifest, live config, or `2mins` branch files were modified.

## Ledger

| experiment_id        | feature_family    |   B_dev |   B_test |   delta_vs137 |   accuracy |   avg_entry |        edge | leakage_passed   | notes                                             |       w1 |       w2 |       w3 |       w4 |       w5 |       w6 |
|:---------------------|:------------------|--------:|---------:|--------------:|-----------:|------------:|------------:|:-----------------|:--------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| F0_frame_qa          | qa                | nan     |  nan     |     nan       | nan        |  nan        | nan         | True             | resumed                                           | nan      | nan      | nan      | nan      | nan      | nan      |
| F1_baseline_replay   | baseline_replay   | 254.043 |  132.436 |      -4.64514 |   0.636439 |    0.618048 |   0.0183913 | True             | resumed; c=0.02; holdout_passed=True              |  62.0644 |  73.9604 |  52.6536 |  65.3642 |  67.0267 |  61.88   |
| F2_trade_path        | trade_path        | 267.873 |  142.343 |       5.26226 |   0.652912 |    0.63359  |   0.0193218 | True             | resumed; c=0.01; holdout_passed=True              |  67.8128 |  77.8729 |  55.4332 |  66.754  |  55.3593 |  49.9773 |
| F3_l2                | l2                | 216.657 |  114.771 |     -22.3098  |   0.639464 |    0.623931 |   0.0155327 | True             | resumed; c=0.01; holdout_passed=True              |  63.0575 |  66.6193 |  41.685  |  45.2955 |  49.8954 |  52.5662 |
| F4_lag1m_price       | lag1m_price       | 330.323 |  178.64  |      41.5588  |   0.655138 |    0.629501 |   0.0256372 | True             | resumed; c=0.00; holdout_passed=True              |  87.2406 |  88.4964 |  69.7875 |  84.7986 |  68.0173 |  74.2443 |
| F5_trade_l2_combined | trade_l2_combined | 267.851 |  150.624 |      13.5427  |   0.653147 |    0.632715 |   0.0204319 | True             | resumed; c=0.00; holdout_passed=True              |  68.5279 |  69.5248 |  62.6717 |  67.127  |  46.3449 |  50.4965 |
| C1_price_cap_grid    | price_cap         | nan     |  178.64  |      41.5588  |   0.655138 |    0.629501 |   0.0256372 | True             | selected_bdev_cap=1.00; holdout_passed=True       | nan      | nan      | nan      | nan      | nan      | nan      |
| C2_price_gate_band   | price_band        | nan     |  178.64  |      41.5588  |   0.655138 |    0.629501 |   0.0256372 | True             | selected_bdev_band=0.00-1.00; holdout_passed=True | nan      | nan      | nan      | nan      | nan      | nan      |
| D1_stable_universe   | diagnostic        | nan     |  nan     |     nan       | nan        |  nan        | nan         | True             | stable_universe_once                              | nan      | nan      | nan      | nan      | nan      | nan      |
