# aws-poly live last 36h submitted vs matched PnL analysis

## Scope

- UTC window: 2026-06-12T15:00:00+00:00 to 2026-06-14T03:15:00+00:00
- Local Beijing window: 2026-06-12 23:00 to 2026-06-13 10:00
- Server: aws-poly
- Data sources: live summaries, Gamma resolved outcomes, Polymarket CLOB get_trades matched fills

## Executive summary

| metric | value |
|---|---:|
| resolved cycles | 311 |
| accepted cycles | 311 |
| signal coverage | 100.00% |
| signal accuracy | 70.42% |
| submitted orders | 311 |
| submitted order IDs | 311 |
| matched order IDs | 246 |
| order fill rate by ID | 79.10% |
| submitted size | 1555.0000 |
| matched size | 1224.9984 |
| size fill rate | 78.78% |
| submitted replay accuracy | 70.42% |
| submitted replay PnL | +176.2500 |
| matched fill accuracy | 62.76% |
| matched fill avg price | 0.5834 |
| matched fill PnL | +50.4023 |
| matched fill ROI | 7.05% |

## Submitted orders by 0.1 limit-price bucket

| limit_price_bucket | orders | accuracy | submit_success_rate | requested_size | avg_limit_price | avg_best_ask_minus_limit | avg_confidence_minus_limit | submitted_replay_pnl | submitted_replay_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2-0.3 | 2 | 0.00% | 100.00% | 10.0000 | 0.2550 | 0.0100 | 0.3290 | -2.5500 | -100.00% |
| 0.3-0.4 | 6 | 33.33% | 100.00% | 30.0000 | 0.3800 | 0.0100 | 0.2358 | -1.4000 | -12.28% |
| 0.4-0.5 | 34 | 47.06% | 100.00% | 170.0000 | 0.4588 | 0.0141 | 0.1598 | +2.0000 | 2.56% |
| 0.5-0.6 | 139 | 68.35% | 100.00% | 695.0000 | 0.5458 | 0.0917 | 0.0834 | +95.7000 | 25.23% |
| 0.6-0.7 | 72 | 76.39% | 100.00% | 360.0000 | 0.6450 | 0.0636 | 0.0622 | +42.8000 | 18.43% |
| 0.7-0.8 | 51 | 86.27% | 100.00% | 255.0000 | 0.7335 | 0.0775 | 0.0511 | +32.9500 | 17.62% |
| 0.8-0.9 | 7 | 100.00% | 100.00% | 35.0000 | 0.8071 | 0.0914 | 0.0536 | +6.7500 | 23.89% |


## Matched fills by 0.1 fill-price bucket

| fill_price_bucket | fills | accuracy | shares | avg_fill_price | avg_best_ask_minus_limit | avg_confidence_minus_fill | pnl | roi_on_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2-0.3 | 2 | 0.00% | 10.0000 | 0.2550 | 0.0100 | 0.3290 | -2.5500 | -100.00% |
| 0.3-0.4 | 7 | 42.86% | 29.9903 | 0.3800 | 0.0100 | 0.2308 | -1.4059 | -12.34% |
| 0.4-0.5 | 41 | 39.02% | 159.9725 | 0.4609 | 0.0134 | 0.1581 | -13.7507 | -18.65% |
| 0.5-0.6 | 119 | 59.66% | 525.0724 | 0.5448 | 0.0794 | 0.0875 | +29.0529 | 10.16% |
| 0.6-0.7 | 72 | 69.44% | 294.9740 | 0.6436 | 0.0478 | 0.0643 | +20.1587 | 10.62% |
| 0.7-0.8 | 44 | 84.09% | 189.9893 | 0.7311 | 0.0743 | 0.0553 | +16.0973 | 11.59% |
| 0.8-0.9 | 5 | 100.00% | 15.0000 | 0.8133 | 0.0780 | 0.0576 | +2.8000 | 22.95% |


## Fill Probability by Limit Bucket

| limit_bucket | submitted_size | matched_size | fill_rate | avg_dist_to_ask |
| :--- | ---: | ---: | ---: | ---: |
| 0.2-0.3 | 10.00 | 10.00 | **100.00%** | 0.0100 |
| 0.3-0.4 | 30.00 | 29.99 | **99.97%** | 0.0100 |
| 0.4-0.5 | 170.00 | 149.97 | **88.22%** | 0.0141 |
| 0.5-0.6 | 695.00 | 535.07 | **76.99%** | 0.0917 |
| 0.6-0.7 | 360.00 | 294.97 | **81.94%** | 0.0636 |
| 0.7-0.8 | 255.00 | 189.99 | **74.51%** | 0.0775 |
| 0.8-0.9 | 35.00 | 15.00 | **42.86%** | 0.0914 |


## Submitted orders by side

| side | orders | accuracy | submit_success_rate | requested_size | avg_limit_price | avg_best_ask_minus_limit | avg_confidence_minus_limit | submitted_replay_pnl | submitted_replay_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 146 | 65.07% | 100.00% | 730.0000 | 0.5799 | 0.0776 | 0.0806 | +51.7000 | 12.21% |
| YES | 165 | 75.15% | 100.00% | 825.0000 | 0.6005 | 0.0676 | 0.0897 | +124.5500 | 25.14% |


## Matched fills by side

| side | fills | accuracy | shares | avg_fill_price | avg_best_ask_minus_limit | avg_confidence_minus_fill | pnl | roi_on_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 139 | 56.12% | 589.9498 | 0.5681 | 0.0668 | 0.0865 | -0.2014 | -0.06% |
| YES | 151 | 68.87% | 635.0487 | 0.5975 | 0.0523 | 0.0960 | +50.6037 | 13.34% |


## Submitted orders by confidence-price edge

| edge_bucket | orders | accuracy | submit_success_rate | requested_size | avg_limit_price | avg_best_ask_minus_limit | avg_confidence_minus_limit | submitted_replay_pnl | submitted_replay_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.10-0.00 | 5 | 60.00% | 100.00% | 25.0000 | 0.6380 | 0.0860 | -0.0079 | -0.9500 | -5.96% |
| 0.00-0.05 | 82 | 81.71% | 100.00% | 410.0000 | 0.6516 | 0.0965 | 0.0317 | +67.8500 | 25.40% |
| 0.05-0.10 | 131 | 70.99% | 100.00% | 655.0000 | 0.6083 | 0.0834 | 0.0717 | +66.5500 | 16.70% |
| 0.10-0.20 | 75 | 61.33% | 100.00% | 375.0000 | 0.5291 | 0.0404 | 0.1348 | +31.6000 | 15.93% |
| >=0.20 | 18 | 55.56% | 100.00% | 90.0000 | 0.4311 | 0.0100 | 0.2497 | +11.2000 | 28.87% |


## Matched fills by confidence-fill edge

| edge_bucket | fills | accuracy | shares | avg_fill_price | avg_best_ask_minus_limit | avg_confidence_minus_fill | pnl | roi_on_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.10-0.00 | 4 | 50.00% | 20.0000 | 0.6300 | 0.0750 | -0.0097 | -2.6000 | -20.63% |
| 0.00-0.05 | 65 | 70.77% | 284.9838 | 0.6451 | 0.0898 | 0.0302 | +26.1542 | 14.23% |
| 0.05-0.10 | 118 | 65.25% | 509.9757 | 0.6069 | 0.0689 | 0.0722 | +10.4993 | 3.39% |
| 0.10-0.20 | 85 | 56.47% | 325.0486 | 0.5306 | 0.0322 | 0.1368 | +12.6047 | 7.31% |
| >=0.20 | 18 | 50.00% | 84.9903 | 0.4265 | 0.0100 | 0.2466 | +3.7441 | 10.33% |


## Matched fills by role

| role | fills | accuracy | shares | avg_fill_price | avg_best_ask_minus_limit | avg_confidence_minus_fill | pnl | roi_on_cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maker | 280 | 62.14% | 1174.9984 | 0.5854 | 0.0610 | 0.0905 | +37.2523 | 5.42% |
| taker | 10 | 80.00% | 50.0000 | 0.5370 | 0.0100 | 0.1177 | +13.1500 | 48.98% |


## Worst matched fills

| t0 | side | actual_side | leg | match_role | limit_price | price | size | confidence | pnl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-13T16:45 | YES | NO | first | maker | 0.7600 | 0.7600 | 5.0000 | 0.8273 | -3.8000 |
| 2026-06-13T22:35 | YES | NO | first | maker | 0.7500 | 0.7500 | 5.0000 | 0.8330 | -3.7500 |
| 2026-06-14T00:05 | YES | NO | first | maker | 0.7400 | 0.7400 | 5.0000 | 0.8189 | -3.7000 |
| 2026-06-12T16:15 | YES | NO | first | maker | 0.7100 | 0.7100 | 5.0000 | 0.7778 | -3.5500 |
| 2026-06-13T02:15 | NO | YES | first | maker | 0.7100 | 0.7100 | 5.0000 | 0.7436 | -3.5500 |
| 2026-06-12T21:00 | YES | NO | first | maker | 0.7000 | 0.7000 | 5.0000 | 0.7805 | -3.5000 |
| 2026-06-13T06:15 | NO | YES | first | maker | 0.7000 | 0.7000 | 5.0000 | 0.7876 | -3.5000 |
| 2026-06-12T22:35 | YES | NO | first | maker | 0.6900 | 0.6900 | 5.0000 | 0.7210 | -3.4500 |
| 2026-06-13T05:30 | YES | NO | first | maker | 0.6900 | 0.6900 | 5.0000 | 0.6999 | -3.4500 |
| 2026-06-13T00:50 | NO | YES | first | maker | 0.6800 | 0.6800 | 5.0000 | 0.7146 | -3.4000 |


## Best matched fills

| t0 | side | actual_side | leg | match_role | limit_price | price | size | confidence | pnl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-13T11:10 | YES | YES | first | taker | 0.3900 | 0.3900 | 5.0000 | 0.5985 | +3.0500 |
| 2026-06-13T00:45 | NO | NO | first | maker | 0.3900 | 0.3900 | 4.8100 | 0.5913 | +2.9341 |
| 2026-06-12T20:05 | NO | NO | first | maker | 0.4300 | 0.4300 | 5.0000 | 0.6352 | +2.8500 |
| 2026-06-12T18:25 | YES | YES | first | taker | 0.4700 | 0.4600 | 5.0000 | 0.6004 | +2.7000 |
| 2026-06-13T07:05 | YES | YES | first | maker | 0.4600 | 0.4600 | 5.0000 | 0.6269 | +2.7000 |
| 2026-06-13T20:25 | NO | NO | first | maker | 0.4600 | 0.4600 | 5.0000 | 0.6845 | +2.7000 |
| 2026-06-14T02:40 | NO | NO | first | maker | 0.4600 | 0.4600 | 5.0000 | 0.5255 | +2.7000 |
| 2026-06-13T05:00 | YES | YES | first | maker | 0.4700 | 0.4700 | 5.0000 | 0.6710 | +2.6500 |
| 2026-06-13T15:40 | NO | NO | first | maker | 0.4800 | 0.4800 | 5.0000 | 0.6018 | +2.6000 |
| 2026-06-13T11:20 | NO | NO | first | maker | 0.4900 | 0.4900 | 5.0000 | 0.6579 | +2.5500 |


## Diagnosis

The submitted-order replay isolates prediction quality before fill selection. The matched-fill tables isolate realized execution quality after the market decided which orders actually traded.

If submitted replay accuracy is materially higher than matched-fill accuracy, the main issue is adverse selection / fill quality. If both are low, the model-side prediction quality is the bottleneck. If accuracy is acceptable but matched avg fill price is too high, the issue is price paid versus break-even accuracy.
