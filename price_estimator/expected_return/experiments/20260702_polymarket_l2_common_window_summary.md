# Polymarket L2 common-window experiment summary

Validation window: 2026-02-26 through 2026-03-27 UTC. The frozen validation
universe contains 8,446 markets. Direction thresholds are tuned on validation
and therefore optimistic. Models, imputers, feature selection, and probability
calibrators do not fit validation rows.

| Experiment | sum_pnl | coverage | accepted accuracy | orders | trades |
|---|---:|---:|---:|---:|---:|
| P0 / A0 common window | -23.98 | 0.7546 | 0.6454 | 2,321 | 1,608 |
| A2 direction L2 | -25.48 | 0.8419 | 0.6372 | 1,549 | 988 |
| A3 price-estimator L2 | -72.19 | 0.7546 | 0.6454 | 3,463 | 2,562 |
| A4 full classifier/price L2 | -36.36 | 0.8419 | 0.6372 | 1,932 | 1,384 |
| A4 market-mid direction diagnostic | -40.34 | 0.7090 | 0.6742 | 2,722 | 1,822 |
| A5 without book depth | -107.37 | 0.8419 | 0.6372 | 3,928 | 2,992 |
| A6 without order flow | -136.75 | 0.8419 | 0.6372 | 4,688 | 3,448 |
| A7 without trade dynamics | -84.26 | 0.8419 | 0.6372 | 3,321 | 2,380 |
| A8 without cross-side features | -41.29 | 0.8419 | 0.6372 | 2,337 | 1,700 |
| Market-mid direction + XGB Q/hazard | -10.92 | 0.7090 | 0.6742 | 1,813 | 1,267 |

The requested validation `sum_pnl > 1000` was not achieved. The best result is
still negative, so no L2 model is promoted. This conclusion uses unit-contract
PnL and is not changed by arbitrary position-size multiplication.

Data QA:

- 11,968 frozen eligible markets; 100% L2 product coverage.
- 43 daily partitions; 11,968 feature, price-reference, and future-low rows.
- first-minute price-reference and future-low coverage are both 100% for UP and DOWN.
- zero post-cutoff feature events and zero future-low boundary violations.
- deterministic 100-market recomputation and future-append invariance audit passed.
- side audit used 100 markets and 260,711 quote-aligned trades; mirror prices are transformed with `1 - raw_price` for DOWN.
- markets lacking reproducible mirror identity are excluded rather than inferred.
