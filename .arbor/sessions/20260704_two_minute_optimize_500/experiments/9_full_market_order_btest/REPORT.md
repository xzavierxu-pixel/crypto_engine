# Full B_test market-order no-gate

No order gate was applied. Every frozen B_test row attempts one market order on `selected_side`; rows without a valid pre-decision market price are counted in the denominator and receive no synthetic fill.

B_test sum_pnl: `159.49876221957948`

Intended order count: `7468.0`; executed order count: `7424.0`; market price coverage: `0.9941081949651848`; late joins: `0.0`.
