# Research Report: Rebuild the accepted first-minute pipeline at decision_time=market_t0+2m using only features time...

## Results

- B_dev baseline: `218.5`
- B_dev final trunk: `218.5`
- B_test baseline: `42.43`
- B_test final trunk: `42.43`

## Exploration

- Nodes total: `1`
- Scored nodes: `1`
- Merged nodes: `0`

### Top Ideas By Score

- **1** `59.9` _done_: Mechanism: End-to-end as-of-time shift rebuild with immutable source inputs and a three-minute selected-side low targ...

## Global Insight

Children findings: [1, done, score=59.9] Two-minute direction accuracy improved, but the shortened-window Gc calibration gap reached 0.2265 and validation PnL fell to -0.75.

## Artifacts

- Idea tree JSON: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260704_two_minute_shift\.coordinator\idea_tree.json`
- Idea tree Markdown: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260704_two_minute_shift\.coordinator\idea_tree.md`
- Experiments: `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260704_two_minute_shift\experiments`
