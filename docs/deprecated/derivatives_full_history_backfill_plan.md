# Derivatives Full-History Backfill Plan

This document defines the implementation plan for backfilling derivatives data from `2024-01-01` forward so the derivatives layer can be trained and evaluated on the same long-history span as the spot baseline.

## Goal

Build a reproducible backfill pipeline that produces archival derivatives datasets covering the full BTC/USDT history used by the project, starting from `2024-01-01`.

The pipeline must:

- preserve the existing spot-centered architecture
- keep train/live parity intact
- normalize all derivatives sources into the shared raw schema
- store backfilled data as local parquet artifacts
- support repeatable re-runs without manual patching

## Scope

The backfill plan covers:

- `funding`
- `basis`
- `open interest`
- `options` proxy or a true options source, depending on availability

The first production-grade target is:

- funding
- basis
- OI

The options layer should remain separately versioned because the current proxy source is not equivalent to a full option-chain archive.

## Data sources

### Binance futures

Use Binance futures market-data endpoints for:

- funding rate history
- perpetual basis history
- open interest history

These sources are the most direct fit for the current derivatives schema and align with the existing raw loaders.

### Options

Options require a separate decision:

- either keep the current Deribit volatility-index proxy
- or add a true historical options source and normalize it into the same schema

The proxy is acceptable for research validation, but a real full-history backfill should treat options as a separate source family with its own archival semantics.

## Required artifacts

Backfill outputs should land in a stable directory structure such as:

```text
artifacts/data/derivatives/
  funding/
    binance_btcusdt_funding_2024-01-01_2026-04-08.parquet
  basis/
    binance_btcusdt_basis_2024-01-01_2026-04-08.parquet
  oi/
    binance_btcusdt_oi_2024-01-01_2026-04-08.parquet
  options/
    deribit_btc_iv_2024-01-01_2026-04-08.parquet
  manifests/
    binance_btcusdt_derivatives_manifest.json
```

The manifest should record:

- source name
- endpoint or provider
- date range
- record counts
- checksum or row-hash summary
- schema version
- generation timestamp

## Backfill pipeline

### 1. Download layer

Create a dedicated script, for example:

- `scripts/backfill_derivatives_history.py`

Responsibilities:

- page through each source from `2024-01-01` to the desired end date
- respect endpoint-specific pagination and window limits
- retry transient failures
- checkpoint intermediate results
- write incremental chunks to disk before final merge

### 2. Normalization layer

Reuse the existing loaders where possible:

- `src/data/derivatives/funding_loader.py`
- `src/data/derivatives/basis_loader.py`
- `src/data/derivatives/oi_loader.py`
- `src/data/derivatives/options_loader.py`

The backfill script should output the same canonical columns already expected by the shared pipeline:

- `timestamp`
- `exchange`
- `symbol`
- `source_version`
- source-specific value columns

### 3. Merge and align layer

Reuse:

- `src/data/derivatives/aligner.py`
- `src/data/derivatives/feature_store.py`

The backfill job should not invent a second feature path. It should only produce a stable raw archive that the existing shared builder can consume.

### 4. Verification layer

After backfill, run:

- schema validation for each parquet file
- row-count checks
- timestamp monotonicity checks
- duplicate timestamp checks
- spot/derivatives overlap checks
- train/live parity checks on sampled windows

## Implementation steps

### Phase A: Historical downloader

Implement a reusable downloader that can:

- fetch a start/end range
- page until the full window is covered
- resume from checkpoints
- stop on missing source coverage with an explicit error

### Phase B: Archive builder

Transform the raw pages into backfilled parquet archives and a manifest.

### Phase C: Integration switch

Extend `config/settings.yaml` and `src/core/config.py` so the training pipeline can point to either:

- live/latest paths
- full-history archive paths

The shared builder should consume whichever archive is selected, without changing feature logic.

### Phase D: Validation

Run full-history training on the archived derivatives data and compare:

- baseline full-history spot-only
- full-history spot + funding
- full-history spot + funding + basis
- full-history spot + funding + basis + OI
- full-history spot + funding + basis + OI + options if the source is mature enough

## Acceptance criteria

The backfill implementation is complete when:

- the repo can generate derivatives archives from `2024-01-01`
- the archives are reproducible
- the shared training pipeline can read them without special-case code
- train/live parity still passes
- the results are reproducible across reruns

## Notes on options

Options should not block the rest of the backfill.

If a full-history options source is not available yet:

- keep the proxy source versioned separately
- exclude it from the canonical full-history benchmark
- treat it as a research-only add-on until a true historical source is added

## Recommended order

1. Backfill `funding`, `basis`, and `OI` first.
2. Wire the archive paths into config.
3. Validate full-history training and parity.
4. Add or replace the options source only after the core three sources are stable.
