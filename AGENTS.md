# AGENTS.md

## Project goal
Implement a BTC/USDT 5-minute direction prediction system using Freqtrade/FreqAI for data, features, training, and inference, with a plugin-friendly architecture and a separate execution layer for Polymarket.

## Non-negotiable rules
- Keep online and offline logic consistent.
- Centralize all business parameters in one config file.
- Do not duplicate feature logic.
- Do not duplicate label logic.
- Keep the Freqtrade strategy thin.
- Shared core must be the single source of truth for:
  - time grid
  - labels
  - feature builders
  - schemas
- Execution must not recompute BTC features.
- Prefer simple, testable modules over large monolithic files.
- Build the minimal complete V1 before adding advanced features.

## Architecture priorities
1. Consistency
2. Readability
3. Maintainability
4. High cohesion
5. Low coupling
6. Plugin-friendly extensions

## Required first version
- BTC/USDT only
- 1m data
- 5m horizon only
- label: y = 1{close[t0+5m] > open[t0]}
- LightGBM baseline
- unified settings file
- shared core + model + thin strategy + execution stubs + tests

## Derivatives integration rules
- Keep the main modeling pipeline spot-centered on BTC/USDT.
- Do not convert the whole system into a futures-first bot.
- Use derivatives data as an additional raw input layer.
- Load and normalize raw derivatives data in `src/data/derivatives/`.
- Build derivatives features only through shared feature packs.
- Execution must not read derivatives raw data.
- Signal layer may recompute features only via the shared builder.
- Phase 1 scope is funding + basis only.
- OI and options may exist as stubs but not full implementations in phase 1.

## Shell tooling rule
- Prefer `rtk` for verbose shell commands.
- Before declaring `rtk` unavailable, first verify it with `where.exe rtk` and `rtk --version`.
- If `where.exe rtk` returns a valid path or `rtk --version` succeeds, continue using `rtk`.
- Only fall back to raw PowerShell commands after those checks fail in the current shell session.

@RTK.md
