# Development Workflow

This project uses `rtk` as the default shell proxy for verbose commands.

## Why

- Reduce shell output sent back into the LLM context
- Keep command output focused on actionable information
- Preserve the project rule that development tooling must not change business logic

## Prerequisites

Verify `rtk` is installed and available:

```powershell
rtk --version
rtk gain
```

For this repository, tests can now run directly without manually setting `PYTHONPATH`.

## Default Rule

Prefer `rtk` for commands that generate noisy or repetitive output.

Use raw commands only when you explicitly need the full unfiltered output.

## Commands That Should Prefer `rtk`

### Git

Use:

```powershell
rtk git status
rtk git diff
rtk git log -n 10
```

Avoid raw output unless you need the complete original diff or status text for debugging.

### Tests

Use:

```powershell
rtk test pytest -q
rtk test pytest tests/test_timegrid.py -vv
rtk pytest -q
```

Prefer `rtk test ...` for test runs because it summarizes failures well and keeps the output compact.

### File Search and Reads

Use:

```powershell
rtk read AGENTS.md
rtk grep "grid" src
rtk find "*.py" .
```

Prefer these when exploring the codebase through the assistant.

### Build, Lint, and Package Commands

If these tools are present, prefer:

```powershell
rtk cargo build
rtk cargo test
rtk npm run build
rtk lint
rtk ruff check
```

## Commands That Can Stay Raw

These usually do not need `rtk`:

```powershell
Get-ChildItem
Get-Content docs\development.md
python --version
where.exe rtk
```

Use raw commands for:

- very small outputs
- PowerShell-native inspection commands
- cases where exact original stdout matters

## Verification

To confirm `rtk` is providing value:

```powershell
rtk gain
rtk gain --history
```

Current local validation showed meaningful reduction on:

- `git status`
- `pytest tests/test_timegrid.py -vv`

## Codex Usage

This repository includes local Codex instructions through:

- `AGENTS.md`
- `RTK.md`

Global Codex instructions may also exist under `C:\Users\ROG\.codex`, but the project-local files remain the source of truth for this repository.

## Practical Rule of Thumb

If a command is likely to print logs, diffs, stack traces, test progress, or large search output, run it through `rtk` first.

If the output is already short and exact formatting matters, use the raw command.
