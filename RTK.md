# RTK - Rust Token Killer (Codex CLI)

**Usage**: High-performance token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk` to reduce token consumption by 60-90%.

## Common Commands

```powershell
rtk git status       # Compact git status
rtk pytest -q        # Shows only failed tests
rtk python <args>    # Compact python output
rtk pip install      # Compact package installation
rtk tree             # Token-optimized directory tree
```

## Meta Commands

```powershell
rtk gain            # Show current session token savings
rtk gain --history  # Show historical savings across sessions
rtk session         # Summary of RTK adoption and effectiveness
rtk discover        # Find missed savings in recent history
rtk proxy <cmd>     # Run command without filtering (usage still tracked)
```

## Project Settings

RTK uses `.codex/config.toml` for project-specific filters.
Trust local filters with:
```powershell
rtk trust
```

## Verification

On Windows, use `where.exe` to locate the binary:

```powershell
rtk --version       # Should be >= 0.42.3
rtk gain            # Verify analytics are working
where.exe rtk       # Locate the binary
```

## Automatic Hooks

To automate RTK for Gemini CLI (auto-prefixing), run:
```powershell
rtk init -g --gemini --auto-patch
```
*Note: Manual prefixing is still recommended in `AGENTS.md` to ensure consistent behavior.*
