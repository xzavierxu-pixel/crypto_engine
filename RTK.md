# RTK - Rust Token Killer (Codex CLI)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk`.

On Windows and in Codex sessions, do not assume `rtk` is unavailable based on a single failed call.
Before falling back to raw commands, verify the current shell with:

```powershell
where.exe rtk
rtk --version
```

If either check succeeds, keep using `rtk` in that session.
Only downgrade to raw commands when both checks fail in the current shell.

Examples:

```bash
rtk git status
rtk cargo test
rtk npm run build
rtk pytest -q
```

## Meta Commands

```bash
rtk gain            # Token savings analytics
rtk gain --history  # Recent command savings history
rtk proxy <cmd>     # Run raw command without filtering
```

## Verification

```bash
rtk --version
rtk gain
where.exe rtk
```
