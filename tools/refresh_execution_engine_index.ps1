$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$source = Join-Path $projectRoot "execution_engine"
$stage = Join-Path $env:LOCALAPPDATA "cbm-staging\execution_engine_pyonly"
$cache = Join-Path $env:LOCALAPPDATA "codebase-memory-mcp-cache"
$cbm = Join-Path $env:LOCALAPPDATA "Programs\codebase-memory-mcp\codebase-memory-mcp.exe"
$log = Join-Path $env:USERPROFILE "Desktop\cbm-execution-engine-refresh.log"

if (-not (Test-Path $cbm)) { throw "CBM executable not found: $cbm" }
if (-not (Test-Path $source)) { throw "Source folder not found: $source" }

# Run this only while Codex is closed.
Get-Process codebase-memory-mcp -ErrorAction SilentlyContinue |
    Stop-Process -Force

$env:CBM_CACHE_DIR = $cache
New-Item -ItemType Directory -Force -Path $cache | Out-Null

# Rebuild Python-only mirror.
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$files = @(
    Get-ChildItem $source -Recurse -File |
    Where-Object { $_.Extension -in @(".py", ".pyi") }
)

if ($files.Count -eq 0) { throw "No Python files found under: $source" }

foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length).TrimStart("\")
    $dest = Join-Path $stage $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
}

Write-Host "Python files staged: $($files.Count)"
Write-Host "Refreshing execution_engine graph..."

$payload = @{
    repo_path = $stage
    mode = "fast"
    persistence = $false
} | ConvertTo-Json -Compress

$payloadEscaped = $payload -replace '"', '\"'

# Launch outside PowerShell's native stderr handling.
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $cbm
$psi.Arguments = "cli index_repository $payloadEscaped"
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()

$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()

if (-not $proc.WaitForExit(180000)) {
    try { $proc.Kill() } catch {}
    throw "Indexing exceeded 180 seconds. See: $log"
}

$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
$output = (@($stderr.Trim(), $stdout.Trim()) | Where-Object { $_ }) -join [Environment]::NewLine

$output | Set-Content -Encoding utf8 -Path $log
$output

if ($proc.ExitCode -ne 0) {
    throw "CBM exited with code $($proc.ExitCode). See: $log"
}

if ($output -notmatch '"status":"indexed"') {
    throw "No indexed status returned. See: $log"
}

Write-Host "`nSUCCESS: execution_engine index refreshed." -ForegroundColor Green
Write-Host "Indexed mirror: $stage"
