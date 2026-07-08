$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$source = Join-Path $projectRoot "execution_engine"
$stage = Join-Path $env:LOCALAPPDATA "cbm-staging\execution_engine_pyonly"

if (-not (Test-Path $source)) {
    throw "execution_engine not found: $source"
}

Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$files = @(
    Get-ChildItem $source -Recurse -File |
    Where-Object { $_.Extension -in @(".py", ".pyi") }
)

if ($files.Count -eq 0) {
    throw "No Python files found under: $source"
}

foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length).TrimStart("\")
    $dest = Join-Path $stage $relative

    New-Item -ItemType Directory -Force `
        -Path (Split-Path $dest -Parent) | Out-Null

    Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
}

Write-Output "CBM execution_engine mirror synced: $($files.Count) Python files"
Write-Output "Mirror: $stage"