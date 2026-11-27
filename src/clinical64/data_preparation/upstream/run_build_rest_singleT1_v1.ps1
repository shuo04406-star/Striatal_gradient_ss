$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptPath = Join-Path $PSScriptRoot 'build_rest_singleT1_v1.py'
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Python build script not found: $scriptPath"
}

$python = Get-Command python -ErrorAction Stop
Write-Host "Python: $($python.Source)"
Write-Host "Build script: $scriptPath"
& $python.Source $scriptPath
if ($LASTEXITCODE -ne 0) {
    throw "BIDS build failed with exit code $LASTEXITCODE"
}
