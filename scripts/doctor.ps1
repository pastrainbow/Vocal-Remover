<#
.SYNOPSIS
    Check the environment without changing anything.

.DESCRIPTION
    Read-only. Run this when something stops working - it reports which layer
    broke rather than making you bisect it by hand. Exits non-zero on failure.
#>
[CmdletBinding()]
param()

$Root = Split-Path $PSScriptRoot -Parent
$VenvPy = Join-Path $Root "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "  [FAIL] no venv at $VenvPy" -ForegroundColor Red
    Write-Host "         run:  .\scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

& $VenvPy (Join-Path $PSScriptRoot "verify_env.py")
exit $LASTEXITCODE
