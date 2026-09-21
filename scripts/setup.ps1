<#
.SYNOPSIS
    One-command setup for the Vocal-Remover environment.

.DESCRIPTION
    Installs everything the app needs into .\venv and verifies it works.
    Safe to re-run: it is idempotent and repairs a broken venv in place.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Force          # rebuild the venv from scratch
    .\scripts\setup.ps1 -Models         # also pre-download the UVR models
    .\scripts\setup.ps1 -Python "C:\Python311\python.exe"
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$Models,
    [string]$Python
)

# NOT "Stop". uv and python write progress to stderr, and in PowerShell 5.1 a
# native command's stderr is wrapped in an ErrorRecord (NativeCommandError),
# which under "Stop" aborts this script even when the exe exited 0. Every
# native call below is checked explicitly via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Root = Split-Path $PSScriptRoot -Parent
$Venv = Join-Path $Root "venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$Lock = Join-Path $Root "requirements\app.lock"

function Write-Step($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Ok($msg)   { Write-Host "  [ok  ] $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red }

# ---------------------------------------------------------------- preflight

Write-Step "1/4  PREFLIGHT"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Bad "uv is not installed."
    Write-Host ""
    Write-Host "  Install it with:" -ForegroundColor Yellow
    Write-Host "    powershell -c ""irm https://astral.sh/uv/install.ps1 | iex""" -ForegroundColor Yellow
    exit 1
}
Write-Ok "uv $((& uv --version).Split(' ')[1])"

if (-not (Test-Path $Lock)) {
    Write-Bad "missing $Lock"
    exit 1
}
Write-Ok "lockfile present"

# ffmpeg lives outside every venv, so no lockfile can catch a missing one.
# audio-separator shells out to it for decode/encode.
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Ok "ffmpeg on PATH"
} else {
    Write-Bad "ffmpeg is NOT on PATH - audio decode/encode will fail."
    Write-Host "    winget install Gyan.FFmpeg" -ForegroundColor Yellow
    exit 1
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | Select-Object -First 1
    Write-Ok "GPU: $gpu"
} else {
    Write-Warn2 "nvidia-smi not found - separation would run on CPU (very slow)."
}

# ------------------------------------------------------------------ python

Write-Step "2/4  INTERPRETER"

# Prefer a real system CPython 3.11 over a uv-managed one. Both work, but the
# uv-managed interpreter lives under AppData\Roaming and has proven flakier to
# launch here; a system install is the more predictable default.
if (-not $Python) {
    $cand = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $cand) {
        $Python = $cand.Trim()
        Write-Ok "system CPython 3.11: $Python"
    } else {
        Write-Warn2 "no system CPython 3.11 found; asking uv to provide one"
        & uv python install 3.11
        $Python = "3.11"
    }
} else {
    Write-Ok "using -Python $Python"
}

# --------------------------------------------------------------------- venv

Write-Step "3/4  ENVIRONMENT"

if ($Force -and (Test-Path $Venv)) {
    Write-Host "  removing existing venv (-Force)"
    Remove-Item -Recurse -Force $Venv
}

& uv venv $Venv --python $Python --allow-existing
if ($LASTEXITCODE -ne 0) { Write-Bad "uv venv failed"; exit 1 }

# --index-strategy unsafe-best-match is REQUIRED at sync time, not just at
# compile time: app.lock pins torch to an explicit +cu128 local version that
# only the PyTorch index carries, and uv otherwise refuses to look past PyPI.
# Without it this fails outright; with the wrong pins it would silently install
# a CPU-only torch instead.
$env:VIRTUAL_ENV = $Venv
& uv pip sync $Lock --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { Write-Bad "uv pip sync failed"; exit 1 }
Write-Ok "packages installed"

if ($Models) {
    Write-Host "  pre-downloading UVR models (about 0.7 GiB) ..."
    & $VenvPy -c "from audio_separator.separator import Separator; s=Separator(model_file_dir=r'$Root\data\models', log_level=40); s.download_model_and_data('model_bs_roformer_ep_317_sdr_12.9755.ckpt'); s.download_model_and_data('UVR-MDX-NET-Voc_FT.onnx')"
    if ($LASTEXITCODE -eq 0) { Write-Ok "models cached" } else { Write-Warn2 "model download failed; they will download on first use" }
}

# ------------------------------------------------------------------- verify

Write-Step "4/4  VERIFY"

& $VenvPy (Join-Path $PSScriptRoot "verify_env.py")
$verifyCode = $LASTEXITCODE

if ($verifyCode -ne 0) {
    Write-Host ""
    Write-Bad "Setup completed but verification failed - see above."
    exit $verifyCode
}

Write-Host ""
Write-Host "Setup complete. Commands (run from the repo root):" -ForegroundColor Green
Write-Host ""
Write-Host "  Log in to Tidal (one time)" -ForegroundColor Gray
Write-Host "    venv\Scripts\python.exe -m smoke_test.tidal_cli login"
Write-Host ""
Write-Host "  Download a track" -ForegroundColor Gray
Write-Host "    venv\Scripts\python.exe -m smoke_test.tidal_cli get <url> -o data\staging\test"
Write-Host ""
Write-Host "  Re-check the environment at any time" -ForegroundColor Gray
Write-Host "    .\scripts\doctor.ps1"
Write-Host ""
exit 0
