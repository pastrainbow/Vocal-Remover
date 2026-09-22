<#
.SYNOPSIS
    Windows prerequisites for the venv. A helper, not an entry point.

.DESCRIPTION
    run.sh calls this the first time it has no venv to work with. It does the
    parts a bash script cannot do well on Windows - finding or installing a
    CPython 3.11, and telling you how to install what is missing - and then
    prints the interpreter to use as a line reading INTERPRETER=<path>, which
    run.sh reads back.

    It installs no packages and creates no venv: run.sh owns both, so there is
    one place where the lockfile is applied.

    ./run.sh is the entry point. Run this by hand only to re-check the tools.
#>
[CmdletBinding()]
param(
    #: Skip interpreter discovery and use this python instead.
    [string]$Python
)

# NOT "Stop". uv and python write progress to stderr, and in PowerShell 5.1 a
# native command's stderr is wrapped in an ErrorRecord (NativeCommandError),
# which under "Stop" aborts this script even when the exe exited 0. Every
# native call below is checked explicitly via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

function Write-Ok($msg)   { Write-Host "  [ok  ] $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red }

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "PREREQUISITES" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

# ------------------------------------------------------------------- tools

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Bad "uv is not installed."
    Write-Host ""
    Write-Host "  Install it with:" -ForegroundColor Yellow
    Write-Host "    powershell -c ""irm https://astral.sh/uv/install.ps1 | iex""" -ForegroundColor Yellow
    exit 1
}
Write-Ok "uv $((& uv --version).Split(' ')[1])"

# ffmpeg lives outside every venv, so no lockfile can catch a missing one.
# audio-separator shells out to it for decode/encode. Fatal here, while
# run.sh only warns on later starts: this is the moment to fix it.
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

# ------------------------------------------------------------- interpreter

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
        if ($LASTEXITCODE -ne 0) { Write-Bad "uv python install 3.11 failed"; exit 1 }
        $Python = "3.11"
    }
} else {
    Write-Ok "using -Python $Python"
}

# The line run.sh parses. Everything above is for you to read.
Write-Host "INTERPRETER=$Python"
exit 0
