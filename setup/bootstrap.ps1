<#
.SYNOPSIS
    Windows prerequisites for the venv. A helper, not an entry point.

.DESCRIPTION
    run.sh calls this the first time it has no venv to work with, and again
    later if ffmpeg has gone missing. It does the parts a bash script cannot
    do well on Windows - installing uv and ffmpeg, and finding or installing a
    CPython 3.11 - and then prints the interpreter to use as a line reading
    INTERPRETER=<path>, which run.sh reads back.

    A missing tool is installed, not reported: the only thing that stops this
    script is an install that actually fails, and then it says what to run by
    hand.

    Whatever it installs goes on PATH for later shells, but not for the bash
    that is running run.sh right now - that process inherited its PATH before
    any of this happened. So each directory is also printed as a line reading
    PATH_ADD=<dir> for run.sh to fold into its own PATH.

    It installs no Python packages and creates no venv: run.sh owns both, so
    there is one place where the lockfile is applied.

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
function Write-Step($msg) { Write-Host "  [ .. ] $msg" -ForegroundColor Cyan }
function Write-Warn2($msg){ Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red }

#: Directories something was installed into, reported to run.sh at the end.
$script:PathAdds = @()

# A tool installed a moment ago is on PATH for the next shell and not for this
# one. Both ends need dealing with: this puts the directory on our own PATH so
# the checks below find it, and queues the PATH_ADD= line run.sh reads.
function Add-Path([string]$dir) {
    if (-not $dir) { return }
    $dir = $dir.TrimEnd('\')
    if (($env:PATH -split ';') -notcontains $dir) { $env:PATH = "$dir;$env:PATH" }
    if ($script:PathAdds -notcontains $dir) { $script:PathAdds += $dir }
}

# An installer that puts itself on PATH does it in the registry; this process
# is still holding the copy it was launched with. Pull the persisted value
# back in, without dropping anything we were handed.
function Sync-PathFromRegistry {
    foreach ($scope in 'Machine', 'User') {
        $stored = [Environment]::GetEnvironmentVariable('Path', $scope)
        if (-not $stored) { continue }
        foreach ($dir in $stored.Split(';')) {
            if ($dir -and (($env:PATH -split ';') -notcontains $dir)) {
                $env:PATH = "$env:PATH;$dir"
            }
        }
    }
}

# Put a directory on the user PATH for good, for the installs that do not do
# it themselves. Read and write through the registry rather than
# [Environment]::GetEnvironmentVariable, which expands any %VARS% in the
# stored value and would write them back flattened; keeping the value's
# original kind matters for the same reason. Shells already open will not see
# it, which is what PATH_ADD= is for.
function Add-UserPath([string]$dir) {
    $key = Get-Item 'HKCU:\Environment'
    if ($key.GetValueNames() -contains 'Path') {
        $cur  = $key.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
        $kind = $key.GetValueKind('Path')
    } else {
        $cur  = ''
        $kind = 'ExpandString'
    }
    if (($cur -split ';') -contains $dir) { return }
    $new = if ($cur) { "$cur;$dir" } else { $dir }
    Set-ItemProperty -Path 'HKCU:\Environment' -Name Path -Value $new -Type $kind
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "PREREQUISITES" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

# -------------------------------------------------------------------- uv

# Where the installer puts uv, in the order it decides between them. Used both
# to notice an install this shell has not picked up yet and to find the one we
# just did.
function Find-Uv {
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $dirs = @($env:UV_INSTALL_DIR,
              $env:XDG_BIN_HOME,
              (Join-Path $env:USERPROFILE '.local\bin'),
              (Join-Path $env:USERPROFILE '.cargo\bin'))
    foreach ($dir in $dirs) {
        if ($dir -and (Test-Path (Join-Path $dir 'uv.exe'))) { return (Join-Path $dir 'uv.exe') }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Step "uv is not installed - installing it from astral.sh"
    # In a child powershell, not piped into iex here: the installer calls exit
    # on failure, and in-process that would take this script with it before it
    # could say anything useful.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Host
    $installed = $LASTEXITCODE
    Sync-PathFromRegistry
    $uv = Find-Uv
    if ($installed -ne 0 -or -not $uv) {
        Write-Bad "could not install uv automatically."
        Write-Host ""
        Write-Host "  Install it by hand, then run ./run.sh again:" -ForegroundColor Yellow
        Write-Host "    powershell -c ""irm https://astral.sh/uv/install.ps1 | iex""" -ForegroundColor Yellow
        exit 1
    }
}
Add-Path (Split-Path $uv -Parent)
Write-Ok "uv $((& uv --version).Split(' ')[1])"

# ---------------------------------------------------------------- ffmpeg

# ffmpeg lives outside every venv, so no lockfile can catch a missing one.
# audio-separator shells out to it for decode/encode, and the Tidal downloader
# remuxes with it, so the app is half broken without one.
function Install-Ffmpeg {
    # winget first when it is there: it is the one route that can update
    # ffmpeg again later.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Step "installing ffmpeg with winget (Gyan.FFmpeg)"
        & winget install --id Gyan.FFmpeg --exact --source winget `
            --accept-package-agreements --accept-source-agreements `
            --disable-interactivity | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Sync-PathFromRegistry
            $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
            if ($cmd) { Add-Path (Split-Path $cmd.Source -Parent); return }
        }
        Write-Warn2 "winget did not provide ffmpeg - falling back to a static build"
    }

    # Otherwise the release build gyan.dev publishes, unzipped under
    # LOCALAPPDATA. Nothing to uninstall later beyond deleting that folder.
    $dest = Join-Path $env:LOCALAPPDATA 'vocal-remover\ffmpeg'
    $zip  = Join-Path $env:TEMP 'ffmpeg-release-essentials.zip'
    try {
        Write-Step "downloading ffmpeg (~40 MiB)"
        # Invoke-WebRequest draws a progress bar that costs more time than the
        # download itself when the output is not a console.
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -OutFile $zip `
            -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    } catch {
        Write-Bad "could not install ffmpeg: $($_.Exception.Message)"
        Write-Host ""
        Write-Host "  Install it by hand, then run ./run.sh again:" -ForegroundColor Yellow
        Write-Host "    winget install Gyan.FFmpeg" -ForegroundColor Yellow
        exit 1
    } finally {
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
    }

    # The archive has a versioned folder at the top, so look the exe up rather
    # than build the path out of a version that changes.
    $exe = Get-ChildItem $dest -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $exe) { Write-Bad "the ffmpeg archive had no ffmpeg.exe in it"; exit 1 }
    $bin = Split-Path $exe.FullName -Parent
    Add-Path $bin
    Add-UserPath $bin   # winget would have done this part itself
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    # It may only be missing from this process's PATH.
    Sync-PathFromRegistry
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Install-Ffmpeg }
}
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Ok "ffmpeg: $($ffmpeg.Source)"
} else {
    Write-Bad "ffmpeg is still not on PATH after installing it."
    exit 1
}

# ------------------------------------------------------------------- gpu

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | Select-Object -First 1
    Write-Ok "GPU: $gpu"
} else {
    Write-Warn2 "nvidia-smi not found - separation would run on CPU (very slow)."
}

# ------------------------------------------------------------- interpreter

# A path is only an interpreter if it starts. uv's managed layout reaches the
# interpreter through a "3.11" alias directory that can end up dangling on
# Windows, and a python that will not launch is much better caught here than
# three steps into building the venv.
function Test-Python([string]$path) {
    if (-not $path) { return $false }
    $ver = & $path -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    return ($LASTEXITCODE -eq 0 -and $ver -and $ver.Trim() -eq '3.11')
}

#: The launcher's answer for 3.11, or $null if it has none that runs.
function Find-SystemPython311 {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { return $null }
    $cand = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $cand) { return $null }
    $cand = $cand.Trim()
    if (Test-Python $cand) { return $cand }
    return $null
}

function Find-Python311 {
    # 1. A system CPython 3.11. Both kinds work, but the uv-managed one lives
    #    under AppData\Roaming and has proven flakier to launch here, so a
    #    system install is the more predictable default.
    $cand = Find-SystemPython311
    if ($cand) { Write-Ok "system CPython 3.11: $cand"; return $cand }

    # 2. One uv manages. Its exit code is not the verdict: uv python install
    #    fails when it cannot put the "3.11" alias in place even though the
    #    interpreter it just downloaded is perfectly good. What uv python find
    #    reports afterwards is the verdict, and the full path it gives back
    #    goes straight to the real directory, alias or no alias.
    Write-Step "no system CPython 3.11 - downloading one with uv (~24 MiB)"
    # Held back rather than shown: the alias failure above prints "error:" in
    # red while the interpreter underneath is fine, and an error nobody needs
    # to act on is worse than no error. It gets printed below if the fallback
    # is what ends up running.
    $log = & uv python install 3.11 2>&1 | Out-String
    $cand = & uv python find 3.11 2>$null
    if ($cand) {
        $cand = $cand.Trim()
        if (Test-Python $cand) { Write-Ok "uv-managed CPython 3.11: $cand"; return $cand }
    }

    # 3. A real installer, which brings the py launcher along with it.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Warn2 "uv could not provide a working 3.11:"
        Write-Host $log.TrimEnd()
        Write-Step "installing CPython 3.11 with winget instead"
        & winget install --id Python.Python.3.11 --exact --source winget --scope user `
            --accept-package-agreements --accept-source-agreements `
            --disable-interactivity | Out-Host
        Sync-PathFromRegistry
        $cand = Find-SystemPython311
        if ($cand) { Write-Ok "system CPython 3.11: $cand"; return $cand }
    }

    return $null
}

if (-not $Python) {
    $Python = Find-Python311
    if (-not $Python) {
        Write-Bad "no CPython 3.11 to build the venv from, and none could be installed."
        Write-Host ""
        Write-Host "  Install one by hand, then run ./run.sh again:" -ForegroundColor Yellow
        Write-Host "    winget install Python.Python.3.11" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Ok "using -Python $Python"
}

# The lines run.sh parses. Everything above is for you to read.
foreach ($dir in $script:PathAdds) { Write-Host "PATH_ADD=$dir" }
Write-Host "INTERPRETER=$Python"
exit 0