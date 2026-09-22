<#
.SYNOPSIS
    Restart the app on this machine and refuse to finish until it is healthy.

.DESCRIPTION
    Driven by .github/workflows/deploy-staging.yml on the self-hosted runner,
    but it is a normal script - run it by hand when you want the same
    stop / start / verify sequence:

        pwsh -File setup/deploy.ps1

    It deliberately knows nothing about installing, checking or prefetching.
    All of that is ./run.sh, which does the same sequence on every run whether
    a human or the runner starts it, so a deploy and a hand-started server are
    the same thing and cannot drift apart. What is left here is the part a
    shell script cannot do for itself: free the port, start run.sh detached
    from the Actions step, and refuse to finish until /api/health agrees.

    Everything that knows *how* the app is supervised therefore lives here, so
    swapping the detached process below for a real service (nssm, or a
    scheduled task) is a one-file change and the workflow does not move.

    Deliberately never cleans the working tree. data/models holds gigabytes of
    downloaded checkpoints and state/ holds the Tidal OAuth token; both are
    gitignored, so a `git clean -xfd` here would cost a model re-download and
    a re-login on every deploy.
#>
[CmdletBinding()]
param(
    # Must agree with PORT in .env - this is only where we look for /api/health.
    [int]$Port = 8000,
    # Half an hour, because this window now covers all of run.sh: a cold
    # lockfile sync pulls ~3 GB of CUDA wheels, the model prefetch another
    # ~0.7 GiB, and only then does the server start loading BS-Roformer and
    # MDX-Net into VRAM. It is a ceiling on a run that has stopped making
    # progress, not a normal wait - a box that is already set up is healthy in
    # well under a minute, and a run.sh that FAILS is noticed the moment it
    # exits rather than at this deadline. See Wait-Healthy.
    [int]$HealthTimeoutSeconds = 1800,

    # Where models, stems, the job database and the Tidal token live.
    #
    # These MUST point outside the source tree when this runs under Actions.
    # The runner checks out into its own workspace - _work\<repo>\<repo> -
    # which is a different directory from any clone made by hand, so
    # repo-relative data means the deploy quietly builds a PARALLEL
    # installation: its own empty data/models (a fresh ~0.7 GiB download),
    # its own state/tidal (logged out), its own job database. Fixed paths are
    # what make the deployed app the same app from one run to the next.
    #
    # Unset falls back to repo-relative, which is still right when running
    # this script by hand inside a working checkout.
    [string]$DataDir  = $env:VR_DATA_DIR,
    [string]$StateDir = $env:VR_STATE_DIR
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
if (-not $DataDir)  { $DataDir  = Join-Path $Root 'data' }
if (-not $StateDir) { $StateDir = Join-Path $Root 'state' }
New-Item -ItemType Directory -Force -Path $DataDir, $StateDir | Out-Null

# app/config.py reads these through pydantic-settings, which has no env prefix
# configured, so DATA_DIR and STATE_DIR are what redirect the app away from
# the runner workspace. They are set here for anything this script runs
# itself; they do NOT reach the app, because Win32_Process.Create gives the
# process it spawns a fresh environment block rather than a copy of this
# one. Start-App writes them into its launcher for that reason - keep the two
# in step.
$env:DATA_DIR  = $DataDir
$env:STATE_DIR = $StateDir

$PidFile = Join-Path $StateDir 'app.pid'
$LogDir  = Join-Path $StateDir 'logs'
$Health  = "http://127.0.0.1:$Port/api/health"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "!!! $msg" -ForegroundColor Red; exit 1 }

function Get-GitBash {
    # Resolve Git Bash by path, never by a PATH lookup.
    #
    # On Windows `bash` resolves to C:\Windows\System32\bash.exe - the WSL
    # launcher - which ships with the OS and takes precedence over Git Bash,
    # and Git for Windows does not put its own bash on PATH at all. Calling
    # plain `bash` therefore runs run.sh under WSL, which on a machine without
    # virtualisation fails with HCS_E_HYPERV_NOT_INSTALLED, and on a machine
    # WITH it would be worse: run.sh would half-work against a Linux
    # filesystem view and a venv it cannot execute.
    # Built from bases that can legitimately be unset - Join-Path throws on a
    # null base, which under Actions' stop preference would fail the deploy
    # rather than move on to the next candidate.
    $bases = @(
        @($env:ProgramFiles,              'Git\bin\bash.exe'),
        @(${env:ProgramFiles(x86)},       'Git\bin\bash.exe'),
        @($env:LOCALAPPDATA,              'Programs\Git\bin\bash.exe')
    )
    $candidates = @()
    foreach ($b in $bases) {
        if ($b[0]) { $candidates += (Join-Path $b[0] $b[1]) }
    }
    # Derive from git.exe too, which covers a non-standard install location:
    # ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($git) {
        $candidates += (Join-Path (Split-Path (Split-Path $git.Source)) 'bin\bash.exe')
    }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    Die ('Git Bash not found - looked in: ' + ($candidates -join '; ') +
         '. Install Git for Windows; the System32 bash.exe is WSL and ' +
         'cannot run run.sh.')
}

$Bash = Get-GitBash

# ---------------------------------------------------------------------- stop

function Get-PortOwners {
    # Pids listening on $Port. Get-NetTCPConnection exists on Windows 8 and
    # Server 2012 onward; netstat is the fallback and parses the last column
    # of a LISTENING line.
    $pids = @()
    try {
        $pids = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
                  Select-Object -ExpandProperty OwningProcess -Unique)
    }
    catch [System.Management.Automation.CommandNotFoundException] {
        $pids = @(netstat -ano -p TCP |
                  Select-String -Pattern "LISTENING" |
                  Select-String -Pattern ":$Port\s" |
                  ForEach-Object { ($_ -split '\s+')[-1] } |
                  Select-Object -Unique)
    }
    catch {
        # No listener at all throws rather than returning empty. Not an error.
        $pids = @()
    }
    return @($pids | Where-Object { $_ -and $_ -ne 0 })
}

function Stop-PortHolders {
    # The pid file is not enough on its own. It is absent on a fresh state
    # directory, it never knew about a server someone started by hand, and a
    # previous deploy can leave an orphan the file no longer names. The port
    # is the authority on what must go - a leftover listener is exactly the
    # WinError 10048 that this whole function exists to prevent.
    foreach ($holderPid in Get-PortOwners) {
        $proc = Get-Process -Id $holderPid -ErrorAction SilentlyContinue
        if ($null -eq $proc) { continue }

        if ($proc.ProcessName -notmatch '^python') {
            # Deliberately fatal rather than killing it. Something that is not
            # our server owning this port is a misconfiguration, and a deploy
            # script that silently kills unknown processes is worse than one
            # that stops and says which process to look at.
            Die ("port $Port is held by pid $holderPid ('$($proc.ProcessName)'), " +
                 'which is not this app. Refusing to kill it - free the port ' +
                 'or set a different PORT.')
        }

        Say "stopping pid $holderPid ('$($proc.ProcessName)') holding port $Port"
        Stop-Process -Id $holderPid -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit(30000) | Out-Null
    }
}

function Wait-PortFree {
    # Killing the process does not free the socket instantly, and binding a
    # port still held by a dying process is the same 10048 by another route.
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        # @() around the call, not just inside it: PowerShell unrolls a
        # single-element array on return, so one listener comes back as a bare
        # pid, and .Count on a scalar is an error under Set-StrictMode Latest.
        if (@(Get-PortOwners).Count -eq 0) { return }
        Start-Sleep -Milliseconds 500
    }
    Die "port $Port is still in use after 30s"
}

function Stop-App {
    # Pid file first: it identifies our own process precisely, including the
    # case where it is somehow not listening yet.
    if (Test-Path $PidFile) {
        $appPid = (Get-Content $PidFile -Raw).Trim()
        $proc   = Get-Process -Id $appPid -ErrorAction SilentlyContinue

        if ($null -eq $proc) {
            Say "pid $appPid is not running (stale pid file)"
        }
        # The pid file survives reboots and Windows reuses pids, so check the
        # number still belongs to a python before killing it.
        elseif ($proc.ProcessName -notmatch '^python') {
            Say "pid $appPid is '$($proc.ProcessName)', not python - leaving it alone"
        }
        else {
            Say "stopping pid $appPid (from pid file)"
            Stop-Process -Id $appPid -Force -ErrorAction SilentlyContinue
            # The worker holds a CUDA context and a SQLite WAL; give the OS a
            # moment to tear both down before the next process claims them.
            $proc.WaitForExit(30000) | Out-Null
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
    else {
        Say 'no pid file'
    }

    # Then the port, which catches everything the pid file cannot know about.
    Stop-PortHolders
    Wait-PortFree
    Say "port $Port is free"
}

# --------------------------------------------------------------------- start

function Start-App {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stamp    = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout   = Join-Path $LogDir "app-$stamp.out.log"
    $stderr   = Join-Path $LogDir "app-$stamp.err.log"
    $launcher = Join-Path $LogDir "launch-$stamp.cmd"

    # One call to ./run.sh is the entire deploy: it installs what is missing,
    # verifies the GPU stack, prefetches the models and only then execs the
    # server. Everything this script used to do step by step is that script's
    # own sequence now, which is why a deploy can no longer install something
    # a hand-started server would not.
    #
    # It has to run detached, and that is the whole reason for the machinery
    # below. A long-lived server started as a descendant of an Actions step
    # keeps that step's stdout/stderr handles open; the runner does not end a
    # step until those handles close, so a PERFECTLY SUCCESSFUL deploy would
    # hang forever, and cancelling would not help - the runner is blocked on a
    # handle, not on a signal. Start-Process -RedirectStandardOutput mostly
    # avoids that, but only while nothing else in the chain reintroduces
    # inheritance.
    #
    # Win32_Process.Create is not a child of this shell at all: the WMI
    # service creates the process, so there is no handle to inherit and no
    # process tree for the runner to wait on.
    #
    # Two things follow from that, and both are what this generated .cmd is
    # for. Redirection has to happen inside the command line, and the new
    # process gets a FRESH environment block built from the registry rather
    # than a copy of this shell's - so DATA_DIR and STATE_DIR, set above,
    # simply would not arrive. Writing them into the launcher is what puts
    # the deployed app's models, stems, job database and Tidal token where
    # the workflow says they go. The file stays next to the logs as an exact
    # record of how this run was started.
    @(
        '@echo off',
        ('set "DATA_DIR={0}"'  -f $DataDir),
        ('set "STATE_DIR={0}"' -f $StateDir),
        ('cd /d "{0}"'         -f $Root),
        ('"{0}" "./run.sh" > "{1}" 2> "{2}"' -f $Bash, $stdout, $stderr)
    ) | Set-Content -Path $launcher -Encoding Oem

    Say "starting run.sh detached [$Bash]"
    $result = ([wmiclass]'Win32_Process').Create(('cmd.exe /c ""{0}""' -f $launcher), $Root)
    if ($result.ReturnValue -ne 0) {
        Die "could not start run.sh: Win32_Process.Create returned $($result.ReturnValue)"
    }

    Say "launched, logs in $LogDir\app-$stamp.*.log"
    # The app's pid is deliberately NOT recorded here: Create returns
    # cmd.exe's, and the python that run.sh eventually execs is what matters.
    # Wait-Healthy records the real one once the port is bound. cmd.exe's pid
    # is still worth keeping - it lives exactly as long as the run does, which
    # is how a failed run is noticed without waiting out the timeout.
    return @{ Stdout = $stdout; Stderr = $stderr; LauncherPid = $result.ProcessId }
}

# -------------------------------------------------------------------- verify

function Show-StartupLog($started) {
    # Nothing to interrogate but the logs: the run is detached, so this is the
    # only account of what it was doing. run.sh narrates its progress on
    # stdout and dies on stderr, and uvicorn logs to stderr, so both matter.
    Write-Host '--- run.sh stdout ---' -ForegroundColor Yellow
    Get-Content $started.Stdout -Tail 30 -ErrorAction SilentlyContinue
    Write-Host '--- run.sh stderr ---' -ForegroundColor Yellow
    Get-Content $started.Stderr -Tail 40 -ErrorAction SilentlyContinue
}

function Wait-Healthy($started) {
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    Say "waiting for $Health"

    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $Health -TimeoutSec 5
        }
        catch {
            # A failed install, a failed environment check and a server that
            # exits during startup all look identical from out here: nothing
            # is listening. What separates them from "still working on it" is
            # the launcher, which outlives run.sh by design and is gone the
            # moment the run ends. Without this check a two-second failure
            # would cost the full half-hour timeout - and on the health
            # workflow, hold the deploy concurrency group for all of it.
            if (-not (Get-Process -Id $started.LauncherPid -ErrorAction SilentlyContinue)) {
                Show-StartupLog $started
                Die 'run.sh exited without bringing the app up'
            }
            Start-Sleep -Seconds 3
            continue
        }

        if (-not $r.worker.running) { Start-Sleep -Seconds 3; continue }

        # main.py aborts startup on an unloadable model, so a served health
        # response almost implies this - but DUPLICATE entries answer 200
        # while meaning a misconfigured preload list, so assert the report.
        $bad = @($r.worker.models | Where-Object { $_.status -ne 'loaded' })
        if ($bad.Count -gt 0) {
            $bad | ForEach-Object {
                Write-Host "    model $($_.name): $($_.status) $($_.error)" -ForegroundColor Red
            }
            Die 'worker is running but not every model loaded'
        }

        # Record the real pid now that something is listening. Start-App
        # cannot: WMI hands back cmd.exe's pid, not python's.
        $owners = @(Get-PortOwners)
        if ($owners.Count -gt 0) {
            $owners[0] | Set-Content $PidFile -NoNewline
            Say "pid $($owners[0]) recorded"
        }

        # Same unrolling hazard: a single preloaded model deserialises to one
        # object rather than a one-element array.
        Say "healthy - $(@($r.worker.models).Count) model(s) resident"
        $r.worker.models | ForEach-Object {
            Write-Host "    $($_.name)  $($_.device)  $($_.load_seconds)s"
        }
        # Reported, never gated on: the box can be perfectly deployed and
        # simply logged out, and that needs a human with a phone, not a
        # failed pipeline.
        if (-not $r.tidal.authenticated) {
            Write-Host "    NOTE: Tidal not authenticated - $($r.tidal.detail)" -ForegroundColor Yellow
        }
        return
    }

    # Still running, still not healthy - a stuck download or a model that
    # will not finish loading. The log is the only account of it.
    Show-StartupLog $started
    Die "not healthy after ${HealthTimeoutSeconds}s"
}

# ----------------------------------------------------------------------- run

Push-Location $Root
try {
    Stop-App
    Wait-Healthy (Start-App)
    Say 'deploy complete'
}
finally {
    Pop-Location
}
