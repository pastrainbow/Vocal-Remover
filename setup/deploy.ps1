<#
.SYNOPSIS
    Restart the app on this machine and refuse to finish until it is healthy.

.DESCRIPTION
    Driven by .github/workflows/deploy-staging.yml on the self-hosted runner,
    but it is a normal script - run it by hand when you want the same
    stop / sync / start / verify sequence:

        pwsh -File setup/deploy.ps1

    Everything that knows *how* the app is supervised lives here, so swapping
    the detached process below for a real service (nssm, or a scheduled task)
    is a one-file change and the workflow does not move.

    Deliberately never cleans the working tree. data/models holds gigabytes of
    downloaded checkpoints and state/ holds the Tidal OAuth token; both are
    gitignored, so a `git clean -xfd` here would cost a model re-download and
    a re-login on every deploy.
#>
[CmdletBinding()]
param(
    # Must agree with PORT in .env - this is only where we look for /api/health.
    [int]$Port = 8000,
    # Generous because startup is dominated by loading BS-Roformer and
    # MDX-Net into VRAM, not by uvicorn binding the socket.
    [int]$HealthTimeoutSeconds = 300,
    # Skip the GPU pre-flight. For iterating on this script only.
    [switch]$SkipVerify,

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

# app/config.py reads these through pydantic-settings, which has no env
# prefix configured, so exporting them here redirects the server and the
# model prefetch alike. Must happen before anything else runs.
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

# --------------------------------------------------------------- sync/verify

function Sync-Env {
    # run.sh --check installs anything missing, then runs setup/verify_env.py,
    # which builds a real ONNX Runtime session rather than trusting
    # get_available_providers(). It exits 1 on a CPU-only torch or a dead
    # CUDA provider - exactly the silent-degradation cases worth failing on.
    Say "run.sh --check (install if needed, then verify) [$Bash]"
    & $Bash './run.sh' '--check'
    if ($LASTEXITCODE -ne 0) { Die "environment check failed (exit $LASTEXITCODE)" }
}

function Get-Models {
    # Must happen HERE, not implicitly inside Worker.start().
    #
    # A model named in preload_models but absent from data/models is not an
    # error - audio-separator downloads it on first load. But that download
    # would then run inside the health-check window below, and a changed
    # preload_models means ~0.7 GiB over a connection this script cannot
    # predict. Overrunning the window would kill the app mid-download, and
    # audio-separator streams to the final path guarded only by isfile() -
    # so the truncated file would look cached forever and the model would
    # never load again without manual deletion.
    #
    # Pulling it into its own step gives the download no deadline and no
    # process waiting to be killed, and setup/fetch_models.py removes its own
    # partial files if it fails anyway.
    #
    # NOT `run.sh --models`. That flag is documented as "pre-download the UVR
    # models, then START" - only --check exits - so calling it here fetched
    # the models and then exec'd uvicorn in the FOREGROUND, attached to this
    # step, which never returns. Every "deploy hangs after a successful
    # startup" was this line. fetch_models.py is the helper run.sh calls for
    # the download half, and calling it directly is the whole of what is
    # wanted here.
    Say 'prefetching models'
    $python = Join-Path $Root 'venv\Scripts\python.exe'
    if (-not (Test-Path $python)) { Die "no venv python at $python" }
    & $python (Join-Path $Root 'setup\fetch_models.py')
    # Deliberately not fatal: a cached model with a dead network is still a
    # perfectly deployable box, and startup will fail loudly if it is not.
    if ($LASTEXITCODE -ne 0) {
        Write-Host '::warning::model prefetch failed; startup will retry the download' -ForegroundColor Yellow
    }
}

# --------------------------------------------------------------------- start

function Start-App {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $LogDir "app-$stamp.out.log"
    $stderr = Join-Path $LogDir "app-$stamp.err.log"

    $python = Join-Path $Root 'venv\Scripts\python.exe'
    if (-not (Test-Path $python)) { Die "no venv python at $python" }

    Say 'starting app'
    # Win32_Process.Create, not Start-Process, and this is the whole reason
    # the job used to hang.
    #
    # A long-lived server started as a descendant of an Actions step keeps
    # that step's stdout/stderr handles open. The runner does not end a step
    # until those handles close, so a PERFECTLY SUCCESSFUL deploy would hang
    # forever, and cancelling would not help - the runner is blocked on a
    # handle, not on a signal. Start-Process -RedirectStandardOutput mostly
    # avoids that, but only while nothing else in the chain reintroduces
    # inheritance.
    #
    # Win32_Process.Create is not a child of this shell at all: the WMI
    # service creates it, so there is no handle to inherit and no process
    # tree for the runner to wait on. Redirection then has to happen inside
    # the command line, which is what the cmd.exe wrapper is for.
    $cmdLine = 'cmd.exe /c ""{0}" -m app.main >"{1}" 2>"{2}""' -f $python, $stdout, $stderr
    $result  = ([wmiclass]'Win32_Process').Create($cmdLine, (Join-Path $Root 'src'))
    if ($result.ReturnValue -ne 0) {
        Die "could not start the app: Win32_Process.Create returned $($result.ReturnValue)"
    }

    Say "launched, logs in $LogDir\app-$stamp.*.log"
    # The pid is deliberately NOT recorded here: Create returns cmd.exe's pid,
    # and the python underneath it is what matters. Wait-Healthy records the
    # real one once the port is bound.
    return @{ Stdout = $stdout; Stderr = $stderr }
}

# -------------------------------------------------------------------- verify

function Wait-Healthy($started) {
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    Say "waiting for $Health"

    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $Health -TimeoutSec 5
        }
        catch {
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

    # No process handle to interrogate any more, so the startup log is the
    # only account of what went wrong. Dumping it here is what keeps a
    # timeout diagnosable.
    Write-Host '--- startup stderr ---' -ForegroundColor Yellow
    Get-Content $started.Stderr -Tail 40 -ErrorAction SilentlyContinue
    Die "not healthy after ${HealthTimeoutSeconds}s"
}

# ----------------------------------------------------------------------- run

Push-Location $Root
try {
    Stop-App
    if (-not $SkipVerify) { Sync-Env; Get-Models }
    Wait-Healthy (Start-App)
    Say 'deploy complete'
}
finally {
    Pop-Location
}
