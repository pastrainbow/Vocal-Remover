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
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root    = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root 'state\app.pid'
$LogDir  = Join-Path $Root 'state\logs'
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

function Stop-App {
    if (-not (Test-Path $PidFile)) { Say 'no pid file, nothing to stop'; return }

    $appPid = (Get-Content $PidFile -Raw).Trim()
    $proc   = Get-Process -Id $appPid -ErrorAction SilentlyContinue

    if ($null -eq $proc) {
        Say "pid $appPid is not running (stale pid file)"
    }
    else {
        # The pid file survives reboots, and Windows reuses pids. Starting
        # with a clean check that this is actually our python avoids killing
        # whatever unrelated process inherited the number.
        if ($proc.ProcessName -notmatch '^python') {
            Say "pid $appPid is '$($proc.ProcessName)', not python - refusing to kill it"
        }
        else {
            Say "stopping pid $appPid"
            Stop-Process -Id $appPid -Force
            # The worker holds CUDA context and a SQLite WAL; give the OS a
            # moment to tear both down before the next process claims them.
            $proc.WaitForExit(30000) | Out-Null
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
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
    Say 'run.sh --models (warm the model cache)'
    & $Bash './run.sh' '--models'
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
    # Redirecting both streams to files is load-bearing, not tidiness: a
    # process still holding the job's console handles is one the Actions
    # runner will reap when the step ends, taking the server with it.
    $proc = Start-Process -FilePath $python `
        -ArgumentList '-m', 'app.main' `
        -WorkingDirectory (Join-Path $Root 'src') `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError  $stderr `
        -WindowStyle Hidden -PassThru

    $proc.Id | Set-Content $PidFile -NoNewline
    Say "pid $($proc.Id), logs in state\logs\app-$stamp.*.log"
    return @{ Proc = $proc; Stdout = $stdout; Stderr = $stderr }
}

# -------------------------------------------------------------------- verify

function Wait-Healthy($started) {
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    Say "waiting for $Health"

    while ((Get-Date) -lt $deadline) {
        # A dead process will never become healthy. Catching it here turns a
        # five-minute timeout into an immediate failure with the real error.
        if ($started.Proc.HasExited) {
            Write-Host '--- stderr ---' -ForegroundColor Yellow
            Get-Content $started.Stderr -Tail 40 -ErrorAction SilentlyContinue
            Die "app exited during startup (code $($started.Proc.ExitCode))"
        }

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

        Say "healthy - $($r.worker.models.Count) model(s) resident"
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
