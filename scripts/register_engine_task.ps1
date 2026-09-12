<#
.SYNOPSIS
    Registers the multi-algo paper-trading engine as a Windows scheduled task.

.DESCRIPTION
    `python -m coinbase.ga.paper_engine` holds a process open: a decision loop
    that wakes on each candle boundary and a price loop that marks open
    positions to market, with the dashboard served off the same event loop.
    That is what keeps http://127.0.0.1:8787 answering continuously, and it is
    why this is registered as a long-lived task rather than a repeating one.

    Contrast `register_paper_task.ps1`, which schedules the ONE-tick
    `paper_trading.py` every N minutes. That form is more robust — nothing
    long-lived to wedge — but it papers a single pair from config.yaml's
    `paper:` section and serves no dashboard. This one runs every algo in
    paper.yaml and keeps the UI up.

    The trigger is AtLogOn, not AtStartup: the engine reads credentials from
    ~/.coinbase and ~/.binance and needs the network, so it has to run as the
    logged-on user. A startup trigger would require storing that account's
    password in the task.

    A second trigger repeats every -HeartbeatMinutes (15 by default) for as
    long as the machine is on. Waking from sleep is not a logon, so AtLogOn by
    itself leaves a killed engine down until the next sign-in; the heartbeat
    starts it instead, and is a free no-op whenever it is already running.

    Re-running this script replaces any existing task of the same name.

    Restarting goes through -Restart, not Stop-ScheduledTask followed by
    Start-ScheduledTask. Stopping the task kills only its powershell wrapper
    and orphans the python underneath, which keeps holding the dashboard port
    long enough for the replacement to die on bind.

    Every run brackets itself in the log with `=== ENGINE START` and
    `=== ENGINE EXIT` lines, and a stop this script performs writes
    `=== ENGINE STOPPED`. Grep those three to see, from the log alone, whether
    the engine is up and how each previous run ended:

        Select-String -Path <log> -Pattern '=== ENGINE'

.EXAMPLE
    ./scripts/register_engine_task.ps1
    ./scripts/register_engine_task.ps1 -Start
    ./scripts/register_engine_task.ps1 -Restart
    ./scripts/register_engine_task.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string] $TaskName = "HippoPaperEngine",
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string] $Python   = "",
    [string] $LogFile  = "",
    [int]    $Port     = 8787,
    [int]    $HeartbeatMinutes = 15,
    [switch] $Start,
    [switch] $Restart,
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

if (-not $LogFile) {
    $LogFile = Join-Path $HOME ".coinbase\ga\paper\logs\engine_task.log"
}
# Guarded: a -LogFile given as a bare filename has no directory part, and
# New-Item on an empty path throws — which under $ErrorActionPreference = Stop
# would take down a -Remove that used to work.
$logDir = Split-Path $LogFile
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# ── The log ────────────────────────────────────────────────────────────
# Every line this script writes to the log goes through here, because the
# task's own `*>>` redirect writes UTF-16LE (that is what Windows PowerShell
# 5.1 does for redirection, whatever PYTHONIOENCODING says) and a file with
# two encodings in it is worse than a file with one awkward one. `unicode` is
# that same UTF-16LE, so the two writers agree.
#
# It retries, and it never throws. The engine being stopped is exactly when
# the log is contended — the wrapper's own redirect still holds the handle for
# a moment after the kill:
#
#     Out-File : The process cannot access the file ... because it is being
#     used by another process.
#
# With $ErrorActionPreference = Stop that aborted a re-registration halfway,
# killing the engine and then refusing to register its replacement. A line
# recording what happened must not be able to change what happens.

function Write-EngineLog {
    param([string] $LogFile, [string] $Message, [int] $Attempts = 10)

    $stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $line  = "=== ENGINE $Message at $stamp"
    for ($i = 0; $i -lt $Attempts; $i++) {
        try {
            $line | Out-File -FilePath $LogFile -Append -Encoding unicode -ErrorAction Stop
            return
        } catch {
            Start-Sleep -Milliseconds 300
        }
    }
    Write-Warning "Could not write to '$LogFile' - it stayed locked. Line dropped: $line"
}

# ── Stopping ───────────────────────────────────────────────────────────
# `Stop-ScheduledTask` kills the task's own process — the powershell wrapper —
# and leaves the python it launched running as an orphan, still holding the
# dashboard port. Starting again then dies on bind:
#
#     OSError: [Errno 10048] error while attempting to bind on ('127.0.0.1', 8787)
#
# So the stop is not complete until the port is free. Engine processes are
# matched on their command line rather than on whoever holds the port, so this
# never kills an unrelated listener that happens to be sitting on it.

function Stop-Engine {
    param([string] $TaskName, [int] $Port, [string] $LogFile, [int] $TimeoutSeconds = 20)

    $task    = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $running = $task -and $task.State -eq 'Running'
    if ($task) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    }

    $orphans = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -and $_.CommandLine -match 'coinbase\.ga\.paper_engine' })
    foreach ($o in $orphans) {
        Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
    }

    $freed    = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
            $freed = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }

    # A forced kill gives the wrapper no chance to run its own `finally`, so
    # without this line a deliberate restart and a crash look identical in the
    # log: an ENGINE START with no ENGINE EXIT above it. Written only when
    # something was actually killed, so a stop against an already-dead engine
    # stays silent — and written after the wait above, by which point the
    # wrapper has released the log file it was redirecting into.
    if ($LogFile -and ($running -or $orphans.Count -gt 0)) {
        Write-EngineLog -LogFile $LogFile -Message "STOPPED by register_engine_task.ps1, killed $($orphans.Count) python process(es)"
    }

    return $freed
}

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        [void] (Stop-Engine -TaskName $TaskName -Port $Port -LogFile $LogFile)
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName'."
    }
    return
}

if ($Restart) {
    if (-not (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
        throw "No scheduled task named '$TaskName' - register it first."
    }
    if (-not (Stop-Engine -TaskName $TaskName -Port $Port -LogFile $LogFile)) {
        $held = (Get-NetTCPConnection -LocalPort $Port -State Listen | Select-Object -First 1).OwningProcess
        throw "Port $Port is still held by PID $held - not starting, it would fail to bind."
    }
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Restarted '$TaskName'. Dashboard: http://127.0.0.1:$Port"
    return
}

# Prefer this tree's virtualenv, then the main checkout's, then PATH.
#
# The middle case is the one that matters: a git worktree has no .venv of its
# own, so $RepoRoot\.venv misses and PATH on Windows resolves to the Microsoft
# Store's python.exe stub — a real file that passes Test-Path and then refuses
# to run. `git rev-parse --git-common-dir` points at the main checkout's .git
# whatever tree we were invoked from, so its parent is where the venv lives.
if (-not $Python) {
    $candidates = @(Join-Path $RepoRoot ".venv\Scripts\python.exe")

    $common = & git -C $RepoRoot rev-parse --git-common-dir 2>$null
    if ($LASTEXITCODE -eq 0 -and $common) {
        if (-not [System.IO.Path]::IsPathRooted($common)) {
            $common = Join-Path $RepoRoot $common
        }
        $main = Split-Path (Resolve-Path $common).Path
        $candidates += (Join-Path $main ".venv\Scripts\python.exe")
    }

    $candidates += (Get-Command python -ErrorAction SilentlyContinue).Source
    $Python = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Python) { throw "No python interpreter found." }

# Test-Path is not enough — the Store stub exists and still fails. Prove the
# interpreter runs before handing it to Task Scheduler, where the failure would
# only surface as a dead task and a log file.
$probe = & $Python -c "import sys; print(sys.executable)" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "'$Python' is not a working interpreter: $probe"
}

# ── What the task actually runs ────────────────────────────────────────
# Task Scheduler captures no stdout of its own, so the engine's output is
# redirected to a log file. `*>>` appends every stream, errors included.
# PYTHONIOENCODING is set because the task's console runs under the system
# codepage, which mangles the non-ASCII characters in the engine's output.
#
# The redirect hangs off `& { ... }` rather than off the python call, and that
# placement is the whole point of this block. Bound to python alone, it caught
# only what python wrote — so every way of dying BEFORE python (a $RepoRoot
# that has gone missing, an interpreter deleted since registration, the
# wrapper being killed at logon) produced an empty log and a task result of
# 0xC000013A with nothing to explain it. That happened on 2026-09-09: the
# engine stopped at 00:43, the logon trigger fired the next morning, the
# wrapper was gone two minutes later, and the log's last line was from the
# night before. Around the whole block, the failure lands in the log.
#
# START and EXIT bracket every run, so the log answers "was it running, and
# how did it stop" by grep rather than by comparing file timestamps:
#
#     === ENGINE START at 2026-09-10T07:01:06Z wrapper-pid 17772 commit c9b2f99
#     === ENGINE EXIT at 2026-09-10T09:15:44Z code 0 after 8078s
#
# A START with no EXIT above it means the previous run was killed outright —
# `finally` does not run when the process is terminated rather than asked to
# stop. Stop-Engine writes its own STOPPED line for exactly that reason, so an
# unexplained gap is a crash and not a restart.
#
# The block is dot-sourced and its exit code re-raised, which is load-bearing
# rather than stylistic. `& { ... }` runs in a child scope, and its last
# statement is the `finally`'s Write-Output — so the wrapper would exit 0 no
# matter how python died, and -RestartCount below would never fire: the
# scheduler only restarts a task whose result was non-zero. The log would say
# `code 1` while Task Scheduler recorded success, and the engine would stay
# down until the next logon. That is the exact 2026-09-09 outage this block
# was written to expose, so making it unrecoverable while documenting it
# would have been a poor trade. Dot-sourcing keeps $code visible afterwards;
# the non-numeric sentinel becomes 1, since `exit 'never-started'` is 0.
#
# The `catch` is not decoration. A terminating error inside `& { ... } *>>`
# is written to the host, not down the redirect — verified: a missing
# $RepoRoot left `code never-started` in the log with the reason nowhere in
# it, since a scheduled task has no console for it to reach. Catching it and
# writing the message ourselves is what puts the cause next to the effect.
#
# `Set-Location` carries -ErrorAction Stop rather than leaning on a
# preference variable. The task's powershell runs at the default Continue, so
# without it a missing $RepoRoot logged its error and then launched python
# anyway, from whatever directory the task happened to start in — a
# ModuleNotFoundError several lines below the real cause. Setting the
# preference to Stop for the whole block is not an option: `*>>` wraps a
# native command's stderr in an ErrorRecord, and the engine writes to stderr
# on every log line, so the first one would terminate the run.
#
# Every string inside is single-quoted: the whole thing is embedded in a
# double-quoted -Command argument below, and a double quote in here would have
# to survive two rounds of parsing. Runtime variables are backtick-escaped so
# they resolve when the task runs; $RepoRoot, $Python and $LogFile are not, so
# they bake in now.
$inner = ". { " +
    "`$started = Get-Date; " +
    "`$env:PYTHONIOENCODING = 'utf-8'; " +
    "`$sha = 'unknown'; " +
    "try { `$sha = (& git -C '$RepoRoot' rev-parse --short HEAD 2>`$null) } catch { }; " +
    "if (-not `$sha) { `$sha = 'unknown' }; " +
    "`$code = 'never-started'; " +
    "try { " +
        "Write-Output ('=== ENGINE START at {0} wrapper-pid {1} commit {2}' -f " +
            "`$started.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), `$PID, `$sha); " +
        "Set-Location -Path '$RepoRoot' -ErrorAction Stop; " +
        "& '$Python' -m coinbase.ga.paper_engine; " +
        "`$code = `$LASTEXITCODE " +
    "} catch { " +
        "Write-Output ('=== ENGINE FAILED {0}' -f `$_.Exception.Message) " +
    "} finally { " +
        "Write-Output ('=== ENGINE EXIT at {0} code {1} after {2}s' -f " +
            "(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), `$code, " +
            "[int]((Get-Date) - `$started).TotalSeconds) " +
    "} " +
"} *>> '$LogFile'; " +
"if (`$code -is [int]) { exit `$code }; exit 1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"$inner`"" `
    -WorkingDirectory $RepoRoot

# ── Triggers ───────────────────────────────────────────────────────────
# Two, and the second is what keeps the engine alive across a closed lid.
#
# AtLogOn alone leaves the engine down for as long as the machine stays logged
# on without it running, which is exactly what happened on 2026-09-11: the
# process was killed at 15:30, the machine slept, and waking it is not a logon,
# so nothing re-fired. It was still down 19 hours later. Restart-on-failure
# does not cover it either — the machine was not running to notice the failure.
#
# So a second trigger fires every $HeartbeatMinutes, forever. It costs nothing
# while the engine is up: MultipleInstances IgnoreNew makes Task Scheduler
# refuse the duplicate before it launches anything, so there is no second
# process, no fight over port 8787, and not even a log line. When the engine is
# down, the same firing starts it. One mechanism covering sleep, hibernate, a
# crash and an outright kill alike, with no liveness check to get wrong.
#
# It does NOT cover a process that is alive but no longer ticking — IgnoreNew
# cannot tell a working engine from a wedged one. That remains this shape's
# known weakness, and needs a freshness check rather than a trigger.
#
# -RepetitionDuration is deliberately NOT passed. An omitted duration is how
# Task Scheduler encodes "repeat indefinitely" — it registers as an empty
# <Duration> and the repetition never lapses. The widely repeated idiom for
# this, `-RepetitionDuration ([TimeSpan]::MaxValue)`, does not work: it
# serializes to P99999999DT23H59M59S and registration fails outright with
#
#     The task XML contains a value which is incorrectly formatted or out of
#     range. (12,42):Duration:P99999999DT23H59M59S
#
# which would have left the engine with no heartbeat at all. ([TimeSpan]::Zero
# is rejected the same way; a finite P3650D registers but eventually lapses.)
$logon     = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$heartbeat = New-ScheduledTaskTrigger `
    -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes $HeartbeatMinutes)
$trigger   = @($logon, $heartbeat)

# ExecutionTimeLimit 0 means "no limit" — the default of 3 days would kill the
# engine mid-run. IgnoreNew keeps a second instance from fighting the first
# over port 8787 and the same state files. The restart settings bring the
# engine back if it exits on its own; they do nothing for a wedged process
# that stays alive, which is this shape's known weakness.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    if (-not (Stop-Engine -TaskName $TaskName -Port $Port -LogFile $LogFile)) {
        $held = (Get-NetTCPConnection -LocalPort $Port -State Listen | Select-Object -First 1).OwningProcess
        throw "Port $Port is still held by PID $held - re-registering now would leave a task that cannot bind."
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Multi-algo paper-trading engine and dashboard. Runs continuously; starts at logon." | Out-Null

Write-Host "Registered '$TaskName'"
Write-Host "  python   : $Python"
Write-Host "  repo     : $RepoRoot"
Write-Host "  starts   : at logon, every $HeartbeatMinutes min if not running, and within 1 min if it exits"
Write-Host "  dashboard: http://127.0.0.1:$Port"
Write-Host "  log      : $LogFile"

if ($Start) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host ""
    Write-Host "Started '$TaskName'."
} else {
    Write-Host ""
    Write-Host "Start now:    Start-ScheduledTask -TaskName '$TaskName'"
}

Write-Host "Restart:      ./scripts/register_engine_task.ps1 -Restart"
Write-Host "Watch log:    Get-Content '$LogFile' -Wait -Tail 20"
Write-Host "Lifecycle:    Select-String -Path '$LogFile' -Pattern '=== ENGINE' | Select-Object -Last 20"
Write-Host "Remove:       ./scripts/register_engine_task.ps1 -Remove"
