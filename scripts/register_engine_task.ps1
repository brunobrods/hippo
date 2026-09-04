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

    Re-running this script replaces any existing task of the same name.

.EXAMPLE
    ./scripts/register_engine_task.ps1
    ./scripts/register_engine_task.ps1 -Start
    ./scripts/register_engine_task.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string] $TaskName = "HippoPaperEngine",
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string] $Python   = "",
    [string] $LogFile  = "",
    [switch] $Start,
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask   -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName'."
    }
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

if (-not $LogFile) {
    $LogFile = Join-Path $HOME ".coinbase\ga\paper\logs\engine_task.log"
}
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

# Task Scheduler captures no stdout of its own, so the engine's output is
# redirected to a log file. `*>>` appends every stream, errors included.
# PYTHONIOENCODING is set because the task's console runs under the system
# codepage, which mangles the non-ASCII characters in the engine's output.
$inner  = "`$env:PYTHONIOENCODING='utf-8'; Set-Location '$RepoRoot'; & '$Python' -m coinbase.ga.paper_engine *>> '$LogFile'"
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"$inner`"" `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

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
    Stop-ScheduledTask       -TaskName $TaskName -ErrorAction SilentlyContinue
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
Write-Host "  starts   : at logon, and restarts within 1 min if it exits"
Write-Host "  dashboard: http://127.0.0.1:8787"
Write-Host "  log      : $LogFile"

if ($Start) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host ""
    Write-Host "Started '$TaskName'."
} else {
    Write-Host ""
    Write-Host "Start now:    Start-ScheduledTask -TaskName '$TaskName'"
}

Write-Host "Stop:         Stop-ScheduledTask  -TaskName '$TaskName'"
Write-Host "Watch log:    Get-Content '$LogFile' -Wait -Tail 20"
Write-Host "Remove:       ./scripts/register_engine_task.ps1 -Remove"
