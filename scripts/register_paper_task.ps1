<#
.SYNOPSIS
    Registers the paper-trading tick as a Windows scheduled task.

.DESCRIPTION
    `python -m coinbase.ga.paper_trading` runs ONE tick and exits. It acts only
    on a candle it has not already acted on, so scheduling it far more often
    than the trading granularity is free: every 30 minutes against SIX_HOUR
    candles means 11 of every 12 runs are no-ops that exit immediately.

    Polling like this beats aligning the schedule to UTC candle closes: no DST
    arithmetic, and a machine that was asleep at the boundary acts late on the
    next run instead of skipping the candle entirely.

    Re-running this script replaces any existing task of the same name.

.EXAMPLE
    ./scripts/register_paper_task.ps1
    ./scripts/register_paper_task.ps1 -IntervalMinutes 15
    ./scripts/register_paper_task.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string] $TaskName        = "HippoPaperTrading",
    [int]    $IntervalMinutes = 30,
    [string] $RepoRoot        = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string] $Python          = "",
    [string] $LogFile         = "",
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName'."
    }
    return
}

# Prefer the repo's virtualenv, then whatever python is on PATH.
if (-not $Python) {
    $venv = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path $venv) { $venv } else { (Get-Command python).Source }
}
if (-not (Test-Path $Python)) { throw "Python not found at '$Python'" }

if (-not $LogFile) {
    $LogFile = Join-Path $HOME ".coinbase\ga\paper_task.log"
}
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

# Task Scheduler captures no stdout of its own, so the tick's output is
# redirected to a log file. `*>>` appends every stream, errors included.
$inner = "Set-Location '$RepoRoot'; & '$Python' -m coinbase.ga.paper_trading *>> '$LogFile'"
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$inner`"" `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)

# StartWhenAvailable catches up after downtime; the battery settings stop
# Windows from silently skipping the task on an unplugged laptop.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Paper-trading tick for the GA strategy. Runs one candle's decision and exits." | Out-Null

Write-Host "Registered '$TaskName'"
Write-Host "  python   : $Python"
Write-Host "  repo     : $RepoRoot"
Write-Host "  every    : $IntervalMinutes minute(s)"
Write-Host "  log      : $LogFile"
Write-Host ""
Write-Host "Run now:      Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Watch log:    Get-Content '$LogFile' -Wait -Tail 20"
Write-Host "Remove:       ./scripts/register_paper_task.ps1 -Remove"
