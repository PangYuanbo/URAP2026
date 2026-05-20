param(
  [string]$AutoresearchDir = "C:\Users\aaron\Desktop\URAP\autoresearch",
  [string]$RunTag = ""
)

$ErrorActionPreference = "Stop"

$runsRoot = Join-Path $AutoresearchDir "runs"
if (-not (Test-Path -Path $runsRoot -PathType Container)) { throw "Runs root not found: $runsRoot" }

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $latestRunFile = Join-Path $runsRoot "latest_run.txt"
  if (-not (Test-Path -Path $latestRunFile -PathType Leaf)) { throw "No latest run file found: $latestRunFile" }
  $RunTag = (Get-Content $latestRunFile | Select-Object -First 1).Trim()
}

$runDir = Join-Path $runsRoot $RunTag
$pidFile = Join-Path $runDir "runner_pid.txt"
$statusFile = Join-Path $runDir "run_status.json"

if (-not (Test-Path -Path $pidFile -PathType Leaf)) { throw "PID file not found: $pidFile" }
$pidValue = (Get-Content $pidFile | Select-Object -First 1).Trim()
if ($pidValue -notmatch '^\d+$') { throw "Invalid PID value: $pidValue" }

$proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
if ($null -eq $proc) {
  Write-Host ("NOT RUNNING pid={0}" -f $pidValue)
  exit 0
}

Stop-Process -Id ([int]$pidValue) -Force

if (Test-Path -Path $statusFile -PathType Leaf) {
  $status = Get-Content -Raw $statusFile | ConvertFrom-Json
  $status.state = "stopped"
  $status.last_updated = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $status.message = "Process stopped by stop_autoresearch_codex.ps1"
  ($status | ConvertTo-Json -Depth 6) | Set-Content -Encoding utf8 -Path $statusFile
}

Write-Host ("Stopped run_tag={0} pid={1}" -f $RunTag, $pidValue)
