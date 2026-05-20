param(
  [string]$AutoresearchDir = "C:\Users\aaron\Desktop\URAP\autoresearch",
  [string]$RunTag = "",
  [int]$TailLines = 20
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
$metaFile = Join-Path $runDir "runner_meta.txt"
$pidFile = Join-Path $runDir "runner_pid.txt"
$statusFile = Join-Path $runDir "run_status.json"
$stdout = Join-Path $runDir "runner_stdout.txt"
$stderr = Join-Path $runDir "runner_stderr.txt"

if (-not (Test-Path -Path $metaFile -PathType Leaf)) { throw "Meta file not found: $metaFile" }

$meta = Get-Content $metaFile
$pidValue = if (Test-Path -Path $pidFile -PathType Leaf) { Get-Content $pidFile | Select-Object -First 1 } else { "" }
$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
}

$status = $null
if (Test-Path -Path $statusFile -PathType Leaf) {
  $status = Get-Content -Raw $statusFile | ConvertFrom-Json
}

$runState = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }
$done = if ($null -ne $status) { $status.done_rounds } else { "?" }
$total = if ($null -ne $status) { $status.total_rounds } else { "?" }
$currentRound = if ($null -ne $status) { $status.current_round } else { "?" }
$lastArtifact = if ($null -ne $status) { $status.last_artifact_path } else { "" }
$lastLog = if ($null -ne $status) { $status.last_log_path } else { "" }

Write-Host ("status={0}" -f $runState)
Write-Host ("done/total={0}/{1}" -f $done, $total)
Write-Host ("pid={0}" -f $pidValue)

$startedLine = $meta | Where-Object { $_ -like "started=*" } | Select-Object -First 1
if ($startedLine) { Write-Host $startedLine }
Write-Host ("current_round={0}" -f $currentRound)
Write-Host ("last_artifact={0}" -f $lastArtifact)
Write-Host ("last_log={0}" -f $lastLog)

if (Test-Path -Path $stdout -PathType Leaf) {
  $stdoutItem = Get-Item $stdout
  Write-Host ("stdout={0}" -f $stdout)
  Write-Host ("stdout_last_write={0}" -f $stdoutItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
}
if (Test-Path -Path $stderr -PathType Leaf) {
  $stderrItem = Get-Item $stderr
  Write-Host ("stderr={0}" -f $stderr)
  Write-Host ("stderr_last_write={0}" -f $stderrItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
}

if ($null -ne $status) {
  Write-Host ("state_file_state={0}" -f $status.state)
  Write-Host ("state_file_message={0}" -f $status.message)
}

if (Test-Path -Path $stdout -PathType Leaf) {
  Write-Host ""
  Write-Host "== stdout tail =="
  Get-Content -Path $stdout -Tail $TailLines -ErrorAction SilentlyContinue
}

if (Test-Path -Path $stderr -PathType Leaf) {
  Write-Host ""
  Write-Host "== stderr tail =="
  Get-Content -Path $stderr -Tail $TailLines -ErrorAction SilentlyContinue
}
