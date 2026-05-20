param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$RunId,
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\baselines\li_tetc_pt_pipeline\runs\detached",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
if (-not $RunId) { throw "RunId is required: -RunId xxx" }
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "NOT RUNNING"
  Write-Host "meta: $metaFile"
  exit 0
}

$metaLines = Get-Content $metaFile -ErrorAction SilentlyContinue
$runPid = if (Test-Path -Path $pidFile -PathType Leaf) { Get-Content $pidFile | Select-Object -First 1 } else { $null }
$proc = $null
if ($runPid -match '^\d+$') { $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $runPid" -ErrorAction SilentlyContinue }

$stdoutPath = (($metaLines | Where-Object { $_ -like "stdout=*" } | Select-Object -First 1) -replace '^stdout=', '')
$stderrPath = (($metaLines | Where-Object { $_ -like "stderr=*" } | Select-Object -First 1) -replace '^stderr=', '')
$startTime = (($metaLines | Where-Object { $_ -like "started=*" } | Select-Object -First 1) -replace '^started=', '')
$totalEpochsLine = ($metaLines | Where-Object { $_ -like "total_epochs=*" } | Select-Object -First 1)
$totalEpochs = if ($totalEpochsLine) { [int]($totalEpochsLine -replace '^total_epochs=', '') } else { 0 }

$doneEpoch = 0
$lastUnit = ""
foreach ($lf in @($stderrPath, $stdoutPath)) {
  if (-not $lf -or -not (Test-Path $lf)) { continue }
  $lines = Get-Content -Path $lf -Tail 400 -ErrorAction SilentlyContinue
  for ($i = $lines.Count - 1; $i -ge 0; $i--) {
    if ($lines[$i] -match 'epoch\s+([0-9]+)/([0-9]+)') {
      $doneEpoch = [int]$matches[1]
      if ($totalEpochs -eq 0) { $totalEpochs = [int]$matches[2] }
      $lastUnit = $lines[$i].Trim()
      break
    }
  }
  if ($lastUnit) { break }
}

if (-not $proc) {
  Write-Host "NOT RUNNING"
  Write-Host "done/total: $doneEpoch/$(if ($totalEpochs -gt 0) { $totalEpochs } else { 'unknown' })"
  Write-Host "pid: $runPid"
  if ($startTime) { Write-Host "start_time: $startTime" }
  if ($stderrPath -and (Test-Path $stderrPath)) { Write-Host "last_output: $((Get-Item $stderrPath).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
  Write-Host "stdout: $stdoutPath"
  Write-Host "stderr: $stderrPath"
  exit 0
}

Write-Host "RUNNING"
Write-Host "done/total: $doneEpoch/$(if ($totalEpochs -gt 0) { $totalEpochs } else { 'unknown' })"
Write-Host "pid: $runPid"
if ($startTime) { Write-Host "start_time: $startTime" }
if ($lastUnit) { Write-Host "last_completed_unit: $lastUnit" }
if ($stderrPath -and (Test-Path $stderrPath)) { Write-Host "last_output: $((Get-Item $stderrPath).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
Write-Host "stdout: $stdoutPath"
Write-Host "stderr: $stderrPath"
