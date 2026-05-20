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
$outJson = (($metaLines | Where-Object { $_ -like "out_json=*" } | Select-Object -First 1) -replace '^out_json=', '')
$startTime = (($metaLines | Where-Object { $_ -like "started=*" } | Select-Object -First 1) -replace '^started=', '')

$doneFrames = "unknown"
$lastUnit = ""
if ($outJson -and (Test-Path $outJson)) {
  try {
    $j = Get-Content -Raw -Path $outJson | ConvertFrom-Json
    if ($null -ne $j.frames) {
      $doneFrames = [string]$j.frames
      $lastUnit = 'results_json_written'
    }
  } catch {}
}

if (-not $proc) {
  Write-Host "NOT RUNNING"
  Write-Host ("done/total: {0}/unknown" -f $doneFrames)
  Write-Host "pid: $runPid"
  if ($startTime) { Write-Host "start_time: $startTime" }
  if ($lastUnit) { Write-Host "last_completed_unit: $lastUnit" }
  if ($stderrPath -and (Test-Path $stderrPath)) { Write-Host "last_output: $((Get-Item $stderrPath).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
  Write-Host "stdout: $stdoutPath"
  Write-Host "stderr: $stderrPath"
  if ($outJson) { Write-Host "out_json: $outJson" }
  exit 0
}

Write-Host "RUNNING"
Write-Host ("done/total: {0}/unknown" -f $doneFrames)
Write-Host "pid: $runPid"
if ($startTime) { Write-Host "start_time: $startTime" }
if ($lastUnit) { Write-Host "last_completed_unit: $lastUnit" }
if ($stderrPath -and (Test-Path $stderrPath)) { Write-Host "last_output: $((Get-Item $stderrPath).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
Write-Host "stdout: $stdoutPath"
Write-Host "stderr: $stderrPath"
if ($outJson) { Write-Host "out_json: $outJson" }
