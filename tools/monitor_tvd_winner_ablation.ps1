param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\runs\ablation\winner_port_v1",
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP"
)

$ErrorActionPreference = "Stop"

$pidFile = Join-Path $OutputRoot "runner_pid.txt"
$metaFile = Join-Path $OutputRoot "runner_meta.txt"
$stateFile = Join-Path $OutputRoot "state.json"
$resultsCsv = Join-Path $OutputRoot "results.csv"

$stderrPath = $null
$stdoutPath = $null

if (Test-Path $metaFile) {
  Write-Host "== Meta =="
  $metaLines = Get-Content $metaFile | Select-Object -First 200
  $metaLines
  $stderrLine = ($metaLines | Where-Object { $_ -like 'stderr=*' } | Select-Object -First 1)
  if ($stderrLine) { $stderrPath = $stderrLine.Substring('stderr='.Length) }
  $stdoutLine = ($metaLines | Where-Object { $_ -like 'stdout=*' } | Select-Object -First 1)
  if ($stdoutLine) { $stdoutPath = $stdoutLine.Substring('stdout='.Length) }
}

Write-Host "== Process =="
$procLine = "NOT RUNNING (no pid file)"
if (Test-Path $pidFile) {
  $runnerPid = (Get-Content $pidFile | Select-Object -First 1)
  if ($runnerPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$runnerPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) {
      $procLine = ("RUNNING pid={0} cpu={1} ws_mb={2:n1}" -f $p.Id, $p.CPU, ($p.WorkingSet64/1MB))
    } else {
      $procLine = ("NOT RUNNING (pid file exists but pid {0} not found)" -f $runnerPid)
    }
  } else {
    $procLine = "NOT RUNNING (pid file exists but cannot parse pid)"
  }
}
Write-Host $procLine

Write-Host "== State =="
$phase = $null
$variant = $null
if (Test-Path $stateFile) {
  try {
    $state = Get-Content $stateFile -Raw | ConvertFrom-Json
    $phase = $state.phase
    $variant = $state.variant
    $state | ConvertTo-Json -Depth 6
  } catch {
    Write-Host ("failed_to_parse_state: {0}" -f $_.Exception.Message)
  }
} else {
  Write-Host "NO STATE FILE"
}

if ($phase -eq "aot" -and $variant) {
  $runName = "fulltest_conf0p2_wport_{0}" -f $variant
  $predDir = Join-Path $URAPRoot ("papers\\TransVisDrone\\runs\\val\\AOT_URAP\\{0}\\aotpredictions" -f $runName)
  $yamlDir = Join-Path $URAPRoot "papers\\TransVisDrone\\data\\AOTTestSplits_URAP"
  $totalSplits = (Get-ChildItem -Path $yamlDir -Filter "AOTTest_*.yaml" -ErrorAction SilentlyContinue | Measure-Object).Count
  $doneSplits = (Get-ChildItem -Path $predDir -Filter "predictions_split_*.pkl" -ErrorAction SilentlyContinue | Measure-Object).Count
  Write-Host "== AOT Progress =="
  Write-Host ("run_name={0}" -f $runName)
  Write-Host ("done_splits={0}/{1}" -f $doneSplits, $totalSplits)
  if (Test-Path $predDir) {
    $latest = Get-ChildItem -Path $predDir -Filter "predictions_split_*.pkl" -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 3 Name,LastWriteTime
    if ($latest) { $latest | Format-Table -AutoSize }
  } else {
    Write-Host ("NO PRED DIR: {0}" -f $predDir)
  }
}

Write-Host "== Results CSV =="
if (Test-Path $resultsCsv) {
  $n = (Get-Content $resultsCsv | Measure-Object).Count
  Write-Host ("rows={0} (incl header)" -f $n)
  Get-Content -Tail 5 $resultsCsv
} else {
  Write-Host "NO results.csv yet"
}

if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  Write-Host "== Log Tail (stderr) =="
  $item = Get-Item $stderrPath
  Write-Host ("stderr_last_write={0}" -f $item.LastWriteTime)
  $tail = Get-Content -Tail 12 $stderrPath
  $last = ($tail | Where-Object { $_.Trim() } | Select-Object -Last 1)
  if ($last) { Write-Host $last }
}

Write-Host "== GPU Signal =="
try {
  $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
  if ($gpu) { Write-Host ("gpu_util_pct,mem_used_mb,mem_total_mb={0}" -f $gpu.Trim()) }
} catch {
  Write-Host ("nvidia-smi_failed: {0}" -f $_.Exception.Message)
}

