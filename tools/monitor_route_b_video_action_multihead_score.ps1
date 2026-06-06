param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'route_b_video_action_multihead_score',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\video_action_multihead_score_runner'),
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

Write-Host '== Meta =='
if (Test-Path $metaFile) {
  Get-Content $metaFile
} else {
  Write-Host "meta missing: $metaFile"
}

$pidValue = $null
if (Test-Path $pidFile) {
  $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
}

$process = $null
if ($pidValue) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}
$workerProcess = $null
$children = @()
if ($pidValue) {
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $pidValue" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    if ($child.CommandLine -like '*score-video-action-multihead-tracklets*') {
      $workerProcess = $child
      break
    }
  }
}
if (-not $workerProcess -and $process -and $process.CommandLine -like '*score-video-action-multihead-tracklets*') {
  $workerProcess = $process
}
if ($workerProcess) {
  $workerPid = [int]$workerProcess.ProcessId
  Write-Host ""
  Write-Host "RUNNING=true PID=$workerPid"
  if ($workerPid -ne [int]$pidValue) { Write-Host "LAUNCHER_PID=$pidValue" }
  Write-Host "PID_START=$((Get-Process -Id $workerPid -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($workerProcess.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf('=')
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}
$stdout = $meta['stdout']
$stderr = $meta['stderr']
$out = $meta['out']
$trackletJsonl = $meta['tracklet_jsonl']

$progress = $null
if ($stdout -and (Test-Path $stdout)) {
  $progressLines = Select-String -Path $stdout -Pattern '"kind": "video_action_multihead_score_progress"' -ErrorAction SilentlyContinue
  if ($progressLines.Count -gt 0) {
    $last = $progressLines[$progressLines.Count - 1].Line
    try { $progress = $last | ConvertFrom-Json } catch { $progress = $null }
  }
}

$done = 0
$total = 0
$lastUnit = ''
if ($progress) {
  $done = [int]$progress.batches_done
  $total = [int]$progress.batches_total
  $lastUnit = "batch_$done"
}
if ($out -and (Test-Path $out)) {
  $lastUnit = 'video_action_multihead_scores_jsonl'
}

$scoreLines = 0
if ($out -and (Test-Path $out)) {
  $scoreLines = (Get-Content $out | Measure-Object -Line).Lines
}
$trackletLines = 0
if ($trackletJsonl -and (Test-Path $trackletJsonl)) {
  $trackletLines = (Get-Content $trackletJsonl | Measure-Object -Line).Lines
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $out)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "score jsonl lines: $scoreLines"
Write-Host "input tracklet lines: $trackletLines"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "scores: $out"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== GPU signal =="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits
} else {
  Write-Host "nvidia-smi not found"
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
