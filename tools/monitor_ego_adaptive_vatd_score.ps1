param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\ego_adaptive_vatd\score_runner',
  [string]$RunId = 'ego_adaptive_vatd_score',
  [int]$TailLines = 40,
  [int]$StaleSeconds = 600
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
if ($process -and $process.CommandLine -like '*score-ego-adaptive-vatd-tracklets*') {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
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

$progress = $null
if ($stdout -and (Test-Path $stdout)) {
  $progressLines = Get-Content $stdout -Tail 2000 | Where-Object {
    $_ -like '*"kind": "ego_adaptive_vatd_score_progress"*' -or
    $_ -like '*"kind": "ego_adaptive_vatd_score_start"*'
  }
  for ($i = $progressLines.Count - 1; $i -ge 0; $i--) {
    try {
      $progress = $progressLines[$i] | ConvertFrom-Json
      break
    } catch {
      $progress = $null
    }
  }
}

$batchesDone = 0
$batchesTotal = 0
$samplesDone = 0
$samplesTotal = 0
if ($progress) {
  if ($progress.kind -eq 'ego_adaptive_vatd_score_progress') {
    $batchesDone = [int]$progress.batches_done
    $batchesTotal = [int]$progress.batches_total
    $samplesDone = [int]$progress.samples_done
    $samplesTotal = [int]$progress.samples_total
  } elseif ($progress.kind -eq 'ego_adaptive_vatd_score_start') {
    $batchesTotal = [int]$progress.batches_total
    $samplesTotal = [int]$progress.samples_total
  }
}

$doneUnits = $batchesDone
$totalUnits = $batchesTotal
if ($out -and (Test-Path $out)) {
  $doneUnits = $totalUnits
  $lastUnit = 'ego_adaptive_vatd_scores'
} else {
  $lastUnit = "batch=$batchesDone/$batchesTotal samples=$samplesDone/$samplesTotal"
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $out)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $doneUnits/$totalUnits"
Write-Host "last output timestamp: $lastWrite"
if ($lastWrite) {
  $ageSeconds = [int]((Get-Date) - $lastWrite).TotalSeconds
  Write-Host "progress_age_seconds: $ageSeconds"
  Write-Host ("progress_stale: {0}" -f ($ageSeconds -gt $StaleSeconds))
  Write-Host "progress_stale_threshold_seconds: $StaleSeconds"
} else {
  Write-Host "progress_age_seconds: missing"
  Write-Host "progress_stale: unknown"
  Write-Host "progress_stale_threshold_seconds: $StaleSeconds"
}
Write-Host "last completed unit: $lastUnit"
Write-Host "score jsonl: $out"
if ($out -and (Test-Path $out)) {
  $scoreLines = (Get-Content $out | Measure-Object -Line).Lines
  Write-Host "score_lines=$scoreLines"
}
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== GPU signal =="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
  Write-Host "-- GPU process sample --"
  & nvidia-smi pmon -c 1 | Select-String -Pattern '(^#|python)'
} else {
  Write-Host "nvidia-smi not found"
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
