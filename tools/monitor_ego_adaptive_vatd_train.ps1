param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\ego_adaptive_vatd\train_runner',
  [string]$RunId = 'ego_adaptive_vatd_train',
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
if ($process -and $process.CommandLine -like '*train-ego-adaptive-vatd-policy*') {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $pidValue" -ErrorAction SilentlyContinue
  if ($children) {
    Write-Host "CHILD_PROCESSES:"
    foreach ($child in $children) {
      $childProc = Get-Process -Id ([int]$child.ProcessId) -ErrorAction SilentlyContinue
      $childCpu = if ($childProc) { $childProc.CPU } else { $null }
      Write-Host ("  PID={0} PPID={1} CPU={2} CMD={3}" -f $child.ProcessId, $child.ParentProcessId, $childCpu, $child.CommandLine)
    }
  }
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
$epochs = 0
if ($meta.ContainsKey('epochs')) { [int]::TryParse($meta['epochs'], [ref]$epochs) | Out-Null }

$progress = $null
if ($stdout -and (Test-Path $stdout)) {
  $progressLines = Get-Content $stdout -Tail 2000 | Where-Object {
    $_ -like '*"kind": "ego_adaptive_vatd_train_progress"*' -or
    $_ -like '*"kind": "ego_adaptive_vatd_train_start"*' -or
    $_ -like '*ego_adaptive_vatd_train*'
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

$epoch = 0
$batchesDone = 0
$batchesTotal = 0
$samplesDone = 0
$samplesTotal = 0
$routerMean = $null
if ($progress) {
  if ($progress.kind -eq 'ego_adaptive_vatd_train_progress') {
    $epoch = [int]$progress.epoch
    $epochs = [int]$progress.epochs
    $batchesDone = [int]$progress.batches_done
    $batchesTotal = [int]$progress.batches_total
    $samplesDone = [int]$progress.samples_done
    $samplesTotal = [int]$progress.samples_total
    $routerMean = $progress.router_mean
  } elseif ($progress.kind -eq 'ego_adaptive_vatd_train_start') {
    $epochs = [int]$progress.epochs
    $batchesTotal = [int]$progress.batches_per_epoch
    $samplesTotal = [int]$progress.samples_total
  } elseif ($progress.ego_adaptive_vatd_train) {
    $epoch = [int]$progress.ego_adaptive_vatd_train.epoch
    $routerMean = $progress.ego_adaptive_vatd_train.router_mean
  }
}

$doneUnits = if ($batchesTotal -gt 0) { (($epoch - 1) * $batchesTotal + $batchesDone) } else { $epoch }
$totalUnits = if ($batchesTotal -gt 0 -and $epochs -gt 0) { $batchesTotal * $epochs } else { $epochs }
if ($out -and (Test-Path $out)) {
  $lastUnit = 'ego_adaptive_vatd_checkpoint'
  $doneUnits = $totalUnits
} else {
  $lastUnit = "epoch=$epoch batch=$batchesDone/$batchesTotal samples=$samplesDone/$samplesTotal"
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
if ($routerMean) { Write-Host "router_mean: $($routerMean -join ',')" }
Write-Host "weights: $out"
if ($out -and (Test-Path $out)) {
  $weightItem = Get-Item $out
  Write-Host "weights_exists=true length=$($weightItem.Length) last_write=$($weightItem.LastWriteTime)"
} else {
  Write-Host "weights_exists=false"
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
