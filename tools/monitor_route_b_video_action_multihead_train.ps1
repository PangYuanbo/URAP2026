param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\route_b_official\aot_part0_video_action_multihead_train_runner',
  [string]$RunId = 'route_b_video_action_multihead_train'
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"
if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "NOT RUNNING: pid file missing: $pidFile"
  exit 0
}

$pidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
$proc = $null
if ($pidText -match '^\d+$') {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}

$meta = @{}
if (Test-Path -Path $metaFile -PathType Leaf) {
  foreach ($line in Get-Content $metaFile) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) { $meta[$parts[0]] = $parts[1] }
  }
}
$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$out = if ($meta.ContainsKey('out')) { [string]$meta['out'] } else { '' }
$epochs = if ($meta.ContainsKey('epochs')) { [int]$meta['epochs'] } else { 0 }
$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) { $lastOutput = $item.LastWriteTime }
  }
}

$epoch = 0
$batchesDone = 0
$batchesTotal = 0
$samplesDone = 0
$samplesTotal = 0
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  $progressLine = Get-Content $stdout | Where-Object { $_ -like '*video_action_multihead_train_progress*' -or $_ -like '*video_action_multihead_train_start*' -or $_ -like '*video_action_multihead_train":*' } | Select-Object -Last 1
  if ($progressLine) {
    try {
      $progress = $progressLine | ConvertFrom-Json
      if ($progress.kind -eq 'video_action_multihead_train_progress') {
        $epoch = [int]$progress.epoch
        $epochs = [int]$progress.epochs
        $batchesDone = [int]$progress.batches_done
        $batchesTotal = [int]$progress.batches_total
        $samplesDone = [int]$progress.samples_done
        $samplesTotal = [int]$progress.samples_total
      } elseif ($progress.kind -eq 'video_action_multihead_train_start') {
        $epochs = [int]$progress.epochs
        $batchesTotal = [int]$progress.batches_per_epoch
        $samplesTotal = [int]$progress.samples_total
      } elseif ($progress.video_action_multihead_train) {
        $epoch = [int]$progress.video_action_multihead_train.epoch
      }
    } catch {}
  }
}

$doneUnits = if ($batchesTotal -gt 0) { (($epoch - 1) * $batchesTotal + $batchesDone) } else { $epoch }
$totalUnits = if ($batchesTotal -gt 0 -and $epochs -gt 0) { $batchesTotal * $epochs } else { $epochs }
if ($out -and (Test-Path -Path $out -PathType Leaf)) {
  $doneUnits = $totalUnits
}

$gpuText = 'nvidia-smi unavailable'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $gpuText = (& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null) -join '; '
}
$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $doneUnits, $totalUnits)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $(if ($out -and (Test-Path -Path $out -PathType Leaf)) { 'checkpoint' } else { "epoch=$epoch batch=$batchesDone/$batchesTotal samples=$samplesDone/$samplesTotal" }))
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out={0}" -f $out)
Write-Host ("gpu={0}" -f $gpuText)
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 30
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 30
}
