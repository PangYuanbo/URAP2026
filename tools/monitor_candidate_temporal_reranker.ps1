param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\candidate_temporal_reranker_yolomg_val_runner",
  [string]$RunId = "candidate_temporal_reranker_yolomg_val",
  [int]$TailLines = 80
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $RunRoot "$RunId.pid"
$metaFile = Join-Path $RunRoot "$RunId.meta.txt"

Write-Host "== Meta =="
if (Test-Path $metaFile) {
  Get-Content $metaFile
} else {
  Write-Host "meta missing: $metaFile"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf("=")
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$pidValue = $null
if (Test-Path $pidFile) {
  $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
}
$process = $null
if ($pidValue) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

if ($process -and $process.CommandLine -like "*train_candidate_temporal_reranker.py*") {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$outSummary = $meta["out_summary"]
$outLabelDir = $meta["out_label_dir"]
$epochs = 0
if ($meta.ContainsKey("epochs") -and $meta["epochs"] -match "^\d+$") { $epochs = [int]$meta["epochs"] }

$done = 0
$total = $epochs
$lastUnit = ""
if ($outSummary -and (Test-Path $outSummary)) {
  $done = $total
  $lastUnit = "summary.json written"
} elseif ($stdout -and (Test-Path $stdout)) {
  $progress = Get-Content $stdout -Tail 2000 -ErrorAction SilentlyContinue | Where-Object { $_ -like '*"kind": "candidate_reranker_progress"*' } | Select-Object -Last 1
  if ($progress) {
    $lastUnit = $progress
    $m = [regex]::Match($progress, '"epoch":\s*(\d+)')
    if ($m.Success) { $done = [int]$m.Groups[1].Value }
    $t = [regex]::Match($progress, '"epochs":\s*(\d+)')
    if ($t.Success) { $total = [int]$t.Groups[1].Value }
  } else {
    $lastUnit = (Get-Content $stdout -Tail 1 -ErrorAction SilentlyContinue)
  }
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $outSummary)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

$labels = 0
if ($outLabelDir -and (Test-Path $outLabelDir)) {
  $labels = (Get-ChildItem -Path $outLabelDir -Filter "*.txt" -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "out summary: $outSummary"
Write-Host "out label dir: $outLabelDir"
Write-Host "pred labels: $labels"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

if ($outSummary -and (Test-Path $outSummary)) {
  Write-Host ""
  Write-Host "== summary =="
  $summary = Get-Content $outSummary -Raw | ConvertFrom-Json
  $lastHistory = $summary.history | Select-Object -Last 1
  [pscustomobject]@{
    train_candidates = $summary.train_candidates
    train_positive = $summary.train_positive
    train_negative = $summary.train_negative
    test_candidates = $summary.test_candidates
    frames_written = $summary.label_summary.frames_written
    last_epoch = $lastHistory.epoch
    last_loss = $lastHistory.loss
  } | Format-List
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
