param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\yolomg_val_eval_runner",
  [string]$RunId = "temporal_recovery_yolomg_val_eval",
  [int]$TailLines = 40
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
if ($process -and ($process.CommandLine -like "*$RunId.runner.ps1*" -or $process.CommandLine -like "*export_temporal_recovery_to_yolo_labels.py*" -or $process.CommandLine -like "*yolomg_eval_pred_labels.py*")) {
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
$predLabelDir = $meta["pred_label_dir"]
$evalDir = $meta["eval_dir"]
$imagesList = $meta["images_list"]
$manifest = if ($evalDir) { Join-Path $evalDir "manifest.json" } else { $null }

$total = 0
if ($imagesList -and (Test-Path $imagesList)) {
  $total = (Get-Content $imagesList | Where-Object { $_.Trim() -ne "" } | Measure-Object).Count
}
$done = 0
$lastUnit = ""
if ($manifest -and (Test-Path $manifest)) {
  $done = $total
  $lastUnit = "eval manifest complete"
} elseif ($predLabelDir -and (Test-Path $predLabelDir)) {
  $done = (Get-ChildItem $predLabelDir -Filter "*.txt" -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $lastUnit = "pred_labels=$done"
} elseif ($stdout -and (Test-Path $stdout)) {
  $progress = Get-Content $stdout -Tail 2000 | Where-Object { $_ -like '*"kind": "yolomg_eval_pred_labels_progress"*' } | Select-Object -Last 1
  if ($progress) {
    $m = [regex]::Match($progress, '"images_done":\s*(\d+)')
    if ($m.Success) { $done = [int]$m.Groups[1].Value }
    $lastUnit = $progress
  }
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $manifest)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "pred label dir: $predLabelDir"
Write-Host "eval dir: $evalDir"
Write-Host "manifest: $manifest"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== GPU signal =="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
} else {
  Write-Host "nvidia-smi not found"
}

if ($manifest -and (Test-Path $manifest)) {
  Write-Host ""
  Write-Host "== eval manifest summary =="
  $json = Get-Content $manifest -Raw | ConvertFrom-Json
  $json.summary | Format-List
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
