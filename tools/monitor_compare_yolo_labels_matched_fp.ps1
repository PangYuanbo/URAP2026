param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\matched_fp_compare_runner",
  [string]$RunId = "compare_yolo_labels_matched_fp",
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

if ($process -and $process.CommandLine -like "*compare_yolo_labels_matched_fp.py*") {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$imagesList = $meta["images_list"]
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$outJson = $meta["out_json"]
$outCsv = $meta["out_csv"]

$total = 0
if ($imagesList -and (Test-Path $imagesList)) {
  $total = (Get-Content $imagesList | Where-Object { $_.Trim() -ne "" } | Measure-Object).Count
}
if ($meta.ContainsKey("max_frames") -and $meta["max_frames"] -match "^\d+$") {
  $maxFrames = [int]$meta["max_frames"]
  if ($maxFrames -gt 0 -and ($total -eq 0 -or $maxFrames -lt $total)) { $total = $maxFrames }
}

$done = 0
$lastUnit = ""
if ($outJson -and (Test-Path $outJson)) {
  $done = $total
  $lastUnit = "compare.json written"
} elseif ($stdout -and (Test-Path $stdout)) {
  $progress = Get-Content $stdout -Tail 2000 -ErrorAction SilentlyContinue | Where-Object { $_ -like '*"kind": "matched_fp_progress"*' } | Select-Object -Last 1
  if ($progress) {
    $lastUnit = $progress
    $m = [regex]::Match($progress, '"done":\s*(\d+)')
    if ($m.Success) { $done = [int]$m.Groups[1].Value }
    $t = [regex]::Match($progress, '"total":\s*(\d+)')
    if ($t.Success) { $total = [int]$t.Groups[1].Value }
  } else {
    $lastUnit = (Get-Content $stdout -Tail 1 -ErrorAction SilentlyContinue)
  }
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $outJson, $outCsv)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "out json: $outJson"
Write-Host "out csv: $outCsv"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
