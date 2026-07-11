param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\yolomg_val_runner",
  [string]$RunId = "temporal_recovery_yolomg_val",
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

if ($process -and $process.CommandLine -like "*run_temporal_recovery_pipeline.py*") {
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
$outDir = $meta["out_dir"]
$imagesList = $meta["images_list"]
$trajectory = if ($outDir) { Join-Path $outDir "trajectory.csv" } else { $null }
$jsonOut = if ($outDir) { Join-Path $outDir "trajectory.json" } else { $null }

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
if ($trajectory -and (Test-Path $trajectory)) {
  $done = [Math]::Max(0, ((Get-Content $trajectory | Measure-Object).Count - 1))
  $lastUnit = "trajectory.csv rows=$done"
} elseif ($stdout -and (Test-Path $stdout)) {
  $progress = Get-Content $stdout -Tail 2000 | Where-Object { $_ -like '*"kind": "temporal_recovery_progress"*' } | Select-Object -Last 1
  if ($progress) {
    $lastUnit = $progress
    $m = [regex]::Match($progress, '"frames_read":\s*(\d+)')
    if ($m.Success) { $done = [int]$m.Groups[1].Value }
  }
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $trajectory, $jsonOut)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "out dir: $outDir"
Write-Host "trajectory: $trajectory"
Write-Host "trajectory json: $jsonOut"
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

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }
