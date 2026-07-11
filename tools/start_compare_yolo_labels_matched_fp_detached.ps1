param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$ImagesList = "U:\URAP_datasets\ARD100_YOLOMG\val.txt",
  [string[]]$Method = @(
    "no_ncc=U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\yolomg_val_full_no_ncc_eval\pred_labels",
    "old_ncc=U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\yolomg_val_full_eval\pred_labels"
  ),
  [string]$BaselineMethod = "no_ncc",
  [double]$BaselineThreshold = 0.001,
  [string]$Thresholds = "0.001 0.01 0.05 0.1 0.2 0.3 0.5 0.7 0.9",
  [int]$ImageWidth = 1920,
  [int]$ImageHeight = 1080,
  [int]$MaxFrames = 0,
  [int]$ProgressEvery = 2000,
  [string]$OutRoot = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\matched_fp_compare",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\matched_fp_compare_runner",
  [string]$RunId = "compare_yolo_labels_matched_fp"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Missing python: $Python" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "Missing images list: $ImagesList" }
foreach ($entry in $Method) {
  $parts = $entry -split "=", 2
  if ($parts.Count -ne 2) { throw "Method must use NAME=PATH: $entry" }
  if (-not (Test-Path -Path $parts[1] -PathType Container)) { throw "Missing method label dir: $($parts[1])" }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$logsDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunRoot "$RunId.pid"
$metaFile = Join-Path $RunRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*compare_yolo_labels_matched_fp.py*" -and $existing.CommandLine -like "*$OutRoot*") {
      Write-Host "Matched-FP comparison already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
    Write-Host "Previous matched-FP comparison PID is NOT RUNNING: pid=$existingPid"
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"
$outJson = Join-Path $OutRoot "compare.json"
$outCsv = Join-Path $OutRoot "sweep.csv"

$argList = @(
  "tools\compare_yolo_labels_matched_fp.py",
  "--images-list", $ImagesList,
  "--baseline-method", $BaselineMethod,
  "--baseline-threshold", [string]$BaselineThreshold,
  "--thresholds"
)
$argList += ($Thresholds -split "\s+" | Where-Object { $_ })
if ($ImageWidth -gt 0 -and $ImageHeight -gt 0) {
  $argList += @("--image-width", [string]$ImageWidth, "--image-height", [string]$ImageHeight)
}
if ($MaxFrames -gt 0) { $argList += @("--max-frames", [string]$MaxFrames) }
$argList += @("--progress-every", [string]$ProgressEvery)
foreach ($entry in $Method) { $argList += @("--method", $entry) }
$argList += @("--out-json", $outJson, "--out-csv", $outCsv)

$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $URAPRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "urap_root=$URAPRoot",
  "images_list=$ImagesList",
  "method=$($Method -join ';')",
  "baseline_method=$BaselineMethod",
  "baseline_threshold=$BaselineThreshold",
  "thresholds=$Thresholds",
  "image_width=$ImageWidth",
  "image_height=$ImageHeight",
  "max_frames=$MaxFrames",
  "progress_every=$ProgressEvery",
  "out_root=$OutRoot",
  "out_json=$outJson",
  "out_csv=$outCsv",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached matched-FP comparison."
Get-Content $metaFile
