param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$ImagesList = "U:\URAP_datasets\ARD100_YOLOMG\val.txt",
  [string]$TrajectoryCsv = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\yolomg_val_full\trajectory.csv",
  [string]$OutRoot = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\yolomg_val_full_eval",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\yolomg_val_eval_runner",
  [string]$RunId = "temporal_recovery_yolomg_val_eval",
  [double]$ConfThres = 0.001,
  [double]$MatchIou = 0.5
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Missing python: $Python" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "Missing images list: $ImagesList" }
if (-not (Test-Path -Path $TrajectoryCsv -PathType Leaf)) { throw "Missing trajectory csv: $TrajectoryCsv" }

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$logsDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunRoot "$RunId.pid"
$metaFile = Join-Path $RunRoot "$RunId.meta.txt"
$predLabelDir = Join-Path $OutRoot "pred_labels"
$evalDir = Join-Path $OutRoot "eval"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*export_temporal_recovery_to_yolo_labels.py*") {
      Write-Host "Temporal recovery YOLOMG eval already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
    Write-Host "Previous temporal recovery eval PID is NOT RUNNING: pid=$existingPid"
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"
$script = @"
`$ErrorActionPreference = "Stop"
Set-Location "$URAPRoot"
& "$Python" tools\export_temporal_recovery_to_yolo_labels.py --trajectory-csv "$TrajectoryCsv" --images-list "$ImagesList" --out-label-dir "$predLabelDir"
if (`$LASTEXITCODE -ne 0) { throw "export_temporal_recovery_to_yolo_labels failed with exit code `$LASTEXITCODE" }
& "$Python" tools\yolomg_eval_pred_labels.py --images-list "$ImagesList" --pred-label-dir "$predLabelDir" --out-dir "$evalDir" --conf-thres "$ConfThres" --match-iou "$MatchIou"
if (`$LASTEXITCODE -ne 0) { throw "yolomg_eval_pred_labels failed with exit code `$LASTEXITCODE" }
"@
$runnerScript = Join-Path $RunRoot "$RunId.runner.ps1"
$script | Set-Content -Path $runnerScript -Encoding utf8

$proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", $runnerScript) -WorkingDirectory $URAPRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "urap_root=$URAPRoot",
  "images_list=$ImagesList",
  "trajectory_csv=$TrajectoryCsv",
  "out_root=$OutRoot",
  "pred_label_dir=$predLabelDir",
  "eval_dir=$evalDir",
  "conf_thres=$ConfThres",
  "match_iou=$MatchIou",
  "stdout=$stdout",
  "stderr=$stderr",
  "runner_script=$runnerScript"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached temporal recovery YOLOMG val eval."
Get-Content $metaFile
