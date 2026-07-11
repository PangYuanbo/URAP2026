param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$YOLOMGRepo = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$DataYaml = "U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\original\original_yolomg.yaml",
  [string]$RunDir = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\yolomg_nps_train50",
  [int]$BatchSize = 8,
  [string]$Device = "0",
  [switch]$ResumeExisting,
  [string]$RunId = "nps_yolomg_train50",
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\train_runner"
)

$ErrorActionPreference = "Stop"
$weights = Join-Path $YOLOMGRepo "yolov5s.pt"
$config = Join-Path $YOLOMGRepo "models\NPS_uav_s.yaml"
foreach ($path in @($Python, $DataYaml, $weights, $config)) { if (-not (Test-Path $path -PathType Leaf)) { throw "Required file not found: $path" } }
New-Item -ItemType Directory -Force -Path $RunnerRoot | Out-Null
$logsDir = Join-Path $RunnerRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
if (Test-Path $pidFile) {
  $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  $oldProcess = if ($oldPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue } else { $null }
  if ($oldProcess -and $oldProcess.CommandLine -like "*train.py*" -and $oldProcess.CommandLine -like "*NPS_uav_s.yaml*") {
    Write-Host "Already running: PID=$oldPid"
    Get-Content $metaFile -ErrorAction SilentlyContinue
    exit 0
  }
  Write-Host "Previous job is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid"
}
$lastCheckpoint = Join-Path $RunDir "weights\last.pt"
if ((Test-Path $RunDir) -and -not $ResumeExisting) { throw "RunDir already exists. Pass -ResumeExisting to explicitly resume: $RunDir" }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "${RunId}_${timestamp}.out.txt"
$stderr = Join-Path $logsDir "${RunId}_${timestamp}.err.txt"
if ($ResumeExisting) {
  if (-not (Test-Path $lastCheckpoint -PathType Leaf)) { throw "Resume requested but checkpoint missing: $lastCheckpoint" }
  Write-Host "Restarting stopped training explicitly at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') from $lastCheckpoint"
  $arguments = @("train.py", "--resume", $lastCheckpoint)
  $resumeFrom = $lastCheckpoint
} else {
  $project = Split-Path $RunDir -Parent
  $name = Split-Path $RunDir -Leaf
  $arguments = @("train.py", "--data", $DataYaml, "--cfg", $config, "--weights", $weights, "--batch-size", [string]$BatchSize,
    "--epochs", "50", "--imgsz", "1280", "--device", $Device, "--project", $project, "--name", $name, "--exist-ok")
  $resumeFrom = "none"
}
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $YOLOMGRepo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$process.Id | Set-Content $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "pid=$($process.Id)", "python=$Python", "data_yaml=$DataYaml", "run_dir=$RunDir",
  "epochs=50", "batch_size=$BatchSize", "device=$Device", "resume_from=$resumeFrom", "stdout=$stdout", "stderr=$stderr", "cmd_args=$($arguments -join ' ')"
) | Set-Content $metaFile -Encoding utf8
Write-Host "Started detached YOLOMG NPS 50-epoch training. PID=$($process.Id)"
Get-Content $metaFile
