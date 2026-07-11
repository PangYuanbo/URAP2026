param(
  [string]$RunName = "two_branch_motion_action_yolomg_20260606",
  [int]$Epochs = 20,
  [double]$Center = 0.20,
  [double]$Beta = 0.30,
  [string]$Mode = "boost-only"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path ".").Path
$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$RunDir = Join-Path $Repo ("artifacts\yolomg_action\" + $RunName)
$LogDir = Join-Path $RunDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stdout = Join-Path $LogDir "runner.out.txt"
$Stderr = Join-Path $LogDir "runner.err.txt"
$PidFile = Join-Path $RunDir "runner.pid"
$MetaFile = Join-Path $RunDir "runner_meta.txt"
$Args = @(
  "tools\run_two_branch_yolomg_experiment.py",
  "--repo-root", $Repo,
  "--out-dir", $RunDir,
  "--epochs", "$Epochs",
  "--center", "$Center",
  "--beta", "$Beta",
  "--mode", $Mode
)

$Process = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -Path $PidFile -Encoding ascii
@(
  "run_name=$RunName",
  "pid=$($Process.Id)",
  "started=$(Get-Date -Format o)",
  "python=$Python",
  "stdout=$Stdout",
  "stderr=$Stderr",
  "status=$(Join-Path $RunDir 'status.json')"
) | Set-Content -Path $MetaFile -Encoding utf8

[pscustomobject]@{
  RunName = $RunName
  PID = $Process.Id
  RunDir = $RunDir
  Stdout = $Stdout
  Stderr = $Stderr
  Status = (Join-Path $RunDir "status.json")
}
