param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$DatasetRoot = "U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1",
  [string]$EvalRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\model_evals",
  [string[]]$Models = @("transvisdrone", "yolomg_native", "yolomg_ard100"),
  [string[]]$Interventions = @("original", "slow_0p5", "fast_2x", "accelerate_g2", "decelerate_g2"),
  [string]$RunId = "nps_motion_model_evals",
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\eval_runner"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path $DatasetRoot -PathType Container)) { throw "DatasetRoot not found: $DatasetRoot" }
New-Item -ItemType Directory -Force -Path $RunnerRoot | Out-Null
$logsDir = Join-Path $RunnerRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
if (Test-Path $pidFile) {
  $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  $oldProcess = if ($oldPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue } else { $null }
  if ($oldProcess -and $oldProcess.CommandLine -like "*run_nps_motion_model_evals.py*") {
    Write-Host "Already running: PID=$oldPid"
    Get-Content $metaFile -ErrorAction SilentlyContinue
    exit 0
  }
  $existing = Get-ChildItem $EvalRoot -Recurse -Filter complete.json -ErrorAction SilentlyContinue | Measure-Object
  Write-Host "Previous job is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid completed_units=$($existing.Count)"
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "${RunId}_${timestamp}.out.txt"
$stderr = Join-Path $logsDir "${RunId}_${timestamp}.err.txt"
$arguments = @("tools\run_nps_motion_model_evals.py", "--dataset-root", $DatasetRoot, "--out-root", $EvalRoot, "--models") + $Models + @("--interventions") + $Interventions
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$process.Id | Set-Content $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "pid=$($process.Id)", "python=$Python", "dataset_root=$DatasetRoot",
  "eval_root=$EvalRoot", "models=$($Models -join ',')", "interventions=$($Interventions -join ',')", "stdout=$stdout", "stderr=$stderr",
  "cmd_args=$($arguments -join ' ')"
) | Set-Content $metaFile -Encoding utf8
Write-Host "Started detached NPS motion evaluations. PID=$($process.Id)"
Get-Content $metaFile
