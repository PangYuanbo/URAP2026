param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$RunId = "nps_motion_full_pipeline",
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\pipeline_runner"
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $RepoRoot "tools\run_nps_motion_full_pipeline.ps1"
if (-not (Test-Path $scriptPath -PathType Leaf)) { throw "Pipeline script missing: $scriptPath" }
New-Item -ItemType Directory -Force -Path $RunnerRoot | Out-Null
$logsDir = Join-Path $RunnerRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
if (Test-Path $pidFile) {
  $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  $oldProcess = if ($oldPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue } else { $null }
  if ($oldProcess -and $oldProcess.CommandLine -like "*run_nps_motion_full_pipeline.ps1*") {
    Write-Host "NPS full pipeline already running: PID=$oldPid"
    Get-Content $metaFile -ErrorAction SilentlyContinue
    exit 0
  }
  Write-Host "Previous pipeline is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid"
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "${RunId}_${timestamp}.out.txt"
$stderr = Join-Path $logsDir "${RunId}_${timestamp}.err.txt"
$arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$scriptPath,'-RepoRoot',$RepoRoot)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$process.Id | Set-Content $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "pid=$($process.Id)", "script=$scriptPath", "repo_root=$RepoRoot",
  "stdout=$stdout", "stderr=$stderr", "cmd_args=$($arguments -join ' ')"
) | Set-Content $metaFile -Encoding utf8
Write-Host "Started detached NPS full pipeline. PID=$($process.Id)"
Get-Content $metaFile
