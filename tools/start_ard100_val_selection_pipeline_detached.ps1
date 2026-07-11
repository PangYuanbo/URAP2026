$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\aaron\Desktop\URAP'
$runDir = Join-Path $repo 'artifacts\detached_ard100_val_selection_pipeline_v2'
$logDir = Join-Path $runDir 'logs'
$pidFile = Join-Path $runDir 'pid.txt'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path $pidFile) {
    $oldPid = [int](Get-Content $pidFile -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "ARD val selection pipeline active PID $oldPid" }
}
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $logDir "pipeline_$timestamp.out.txt"
$stderrLog = Join-Path $logDir "pipeline_$timestamp.err.txt"
$python = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$process = Start-Process -FilePath $python -ArgumentList @('tools\run_ard100_val_selection_pipeline.py') -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$process.Id | Set-Content $pidFile -Encoding ASCII
@{pid=$process.Id;start_time=(Get-Date).ToString('o');stdout_log=$stdoutLog;stderr_log=$stderrLog}|ConvertTo-Json|Set-Content (Join-Path $runDir 'meta.json') -Encoding UTF8
Write-Host "RUNNING PID: $($process.Id)"
