$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\aaron\Desktop\URAP'
$runDir = Join-Path $repo 'artifacts\detached_tvd_temporal_nms_v154'
$logDir = Join-Path $runDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $logDir "v154_$timestamp.out.txt"
$stderrLog = Join-Path $logDir "v154_$timestamp.err.txt"
$python = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$process = Start-Process -FilePath $python -ArgumentList @('tools\run_tvd_temporal_nms_v154.py') -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$process.Id | Set-Content (Join-Path $runDir 'pid.txt') -Encoding ASCII
@{pid=$process.Id;start_time=(Get-Date).ToString('o');stdout_log=$stdoutLog;stderr_log=$stderrLog}|ConvertTo-Json|Set-Content (Join-Path $runDir 'meta.json') -Encoding UTF8
Write-Host "RUNNING PID: $($process.Id)"
