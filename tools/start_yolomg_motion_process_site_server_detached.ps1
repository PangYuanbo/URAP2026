param(
    [string]$RunId = "yolomg_motion_process_site_server",
    [string]$WebRoot = "C:\Users\aaron\Desktop\URAP\artifacts\yolomg_motion_process_site",
    [int]$Port = 8782
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$scriptPath = Join-Path $repoRoot "tools\yolomg_motion_process_site_server.py"
$pythonExe = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_site_server\" + $RunId)
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdoutLog = Join-Path $logDir ("runner_{0}.out.txt" -f $RunId)
$stderrLog = Join-Path $logDir ("runner_{0}.err.txt" -f $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

$startTime = Get-Date
$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList @($scriptPath, "--directory", $WebRoot, "--port", "$Port") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
    "run_id=$RunId"
    "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "pid=$($proc.Id)"
    "web_root=$WebRoot"
    "port=$Port"
    "url=http://127.0.0.1:$Port/"
    "stdout=$stdoutLog"
    "stderr=$stderrLog"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Output "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "pid=$($proc.Id)"
Write-Output "url=http://127.0.0.1:$Port/"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"
