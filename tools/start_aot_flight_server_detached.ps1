param(
    [string]$FlightDir = "D:\URAP_datasets\AOT\part1\Images\0001ba865c8e410e88609541b8f55ffc",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$root = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking"
$runDir = Join-Path $root "runs\aot_flight_server"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "aot_flight_server_$ts.out.txt"
$errLog = Join-Path $logDir "aot_flight_server_$ts.err.txt"
$pidFile = Join-Path $runDir "pid.txt"
$metaFile = Join-Path $runDir "meta.json"

$script = "C:\Users\aaron\Desktop\URAP\tools\run_aot_flight_server.ps1"
$argList = @(
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $script,
    "-FlightDir", $FlightDir,
    "-Port", "$Port"
)

$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog

$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
@{
    pid = $proc.Id
    start_time = (Get-Date).ToString("s")
    flight_dir = $FlightDir
    port = $Port
    url = "http://localhost:$Port/"
    stdout_log = $outLog
    stderr_log = $errLog
} | ConvertTo-Json | Set-Content -Path $metaFile -Encoding UTF8

Write-Host "RUNNING"
Write-Host "PID: $($proc.Id)"
Write-Host "URL: http://localhost:$Port/"
Write-Host "stdout: $outLog"
Write-Host "stderr: $errLog"
