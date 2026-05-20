param()

$runDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\aot_flight_server"
$pidFile = Join-Path $runDir "pid.txt"
$metaFile = Join-Path $runDir "meta.json"

if (-not (Test-Path $pidFile) -or -not (Test-Path $metaFile)) {
    Write-Host "NOT RUNNING"
    exit 0
}

$pid = Get-Content $pidFile | Select-Object -First 1
$meta = Get-Content $metaFile | ConvertFrom-Json
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pid" -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "NOT RUNNING"
    Write-Host "last pid: $pid"
    Write-Host "url: $($meta.url)"
    Write-Host "stdout: $($meta.stdout_log)"
    Write-Host "stderr: $($meta.stderr_log)"
    exit 0
}

$outInfo = Get-Item $meta.stdout_log -ErrorAction SilentlyContinue

Write-Host "RUNNING"
Write-Host "pid: $pid"
Write-Host "start_time: $($meta.start_time)"
Write-Host "url: $($meta.url)"
Write-Host "flight_dir: $($meta.flight_dir)"
if ($outInfo) {
    Write-Host "last_output: $($outInfo.LastWriteTime.ToString('s'))"
}
Write-Host "stdout: $($meta.stdout_log)"
Write-Host "stderr: $($meta.stderr_log)"
