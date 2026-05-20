$runDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\transvisdrone_ard100_train"
$pidFile = Join-Path $runDir "pid.txt"
$metaFile = Join-Path $runDir "meta.json"
if (-not (Test-Path $pidFile) -or -not (Test-Path $metaFile)) {
  Write-Host "NOT RUNNING"
  exit 0
}
$jobPid = [int](Get-Content $pidFile | Select-Object -First 1)
$meta = Get-Content $metaFile | ConvertFrom-Json
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $jobPid" -ErrorAction SilentlyContinue
if (-not $proc) {
  Write-Host "NOT RUNNING"
  Write-Host "last pid: $jobPid"
  Write-Host "stdout: $($meta.stdout_log)"
  Write-Host "stderr: $($meta.stderr_log)"
  exit 0
}
$outInfo = Get-Item $meta.stdout_log -ErrorAction SilentlyContinue
$lastEpoch = ""
if (Test-Path $meta.stdout_log) {
  $epochLine = Select-String -Path $meta.stdout_log -Pattern '^\s*\d+/\d+' | Select-Object -Last 1
  if ($epochLine) { $lastEpoch = $epochLine.Line.Trim() }
}
Write-Host "RUNNING"
Write-Host "done/total: unknown/unknown"
Write-Host "pid: $jobPid"
Write-Host "start_time: $($meta.start_time)"
if ($lastEpoch) { Write-Host "last_completed_unit: $lastEpoch" }
if ($outInfo) { Write-Host "last_output: $($outInfo.LastWriteTime.ToString('s'))" }
Write-Host "stdout: $($meta.stdout_log)"
Write-Host "stderr: $($meta.stderr_log)"
