$runDir = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\transvisdrone_ard100_overnight_pipeline'
$pidFile = Join-Path $runDir 'pid.txt'
$metaFile = Join-Path $runDir 'meta.json'
if (-not (Test-Path $pidFile)) { Write-Host 'NOT RUNNING'; exit 0 }
$procId = [int](Get-Content $pidFile | Select-Object -First 1)
$meta = if (Test-Path $metaFile) { Get-Content $metaFile -Raw | ConvertFrom-Json } else { $null }
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
if (-not $proc) {
  Write-Host 'NOT RUNNING'
  Write-Host "last pid: $procId"
  if ($meta) {
    Write-Host "stdout: $($meta.stdout_log)"
    Write-Host "stderr: $($meta.stderr_log)"
  }
  exit 0
}
$lastOutput = ''
if ($meta -and (Test-Path $meta.stdout_log)) { $lastOutput = (Get-Item $meta.stdout_log).LastWriteTime.ToString('s') }
Write-Host 'RUNNING'
Write-Host 'done/total: staged/3'
Write-Host "pid: $procId"
if ($meta) { Write-Host "start_time: $($meta.start_time)" }
Write-Host "last_output: $lastOutput"
if ($meta) {
  Write-Host "stdout: $($meta.stdout_log)"
  Write-Host "stderr: $($meta.stderr_log)"
}
