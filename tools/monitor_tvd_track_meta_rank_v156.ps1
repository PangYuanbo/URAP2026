$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_track_meta_rank_v156'
$summaryPath = 'D:\URAP_vatd_rank_results\tvd_track_meta_rank_v156\official_summary.json'
if (-not (Test-Path (Join-Path $runDir 'pid.txt'))) { Write-Host 'NOT RUNNING: PID file is missing.'; exit 1 }
$pidValue = [int](Get-Content (Join-Path $runDir 'pid.txt'))
$meta = Get-Content (Join-Path $runDir 'meta.json') -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
if ($process) { Write-Host "RUNNING PID: $pidValue"; Write-Host "Command: $($process.CommandLine)" } else { Write-Host "NOT RUNNING PID: $pidValue" }
Write-Host "Start time: $($meta.start_time)"
Write-Host "Stdout log: $($meta.stdout_log)"
Write-Host "Stderr log: $($meta.stderr_log)"
$progressPath = Join-Path $runDir 'progress.json'
if (Test-Path $progressPath) { $progress=Get-Content $progressPath -Raw|ConvertFrom-Json; Write-Host "Progress: $($progress.done)/$($progress.total)"; Write-Host "Stage: $($progress.stage)"; Write-Host "Last output: $($progress.updated)" } else { Write-Host 'Progress: 0/5'; Write-Host 'Last output: no progress file' }
if (Test-Path $summaryPath) { $summary=Get-Content $summaryPath -Raw|ConvertFrom-Json; Write-Host "Test mAP@0.5: $($summary.test_fixed.map50)"; Write-Host "Gain over VATD: $($summary.gain_over_vatd_points) points"; Write-Host "Target met: $($summary.target_3_to_5_met)" }
