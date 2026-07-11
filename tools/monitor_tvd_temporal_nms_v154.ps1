$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\aaron\Desktop\URAP'
$runDir = Join-Path $repo 'artifacts\detached_tvd_temporal_nms_v154'
$pidPath = Join-Path $runDir 'pid.txt'
$metaPath = Join-Path $runDir 'meta.json'
$progressPath = Join-Path $runDir 'progress.json'
$summaryPath = 'D:\URAP_vatd_rank_results\tvd_temporal_nms_v154\official_summary.json'

if (-not (Test-Path $pidPath)) {
    Write-Host 'NOT RUNNING: PID file is missing.'
    exit 1
}

$pidValue = [int](Get-Content $pidPath)
$meta = if (Test-Path $metaPath) { Get-Content $metaPath -Raw | ConvertFrom-Json } else { $null }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$progress = if (Test-Path $progressPath) { Get-Content $progressPath -Raw | ConvertFrom-Json } else { $null }

if ($process) {
    Write-Host "RUNNING PID: $pidValue"
    Write-Host "Command: $($process.CommandLine)"
} else {
    Write-Host "NOT RUNNING PID: $pidValue"
}

if ($meta) {
    Write-Host "Start time: $($meta.start_time)"
    Write-Host "Stdout log: $($meta.stdout_log)"
    Write-Host "Stderr log: $($meta.stderr_log)"
}

if ($progress) {
    Write-Host "Progress: $($progress.done)/$($progress.total)"
    Write-Host "Stage: $($progress.stage)"
    Write-Host "Last output: $($progress.updated)"
} else {
    Write-Host 'Progress: 0/3'
    Write-Host 'Last output: no progress file'
}

if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Host "Test mAP@0.5: $($summary.test_fixed.map50)"
    Write-Host "Gain over VATD: $($summary.gain_over_vatd_points) points"
    Write-Host "Target met: $($summary.target_3_to_5_met)"
}
