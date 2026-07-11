$ErrorActionPreference = 'SilentlyContinue'
$Repo = 'C:\Users\aaron\Desktop\URAP'
$Run = Join-Path $Repo 'artifacts\detached_tvd_detector_hard_replay_v165_posteval'
$PidFile = Join-Path $Run 'pid.txt'
$MetaFile = Join-Path $Run 'meta.json'
$ProgressFile = Join-Path $Run 'progress.json'
$Summary = 'D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165_posteval\official_summary.json'
if (!(Test-Path $PidFile)) { Write-Output 'NOT RUNNING: PID file missing'; exit 1 }
$JobPid = [int](Get-Content $PidFile -Raw).Trim()
$Process = Get-CimInstance Win32_Process -Filter "ProcessId=$JobPid"
if ($Process) { Write-Output "RUNNING PID: $JobPid"; Write-Output "Command: $($Process.CommandLine)" } else { Write-Output "NOT RUNNING PID: $JobPid" }
if (Test-Path $MetaFile) {
    $Meta = Get-Content $MetaFile -Raw | ConvertFrom-Json
    Write-Output "Start time: $($Meta.start_time)"
    Write-Output "Stdout log: $($Meta.stdout_log)"
    Write-Output "Stderr log: $($Meta.stderr_log)"
}
if (Test-Path $ProgressFile) {
    $Progress = Get-Content $ProgressFile -Raw | ConvertFrom-Json
    Write-Output "Progress: $($Progress.done)/$($Progress.total)"
    Write-Output "Stage: $($Progress.stage)"
    Write-Output "Last output: $($Progress.updated)"
    if ($Progress.child_pid) { Write-Output "Child PID: $($Progress.child_pid)" }
    if ($Progress.error) { Write-Output "Error: $($Progress.error)" }
} else { Write-Output 'Progress: 0/4' }
if (Test-Path $Summary) {
    $Result = Get-Content $Summary -Raw | ConvertFrom-Json
    Write-Output "Test mAP@0.5: $($Result.test_fixed.map50)"
    Write-Output "Gain over VATD: $($Result.gain_over_vatd_points) points"
    Write-Output "Target met: $($Result.target_at_least_3_met)"
}
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
