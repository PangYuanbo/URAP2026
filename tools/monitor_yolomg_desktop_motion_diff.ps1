param([string]$RunId = "desktop_yolomg_motion_diff_samples")
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot ("artifacts\detached_yolomg_motion_diff\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"
$progressPath = Join-Path $runRoot "progress.json"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
if (-not (Test-Path $pidPath)) { throw "PID file not found: $pidPath" }
$pidValue = [int](Get-Content $pidPath -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$matching = $process -and $process.CommandLine -like "*yolomg_video_motion_diff.py*"
if ($matching) { Write-Output "RUNNING PID=$pidValue start=$((Get-Process -Id $pidValue).StartTime.ToString('o'))" } else { Write-Output "NOT RUNNING PID=$pidValue" }
if (Test-Path $progressPath) {
    $progress = Get-Content $progressPath -Raw | ConvertFrom-Json
    Write-Output "status=$($progress.status) done=$($progress.done)/$($progress.total) video=$($progress.video)"
    Write-Output "last_output_timestamp=$($progress.last_output_timestamp) last_output=$($progress.last_output)"
} else { Write-Output "progress=not-created" }
Write-Output "stdout=$stdoutPath"
Write-Output "stderr=$stderrPath"
if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Tail 5 }
if ((Test-Path $stderrPath) -and (Get-Item $stderrPath).Length -gt 0) { Write-Output "--- stderr ---"; Get-Content $stderrPath -Tail 10 }