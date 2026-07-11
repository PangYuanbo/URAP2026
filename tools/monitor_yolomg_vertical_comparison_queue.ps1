param([string]$RunId = "yolomg_nps_vertical_full_queue")
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot ("artifacts\detached_yolomg_nps_vertical\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"
$progressPath = Join-Path $runRoot "progress.json"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
if (-not (Test-Path $pidPath)) { throw "PID file not found: $pidPath" }
$pidValue = [int](Get-Content $pidPath -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$matching = $process -and $process.CommandLine -like "*run_yolomg_vertical_comparison_queue.py*"
if ($matching) { Write-Output "RUNNING PID=$pidValue start=$((Get-Process -Id $pidValue).StartTime.ToString('o'))" } else { Write-Output "NOT RUNNING PID=$pidValue" }
if (Test-Path $progressPath) { Get-Content $progressPath -Raw } else { Write-Output "progress=not-created" }
Write-Output "stdout=$stdoutPath"
Write-Output "stderr=$stderrPath"
if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Tail 5 }
if ((Test-Path $stderrPath) -and (Get-Item $stderrPath).Length -gt 0) { Write-Output "--- stderr ---"; Get-Content $stderrPath -Tail 12 }