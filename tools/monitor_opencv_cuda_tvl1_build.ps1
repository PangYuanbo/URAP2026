$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\opencv_cuda_tvl1_build"
$buildPid = [int](Get-Content (Join-Path $runRoot "build.pid") -Raw)
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$buildPid" -ErrorAction SilentlyContinue
if ($proc -and $proc.CommandLine -like "*build_opencv_cuda_tvl1_worker.ps1*") { Write-Output "RUNNING PID=$buildPid start=$((Get-Process -Id $buildPid).StartTime.ToString('o'))" } else { Write-Output "NOT RUNNING PID=$buildPid" }
$status = Join-Path $runRoot "status.json"
if (Test-Path $status) { Get-Content $status -Raw }
$stdout = Join-Path $runRoot "stdout.log"; $stderr = Join-Path $runRoot "stderr.log"
Write-Output "stdout=$stdout"; Write-Output "stderr=$stderr"
if (Test-Path $stdout) { Get-Content $stdout -Tail 12 }
if ((Test-Path $stderr) -and (Get-Item $stderr).Length -gt 0) { Write-Output "--- stderr ---"; Get-Content $stderr -Tail 20 }