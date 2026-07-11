param([string]$RunId = "yolomg_nps_vertical_full_queue")
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "artifacts\venvs\nps_flow\Scripts\python.exe"
$script = Join-Path $repoRoot "tools\run_yolomg_vertical_comparison_queue.py"
$runRoot = Join-Path $repoRoot ("artifacts\detached_yolomg_nps_vertical\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
$progressPath = Join-Path $runRoot "progress.json"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*run_yolomg_vertical_comparison_queue.py*") { throw "A matching queue is already running with PID $oldPid" }
}
$arguments = @($script, "--repo", $repoRoot, "--python", $python, "--progress-json", $progressPath)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Set-Content -LiteralPath (Join-Path $runRoot "started_at.txt") -Value (Get-Date).ToString("o") -Encoding ascii
Write-Output "Started detached vertical-comparison queue."
Write-Output "PID=$($process.Id)"
Write-Output "Progress=$progressPath"
Write-Output "Logs=$stdoutPath ; $stderrPath"