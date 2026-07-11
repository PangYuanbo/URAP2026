param(
    [string]$RunId = "nps_tvl1_yolomg_dji0619_sample",
    [string]$InputVideo = "C:\Users\aaron\Desktop\DJI_0619_W.MP4",
    [double]$DurationSeconds = 10,
    [int]$ProcessWidth = 960,
    [int]$DisplayWidth = 480
)
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "artifacts\venvs\nps_flow\Scripts\python.exe"
$script = Join-Path $repoRoot "tools\nps_tvl1_yolomg_diff.py"
$runRoot = Join-Path $repoRoot ("artifacts\detached_nps_tvl1_yolomg\" + $RunId)
$outputDir = Join-Path $repoRoot ("artifacts\nps_tvl1_yolomg\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
$progressPath = Join-Path $runRoot "progress.json"
New-Item -ItemType Directory -Force -Path $runRoot, $outputDir | Out-Null
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*nps_tvl1_yolomg_diff.py*") { throw "A matching job is already running with PID $oldPid" }
}
$arguments = @($script, "--input", $InputVideo, "--output-dir", $outputDir, "--duration-seconds", $DurationSeconds, "--process-width", $ProcessWidth, "--display-width", $DisplayWidth, "--progress-json", $progressPath)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Set-Content -LiteralPath (Join-Path $runRoot "started_at.txt") -Value (Get-Date).ToString("o") -Encoding ascii
Write-Output "Started detached NPS Dual TV-L1 + YOLOMG render."
Write-Output "PID=$($process.Id)"
Write-Output "Output=$outputDir"
Write-Output "Progress=$progressPath"
Write-Output "Logs=$stdoutPath ; $stderrPath"