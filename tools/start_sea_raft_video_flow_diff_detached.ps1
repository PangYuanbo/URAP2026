param(
    [string]$RunId = "sea_raft_dji0619_sample",
    [string]$InputVideo = "C:\Users\aaron\Desktop\DJI_0619_W.MP4",
    [double]$DurationSeconds = 10,
    [int]$ProcessWidth = 960,
    [int]$DisplayWidth = 480
)
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$script = Join-Path $repoRoot "tools\sea_raft_video_flow_diff.py"
$seaRoot = Join-Path $repoRoot "third_party\SEA-RAFT"
$cfg = Join-Path $seaRoot "config\eval\spring-M.json"
$runRoot = Join-Path $repoRoot ("artifacts\detached_sea_raft\" + $RunId)
$outputDir = Join-Path $repoRoot ("artifacts\sea_raft_flow_diff\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
$progressPath = Join-Path $runRoot "progress.json"
New-Item -ItemType Directory -Force -Path $runRoot, $outputDir | Out-Null
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*sea_raft_video_flow_diff.py*") { throw "A matching SEA-RAFT job is already running with PID $oldPid" }
}
$arguments = @($script, "--input", $InputVideo, "--output-dir", $outputDir, "--sea-raft-root", $seaRoot, "--cfg", $cfg, "--duration-seconds", $DurationSeconds, "--process-width", $ProcessWidth, "--display-width", $DisplayWidth, "--progress-json", $progressPath)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $seaRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Set-Content -LiteralPath (Join-Path $runRoot "started_at.txt") -Value (Get-Date).ToString("o") -Encoding ascii
Write-Output "Started detached SEA-RAFT flow render."
Write-Output "PID=$($process.Id)"
Write-Output "Output=$outputDir"
Write-Output "Progress=$progressPath"
Write-Output "Logs=$stdoutPath ; $stderrPath"