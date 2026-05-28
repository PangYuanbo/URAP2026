param(
    [string]$RunId = "yolomg_motion_diff_compare_demo5",
    [string]$Videos = "phantom57 phantom102 phantom02 phantom05 phantom61",
    [string]$OutputDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_diff_maps_paper\compare_demo5_full",
    [int]$MaxFrames = 0,
    [switch]$SideBySide,
    [int]$DisplayWidth = 960
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$scriptPath = Join-Path $repoRoot "tools\yolomg_motion_diff_video.py"
$pythonExe = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_diff\" + $RunId)
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stdoutLog = Join-Path $logDir ("runner_{0}.out.txt" -f $RunId)
$stderrLog = Join-Path $logDir ("runner_{0}.err.txt" -f $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

$videoList = $Videos -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$argList = @(
    $scriptPath,
    "--output-dir", $OutputDir,
    "--videos"
) + $videoList
if ($MaxFrames -gt 0) {
    $argList += @("--max-frames", "$MaxFrames")
}
if ($SideBySide) {
    $argList += @("--side-by-side", "--display-width", "$DisplayWidth")
}

$startTime = Get-Date
$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList $argList `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
    "run_id=$RunId"
    "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "pid=$($proc.Id)"
    "videos=$Videos"
    "max_frames=$MaxFrames"
    "side_by_side=$SideBySide"
    "display_width=$DisplayWidth"
    "output_dir=$OutputDir"
    "stdout=$stdoutLog"
    "stderr=$stderrLog"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Output "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "pid=$($proc.Id)"
Write-Output "run_id=$RunId"
Write-Output "output_dir=$OutputDir"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"
