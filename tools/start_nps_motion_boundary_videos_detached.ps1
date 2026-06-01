param(
    [string]$RunId = "nps_motion_boundary_full",
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
    [string]$FramesRoot = "D:\URAP_datasets\TransVisDrone\NPS\AllFrames",
    [string]$OutputDir = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_boundary_site",
    [int]$DisplayWidth = 640,
    [int]$FlowWidth = 640,
    [double]$Fps = 30.0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $RepoRoot "tools\nps_motion_boundary_videos.py"
$runRoot = Join-Path $RepoRoot ("artifacts\detached_nps_motion_boundary\" + $RunId)
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stdoutLog = Join-Path $logDir ("runner_{0}.out.txt" -f $RunId)
$stderrLog = Join-Path $logDir ("runner_{0}.err.txt" -f $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

$argsList = @(
    $scriptPath,
    "--frames-root", $FramesRoot,
    "--out", $OutputDir,
    "--display-width", "$DisplayWidth",
    "--flow-width", "$FlowWidth",
    "--fps", "$Fps"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}

$startTime = Get-Date
$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList $argsList `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
    "run_id=$RunId"
    "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "pid=$($proc.Id)"
    "python=$PythonExe"
    "script=$scriptPath"
    "frames_root=$FramesRoot"
    "output_dir=$OutputDir"
    "stdout=$stdoutLog"
    "stderr=$stderrLog"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Output "Started detached NPS motion-boundary render."
Write-Output "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "pid=$($proc.Id)"
Write-Output "output_dir=$OutputDir"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"
