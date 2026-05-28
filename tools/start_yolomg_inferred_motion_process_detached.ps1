param(
    [string]$RunId = "yolomg_inferred_motion_process_demo",
    [string]$Videos = "phantom02",
    [string]$OutputDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_process_gradcam\demo",
    [int]$MaxFrames = 120,
    [int]$DisplayWidth = 480,
    [int]$MotionLayer = 3,
    [int]$FusionLayer = 5,
    [string]$Device = "0",
    [string]$List = "D:\URAP_datasets\ARD100_YOLOMG\test.txt",
    [string]$List2 = "D:\URAP_datasets\ARD100_YOLOMG\test2.txt"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$scriptPath = Join-Path $repoRoot "tools\yolomg_inferred_motion_process_video.py"
$pythonExe = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_process\" + $RunId)
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
) + $videoList + @(
    "--max-frames", "$MaxFrames",
    "--display-width", "$DisplayWidth",
    "--motion-layer", "$MotionLayer",
    "--fusion-layer", "$FusionLayer",
    "--device", "$Device",
    "--test-list", "$List",
    "--test2-list", "$List2"
)

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
    "display_width=$DisplayWidth"
    "motion_layer=$MotionLayer"
    "fusion_layer=$FusionLayer"
    "device=$Device"
    "list=$List"
    "list2=$List2"
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
