param(
    [string]$RunId = "yolomg_motion_heatmap_02_05",
    [string]$Videos = "phantom02 phantom05",
    [string]$Weights = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt",
    [string]$OutputDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\motion_heatmaps\test_02_05"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$scriptPath = Join-Path $repoRoot "tools\yolomg_motion_heatmap_video.py"
$pythonExe = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_heatmap\" + $RunId)
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stdoutLog = Join-Path $logDir ("runner_{0}.out.txt" -f $RunId)
$stderrLog = Join-Path $logDir ("runner_{0}.err.txt" -f $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"
$statusFile = Join-Path $runRoot "run_status.json"

$videoList = $Videos -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$argList = @(
    $scriptPath,
    "--weights", $Weights,
    "--output-dir", $OutputDir,
    "--skip-overlay",
    "--videos"
) + $videoList
$startTime = Get-Date

$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList $argList `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
    "run_id=$RunId"
    "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "pid=$($proc.Id)"
    "weights=$Weights"
    "videos=$Videos"
    "output_dir=$OutputDir"
    "stdout=$stdoutLog"
    "stderr=$stderrLog"
    "status_file=$statusFile"
) | Set-Content -Path $metaFile -Encoding utf8

$status = [ordered]@{
    run_id = $RunId
    state = "running"
    done = 0
    total = ($Videos -split '\s+').Count
    current_video = ""
    started = $startTime.ToString("yyyy-MM-dd HH:mm:ss")
    pid = $proc.Id
    output_dir = $OutputDir
    stdout = $stdoutLog
    stderr = $stderrLog
    last_updated = $startTime.ToString("yyyy-MM-dd HH:mm:ss")
    message = "Started detached heatmap rendering"
}
$status | ConvertTo-Json | Set-Content -Path $statusFile -Encoding utf8

Write-Output "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "pid=$($proc.Id)"
Write-Output "run_id=$RunId"
Write-Output "output_dir=$OutputDir"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"
Write-Output "status_file=$statusFile"
