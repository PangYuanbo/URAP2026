param(
    [int]$BatchSize = 32,
    [int]$ImgSz = 1280
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_yolomg_corrected_v2'
$logDir = Join-Path $runDir 'logs'
$pidFile = Join-Path $runDir 'pid.txt'
$weights = Join-Path $repo 'runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt'
$data = 'D:\URAP_datasets\ARD100_YOLOMG\ARD100_mask32_local.yaml'
$project = 'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2'
$name = 'yolomg_ard100_test_candidates'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path $pidFile) {
    $oldPid = [int](Get-Content $pidFile -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        throw "Corrected YOLOMG ARD100 run is already active with PID $oldPid"
    }
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $logDir "yolomg_corrected_$timestamp.out.txt"
$stderrLog = Join-Path $logDir "yolomg_corrected_$timestamp.err.txt"
$python = Join-Path $repo '.venv\Scripts\python.exe'
$arguments = @(
    'val.py', '--data', $data, '--weights', $weights,
    '--task', 'test', '--task2', 'test2', '--imgsz', "$ImgSz",
    '--batch-size', "$BatchSize", '--device', '0', '--workers', '4', '--half',
    '--conf-thres', '0.001', '--iou-thres', '0.45', '--save-txt', '--save-conf',
    '--project', $project, '--name', $name, '--exist-ok'
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$process.Id | Set-Content $pidFile -Encoding ASCII
@{
    pid = $process.Id
    start_time = (Get-Date).ToString('o')
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    output_dir = (Join-Path $project $name)
    weights = $weights
    data = $data
    total_images = 71608
    total_batches = [math]::Ceiling(71608 / $BatchSize)
    batch_size = $BatchSize
} | ConvertTo-Json | Set-Content (Join-Path $runDir 'meta.json') -Encoding UTF8

Write-Host 'RUNNING'
Write-Host "PID: $($process.Id)"
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
