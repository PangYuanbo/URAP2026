param(
    [string]$Weights = "D:\TransVisDrone_official_weights\NPS\best.pt",
    [string]$Data = "D:\URAP_datasets\TransVisDrone\ARD100\ARD100_TVD.yaml",
    [int]$BatchSize = 4,
    [int]$ImgSz = 1280
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\TransVisDrone"
$runDir = "C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_tvd_candidates_v1"
$outputRoot = "D:\URAP_vatd_rank_results\ard100_generalization_v1"
$name = "tvd_nps_zero_shot_candidates"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$pidFile = Join-Path $runDir "pid.txt"
if (Test-Path $pidFile) {
    $oldPid = [int](Get-Content $pidFile -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "ARD100 candidate job already running with PID $oldPid" }
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "ard100_candidates_$ts.out.txt"
$errLog = Join-Path $logDir "ard100_candidates_$ts.err.txt"
$metaFile = Join-Path $runDir "meta.json"
$py = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$args = @(
  "inference.py", "--data", $Data, "--weights", $Weights, "--task", "test",
  "--imgsz", "$ImgSz", "--batch-size", "$BatchSize", "--device", "0", "--half",
  "--conf-thres", "0.001", "--iou-thres", "0.6", "--save-aot-predictions",
  "--project", $outputRoot, "--name", $name, "--exist-ok"
)
$proc = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id | Set-Content $pidFile -Encoding ASCII
@{
  pid = $proc.Id; start_time = (Get-Date).ToString("o"); weights = $Weights; data = $Data;
  output_dir = (Join-Path $outputRoot $name); stdout_log = $outLog; stderr_log = $errLog;
  total_images = 71608; total_batches = [math]::Ceiling(71608 / $BatchSize)
} | ConvertTo-Json | Set-Content $metaFile -Encoding UTF8
Write-Host "RUNNING"
Write-Host "PID: $($proc.Id)"
Write-Host "stdout: $outLog"
Write-Host "stderr: $errLog"
