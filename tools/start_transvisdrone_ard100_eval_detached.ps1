param(
    [string]$Weights = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt",
    [string]$Data = "D:\URAP_datasets\TransVisDrone\ARD100\ARD100_TVD.yaml",
    [string]$Name = "nps_best_on_ard100_test",
    [int]$BatchSize = 4,
    [int]$ImgSz = 1280
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\TransVisDrone"
$runDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\transvisdrone_ard100_eval"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "transvisdrone_ard100_eval_$ts.out.txt"
$errLog = Join-Path $logDir "transvisdrone_ard100_eval_$ts.err.txt"
$pidFile = Join-Path $runDir "pid.txt"
$metaFile = Join-Path $runDir "meta.json"

$py = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$args = @(
  "inference.py",
  "--data", $Data,
  "--weights", $Weights,
  "--task", "test",
  "--imgsz", "$ImgSz",
  "--batch-size", "$BatchSize",
  "--device", "0",
  "--half",
  "--project", "runs\inference\ARD100",
  "--name", $Name,
  "--exist-ok",
  "--save-txt",
  "--save-conf"
)

$proc = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id | Set-Content $pidFile -Encoding ASCII
@{
  pid = $proc.Id
  start_time = (Get-Date).ToString("s")
  weights = $Weights
  data = $Data
  name = $Name
  stdout_log = $outLog
  stderr_log = $errLog
} | ConvertTo-Json | Set-Content $metaFile -Encoding UTF8

Write-Host "RUNNING"
Write-Host "PID: $($proc.Id)"
Write-Host "stdout: $outLog"
Write-Host "stderr: $errLog"
