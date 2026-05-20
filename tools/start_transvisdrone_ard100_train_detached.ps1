param(
    [string]$Weights = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt",
    [string]$Data = "D:\URAP_datasets\TransVisDrone\ARD100\ARD100_TVD.yaml",
    [string]$Hyp = "data\\hyps\\hyp.VisDrone_3.yaml",
    [string]$Cfg = "models\\yolov5l-xs-tph-temporal.yaml",
    [string]$Name = "ard100_temporal3_from_nps",
    [int]$Epochs = 50,
    [int]$BatchSize = 4,
    [int]$ImgSz = 1280
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\TransVisDrone"
$runDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\transvisdrone_ard100_train"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "transvisdrone_ard100_train_$ts.out.txt"
$errLog = Join-Path $logDir "transvisdrone_ard100_train_$ts.err.txt"
$pidFile = Join-Path $runDir "pid.txt"
$metaFile = Join-Path $runDir "meta.json"

$py = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$args = @(
  "train.py",
  "--data", $Data,
  "--weights", $Weights,
  "--cfg", $Cfg,
  "--hyp", $Hyp,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--imgsz", "$ImgSz",
  "--device", "0",
  "--workers", "4",
  "--project", "runs\train\ARD100",
  "--name", $Name,
  "--exist-ok"
)

$proc = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id | Set-Content $pidFile -Encoding ASCII
@{
  pid = $proc.Id
  start_time = (Get-Date).ToString("s")
  weights = $Weights
  data = $Data
  cfg = $Cfg
  hyp = $Hyp
  name = $Name
  stdout_log = $outLog
  stderr_log = $errLog
} | ConvertTo-Json | Set-Content $metaFile -Encoding UTF8

Write-Host "RUNNING"
Write-Host "PID: $($proc.Id)"
Write-Host "stdout: $outLog"
Write-Host "stderr: $errLog"
