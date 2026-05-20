param(
    [int]$WaitEvalPid = 0,
    [string]$Repo = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\TransVisDrone',
    [string]$PythonExe = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe',
    [string]$Data = 'D:\URAP_datasets\TransVisDrone\ARD100\ARD100_TVD.yaml',
    [string]$Weights = 'C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt',
    [string]$Cfg = 'models\\yolov5l-xs-tph-temporal.yaml',
    [string]$Hyp = 'data\\hyps\\hyp.VisDrone_3.yaml',
    [int]$Epochs = 1,
    [int]$BatchSize = 4,
    [int]$ImgSz = 1280,
    [string]$TrainName = 'ard100_temporal3_from_nps_e1',
    [string]$EvalName = 'ard100_temporal3_from_nps_e1_on_ard100_test'
)

$ErrorActionPreference = 'Stop'

function Write-Stage($msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Output "[$ts] $msg"
}

if ($WaitEvalPid -gt 0) {
    Write-Stage "Waiting for zero-shot eval PID $WaitEvalPid to finish"
    while (Get-Process -Id $WaitEvalPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-Stage "Zero-shot eval PID $WaitEvalPid finished"
}

$trainArgs = @(
  'train.py',
  '--data', $Data,
  '--weights', $Weights,
  '--cfg', $Cfg,
  '--hyp', $Hyp,
  '--epochs', "$Epochs",
  '--batch-size', "$BatchSize",
  '--imgsz', "$ImgSz",
  '--device', '0',
  '--workers', '4',
  '--project', 'runs\train\ARD100',
  '--name', $TrainName,
  '--exist-ok'
)

Write-Stage "Starting short fine-tune: $TrainName"
& $PythonExe @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Short fine-tune failed with exit code $LASTEXITCODE"
}
Write-Stage "Short fine-tune finished: $TrainName"

$trainedWeights = Join-Path $Repo "runs\train\ARD100\$TrainName\weights\best.pt"
if (-not (Test-Path $trainedWeights)) {
    $trainedWeights = Join-Path $Repo "runs\train\ARD100\$TrainName\weights\last.pt"
}
if (-not (Test-Path $trainedWeights)) {
    throw "No trained checkpoint found under $TrainName"
}
Write-Stage "Using trained weights: $trainedWeights"

$evalArgs = @(
  'inference.py',
  '--data', $Data,
  '--weights', $trainedWeights,
  '--task', 'test',
  '--imgsz', "$ImgSz",
  '--batch-size', "$BatchSize",
  '--device', '0',
  '--half',
  '--project', 'runs\inference\ARD100',
  '--name', $EvalName,
  '--exist-ok',
  '--save-txt',
  '--save-conf'
)

Write-Stage "Starting post-train eval: $EvalName"
& $PythonExe @evalArgs
if ($LASTEXITCODE -ne 0) {
    throw "Post-train eval failed with exit code $LASTEXITCODE"
}
Write-Stage "Post-train eval finished: $EvalName"
