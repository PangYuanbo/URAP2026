param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone",
  [string]$DataYaml = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\data\NPS_URAP_D.yaml",
  [string]$Weights = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt",
  [ValidateSet("train", "val", "test")]
  [string]$Task = "val",
  [string]$Project = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\runs\val\NPS_URAP_D",
  [string]$RunName = "nps_val_detached",
  [int]$BatchSize = 2,
  [int]$Img = 1280,
  [int]$NumFrames = 5,
  [double]$ConfThres = 0.001,
  [double]$IouThres = 0.6,
  [switch]$SaveJsonGt,
  [string]$RunId = "tvd_nps_eval",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\tvd_nps_eval_runner"
)

$ErrorActionPreference = "Stop"
$py = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $py -PathType Leaf)) { throw "Missing python venv: $py" }
if (-not (Test-Path -Path $DataYaml -PathType Leaf)) { throw "Missing data yaml: $DataYaml" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Missing weights: $Weights" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*val.py*" -and $existing.CommandLine -like "*--task $Task*") {
      Write-Host "TransVisDrone NPS eval already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

New-Item -ItemType Directory -Force -Path $Project | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"

$argList = @(
  ".\val.py",
  "--task", $Task,
  "--data", $DataYaml,
  "--weights", $Weights,
  "--img", [string]$Img,
  "--batch-size", [string]$BatchSize,
  "--half",
  "--num-frames", [string]$NumFrames,
  "--conf-thres", [string]$ConfThres,
  "--iou-thres", [string]$IouThres,
  "--project", $Project,
  "--name", $RunName,
  "--exist-ok"
)
if ($SaveJsonGt) { $argList += "--save-json-gt" }

$proc = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $RepoDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$py",
  "run_id=$RunId",
  "repo_dir=$RepoDir",
  "data_yaml=$DataYaml",
  "weights=$Weights",
  "task=$Task",
  "project=$Project",
  "run_name=$RunName",
  "batch_size=$BatchSize",
  "img=$Img",
  "num_frames=$NumFrames",
  "conf_thres=$ConfThres",
  "iou_thres=$IouThres",
  "save_json_gt=$SaveJsonGt",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached TransVisDrone NPS eval."
Get-Content $metaFile
