param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$Yolov5Repo = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG",
  [string]$ImagesList = "U:\URAP_datasets\ARD100_YOLOMG\val.txt",
  [string]$Weights = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\ARD100_mask32-1280_uavs\weights\best.pt",
  [string]$OutDir = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_full",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_runner",
  [string]$RunId = "samurai_memory_gate_yolomg_val",
  [ValidateSet("temporal", "raw")]
  [string]$FinalSelectionScore = "temporal",
  [ValidateSet("temporal", "raw")]
  [string]$FinalOutputScore = "temporal",
  [ValidateSet("final", "temporal")]
  [string]$MemoryUpdateSelection = "final",
  [bool]$ApplyOutputGate = $true,
  [string]$Device = "0",
  [double]$Conf = 0.001,
  [double]$IouThres = 0.45,
  [int]$ImgSize = 1280,
  [int]$TopK = 80,
  [double]$NccMinScore = 0.62,
  [double]$NccScore = 0.34,
  [double]$MemoryQualityMin = 0.38,
  [double]$MemoryDetectorMin = 0.05,
  [double]$MemoryMotionMin = 0.08,
  [bool]$AllowSupportOnlyOutput = $true,
  [double]$SupportOnlyOutputMinQuality = 0.72,
  [int]$SupportOnlyMinDetectorUpdates = 2,
  [int]$SupportOnlyMaxMisses = 1,
  [int]$MaxFrames = 0,
  [int]$ProgressEvery = 500,
  [switch]$WriteCandidateLabels = $true
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Missing python: $Python" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "Missing images list: $ImagesList" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Missing weights: $Weights" }
if (-not (Test-Path -Path $Yolov5Repo -PathType Container)) { throw "Missing YOLOv5 repo: $Yolov5Repo" }

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunRoot "$RunId.pid"
$metaFile = Join-Path $RunRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*run_temporal_recovery_pipeline.py*" -and $existing.CommandLine -like "*$OutDir*") {
      Write-Host "SAMURAI memory-gate YOLOMG val already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
    Write-Host "Previous SAMURAI memory-gate PID is NOT RUNNING: pid=$existingPid"
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"

$argList = @(
  "tools\run_temporal_recovery_pipeline.py",
  "--detector-backend", "yolov5-dual",
  "--yolov5-repo", $Yolov5Repo,
  "--image-list", $ImagesList,
  "--out-dir", $OutDir,
  "--yolo-weights", $Weights,
  "--profile", "dji-tiny",
  "--final-selection-score", $FinalSelectionScore,
  "--final-output-score", $FinalOutputScore,
  "--memory-update-selection", $MemoryUpdateSelection,
  "--device", $Device,
  "--conf", [string]$Conf,
  "--iou-thres", [string]$IouThres,
  "--img-size", [string]$ImgSize,
  "--top-k", [string]$TopK,
  "--ncc-min-score", [string]$NccMinScore,
  "--ncc-score", [string]$NccScore,
  "--memory-quality-min", [string]$MemoryQualityMin,
  "--memory-detector-min", [string]$MemoryDetectorMin,
  "--memory-motion-min", [string]$MemoryMotionMin,
  "--support-only-output-min-quality", [string]$SupportOnlyOutputMinQuality,
  "--support-only-min-detector-updates", [string]$SupportOnlyMinDetectorUpdates,
  "--support-only-max-misses", [string]$SupportOnlyMaxMisses,
  "--progress-every", [string]$ProgressEvery
)
if (-not $AllowSupportOnlyOutput) { $argList += @("--no-allow-support-only-output") }
if (-not $ApplyOutputGate) { $argList += @("--no-apply-output-gate") }
if ($MaxFrames -gt 0) { $argList += @("--max-frames", [string]$MaxFrames) }
if ($WriteCandidateLabels) { $argList += @("--write-candidate-labels") }

$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $URAPRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "urap_root=$URAPRoot",
  "yolov5_repo=$Yolov5Repo",
  "images_list=$ImagesList",
  "weights=$Weights",
  "out_dir=$OutDir",
  "final_selection_score=$FinalSelectionScore",
  "final_output_score=$FinalOutputScore",
  "memory_update_selection=$MemoryUpdateSelection",
  "apply_output_gate=$ApplyOutputGate",
  "device=$Device",
  "conf=$Conf",
  "iou_thres=$IouThres",
  "img_size=$ImgSize",
  "top_k=$TopK",
  "ncc_min_score=$NccMinScore",
  "ncc_score=$NccScore",
  "memory_quality_min=$MemoryQualityMin",
  "memory_detector_min=$MemoryDetectorMin",
  "memory_motion_min=$MemoryMotionMin",
  "allow_support_only_output=$AllowSupportOnlyOutput",
  "support_only_output_min_quality=$SupportOnlyOutputMinQuality",
  "support_only_min_detector_updates=$SupportOnlyMinDetectorUpdates",
  "support_only_max_misses=$SupportOnlyMaxMisses",
  "max_frames=$MaxFrames",
  "progress_every=$ProgressEvery",
  "write_candidate_labels=$WriteCandidateLabels",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached SAMURAI memory-gate YOLOMG val."
Get-Content $metaFile
