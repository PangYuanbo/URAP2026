param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$TrainCandidateJsonl = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_default_candidate_pool_yolomg_train_full\candidate_predictions.jsonl",
  [string]$TestCandidateJsonl = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_default_candidate_pool_yolomg_val_full\candidate_predictions.jsonl",
  [string]$OutRoot = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\candidate_temporal_reranker_yolomg_val",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\candidate_temporal_reranker_yolomg_val_runner",
  [string]$RunId = "candidate_temporal_reranker_yolomg_val",
  [double]$MatchIou = 0.5,
  [int]$Epochs = 20,
  [int]$BatchSize = 16384,
  [double]$Lr = 0.03,
  [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Missing python: $Python" }
if (-not (Test-Path -Path $TrainCandidateJsonl -PathType Leaf)) { throw "Missing train candidate jsonl: $TrainCandidateJsonl" }
if (-not (Test-Path -Path $TestCandidateJsonl -PathType Leaf)) { throw "Missing test candidate jsonl: $TestCandidateJsonl" }

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$logsDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $RunRoot "$RunId.pid"
$metaFile = Join-Path $RunRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*train_candidate_temporal_reranker.py*" -and $existing.CommandLine -like "*$OutRoot*") {
      Write-Host "Candidate temporal reranker already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
    Write-Host "Previous candidate temporal reranker PID is NOT RUNNING: pid=$existingPid"
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"
$outLabelDir = Join-Path $OutRoot "pred_labels"
$outModel = Join-Path $OutRoot "candidate_temporal_reranker.npz"
$outSummary = Join-Path $OutRoot "summary.json"

$argList = @(
  "tools\train_candidate_temporal_reranker.py",
  "--train-candidate-jsonl", $TrainCandidateJsonl,
  "--test-candidate-jsonl", $TestCandidateJsonl,
  "--out-label-dir", $outLabelDir,
  "--out-model", $outModel,
  "--out-summary", $outSummary,
  "--match-iou", [string]$MatchIou,
  "--epochs", [string]$Epochs,
  "--batch-size", [string]$BatchSize,
  "--lr", [string]$Lr,
  "--seed", [string]$Seed
)

$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $URAPRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "urap_root=$URAPRoot",
  "train_candidate_jsonl=$TrainCandidateJsonl",
  "test_candidate_jsonl=$TestCandidateJsonl",
  "out_root=$OutRoot",
  "out_label_dir=$outLabelDir",
  "out_model=$outModel",
  "out_summary=$outSummary",
  "match_iou=$MatchIou",
  "epochs=$Epochs",
  "batch_size=$BatchSize",
  "lr=$Lr",
  "seed=$Seed",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached candidate temporal reranker."
Get-Content $metaFile
