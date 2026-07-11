param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$DatasetRoot = "U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1",
  [string]$ArtifactRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness",
  [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

function Write-PipelineProgress {
  param([string]$Stage, [string]$Status, [string]$Detail)
  $payload = [ordered]@{
    updated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    stage = $Stage
    status = $Status
    detail = $Detail
  }
  $payload | ConvertTo-Json | Set-Content (Join-Path $ArtifactRoot "pipeline_progress.json") -Encoding utf8
  Write-Host "[$($payload.updated)] stage=$Stage status=$Status detail=$Detail"
}

function Read-Meta {
  param([string]$Path)
  $meta = @{}
  if (Test-Path $Path) {
    foreach ($line in Get-Content $Path) {
      $index = $line.IndexOf('=')
      if ($index -gt 0) { $meta[$line.Substring(0, $index)] = $line.Substring($index + 1) }
    }
  }
  return $meta
}

function Wait-DetachedJob {
  param(
    [string]$RunnerRoot,
    [string]$RunId,
    [string]$ExpectedCommand,
    [scriptblock]$SuccessCheck,
    [string]$Stage
  )
  $pidFile = Join-Path $RunnerRoot "$RunId.pid"
  $metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
  while ($true) {
    $pidValue = if (Test-Path $pidFile) { Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
    $process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
    $meta = Read-Meta $metaFile
    $latest = @($meta['stdout'], $meta['stderr']) | Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($process -and $process.CommandLine -like "*$ExpectedCommand*") {
      Write-PipelineProgress $Stage "running" "pid=$pidValue last_output=$(if($latest){$latest.LastWriteTime}else{'none'})"
      Start-Sleep -Seconds $PollSeconds
      continue
    }
    if (& $SuccessCheck) {
      Write-PipelineProgress $Stage "completed" "pid=$pidValue stopped; outputs verified"
      return
    }
    $stderrTail = if ($meta['stderr'] -and (Test-Path $meta['stderr'])) { (Get-Content $meta['stderr'] -Tail 30) -join "`n" } else { "stderr unavailable" }
    throw "$Stage stopped without verified outputs. pid=$pidValue`n$stderrTail"
  }
}

function Test-InterventionsReady {
  param([string[]]$Names)
  foreach ($name in $Names) {
    $integrity = Join-Path $DatasetRoot "$name\integrity.json"
    if (-not (Test-Path $integrity)) { return $false }
    $payload = Get-Content $integrity -Raw | ConvertFrom-Json
    if (-not $payload.valid) { return $false }
  }
  return $true
}

New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null

$originalRunner = Join-Path $ArtifactRoot "dataset_builder_original"
$originalRunId = "nps_motion_original_full"
if (-not (Test-InterventionsReady @('original'))) {
  Write-PipelineProgress "original_dataset" "starting" "train,val,test"
  & (Join-Path $RepoRoot "tools\start_nps_motion_interventions_detached.ps1") `
    -OutRoot $DatasetRoot -Splits @('train','val','test') -Interventions @('original') `
    -RunId $originalRunId -RunnerRoot $originalRunner
}
Wait-DetachedJob $originalRunner $originalRunId "build_nps_motion_interventions.py" { Test-InterventionsReady @('original') } "original_dataset"

$trainRunner = Join-Path $ArtifactRoot "train_runner"
$trainRunId = "nps_yolomg_train50"
$bestWeights = Join-Path $ArtifactRoot "yolomg_nps_train50\weights\best.pt"
$lastWeights = Join-Path $ArtifactRoot "yolomg_nps_train50\weights\last.pt"
if (-not (Test-Path $bestWeights)) {
  if (Test-Path $lastWeights) {
    Write-PipelineProgress "training_and_interventions" "resuming_training" "YOLOMG NPS from $lastWeights"
    & (Join-Path $RepoRoot "tools\start_nps_yolomg_train50_detached.ps1") -RunId $trainRunId -RunnerRoot $trainRunner -ResumeExisting
  } else {
    Write-PipelineProgress "training_and_interventions" "starting_training" "YOLOMG NPS 50 epochs"
    & (Join-Path $RepoRoot "tools\start_nps_yolomg_train50_detached.ps1") -RunId $trainRunId -RunnerRoot $trainRunner
  }
}

$interventionRunner = Join-Path $ArtifactRoot "dataset_builder_test_interventions"
$interventionRunId = "nps_motion_test_interventions_full"
$changedNames = @('slow_0p5','fast_2x','accelerate_g2','decelerate_g2')
if (-not (Test-InterventionsReady $changedNames)) {
  Write-PipelineProgress "training_and_interventions" "starting_interventions" ($changedNames -join ',')
  & (Join-Path $RepoRoot "tools\start_nps_motion_interventions_detached.ps1") `
    -OutRoot $DatasetRoot -Splits @('test') -Interventions $changedNames `
    -RunId $interventionRunId -RunnerRoot $interventionRunner
}

Wait-DetachedJob $trainRunner $trainRunId "train.py" { Test-Path $bestWeights } "yolomg_train50"
Wait-DetachedJob $interventionRunner $interventionRunId "build_nps_motion_interventions.py" { Test-InterventionsReady $changedNames } "test_interventions"

$evalRunner = Join-Path $ArtifactRoot "eval_runner"
$evalRunId = "nps_motion_model_evals_full"
$evalRoot = Join-Path $ArtifactRoot "model_evals"
$evalSummary = Join-Path $evalRoot "summary.json"
$evalReady = {
  if (-not (Test-Path $evalSummary)) { return $false }
  $payload = Get-Content $evalSummary -Raw | ConvertFrom-Json
  return $payload.done -eq 15 -and $payload.total -eq 15 -and $payload.skipped.Count -eq 0
}
if (-not (& $evalReady)) {
  Write-PipelineProgress "model_evaluations" "starting" "3 models x 5 interventions"
  & (Join-Path $RepoRoot "tools\start_nps_motion_model_evals_detached.ps1") -DatasetRoot $DatasetRoot -EvalRoot $evalRoot -RunId $evalRunId -RunnerRoot $evalRunner
}
Wait-DetachedJob $evalRunner $evalRunId "run_nps_motion_model_evals.py" $evalReady "model_evaluations"

$reportRoot = Join-Path $ArtifactRoot "report"
$python = Join-Path $RepoRoot "papers\TransVisDrone\.venv\Scripts\python.exe"
Write-PipelineProgress "report" "running" $reportRoot
& $python (Join-Path $RepoRoot "tools\summarize_nps_motion_robustness.py") --dataset-root $DatasetRoot --eval-root $evalRoot --out-dir $reportRoot --bootstrap-samples 5000
if ($LASTEXITCODE -ne 0) { throw "Report generation failed with exit code $LASTEXITCODE" }
$reportSummary = Join-Path $reportRoot "summary.json"
if (-not (Test-Path $reportSummary)) { throw "Report summary missing: $reportSummary" }
Write-PipelineProgress "complete" "completed" $reportSummary
