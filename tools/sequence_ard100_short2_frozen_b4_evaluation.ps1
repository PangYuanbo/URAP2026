param(
    [string]$ExperimentRoot = "U:\URAP_runs\samurai\ard100_short2_frozen_b4_experiment_v1",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$statePath = Join-Path $controlRoot "ard100_short2_frozen_b4_evaluation.state.json"
$trainName = "finetune_base_plus_ard100_short2_frozen_b4_stage1"
$trainMetaPath = Join-Path $controlRoot "$trainName.meta.json"
$trainPidPath = Join-Path $controlRoot "$trainName.pid"
$checkpoint = "U:\URAP_runs\samurai\finetune_base_plus_ard100_short2_frozen_b4_stage1\checkpoints\checkpoint.pt"
$datasetRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166\test_v1"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $controlRoot, $ExperimentRoot | Out-Null

function Write-State([string]$Phase, [int]$Done, [int]$Total, [hashtable]$Extra = @{}) {
    $state = [ordered]@{
        status = $Phase
        done = $Done
        total = $Total
        updated_at = (Get-Date).ToString("o")
    }
    foreach ($key in $Extra.Keys) { $state[$key] = $Extra[$key] }
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding utf8
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
        catch { Start-Sleep -Milliseconds 250 }
    }
    return $null
}

function Wait-For-Training {
    if (-not (Test-Path -LiteralPath $trainMetaPath) -or -not (Test-Path -LiteralPath $trainPidPath)) {
        throw "Missing detached training metadata or PID file"
    }
    $trainMeta = Read-Json $trainMetaPath
    $trainPid = [int](Get-Content -LiteralPath $trainPidPath -Raw)
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$trainPid" -ErrorAction SilentlyContinue
        $matches = $process -and $process.CommandLine -like "*run_sam2_train_with_memory_cap.py*ARD100_short2_frozen_b4_local_stage1*"
        if (-not $matches) { break }
        $latest = Get-Content -LiteralPath $trainMeta.stdout_log -Tail 300 -ErrorAction SilentlyContinue |
            Select-String -Pattern 'Train Epoch: \[(\d+)\]\[\s*(\d+)/(\s*\d+)\]' |
            Select-Object -Last 1
        Write-State "waiting_for_training" 0 3 @{
            training_pid = $trainPid
            last_training_line = if ($latest) { $latest.Line.Trim() } else { "initializing" }
            training_stdout = $trainMeta.stdout_log
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Training PID ended without final checkpoint: $checkpoint"
    }
    & $python (Join-Path $PSScriptRoot "verify_samurai_checkpoint.py") --checkpoint $checkpoint --expected-final-epoch 16
    if ($LASTEXITCODE -ne 0) { throw "Final checkpoint verification failed" }
    Write-State "training_verified" 1 3 @{ checkpoint = $checkpoint; training_pid = $trainPid }
}

function Run-Evaluation(
    [string]$Name,
    [string]$ModelConfig,
    [int]$DoneBefore
) {
    $runRoot = Join-Path $ExperimentRoot $Name
    $metricsPath = Join-Path $runRoot "metrics.json"
    $controlName = "ard100_short2_frozen_b4_$Name"
    if (Test-Path -LiteralPath $metricsPath) {
        $existing = Read-Json $metricsPath
        if ($existing -and [int]$existing.sequences -eq 462) {
            Write-State "evaluation_completed" ($DoneBefore + 1) 3 @{ run = $Name; metrics = $metricsPath }
            return
        }
    }
    $evalParams = @{
        DatasetRoot = $datasetRoot
        Split = "test"
        Checkpoint = $checkpoint
        ModelConfig = $ModelConfig
        RunRoot = $runRoot
        Device = "cuda:0"
        Dtype = "bfloat16"
        PropagationMode = "video"
        Resume = $true
        AsyncLoadingFrames = $true
        ControlName = $controlName
    }
    $metadataJson = & (Join-Path $PSScriptRoot "start_samurai_nps_eval_detached.ps1") @evalParams
    if ($LASTEXITCODE -ne 0) { throw "Could not launch evaluation $Name" }
    $metadata = $metadataJson | ConvertFrom-Json
    $childPid = [int]$metadata.pid
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$childPid" -ErrorAction SilentlyContinue
        $matches = $process -and $process.CommandLine -like "*eval_samurai_nps.py*"
        $progress = Read-Json (Join-Path $runRoot "progress.json")
        Write-State "evaluating" $DoneBefore 3 @{
            run = $Name
            child_pid = $childPid
            evaluation_done = if ($progress) { [int]$progress.done_sequences } else { 0 }
            evaluation_total = if ($progress) { [int]$progress.total_sequences } else { 462 }
            last_sequence = if ($progress) { $progress.last_completed_sequence } else { $null }
            progress_file = (Join-Path $runRoot "progress.json")
        }
        if (-not $matches) { break }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $metricsPath)) { throw "Evaluation $Name ended without metrics.json" }
    $metrics = Read-Json $metricsPath
    if ([int]$metrics.sequences -ne 462) { throw "Evaluation $Name incomplete: $($metrics.sequences)/462" }
    Write-State "evaluation_completed" ($DoneBefore + 1) 3 @{ run = $Name; metrics = $metricsPath }
}

try {
    Wait-For-Training
    Run-Evaluation "sam2_video_finetuned_short2_frozen_b4" "configs/sam2.1/sam2.1_hiera_b+.yaml" 1
    Run-Evaluation "samurai_finetuned_short2_frozen_b4" "configs/samurai/sam2.1_hiera_b+.yaml" 2
    $sam2 = Read-Json (Join-Path $ExperimentRoot "sam2_video_finetuned_short2_frozen_b4\metrics.json")
    $samurai = Read-Json (Join-Path $ExperimentRoot "samurai_finetuned_short2_frozen_b4\metrics.json")
    $summary = [ordered]@{
        protocol = "ARD100 test_v1, 462 trajectories, maximum 166 frames, first-frame GT only"
        training = "ARD100 full train split, 53450 two-frame windows, frozen image encoder"
        sam2_video_finetuned = $sam2
        samurai_finetuned = $samurai
        ard_short_zero_shot = [ordered]@{
            sam2_success_auc = 0.3469309864
            samurai_success_auc = 0.3664677220
        }
        nps_reference = [ordered]@{
            sam2_zero_shot_success_auc = 0.5954171361
            samurai_zero_shot_success_auc = 0.6014026323
            sam2_finetuned_success_auc = 0.6765270148
            samurai_finetuned_success_auc = 0.6572430036
        }
    }
    $summaryPath = Join-Path $ExperimentRoot "ARD100_SHORT2_FROZEN_B4_VS_NPS.json"
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-State "completed" 3 3 @{ summary = $summaryPath; checkpoint = $checkpoint }
} catch {
    Write-State "failed" 0 3 @{ error = $_.Exception.Message; stack = $_.ScriptStackTrace }
    throw
}
