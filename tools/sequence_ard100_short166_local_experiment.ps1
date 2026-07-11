param(
    [string]$DatasetRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166",
    [string]$RunRoot = "U:\URAP_runs\samurai\ard100_short166_experiment_v1",
    [int]$MinimumFreeMiB = 29000,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\ard100_short166_local_experiment_v1"
$statePath = Join-Path $controlRoot "state.json"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$baseCheckpoint = Join-Path $repoRoot "artifacts\samurai_models\sam2.1_hiera_base_plus.pt"
$fineTunedCheckpoint = "U:\URAP_runs\samurai\finetune_base_plus_ard100_short166_stage1\checkpoints\checkpoint.pt"
$trainRunName = "finetune_base_plus_ard100_short166_stage1"
$expectedVideos = @{ train = 55; val = 10; test = 35 }
New-Item -ItemType Directory -Force -Path $controlRoot, $RunRoot | Out-Null

function Write-State([string]$Stage, [int]$Done, [int]$Total, [string]$LastUnit, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{
        status = if ($Stage -eq "completed") { "completed" } elseif ($Stage -eq "failed") { "failed" } else { "running" }
        stage = $Stage
        done = $Done
        total = $Total
        last_completed_unit = $LastUnit
        updated_at = (Get-Date).ToString("o")
    }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $temporary = "$statePath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
    $payload | ConvertTo-Json -Compress -Depth 8 | Write-Output
}

function Get-ConflictingGpuJobs {
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match 'train\.py|eval_samurai_nps\.py|train_joint_detector\.py'
    })
}

function Wait-GpuReady([string]$NextStage, [int]$Done, [int]$RequiredFreeMiB, [bool]$AllowConflictingTraining) {
    $consecutive = 0
    while ($consecutive -lt 3) {
        $gpuFields = (& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits).Trim().Split(',')
        $freeMiB = [int]$gpuFields[0].Trim()
        $utilization = [int]$gpuFields[1].Trim()
        $conflicts = @(Get-ConflictingGpuJobs)
        $conflictGate = $AllowConflictingTraining -or $conflicts.Count -eq 0
        if ($freeMiB -ge $RequiredFreeMiB -and $conflictGate) { $consecutive += 1 } else { $consecutive = 0 }
        Write-State "waiting_for_gpu" $Done 6 $NextStage @{
            gpu_free_mib = $freeMiB
            required_free_mib = $RequiredFreeMiB
            allow_conflicting_training = $AllowConflictingTraining
            gpu_utilization = $utilization
            consecutive_ready_checks = $consecutive
            conflicting_jobs = @($conflicts | Select-Object ProcessId, CreationDate, CommandLine)
        }
        if ($consecutive -lt 3) { Start-Sleep -Seconds $PollSeconds }
    }
}

function Wait-Dataset {
    while (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot "LOCAL_MATERIALIZE_COMPLETE.json"))) {
        Write-State "waiting_for_dataset" 0 6 "LOCAL_MATERIALIZE_COMPLETE.json"
        Start-Sleep -Seconds $PollSeconds
    }
    foreach ($split in @("train", "val", "test")) {
        $manifestPath = Join-Path $DatasetRoot "${split}_v1\manifest.json"
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ([int]$manifest.source_video_count -ne $expectedVideos[$split]) {
            throw "Incomplete $split source videos: $($manifest.source_video_count)/$($expectedVideos[$split])"
        }
        if ([int]$manifest.sequence_count -le 0) { throw "No $split tracklets in $manifestPath" }
    }
}

function Read-JsonWithRetry([string]$Path, [int]$Attempts = 5) {
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            if (-not (Test-Path -LiteralPath $Path)) { return $null }
            return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        } catch [System.IO.IOException] {
            if ($attempt -eq $Attempts) { return $null }
            Start-Sleep -Milliseconds 250
        }
    }
    return $null
}

function Start-And-Wait-Evaluation(
    [string]$RunName,
    [string]$ModelConfig,
    [string]$PropagationMode,
    [string]$Checkpoint,
    [int]$DoneBefore
) {
    $outputRoot = Join-Path $RunRoot "eval\$RunName\canonical"
    $metricsPath = Join-Path $outputRoot "metrics.json"
    $testManifest = Read-JsonWithRetry (Join-Path $DatasetRoot "test_v1\manifest.json")
    $existingMetrics = Read-JsonWithRetry $metricsPath
    if ($existingMetrics -and [int]$existingMetrics.sequences -eq [int]$testManifest.sequence_count) {
        Write-State "evaluation_completed" ($DoneBefore + 1) 6 $RunName @{
            resumed_complete = $true
            success_auc = [double]$existingMetrics.success_auc
            success_50 = [double]$existingMetrics.success_50
            precision_20 = [double]$existingMetrics.precision_20
        }
        return
    }
    # A measured base-plus 166-frame SAMURAI smoke added only ~1.82 GiB.
    # Zero-shot evaluation may share the remaining memory, but training may not.
    Wait-GpuReady $RunName $DoneBefore 16000 $true
    $metadataJson = & (Join-Path $PSScriptRoot "start_samurai_nps_eval_detached.ps1") `
        -DatasetRoot (Join-Path $DatasetRoot "test_v1") `
        -Split "test" `
        -Checkpoint $Checkpoint `
        -ModelConfig $ModelConfig `
        -RunRoot $outputRoot `
        -Device "cuda:0" `
        -Dtype "bfloat16" `
        -PropagationMode $PropagationMode `
        -Resume `
        -AsyncLoadingFrames `
        -ControlName "ard100_short166_$RunName"
    if ($LASTEXITCODE -ne 0) { throw "Could not start evaluation $RunName" }
    $metadata = $metadataJson | ConvertFrom-Json
    $pidValue = [int]$metadata.pid
    Write-State "evaluating" $DoneBefore 6 $RunName @{ child_pid = $pidValue; run = $RunName }
    while (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
        $progressPath = Join-Path $outputRoot "progress.json"
        $progress = Read-JsonWithRetry $progressPath
        Write-State "evaluating" $DoneBefore 6 $RunName @{
            child_pid = $pidValue
            run = $RunName
            evaluation_done = if ($progress) { [int]$progress.done_sequences } else { 0 }
            evaluation_total = if ($progress) { [int]$progress.total_sequences } else { 0 }
            last_sequence = if ($progress) { $progress.last_completed_sequence } else { $null }
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $metricsPath)) { throw "Evaluation $RunName ended without $metricsPath" }
    $metrics = Read-JsonWithRetry $metricsPath
    $testManifest = Read-JsonWithRetry (Join-Path $DatasetRoot "test_v1\manifest.json")
    if ([int]$metrics.sequences -ne [int]$testManifest.sequence_count) {
        throw "Evaluation $RunName is incomplete: $($metrics.sequences)/$($testManifest.sequence_count)"
    }
    Write-State "evaluation_completed" ($DoneBefore + 1) 6 $RunName @{
        success_auc = [double]$metrics.success_auc
        success_50 = [double]$metrics.success_50
        precision_20 = [double]$metrics.precision_20
    }
}

function Start-And-Wait-Training([int]$DoneBefore) {
    Wait-GpuReady $trainRunName $DoneBefore $MinimumFreeMiB $false
    $metadataJson = & (Join-Path $PSScriptRoot "start_ard100_short166_local_train_detached.ps1") -RunName $trainRunName
    if ($LASTEXITCODE -ne 0) { throw "Could not start ARD100 short166 training" }
    $metadata = $metadataJson | ConvertFrom-Json
    $pidValue = [int]$metadata.pid
    Write-State "training" $DoneBefore 6 $trainRunName @{ child_pid = $pidValue }
    while (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
        $log = Get-Content -LiteralPath $metadata.stdout_log -Tail 200 -ErrorAction SilentlyContinue
        $latest = @($log | Select-String -Pattern 'Train Epoch: \[(\d+)\]\[(\s*\d+)/(\s*\d+)\]') | Select-Object -Last 1
        Write-State "training" $DoneBefore 6 $trainRunName @{
            child_pid = $pidValue
            latest_train_line = if ($latest) { $latest.Line.Trim() } else { "initializing" }
            stdout_log = $metadata.stdout_log
            stderr_log = $metadata.stderr_log
        }
        Start-Sleep -Seconds $PollSeconds
    }
    & $python (Join-Path $PSScriptRoot "verify_samurai_checkpoint.py") --checkpoint $fineTunedCheckpoint --expected-final-epoch 16
    if ($LASTEXITCODE -ne 0) { throw "Fine-tuned checkpoint verification failed" }
    Write-State "training_completed" ($DoneBefore + 1) 6 $trainRunName @{ checkpoint = $fineTunedCheckpoint }
}

try {
    Wait-Dataset
    if (-not (Test-Path -LiteralPath $baseCheckpoint)) { throw "Missing base checkpoint: $baseCheckpoint" }
    Start-And-Wait-Evaluation "image_box_zero_shot" "configs/sam2.1/sam2.1_hiera_b+.yaml" "image-box" $baseCheckpoint 0
    Start-And-Wait-Evaluation "sam2_video_zero_shot" "configs/sam2.1/sam2.1_hiera_b+.yaml" "video" $baseCheckpoint 1
    Start-And-Wait-Evaluation "samurai_zero_shot" "configs/samurai/sam2.1_hiera_b+.yaml" "video" $baseCheckpoint 2
    Start-And-Wait-Training 3
    Start-And-Wait-Evaluation "sam2_video_finetuned" "configs/sam2.1/sam2.1_hiera_b+.yaml" "video" $fineTunedCheckpoint 4
    Start-And-Wait-Evaluation "samurai_finetuned" "configs/samurai/sam2.1_hiera_b+.yaml" "video" $fineTunedCheckpoint 5
    $summaryPath = Join-Path $RunRoot "ARD100_SHORT166_VS_NPS.json"
    & $python (Join-Path $PSScriptRoot "summarize_ard100_short166_results.py") --results-root $RunRoot --output $summaryPath
    if ($LASTEXITCODE -ne 0) { throw "Result summary failed" }
    Write-State "completed" 6 6 "ARD100_SHORT166_VS_NPS.json" @{ summary = $summaryPath }
} catch {
    Write-State "failed" 0 6 $_.Exception.Message @{ error = $_.ScriptStackTrace }
    throw
}
