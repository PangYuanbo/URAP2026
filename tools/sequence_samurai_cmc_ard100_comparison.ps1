param(
    [string]$RunRoot = "U:\URAP_runs\samurai\ard100_samurai_cmc_comparison_v1",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$statePath = Join-Path $controlRoot "samurai_cmc_ard100_comparison.state.json"
$upstreamPidPath = Join-Path $controlRoot "ard100_short2_frozen_b4_evaluation.pid"
$datasetRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166\test_v1"
$checkpoint = "U:\URAP_runs\samurai\finetune_base_plus_ard100_short2_frozen_b4_stage1\checkpoints\checkpoint.pt"
New-Item -ItemType Directory -Force -Path $controlRoot, $RunRoot | Out-Null

function Write-State([string]$Status, [int]$Done, [hashtable]$Extra = @{}) {
    $state = [ordered]@{ status = $Status; done = $Done; total = 2; updated_at = (Get-Date).ToString("o") }
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

function Wait-Upstream {
    if (-not (Test-Path -LiteralPath $upstreamPidPath)) { return }
    $upstreamPid = [int](Get-Content -LiteralPath $upstreamPidPath -Raw)
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$upstreamPid" -ErrorAction SilentlyContinue
        $matches = $process -and $process.CommandLine -like "*sequence_ard100_short2_frozen_b4_evaluation.ps1*"
        if (-not $matches) { return }
        Write-State "waiting_for_upstream_evaluation" 0 @{ upstream_pid = $upstreamPid }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Run-Evaluation([string]$Name, [string]$ModelConfig, [int]$DoneBefore) {
    $outputRoot = Join-Path $RunRoot $Name
    $metricsPath = Join-Path $outputRoot "metrics.json"
    if (Test-Path -LiteralPath $metricsPath) {
        $metrics = Read-Json $metricsPath
        if ($metrics -and [int]$metrics.sequences -eq 462) {
            Write-State "evaluation_completed" ($DoneBefore + 1) @{ run = $Name; metrics = $metricsPath }
            return
        }
    }
    $params = @{
        DatasetRoot = $datasetRoot
        Split = "test"
        Checkpoint = $checkpoint
        ModelConfig = $ModelConfig
        RunRoot = $outputRoot
        Device = "cuda:0"
        Dtype = "bfloat16"
        PropagationMode = "video"
        Resume = $true
        AsyncLoadingFrames = $true
        ControlName = "ard100_$Name"
    }
    $metadataJson = & (Join-Path $PSScriptRoot "start_samurai_nps_eval_detached.ps1") @params
    if ($LASTEXITCODE -ne 0) { throw "Could not launch $Name" }
    $metadata = $metadataJson | ConvertFrom-Json
    $childPid = [int]$metadata.pid
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$childPid" -ErrorAction SilentlyContinue
        $matches = $process -and $process.CommandLine -like "*eval_samurai_nps.py*"
        $progress = Read-Json (Join-Path $outputRoot "progress.json")
        Write-State "evaluating" $DoneBefore @{
            run = $Name
            child_pid = $childPid
            evaluation_done = if ($progress) { [int]$progress.done_sequences } else { 0 }
            evaluation_total = if ($progress) { [int]$progress.total_sequences } else { 462 }
            last_sequence = if ($progress) { $progress.last_completed_sequence } else { $null }
            progress_file = (Join-Path $outputRoot "progress.json")
        }
        if (-not $matches) { break }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $metricsPath)) { throw "$Name ended without metrics.json" }
    $metrics = Read-Json $metricsPath
    if ([int]$metrics.sequences -ne 462) { throw "$Name incomplete: $($metrics.sequences)/462" }
    Write-State "evaluation_completed" ($DoneBefore + 1) @{ run = $Name; metrics = $metricsPath }
}

try {
    Wait-Upstream
    Run-Evaluation "samurai_reset" "configs/samurai/sam2.1_hiera_b+.yaml" 0
    Run-Evaluation "samurai_cmc" "configs/samurai_cmc/sam2.1_hiera_b+.yaml" 1
    $plain = Read-Json (Join-Path $RunRoot "samurai_reset\metrics.json")
    $cmc = Read-Json (Join-Path $RunRoot "samurai_cmc\metrics.json")
    $summary = [ordered]@{
        protocol = "ARD100 test_v1, 462 trajectories, first-frame GT only"
        checkpoint = $checkpoint
        samurai_reset = $plain
        samurai_cmc = $cmc
        delta = [ordered]@{
            success_auc = [double]$cmc.success_auc - [double]$plain.success_auc
            mean_iou = [double]$cmc.mean_iou - [double]$plain.mean_iou
            success_50 = [double]$cmc.success_50 - [double]$plain.success_50
            precision_20 = [double]$cmc.precision_20 - [double]$plain.precision_20
        }
    }
    $summaryPath = Join-Path $RunRoot "SAMURAI_CMC_VS_RESET.json"
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-State "completed" 2 @{ summary = $summaryPath }
} catch {
    Write-State "failed" 0 @{ error = $_.Exception.Message; stack = $_.ScriptStackTrace }
    throw
}
