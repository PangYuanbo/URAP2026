param(
    [string]$ProgressPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_runs\ard100_finetune_restart1_sequencer.progress.json"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$mergeProgress = Join-Path $repoRoot "artifacts\samurai_runs\ard100_zero_shot_merge_watcher.progress.json"
$startScript = Join-Path $repoRoot "tools\start_samurai_finetune_detached.ps1"
$preflightScript = Join-Path $repoRoot "tools\check_samurai_ard100_restart_ready.ps1"
function Write-State($payload) {
    $tmp = "$ProgressPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $ProgressPath -Force
}
& $preflightScript | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-State ([ordered]@{status="failed_preflight";observed_at=(Get-Date).ToString("o");preflight=(Join-Path $repoRoot "artifacts\samurai_runs\ard100_restart_preflight.json")})
    throw "ARD100 restart preflight failed"
}

function Wait-Run([string]$RunName, [string]$RunRoot, [int]$Epochs, [string]$CheckpointDir = "") {
    $control = Join-Path $repoRoot "artifacts\samurai_runs"
    if (-not $CheckpointDir) { $CheckpointDir = Join-Path $RunRoot "checkpoints" }
    $metaPath = Join-Path $control "$RunName.meta.json"
    while (-not (Test-Path $metaPath)) { Start-Sleep -Seconds 5 }
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
        $running = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*training*train.py*"
        $checkpoints = @(Get-ChildItem (Join-Path $CheckpointDir "checkpoint_*.pt") -ErrorAction SilentlyContinue)
        $complete = (-not $running) -and (
            $checkpoints.Count -ge $Epochs -or (
                $Epochs -eq 1 -and (Test-Path (Join-Path $CheckpointDir "checkpoint.pt"))
            )
        )
        $logs = @(Get-Item $meta.stdout_log,$meta.stderr_log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
        Write-State ([ordered]@{status=if($running){"running_$RunName"}elseif($complete){"completed_$RunName"}else{"failed_$RunName"};run_name=$RunName;pid=$meta.pid;started_at=$meta.started_at;last_output_timestamp=if($logs.Count){$logs[0].LastWriteTime.ToString("o")}else{$null};stdout_log=$meta.stdout_log;stderr_log=$meta.stderr_log;checkpoint_dir=$CheckpointDir})
        if ($complete) { return }
        if (-not $running) { throw "Fine-tune stopped before checkpoint: $RunName" }
        Start-Sleep -Seconds 60
    }
}
while ($true) {
    if (Test-Path $mergeProgress) {
        $merge = Get-Content $mergeProgress -Raw | ConvertFrom-Json
        if ($merge.status -eq "failed") { throw "Zero-shot stage failed; fine-tune not started" }
        if ($merge.status -eq "completed") { break }
    }
    Write-State ([ordered]@{status="waiting_for_zero_shot_merge";observed_at=(Get-Date).ToString("o");dependency=$mergeProgress})
    Start-Sleep -Seconds 60
}
$smokeName = "finetune_base_plus_ard100_fullframe_smoke_restart2"
$smokeRoot = "U:\URAP_runs\samurai\finetune_base_plus_ard100_fullframe_smoke_restart2"
$smokeCheckpointDir = Join-Path $repoRoot "artifacts\samurai_checkpoints\finetune_base_plus_ard100_fullframe_smoke_restart2"
& $startScript -Config "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_fullframe_smoke_restart2.yaml" -RunName $smokeName -TotalEpochs 1 -RunRoot $smokeRoot
Wait-Run $smokeName $smokeRoot 1 $smokeCheckpointDir
$fullName = "finetune_base_plus_ard100_fullframe_stage1_restart1"
$fullRoot = "U:\URAP_runs\samurai\finetune_base_plus_ard100_fullframe_stage1_restart1"
$fullCheckpointDir = Join-Path $repoRoot "artifacts\samurai_checkpoints\finetune_base_plus_ard100_fullframe_stage1_restart1"
& $startScript -Config "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_fullframe_stage1_restart1.yaml" -RunName $fullName -TotalEpochs 4 -RunRoot $fullRoot
Wait-Run $fullName $fullRoot 4 $fullCheckpointDir
Write-State ([ordered]@{status="completed";observed_at=(Get-Date).ToString("o");checkpoint=(Join-Path $fullCheckpointDir "checkpoint.pt")})
