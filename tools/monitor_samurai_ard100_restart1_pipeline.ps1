param()

$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$jobs = @(
    "ard100_finetune_restart1_sequencer",
    "ard100_post_finetune_restart1_pipeline",
    "finetune_base_plus_ard100_fullframe_smoke_restart2",
    "finetune_base_plus_ard100_fullframe_stage1_restart1"
)

function Get-ProcessState($Meta, [string]$Needle) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($Meta.pid)" -ErrorAction SilentlyContinue
    return [bool]($proc -and $proc.CommandLine -like "*$Needle*")
}

$rows = foreach ($name in $jobs) {
    $metaPath = Join-Path $controlRoot "$name.meta.json"
    if (-not (Test-Path -LiteralPath $metaPath)) {
        [pscustomobject]@{ Name=$name; Status="NOT STARTED"; PID=$null; Started=$null; LastOutput=$null; Progress=$null; Log=$null }
        continue
    }
    $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
    $commandNeedle = if ($name -like "*sequencer*" -or $name -like "*pipeline*") { ".ps1" } else { "training\train.py" }
    $running = Get-ProcessState $meta $commandNeedle
    $progress = $null
    if ($meta.progress_file -and (Test-Path -LiteralPath $meta.progress_file)) {
        $progress = Get-Content -LiteralPath $meta.progress_file -Raw | ConvertFrom-Json
    }
    $logItems = @($meta.stdout_log, $meta.stderr_log | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | ForEach-Object { Get-Item -LiteralPath $_ } | Sort-Object LastWriteTime -Descending)
    [pscustomobject]@{
        Name = $name
        Status = if ($running) { "RUNNING" } else { "NOT RUNNING" }
        PID = $meta.pid
        Started = $meta.started_at
        LastOutput = if ($logItems.Count) { $logItems[0].LastWriteTime } else { $null }
        Progress = if ($progress) { $progress.status } else { $null }
        Log = $meta.stderr_log
    }
}
$rows | Format-Table -AutoSize

$trainName = "finetune_base_plus_ard100_fullframe_stage1_restart1"
$trainMetaPath = Join-Path $controlRoot "$trainName.meta.json"
$phaseSizes = @(3352, 3352, 3351, 3351)
$totalSteps = ($phaseSizes | Measure-Object -Sum).Sum
if (Test-Path -LiteralPath $trainMetaPath) {
    $trainMeta = Get-Content -LiteralPath $trainMetaPath -Raw | ConvertFrom-Json
    $trainLog = Join-Path $trainMeta.run_root "logs\log.txt"
    $lastTrain = $null
    if (Test-Path -LiteralPath $trainLog) {
        $lastTrain = Get-Content -LiteralPath $trainLog -Tail 500 | Where-Object { $_ -match 'Train Epoch:s+[(d+)][s*(d+)/(d+)]' } | Select-Object -Last 1
    }
    if ($lastTrain -and $lastTrain -match 'Train Epoch:s+[(d+)][s*(d+)/(d+)]') {
        $phase = [int]$Matches[1]
        $phaseDone = [int]$Matches[2]
        $phaseTotal = [int]$Matches[3]
        $offset = 0
        for ($i = 0; $i -lt [math]::Min($phase, $phaseSizes.Count); $i++) { $offset += $phaseSizes[$i] }
        $globalDone = [math]::Min($totalSteps, $offset + $phaseDone)
        Write-Output "training_done=$globalDone/$totalSteps phase=$phase phase_done=$phaseDone/$phaseTotal last_unit=$lastTrain log=$trainLog"
    } else {
        Write-Output "training_done=0/$totalSteps last_unit=none log=$trainLog"
    }
} else {
    Write-Output "training_done=0/$totalSteps status=NOT STARTED"
}

$checkpointDir = Join-Path $repoRoot "artifacts\samurai_checkpoints\finetune_base_plus_ard100_fullframe_stage1_restart1"
$phaseCheckpoints = @(Get-ChildItem -LiteralPath $checkpointDir -Filter "checkpoint_*.pt" -ErrorAction SilentlyContinue)
$latestCheckpoint = Get-Item -LiteralPath (Join-Path $checkpointDir "checkpoint.pt") -ErrorAction SilentlyContinue
Write-Output "phase_checkpoints=$($phaseCheckpoints.Count)/4 latest_checkpoint=$([bool]$latestCheckpoint) directory=$checkpointDir"
if ($phaseCheckpoints.Count -or $latestCheckpoint) {
    @($phaseCheckpoints + $latestCheckpoint) | Where-Object { $_ } | Select-Object Name, Length, LastWriteTime | Sort-Object Name | Format-Table -AutoSize
}

$postProgressPath = Join-Path $controlRoot "ard100_post_finetune_restart1_pipeline.progress.json"
if (Test-Path -LiteralPath $postProgressPath) {
    $post = Get-Content -LiteralPath $postProgressPath -Raw | ConvertFrom-Json
    if ($post.jobs) {
        $doneSequences = 0
        $totalSequences = 0
        $doneFrames = 0
        foreach ($job in $post.jobs) {
            if ($job.done_total -match '^(d+)/(d+)$') {
                $doneSequences += [int]$Matches[1]
                $totalSequences += [int]$Matches[2]
            }
            $doneFrames += [int]$job.done_frames
        }
        Write-Output "post_status=$($post.status) shard_sequences=$doneSequences/$totalSequences done_frames=$doneFrames progress=$postProgressPath"
    } else {
        Write-Output "post_status=$($post.status) progress=$postProgressPath"
    }
}

$preflight = Join-Path $controlRoot "ard100_restart_preflight.json"
if (Test-Path -LiteralPath $preflight) {
    $preflightState = Get-Content -LiteralPath $preflight -Raw | ConvertFrom-Json
    Write-Output "preflight=$($preflightState.status) observed_at=$($preflightState.observed_at) report=$preflight"
}
& nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader
