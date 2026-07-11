param(
    [string]$ProgressPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_runs\ard100_post_finetune_restart1_pipeline.progress.json"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$startEval = Join-Path $repoRoot "tools\start_samurai_nps_eval_detached.ps1"
$mergeEval = Join-Path $repoRoot "tools\merge_samurai_eval_shards.py"
$mergeFeatures = Join-Path $repoRoot "tools\merge_samurai_feature_chunks.py"
$startBbox = Join-Path $repoRoot "tools\start_samurai_bbox_readout_detached.ps1"
$evalBbox = Join-Path $repoRoot "tools\eval_samurai_bbox_readout.py"
$summarize = Join-Path $repoRoot "tools\summarize_samurai_ard100_ablation.py"
$finetuneProgress = Join-Path $repoRoot "artifacts\samurai_runs\ard100_finetune_restart1_sequencer.progress.json"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
function Write-State($payload) {
    $tmp = "$ProgressPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $ProgressPath -Force
}
function Wait-Evals($names) {
    while ($true) {
        $rows = @(); $all = $true
        foreach ($name in $names) {
            $meta = Get-Content (Join-Path $controlRoot "$name.meta.json") -Raw | ConvertFrom-Json
            $progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
            $running = $proc -and $proc.Name -eq "python.exe" -and $proc.CommandLine -like "*eval_samurai_nps.py*"
            $complete = $progress -and $progress.status -eq "completed" -and (Test-Path (Join-Path $meta.run_root "metrics.json"))
            if (-not $complete) { $all = $false }
            if (-not $running -and -not $complete) { Write-State ([ordered]@{status="failed_eval";failed_job=$name;observed_at=(Get-Date).ToString("o");jobs=$rows}); throw "Evaluation stopped: $name" }
            $rows += [ordered]@{name=$name;pid=$meta.pid;running=[bool]$running;complete=[bool]$complete;done_total=if($progress){"$($progress.done_sequences)/$($progress.total_sequences)"}else{"0/?"};done_frames=if($progress){$progress.done_frames}else{0}}
        }
        Write-State ([ordered]@{status=if($all){"merging_finetuned_outputs"}else{"running_finetuned_evals"};observed_at=(Get-Date).ToString("o");jobs=$rows})
        if ($all) { return }
        Start-Sleep -Seconds 60
    }
}
while ($true) {
    if (Test-Path $finetuneProgress) {
        $state = Get-Content $finetuneProgress -Raw | ConvertFrom-Json
        if ($state.status -like "failed*") { throw "Fine-tune stage failed" }
        if ($state.status -eq "completed") { break }
    }
    Write-State ([ordered]@{status="waiting_for_finetune";observed_at=(Get-Date).ToString("o");dependency=$finetuneProgress})
    Start-Sleep -Seconds 60
}
$checkpoint = Join-Path $repoRoot "artifacts\samurai_checkpoints\finetune_base_plus_ard100_fullframe_stage1_restart1\checkpoint.pt"
if (-not (Test-Path $checkpoint)) { throw "Missing fine-tuned checkpoint: $checkpoint" }
$jobs = @()
for ($i=0; $i -lt 3; $i++) {
    $name="ard100_ablation_sam2_video_finetuned_ard100_restart1_test_v1_shard$i"; $root="U:\URAP_runs\samurai\$name"
    & $startEval -DatasetRoot "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI\test_v1" -Split test -Checkpoint $checkpoint -ModelConfig "configs/sam2.1/sam2.1_hiera_b+.yaml" -RunRoot $root -PropagationMode video -Resume -AsyncLoadingFrames -OffloadStateToCpu -SequenceShardCount 3 -SequenceShardIndex $i -ControlName $name
    $jobs += $name
    $name="ard100_ablation_samurai_finetuned_ard100_restart1_test_v1_shard$i"; $root="U:\URAP_runs\samurai\$name"
    & $startEval -DatasetRoot "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI\test_v1" -Split test -Checkpoint $checkpoint -ModelConfig "configs/samurai/sam2.1_hiera_b+.yaml" -RunRoot $root -PropagationMode video -FeatureOutput "$root\features.npz" -Resume -AsyncLoadingFrames -OffloadStateToCpu -SequenceShardCount 3 -SequenceShardIndex $i -ControlName $name
    $jobs += $name
    $name="ard100_ablation_feature_train_finetuned_ard100_restart1_shard$i"; $root="U:\URAP_runs\samurai\$name"
    & $startEval -DatasetRoot "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI\train_v1" -Split train -Checkpoint $checkpoint -ModelConfig "configs/samurai/sam2.1_hiera_b+.yaml" -RunRoot $root -PropagationMode video -FeatureOutput "$root\features.npz" -Resume -AsyncLoadingFrames -OffloadStateToCpu -SequenceShardCount 3 -SequenceShardIndex $i -ControlName $name
    $jobs += $name
}
Wait-Evals $jobs
foreach ($spec in @(
    @{prefix="ard100_ablation_sam2_video_finetuned_ard100_restart1_test_v1_shard";out="U:\URAP_runs\samurai\ard100_ablation_sam2_video_finetuned_ard100_restart1_test_v1"},
    @{prefix="ard100_ablation_samurai_finetuned_ard100_restart1_test_v1_shard";out="U:\URAP_runs\samurai\ard100_ablation_samurai_finetuned_ard100_restart1_test_v1"}
)) {
    $args=@($mergeEval)
    for($i=0;$i -lt 3;$i++){$args+=@("--shard-root","U:\URAP_runs\samurai\$($spec.prefix)$i")}
    $args+=@("--output-root",$spec.out,"--expected-sequences","35"); & $python @args
    if($LASTEXITCODE -ne 0){throw "Evaluation merge failed: $($spec.out)"}
}
$trainFeatures="U:\URAP_runs\samurai\ard100_ablation_feature_train_finetuned_ard100_restart1\features.npz"
$testFeatures="U:\URAP_runs\samurai\ard100_ablation_feature_test_finetuned_ard100_restart1\features.npz"
$args=@($mergeFeatures);for($i=0;$i -lt 3;$i++){$args+=@("--chunk-root","U:\URAP_runs\samurai\ard100_ablation_feature_train_finetuned_ard100_restart1_shard$i")};$args+=@("--output",$trainFeatures,"--expected-sequences","55");& $python @args;if($LASTEXITCODE -ne 0){throw "Train feature merge failed"}
$args=@($mergeFeatures);for($i=0;$i -lt 3;$i++){$args+=@("--chunk-root","U:\URAP_runs\samurai\ard100_ablation_samurai_finetuned_ard100_restart1_test_v1_shard$i")};$args+=@("--output",$testFeatures,"--expected-sequences","35");& $python @args;if($LASTEXITCODE -ne 0){throw "Test feature merge failed"}
$bboxName="ard100_ablation_bbox_readout_finetuned_ard100_restart1";$bboxCheckpoint="U:\URAP_models\samurai\bbox_readout_ard100_restart1.pt"
& $startBbox -Features $trainFeatures -Checkpoint $bboxCheckpoint -RunName $bboxName -Epochs 80 -BatchSize 2048 -Device cuda:0
$bboxMetaPath=Join-Path $repoRoot "artifacts\samurai_ablation\$bboxName.meta.json"
while($true){$meta=Get-Content $bboxMetaPath -Raw|ConvertFrom-Json;$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue;$running=$proc -and $proc.Name -eq "python.exe" -and $proc.CommandLine -like "*train_samurai_bbox_readout.py*";$complete=Test-Path $bboxCheckpoint;Write-State ([ordered]@{status=if($running){"training_bbox_readout"}elseif($complete){"evaluating_bbox_readout"}else{"failed_bbox_readout"};pid=$meta.pid;checkpoint=$bboxCheckpoint;observed_at=(Get-Date).ToString("o")});if($complete){break};if(-not $running){throw "BBox readout stopped"};Start-Sleep -Seconds 30}
$bboxMetrics=Join-Path $repoRoot "artifacts\samurai_ard100_ablation\bbox_readout_metrics_restart1.json";New-Item -ItemType Directory -Force (Split-Path $bboxMetrics)|Out-Null
& $python $evalBbox --features $testFeatures --checkpoint $bboxCheckpoint --output $bboxMetrics --device cuda:0
if($LASTEXITCODE -ne 0){throw "BBox evaluation failed"}
$summaryRoot=Join-Path $repoRoot "artifacts\samurai_ard100_ablation";New-Item -ItemType Directory -Force $summaryRoot|Out-Null
& $python $summarize --bbox-metrics $bboxMetrics --output-json (Join-Path $summaryRoot "summary_restart1.json") --output-md (Join-Path $repoRoot "doc\samurai_ard100_ablation_generated_restart1_2026_06_26.md")
if($LASTEXITCODE -ne 0){throw "Summary failed"}
Write-State ([ordered]@{status="completed";observed_at=(Get-Date).ToString("o");summary=(Join-Path $summaryRoot "summary_restart1.json");report=(Join-Path $repoRoot "doc\samurai_ard100_ablation_generated_restart1_2026_06_26.md")})
