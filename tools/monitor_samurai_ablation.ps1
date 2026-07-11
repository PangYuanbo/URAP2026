$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$names = @(
    "ablation_sam2_video_zero_shot_test_v1",
    "ablation_sam2_video_finetuned1_test_v1",
    "ablation_image_box_zero_shot_test_v1",
    "ablation_feature_train_finetuned1",
    "ablation_feature_train_finetuned1_shard0",
    "ablation_feature_train_finetuned1_shard1",
    "ablation_feature_train_finetuned1_shard2",
    "ablation_feature_test_finetuned1"
)
$jobs = foreach ($name in $names) {
    $metaPath = Join-Path $controlRoot "$name.meta.json"
    if (-not (Test-Path $metaPath)) {
        [ordered]@{ name = $name; status = "NOT STARTED"; done_total = "0/?" }
        continue
    }
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
    $processMatches = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*eval_samurai_nps.py*"
    $progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
    $lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.progress_file -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    [ordered]@{
        name = $name
        status = if ($processMatches) { "RUNNING" } elseif ($progress.status -eq "completed") { "COMPLETE" } else { "NOT RUNNING" }
        done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } else { "0/?" }
        pid = $meta.pid
        start_time = $meta.started_at
        last_completed_unit = $progress.last_completed_sequence
        done_frames = $progress.done_frames
        last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
        stdout_log = $meta.stdout_log
        stderr_log = $meta.stderr_log
    }
}
[ordered]@{
    jobs = $jobs
    gpu_utilization_memory_mib = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null)
} | ConvertTo-Json -Depth 5
