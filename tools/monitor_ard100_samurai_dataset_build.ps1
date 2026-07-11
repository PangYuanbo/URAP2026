$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$jobs = foreach ($split in "train", "val", "test") {
    $name = "ard100_samurai_dataset_${split}_v1"
    $metaPath = Join-Path $controlRoot "$name.meta.json"
    if (-not (Test-Path $metaPath)) { [ordered]@{name=$name;status="NOT STARTED";done_total="0/?"}; continue }
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
    $running = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*build_ard100_samurai_dataset.py*"
    $progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
    $latest = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.progress_file -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    [ordered]@{name=$name;status=if($running){"RUNNING"}elseif($progress.status -eq "completed"){"COMPLETE"}else{"NOT RUNNING"};done_total=if($progress){"$($progress.done_sequences)/$($progress.total_sequences)"}else{"0/?"};done_frames=$progress.done_frames;pid=$meta.pid;start_time=$meta.started_at;last_completed_unit=$progress.last_completed_sequence;last_output_timestamp=if($latest){$latest.LastWriteTime.ToString("o")}else{$null};stdout_log=$meta.stdout_log;stderr_log=$meta.stderr_log}
}
$jobs | ConvertTo-Json -Depth 4
