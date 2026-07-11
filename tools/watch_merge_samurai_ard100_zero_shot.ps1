param(
    [string]$ProgressPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_runs\ard100_zero_shot_merge_watcher.progress.json"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$merge = Join-Path $repoRoot "tools\merge_samurai_eval_shards.py"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
function Write-ProgressJson($payload) {
    $tmp = "$ProgressPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $ProgressPath -Force
}
$modes = @(
    @{ key = "image_box"; canonical = "ard100_ablation_image_box_zero_shot_test_v1" },
    @{ key = "sam2_video"; canonical = "ard100_ablation_sam2_video_zero_shot_test_v1" },
    @{ key = "samurai"; canonical = "ard100_ablation_samurai_zero_shot_test_v1" }
)
while ($true) {
    $jobs = @()
    $allComplete = $true
    foreach ($mode in $modes) {
        for ($index = 0; $index -lt 3; $index++) {
            $name = "ard100_ablation_$($mode.key)_zero_shot_test_v1_shard$index"
            $metaPath = Join-Path $controlRoot "$name.meta.json"
            if (-not (Test-Path $metaPath)) { throw "Missing shard metadata: $metaPath" }
            $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
            $progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
            $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
            $running = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*eval_samurai_nps.py*"
            $complete = $progress -and $progress.status -eq "completed" -and (Test-Path (Join-Path $meta.run_root "metrics.json"))
            if (-not $complete) { $allComplete = $false }
            if (-not $running -and -not $complete) {
                $payload = [ordered]@{ status = "failed"; failed_job = $name; observed_at = (Get-Date).ToString("o"); jobs = $jobs }
                Write-ProgressJson $payload
                throw "Shard stopped before completion: $name"
            }
            $jobs += [ordered]@{ name=$name; running=[bool]$running; complete=[bool]$complete; done_total=if($progress){"$($progress.done_sequences)/$($progress.total_sequences)"}else{"0/?"}; done_frames=if($progress){$progress.done_frames}else{0}; pid=$meta.pid; progress_file=$meta.progress_file }
        }
    }
    Write-ProgressJson ([ordered]@{status=if($allComplete){"merging"}else{"waiting"};observed_at=(Get-Date).ToString("o");jobs=$jobs})
    if ($allComplete) { break }
    Start-Sleep -Seconds 60
}
$merged = @()
foreach ($mode in $modes) {
    $args = @($merge)
    for ($index = 0; $index -lt 3; $index++) {
        $args += @("--shard-root", "U:\URAP_runs\samurai\ard100_ablation_$($mode.key)_zero_shot_test_v1_shard$index")
    }
    $output = "U:\URAP_runs\samurai\$($mode.canonical)"
    $args += @("--output-root", $output, "--expected-sequences", "35")
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Merge failed for $($mode.key)" }
    $merged += $output
}
Write-ProgressJson ([ordered]@{status="completed";observed_at=(Get-Date).ToString("o");merged_outputs=$merged})
