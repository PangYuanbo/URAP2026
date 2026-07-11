param([string]$RunName = "finetune_base_plus_ard100_short2_frozen_b4_stage1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$meta = Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.meta.json") -Raw | ConvertFrom-Json
$pidValue = [int](Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.pid") -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*run_sam2_train_with_memory_cap.py*ARD100_short2_frozen_b4_local_stage1*"
$stdoutItem = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$stderrItem = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
$lines = @()
if ($stdoutItem) { $lines += Get-Content -LiteralPath $meta.stdout_log -Tail 300 }
if ($stderrItem) { $lines += Get-Content -LiteralPath $meta.stderr_log -Tail 120 }
$matchesProgress = @($lines | Select-String -Pattern 'Train Epoch: \[(\d+)\]\[\s*(\d+)/(\s*\d+)\]')
$latest = if ($matchesProgress.Count) { $matchesProgress[-1].Matches[0] } else { $null }
$phase = if ($latest) { [int]$latest.Groups[1].Value } else { 0 }
$step = if ($latest) { [int]$latest.Groups[2].Value.Trim() } else { 0 }
$stepsTotal = if ($latest) { [int]$latest.Groups[3].Value.Trim() } else { 0 }
$checkpointDir = Join-Path $meta.run_root "checkpoints"
$checkpoint = Join-Path $checkpointDir "checkpoint.pt"
$checkpointEpoch = 0
if (Test-Path -LiteralPath $checkpoint) {
    $python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
    $checkpointEpoch = [int](& $python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch',0)))" $checkpoint)
}
$completed = $checkpointEpoch -ge [int]$meta.total_phases
$lastOutput = @($stdoutItem, $stderrItem) | Where-Object { $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null
$recent = @($lines | Select-String -Pattern 'Train Epoch:|OutOfMemoryError|CUDA out of memory|Traceback' | ForEach-Object { $_.Line } | Select-Object -Last 12)

[ordered]@{
    status = if ($matches) { "RUNNING" } elseif ($completed) { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = "$checkpointEpoch/$($meta.total_phases) completed phases; $step/$stepsTotal current-phase batches"
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$matches
    last_completed_unit = if ($checkpointEpoch -gt 0) { "checkpoint_epoch=$checkpointEpoch" } elseif ($latest) { "phase=$phase batch=$step" } else { "initializing" }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    total_windows = $meta.total_windows
    expected_batches = $meta.expected_batches
    memory_fraction = $meta.memory_fraction
    gpu_utilization_used_free_mib = $gpu
    checkpoint = if (Test-Path -LiteralPath $checkpoint) { $checkpoint } else { $null }
    recent_progress = $recent
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 5
