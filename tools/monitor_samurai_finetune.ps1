param([string]$RunName = "finetune_base_plus_nps_weakmask_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
if (-not (Test-Path $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$runRoot = if ($meta.run_root) { $meta.run_root } else { "U:\URAP_runs\samurai\$RunName" }
$pidValue = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*training*train.py*"
$logFiles = Get-ChildItem $meta.stdout_log, $meta.stderr_log -ErrorAction SilentlyContinue
$lastOutput = $logFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lines = @()
foreach ($file in $logFiles) { $lines += Get-Content $file.FullName -Tail 200 -ErrorAction SilentlyContinue }
$matches = @($lines | Select-String -Pattern 'Train Epoch: \[(\d+)\]\[(\s*\d+)/(\s*\d+)\]')
$latest = if ($matches.Count) { $matches[-1].Matches[0] } else { $null }
$epoch = if ($latest) { [int]$latest.Groups[1].Value } else { 0 }
$step = if ($latest) { [int]$latest.Groups[2].Value.Trim() } else { 0 }
$stepsTotal = if ($latest) { [int]$latest.Groups[3].Value.Trim() } else { 0 }
$startedAt = [datetimeoffset]::Parse($meta.started_at)
$epochCheckpoints = @(
    Get-ChildItem (Join-Path $runRoot "checkpoints\checkpoint_*.pt") -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTimeUtc -gt $startedAt.UtcDateTime }
)
$completedEpoch = 0
foreach ($checkpoint in $epochCheckpoints) {
    if ($checkpoint.BaseName -match '^checkpoint_(\d+)$') {
        $completedEpoch = [math]::Max($completedEpoch, [int]$Matches[1])
    }
}
$completed = $completedEpoch -ge [int]$meta.total_epochs
$reportedEpoch = [math]::Max($epoch, $completedEpoch)
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } elseif ($completed) { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = "$reportedEpoch/$($meta.total_epochs) epochs; $step/$stepsTotal current-epoch steps"
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$commandMatches
    last_completed_unit = if ($completedEpoch -gt 0) { "checkpoint_epoch=$completedEpoch" } elseif ($latest) { "epoch=$epoch step=$step" } else { "initializing" }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    gpu_utilization_memory_mib = $gpu
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    checkpoint_dir = (Join-Path $runRoot "checkpoints")
} | ConvertTo-Json -Depth 4
