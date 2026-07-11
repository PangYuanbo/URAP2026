$ErrorActionPreference = 'Stop'

function Read-JobStatus {
    param([string]$Name, [string]$RunDir, [int]$DefaultTotal)
    $metaPath = Join-Path $RunDir 'meta.json'
    $progressPath = Join-Path $RunDir 'progress.json'
    if (-not (Test-Path $metaPath)) {
        return [pscustomobject]@{name=$Name;status='NOT STARTED';done=0;total=$DefaultTotal}
    }
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    $progress = if (Test-Path $progressPath) { Get-Content $progressPath -Raw | ConvertFrom-Json } else { $null }
    $all = @(Get-CimInstance Win32_Process)
    $ids = New-Object System.Collections.Generic.HashSet[int]
    [void]$ids.Add([int]$meta.pid)
    do {
        $before = $ids.Count
        foreach ($process in $all) { if ($ids.Contains([int]$process.ParentProcessId)) { [void]$ids.Add([int]$process.ProcessId) } }
    } while ($ids.Count -gt $before)
    $matching = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
    $logPath = if ($meta.stdout_log) { [string]$meta.stdout_log } elseif ($meta.stderr_log) { [string]$meta.stderr_log } else { $null }
    return [pscustomobject]@{
        name = $Name
        status = if ($matching.Count) { 'RUNNING' } else { 'NOT RUNNING' }
        done = if ($progress) { [int]$progress.done } else { 0 }
        total = if ($progress) { [int]$progress.total } else { $DefaultTotal }
        launcher_pid = [int]$meta.pid
        compute_pids = @($matching | Select-Object -ExpandProperty ProcessId -Unique)
        start_time = $meta.start_time
        last_output_timestamp = if ($logPath -and (Test-Path $logPath)) { (Get-Item $logPath).LastWriteTime.ToString('o') } else { $null }
        last_completed_unit = if ($progress) { $progress.stage } else { 'not_started' }
        stdout_log = $meta.stdout_log
        stderr_log = $meta.stderr_log
    }
}

$root = 'C:\Users\aaron\Desktop\URAP\artifacts'
$output = 'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2'
$jobs = @(
    Read-JobStatus 'frozen_action_bank_test' (Join-Path $root 'detached_ard100_yolomg_action_bank_v2') 6
    Read-JobStatus 'val_selected_action_bank' (Join-Path $root 'detached_ard100_val_selection_pipeline_v2') 3
    Read-JobStatus 'vatd_test' (Join-Path $root 'detached_ard100_yolomg_vatd_v2') 6
)
$summaries = @{}
foreach ($name in @('detector_baseline.json','action_bank_summary.json','action_bank_val_selected_summary.json','vatd_summary.json','official_adapted_comparison.json', 'official_comparison.json')) {
    $path = Join-Path $output $name
    if (Test-Path $path) { $summaries[$name] = Get-Content $path -Raw | ConvertFrom-Json }
}
[pscustomobject]@{
    jobs = $jobs
    summaries = $summaries
    gpu = (& nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>$null) -join '; '
} | ConvertTo-Json -Depth 12

