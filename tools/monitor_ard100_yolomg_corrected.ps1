$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_yolomg_corrected_v2'
$meta = Get-Content (Join-Path $runDir 'meta.json') -Raw | ConvertFrom-Json
$allProcesses = @(Get-CimInstance Win32_Process)
$ids = New-Object System.Collections.Generic.HashSet[int]
[void]$ids.Add([int]$meta.pid)
do {
    $before = $ids.Count
    foreach ($process in $allProcesses) {
        if ($ids.Contains([int]$process.ParentProcessId)) {
            [void]$ids.Add([int]$process.ProcessId)
        }
    }
} while ($ids.Count -gt $before)

$matching = @($allProcesses | Where-Object { $ids.Contains([int]$_.ProcessId) })
$alive = $matching.Count -gt 0
$done = 0
$lastUnit = 'not_started'
$finalMetrics = $null
if (Test-Path $meta.stderr_log) {
    $text = Get-Content $meta.stderr_log -Raw -ErrorAction SilentlyContinue
    $progress = [regex]::Matches($text, '(\d+)/(' + $meta.total_batches + ')')
    if ($progress.Count) {
        $done = [int]$progress[$progress.Count - 1].Groups[1].Value
        $lastUnit = "batch $done/$($meta.total_batches)"
    } else {
        $scanProgress = [regex]::Matches($text, '(\d+) found, 0 missing, 0 empty, 0 corrupt')
        if ($scanProgress.Count) {
            $scanned = [int]$scanProgress[$scanProgress.Count - 1].Groups[1].Value
            $lastUnit = "secondary cache scan $scanned/$($meta.total_images)"
        }
    }
    $metrics = [regex]::Matches($text, 'all\s+71608\s+71616\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+')
    if ($metrics.Count) {
        $done = [int]$meta.total_batches
        $lastUnit = 'evaluation complete'
        $finalMetrics = $metrics[$metrics.Count - 1].Value.Trim()
    }
}

$lastOutput = if (Test-Path $meta.stderr_log) { (Get-Item $meta.stderr_log).LastWriteTime.ToString('o') } else { $null }
$pids = @($matching | Select-Object -ExpandProperty ProcessId -Unique)
$gpuRows = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null)
$gpu = @($gpuRows | Where-Object { $row = $_; $pids | Where-Object { $row -match "^$($_)," } })
[pscustomobject]@{
    status = if ($alive) { 'RUNNING' } else { 'NOT RUNNING' }
    done = $done
    total = [int]$meta.total_batches
    launcher_pid = [int]$meta.pid
    compute_pids = $pids
    start_time = $meta.start_time
    last_output_timestamp = $lastOutput
    last_completed_unit = $lastUnit
    final_metrics = $finalMetrics
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    output_dir = $meta.output_dir
    weights = $meta.weights
    gpu_process_signal = ($gpu -join '; ')
} | ConvertTo-Json -Depth 3
