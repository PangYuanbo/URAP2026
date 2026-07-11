$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$latestFile = Join-Path $repo 'artifacts\runs\yolomg_pure_1080p_new_batch_latest.json'
if (-not (Test-Path -LiteralPath $latestFile)) {
    throw "No batch run pointer found: $latestFile"
}

$latest = Get-Content -LiteralPath $latestFile -Raw | ConvertFrom-Json
$processId = [int]$latest.coordinator_pid
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
$status = $null
if (Test-Path -LiteralPath $latest.status_json) {
    try {
        $status = Get-Content -LiteralPath $latest.status_json -Raw | ConvertFrom-Json
    } catch {
        Write-Warning "Status JSON is temporarily unreadable: $($_.Exception.Message)"
    }
}

Write-Output "Coordinator: $(if ($process) { 'RUNNING' } else { 'NOT RUNNING' })"
Write-Output "PID: $processId"
Write-Output "Start time: $($latest.started_at)"
if ($process) {
    Write-Output "Command: $($process.CommandLine)"
}
if ($status) {
    Write-Output "Batch: $($status.done)/$($status.total) complete; running=$($status.running); pending=$($status.pending); failed=$($status.failed)"
    Write-Output "Last status update: $($status.updated_at)"
    foreach ($item in $status.items) {
        $lastOutput = if ($item.last_output_timestamp) { [DateTimeOffset]::FromUnixTimeSeconds([long]$item.last_output_timestamp).ToLocalTime().ToString('o') } else { '-' }
        Write-Output ("{0} state={1} progress={2}/{3} PID={4} last_output={5}" -f (Split-Path $item.input -Leaf), $item.state, $item.done, $item.total, $item.pid, $lastOutput)
    }
}
Write-Output "Coordinator logs: $($latest.stdout_log) ; $($latest.stderr_log)"
Write-Output "Output directory: $($latest.output_dir)"
if (Test-Path -LiteralPath $latest.stdout_log) {
    Write-Output '--- coordinator log tail ---'
    Get-Content -LiteralPath $latest.stdout_log -Tail 12
}
if (Test-Path -LiteralPath $latest.stderr_log) {
    $stderrTail = Get-Content -LiteralPath $latest.stderr_log -Tail 12
    if ($stderrTail) {
        Write-Output '--- coordinator stderr tail ---'
        $stderrTail
    }
}
