param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [ValidateSet('run', 'smoke')][string]$Mode = 'run',
    [int]$IntervalSeconds = 5
)

$ErrorActionPreference = 'Stop'
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$configuration = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$metadataPath = Join-Path $configuration.run_dir ($Mode + '.json')
if (-not (Test-Path -LiteralPath $metadataPath)) { Write-Output 'NOT RUNNING: no launch metadata'; return }
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json

function Read-JsonRetry([string]$Path) {
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json) }
        catch {
            if ($attempt -eq 19) { throw }
            Start-Sleep -Milliseconds (10 * ($attempt + 1))
        }
    }
}

function Read-Snapshot {
    $state = if (Test-Path -LiteralPath $metadata.progress) { Read-JsonRetry $metadata.progress } else { $null }
    $active = @()
    if ($Mode -eq 'smoke' -and $state.worker_progress_path -and (Test-Path -LiteralPath $state.worker_progress_path)) {
        $active += Read-JsonRetry $state.worker_progress_path
    }
    foreach ($worker in $state.active) {
        $progressPath = Join-Path $configuration.output_root ($worker.sequence + '/progress.json')
        if (Test-Path -LiteralPath $progressPath) { $active += Read-JsonRetry $progressPath }
    }
    $counter = [ordered]@{ completed = $state.done; completed_frames = $state.completed_frames; units = @($active | ForEach-Object { "$($_.sequence):$($_.phase):$($_.done)" }) }
    return @{ state = $state; active = $active; signature = ($counter | ConvertTo-Json -Compress) }
}

$first = Read-Snapshot
Start-Sleep -Seconds ([Math]::Max(1, $IntervalSeconds))
$second = Read-Snapshot
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($metadata.pid)" -ErrorAction SilentlyContinue
$matched = $process -and $process.CommandLine -and $process.CommandLine.Contains('dataset_videos.py') -and $process.CommandLine.Contains($manifestPath)
$advanced = $first.signature -ne $second.signature
$status = if (-not $matched) { 'NOT RUNNING (coordinator)' } elseif ($advanced) { 'PID MATCHED; PROGRESS ADVANCED' } else { 'PID MATCHED; NO PROGRESS ADVANCE OBSERVED' }
Write-Output "observed=$(Get-Date -Format o) status=$status"
Write-Output "done=$($second.state.done)/$($second.state.total) phase=$($second.state.phase) completed_frames=$($second.state.completed_frames)"
Write-Output "pid=$($metadata.pid) start=$($metadata.started_at) command=$($process.CommandLine)"
Write-Output "last_completed=$($second.state.last_completed) last_completed_at=$($second.state.last_completed_at)"
foreach ($worker in $second.active) {
    $workerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.pid)" -ErrorAction SilentlyContinue
    Write-Output "worker_pid=$($worker.pid) sequence=$($worker.sequence) done=$($worker.done)/$($worker.total) unit=$($worker.unit) last_output=$($worker.updated_at) fps=$($worker.processing_fps)"
    Write-Output "worker_command=$($workerProcess.CommandLine)"
    foreach ($encoderPid in $worker.encoder_pids) {
        $encoder = Get-CimInstance Win32_Process -Filter "ProcessId=$encoderPid" -ErrorAction SilentlyContinue
        Write-Output "encoder_pid=$encoderPid command=$($encoder.CommandLine)"
    }
}
$failureCount = @($second.state.failures | Where-Object { $null -ne $_ }).Count
Write-Output "failures=$failureCount stdout=$($metadata.stdout) stderr=$($metadata.stderr)"
if (Test-Path -LiteralPath $metadata.stdout) { Get-Content -LiteralPath $metadata.stdout -Tail 4 }
if (Test-Path -LiteralPath $metadata.stderr) { Get-Content -LiteralPath $metadata.stderr -Tail 4 }
& nvidia-smi --query-gpu=name,utilization.gpu,utilization.encoder,memory.used,memory.total --format=csv
