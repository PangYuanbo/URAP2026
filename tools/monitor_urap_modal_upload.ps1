param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_urap_upload",
    [int]$TailLines = 15
)

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$pidPath = Join-Path $RunnerDir "modal_upload.pid"
$metaPath = Join-Path $RunnerDir "modal_upload.meta.txt"
$progressPath = Join-Path $RunnerDir "progress.json"
$pidValue = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) {
    Write-Host "RUNNING=true PID=$pidValue"
    Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
    Write-Host "NOT RUNNING PID=$pidValue"
}
if (Test-Path $progressPath) {
    $progress = Get-Content $progressPath -Raw | ConvertFrom-Json
    Write-Host "done/total: $($progress.done)/$($progress.total)"
    Write-Host "status: $($progress.status)"
    Write-Host "current job: $($progress.current_job)"
    Write-Host "volume: $($progress.volume)"
    Write-Host "local: $($progress.local)"
    Write-Host "remote: $($progress.remote)"
    Write-Host "last output timestamp: $($progress.updated)"
    Write-Host "message: $($progress.message)"
} else {
    Write-Host "done/total: 0/?"
    Write-Host "progress file: MISSING"
}
if (Test-Path $metaPath) {
    $meta = @{}
    foreach ($line in Get-Content $metaPath) {
        if ($line -match "^([^=]+)=(.*)$") { $meta[$Matches[1]] = $Matches[2] }
    }
    Write-Host "start time: $($meta.started)"
    Write-Host "stdout log: $($meta.stdout)"
    Write-Host "stderr log: $($meta.stderr)"
    if ($meta.stdout -and (Test-Path $meta.stdout)) { Write-Host "== stdout tail =="; Get-Content $meta.stdout -Tail $TailLines }
    if ($meta.stderr -and (Test-Path $meta.stderr)) { Write-Host "== stderr tail =="; Get-Content $meta.stderr -Tail $TailLines }
}
