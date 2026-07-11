param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_urap_upload",
    [int]$TailLines = 5
)

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$PidDir = Join-Path $RunnerDir "pids"
$CompletedDir = Join-Path $RunnerDir "completed"
$completedGlobal = @(Get-ChildItem $CompletedDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
Write-Host "global done/total: $completedGlobal/44"
$runningCount = 0
foreach ($metaFile in @(Get-ChildItem $PidDir -Filter "*.meta.txt" -File -ErrorAction SilentlyContinue | Sort-Object Name)) {
    $metaPath = $metaFile.FullName
    $meta = @{}
    foreach ($line in Get-Content $metaPath) { if ($line -match "^([^=]+)=(.*)$") { $meta[$Matches[1]] = $Matches[2] } }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
    $state = if ($meta.progress -and (Test-Path $meta.progress)) { Get-Content $meta.progress -Raw | ConvertFrom-Json } else { $null }
    if ($process) { $runningCount++; $runState = "RUNNING" } else { $runState = "NOT RUNNING" }
    $doneText = if ($state) { "$($state.done)/$($state.total)" } else { "0/?" }
    $jobText = if ($state) { $state.current_job } else { "none" }
    $updatedText = if ($state) { $state.updated } else { "none" }
    Write-Host "[$runState] volume=$($meta.volume) PID=$($meta.pid) done/total=$doneText current=$jobText updated=$updatedText"
    if (-not $process -and $meta.stderr -and (Test-Path $meta.stderr) -and (Get-Item $meta.stderr).Length) {
        Write-Host "  stderr=$($meta.stderr)"
        Get-Content $meta.stderr -Tail $TailLines | ForEach-Object { Write-Host "  $_" }
    }
}
Write-Host "active workers: $runningCount"
