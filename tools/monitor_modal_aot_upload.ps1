param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$runnerDir = Join-Path $RepoRoot "artifacts\modal_aot_upload"
$pidPath = Join-Path $runnerDir "upload.pid"
$metaPath = Join-Path $runnerDir "upload.meta.txt"
$progressPath = Join-Path $runnerDir "progress.json"
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
    Write-Host "done/total=$($progress.done)/$($progress.total) status=$($progress.status) current=$($progress.current) updated=$($progress.updated)"
} else {
    Write-Host "done/total=0/174 status=not_started"
}
if (Test-Path $metaPath) { Get-Content $metaPath }
$latest = Get-ChildItem (Join-Path $runnerDir "logs") -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 2
if ($latest) {
    $latest | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
    foreach ($log in $latest) { Get-Content -LiteralPath $log.FullName -Tail 8 -ErrorAction SilentlyContinue }
}
