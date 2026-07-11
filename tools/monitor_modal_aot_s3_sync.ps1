param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$runnerDir = Join-Path $RepoRoot "artifacts\modal_aot_s3_sync"
$pidPath = Join-Path $runnerDir "sync.pid"
$metaPath = Join-Path $runnerDir "sync.meta.txt"
$pidValue = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) { Write-Host "RUNNING=true PID=$pidValue"; Write-Host "PROCESS_COMMAND=$($process.CommandLine)" } else { Write-Host "NOT RUNNING PID=$pidValue" }
if (Test-Path $metaPath) { Get-Content $metaPath }
$logs = Get-ChildItem (Join-Path $runnerDir "logs") -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
$stdout = $logs | Where-Object Name -Like '*.out.txt' | Select-Object -First 1
$stderr = $logs | Where-Object Name -Like '*.err.txt' | Select-Object -First 1
$progressLines = if ($stdout) { @(Get-Content -LiteralPath $stdout.FullName -ErrorAction SilentlyContinue | Select-String '"done":') } else { @() }
$lastProgress = $progressLines | Select-Object -Last 1
if ($lastProgress) { Write-Host "latest progress: $($lastProgress.Line.Trim())" } else { Write-Host "done/total=0/206182" }
if ($stdout) { Write-Host "stdout log: $($stdout.FullName) updated=$($stdout.LastWriteTime.ToString('o'))"; Get-Content -LiteralPath $stdout.FullName -Tail 18 }
if ($stderr) { Write-Host "stderr log: $($stderr.FullName) updated=$($stderr.LastWriteTime.ToString('o'))"; Get-Content -LiteralPath $stderr.FullName -Tail 18 }
