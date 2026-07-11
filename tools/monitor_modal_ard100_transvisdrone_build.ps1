param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$runnerDir = Join-Path $RepoRoot "artifacts\modal_ard100_transvisdrone_build"
$pidPath = Join-Path $runnerDir "build.pid"
$metaPath = Join-Path $runnerDir "build.meta.txt"
$pidValue = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) { Write-Host "RUNNING=true PID=$pidValue"; Write-Host "PROCESS_COMMAND=$($process.CommandLine)" } else { Write-Host "NOT RUNNING PID=$pidValue" }
if (Test-Path $metaPath) { Get-Content $metaPath }
$logs = Get-ChildItem (Join-Path $runnerDir "logs") -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
$stdout = $logs | Where-Object Name -Like '*.out.txt' | Select-Object -First 1
$stderr = $logs | Where-Object Name -Like '*.err.txt' | Select-Object -First 1
$progress = if ($stdout) { @(Get-Content -LiteralPath $stdout.FullName | Select-String '"split":') } else { @() }
$last = $progress | Select-Object -Last 1
if ($last) { Write-Host "latest progress: $($last.Line.Trim())" } else { Write-Host "done/total=0/199104" }
if ($stdout) { Write-Host "stdout log: $($stdout.FullName) updated=$($stdout.LastWriteTime.ToString('o'))"; Get-Content $stdout.FullName -Tail 15 }
if ($stderr) { Write-Host "stderr log: $($stderr.FullName) updated=$($stderr.LastWriteTime.ToString('o'))"; Get-Content $stderr.FullName -Tail 15 }
