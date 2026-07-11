$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\aaron\Desktop\URAP'
$run = Join-Path $repo 'artifacts\detached_action_chunk_conditional_router_v73'
$meta = Get-Content -LiteralPath (Join-Path $run 'run.json') -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($meta.pid)" -ErrorAction SilentlyContinue
$progressPath = Join-Path $run 'progress.json'
$progress = if (Test-Path -LiteralPath $progressPath) { Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json } else { $null }
$status = if ($process) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Output "status: $status"
if ($progress) { Write-Output "done/total: $($progress.done)/$($progress.total)"; Write-Output "stage: $($progress.stage)"; Write-Output "last_output_timestamp: $((Get-Item -LiteralPath $progressPath).LastWriteTime)" } else { Write-Output 'done/total: 0/2'; Write-Output 'stage: starting'; Write-Output 'last_output_timestamp: none' }
Write-Output "pid: $($meta.pid)"
Write-Output "start_time: $($meta.start_time)"
Write-Output "stdout: $($meta.stdout)"
Write-Output "stderr: $($meta.stderr)"
if (Test-Path -LiteralPath $meta.stdout) { Get-Content -LiteralPath $meta.stdout -Tail 8 }
if (Test-Path -LiteralPath $meta.stderr) { Get-Content -LiteralPath $meta.stderr -Tail 8 }
