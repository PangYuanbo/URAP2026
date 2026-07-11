$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_otb100_full_download'
$StatePath=Join-Path $Run 'state.json'
if (-not (Test-Path $StatePath)) { Write-Host 'status: NOT RUNNING'; Write-Host 'done/total: 0/2815880168'; exit 0 }
$State=Get-Content $StatePath -Raw | ConvertFrom-Json
$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$($State.pid)" -ErrorAction SilentlyContinue
$Progress=$null
if (Test-Path $State.progress) { $Progress=Get-Content $State.progress -Raw | ConvertFrom-Json }
Write-Host ("status: " + $(if ($Process) {'RUNNING'} else {'NOT RUNNING'}))
Write-Host ("done/total: " + $(if ($Progress) {"$($Progress.done)/$($Progress.total)"} else {'0/2815880168'}))
Write-Host ("stage: " + $(if ($Progress) {$Progress.stage} else {'starting'}))
Write-Host "pid: $($State.pid)"
Write-Host "start_time: $($State.start_time)"
if ($Process) { Write-Host "command: $($Process.CommandLine)" }
$Paths=@($State.stdout,$State.stderr,$State.progress) | Where-Object { Test-Path $_ }
$Latest=$Paths | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host ("last_output_timestamp: " + $(if ($Latest) {$Latest.LastWriteTime} else {'none'}))
Write-Host ("last_completed_unit: " + $(if ($Progress) {($Progress | ConvertTo-Json -Compress)} else {'none'}))
Write-Host "stdout: $($State.stdout)"
Write-Host "stderr: $($State.stderr)"
Write-Host "progress: $($State.progress)"
