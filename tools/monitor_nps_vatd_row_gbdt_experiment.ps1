param(
  [string]$RunId = 'nps_vatd_row_gbdt_trainval_v1',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_sota_research\nps_vatd_row_gbdt_trainval_v1_runner')
)
$pidFile = Join-Path $OutputRoot "$RunId.pid"; $metaFile = Join-Path $OutputRoot "$RunId.meta.txt"; $progressFile = Join-Path $OutputRoot 'progress.json'
$pidValue = if (Test-Path $pidFile) { Get-Content $pidFile | Select-Object -First 1 } else { $null }; $process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }; $status = if ($process -and $process.CommandLine -like "*$RunId*") { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host "$status PID=$pidValue"
if (Test-Path $progressFile) { $progress = Get-Content $progressFile -Raw | ConvertFrom-Json; Write-Host "status=$($progress.stage) done=$($progress.done)/$($progress.total) last_output_timestamp=$($progress.updated)" } else { Write-Host 'status=not_started done=0/2 last_output_timestamp=none' }
if (Test-Path $metaFile) { Get-Content $metaFile }
$parent = Split-Path $OutputRoot -Parent; $summary = Join-Path $parent 'nps_vatd_row_gbdt_trainval_v1\fusion_sweep.json'
if (Test-Path $summary) { $data = Get-Content $summary -Raw | ConvertFrom-Json; Write-Host "last_completed_unit=evaluation best_map50=$($data.best.map50) best_recall=$($data.best.recall)" } elseif (Test-Path (Join-Path $parent 'nps_vatd_row_gbdt_trainval_v1\train_summary.json')) { Write-Host 'last_completed_unit=train' } else { Write-Host 'last_completed_unit=none' }
$meta = if (Test-Path $metaFile) { Get-Content $metaFile } else { @() }; foreach ($kind in @('stdout','stderr')) { $line = $meta | Where-Object { $_ -like "$kind=*" } | Select-Object -First 1; if ($line) { $path = $line.Substring($kind.Length + 1); Write-Host "${kind}=$path"; if (Test-Path $path) { Get-Content $path -Tail 8 } } }

