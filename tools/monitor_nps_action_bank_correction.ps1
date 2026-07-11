$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\aaron\Desktop\URAP'
$Run = Join-Path $Repo 'artifacts\detached_nps_action_bank_correction_v6'
$StatePath = Join-Path $Run 'state.json'
if (-not (Test-Path $StatePath)) { Write-Host 'status: NOT RUNNING'; Write-Host 'done/total: 0/18'; exit 0 }
$State = Get-Content $StatePath -Raw | ConvertFrom-Json
$Process = Get-CimInstance Win32_Process -Filter "ProcessId=$($State.pid)" -ErrorAction SilentlyContinue
$Children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $State.pid -and $_.CommandLine -like '*train_action_bank_correction_head.py*' })
$Alive = $null -ne $Process
$Epoch = 0
$Last = $null
if (Test-Path $State.stdout) {
  $Lines = Get-Content $State.stdout -Tail 80
  $ProgressLines = @($Lines | Where-Object { $_ -like '*correction_train_progress*' })
  if ($ProgressLines.Count -gt 0) {
    $Last = $ProgressLines[-1] | ConvertFrom-Json
    $Epoch = [int]$Last.epoch
  }
}
$Summary = Join-Path $State.output 'train_summary.json'
if (Test-Path $Summary) { $Epoch = 18 }
Write-Host ("status: " + $(if ($Alive) { 'RUNNING' } else { 'NOT RUNNING' }))
Write-Host "done/total: $Epoch/18"
Write-Host ("stage: " + $(if ($Epoch -eq 18) { 'training_complete' } elseif ($Epoch -gt 0) { 'gpu_training' } else { 'loading_tracklet_features' }))
Write-Host "pid: $($State.pid)"
Write-Host "start_time: $($State.start_time)"
if ($Alive) { Write-Host "command: $($Process.CommandLine)" }
if ($Children.Count -gt 0) { Write-Host "runtime_pid: $($Children[0].ProcessId)"; Write-Host "runtime_command: $($Children[0].CommandLine)" } else { Write-Host 'runtime_pid: none' }
$ExistingPaths = @($State.stdout, $State.stderr, $Summary) | Where-Object { Test-Path $_ }
$Latest = $ExistingPaths | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($Latest) { Write-Host "last_output_timestamp: $($Latest.LastWriteTime)" } else { Write-Host 'last_output_timestamp: none' }
if ($Last) { Write-Host ("last_completed_unit: " + ($Last | ConvertTo-Json -Compress)) } else { Write-Host 'last_completed_unit: none' }
$GpuPids = @($State.pid) + @($Children | ForEach-Object { $_.ProcessId })
$GpuRows = & nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
$Gpu = @($GpuRows | Where-Object { $row = $_; $GpuPids | Where-Object { $row -match "^\s*$_," } })
Write-Host ("gpu_signal: " + $(if ($Gpu) { $Gpu } else { 'NO GPU SIGNAL YET' }))
Write-Host "stdout: $($State.stdout)"
Write-Host "stderr: $($State.stderr)"
Write-Host "summary: $Summary"
