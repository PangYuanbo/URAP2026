param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$WatcherRunId = "native_video_continuous_val_watcher",
  [string]$WatcherSubdir = "continuous_val_watcher",
  [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RunDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
$OutputRoot = Join-Path $RunDir $WatcherSubdir
$PidFile = Join-Path $OutputRoot "$WatcherRunId.pid"
$MetaFile = Join-Path $OutputRoot "$WatcherRunId.meta.json"

if (-not (Test-Path $MetaFile)) {
  throw "Meta file not found: $MetaFile"
}

$Meta = Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json
$PidValue = $null
if (Test-Path $PidFile) {
  $PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
}
$Proc = if ($PidValue) { Get-Process -Id $PidValue -ErrorAction SilentlyContinue } else { $null }
$ProcInfo = if ($PidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue } else { $null }
$Running = $null -ne $ProcInfo
$CommandLine = if ($ProcInfo) { [string]$ProcInfo.CommandLine } else { "" }
$RunnerFile = [string]$Meta.runner_file
$CommandLineMatches = $Running -and $RunnerFile -and ($CommandLine -like "*$RunnerFile*")
$TrackedPids = @()
if ($PidValue) {
  $TrackedPids += $PidValue
  try {
    $ChildPids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$PidValue" |
      Select-Object -ExpandProperty ProcessId
    if ($ChildPids) {
      $TrackedPids = @($TrackedPids + $ChildPids) | Select-Object -Unique
    }
  } catch {
  }
}
$Stdout = [string]$Meta.stdout_log
$Stderr = [string]$Meta.stderr_log
$BestJson = [string]$Meta.best_json
$SeenFile = [string]$Meta.seen_file
$SeenCount = 0
if (Test-Path $SeenFile) {
  try {
    $Seen = Get-Content -LiteralPath $SeenFile -Raw | ConvertFrom-Json
    $SeenCount = @($Seen).Count
  } catch {
  }
}

$LastWrite = $null
foreach ($Path in @($Stdout, $Stderr, $BestJson, $SeenFile)) {
  if ($Path -and (Test-Path $Path)) {
    $Time = (Get-Item -LiteralPath $Path).LastWriteTime
    if (-not $LastWrite -or $Time -gt $LastWrite) {
      $LastWrite = $Time
    }
  }
}
$GpuProcess = $null
$GpuSummary = $null
try {
  $GpuRows = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
  $GpuProcess = $GpuRows | Where-Object {
    $Row = $_
    $TrackedPids | Where-Object { $Row -match "^\s*$_\s*," }
  }
  $GpuSummary = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
} catch {
}

Write-Output "RunId: $RunId"
if ($Running) {
  Write-Output "Status: RUNNING"
  Write-Output "PID: $PidValue"
  Write-Output "ProcessStartTime: $($Proc.StartTime.ToString('o'))"
} else {
  Write-Output "Status: NOT RUNNING"
  Write-Output "PID: $PidValue"
}
Write-Output "command_line_matches: $CommandLineMatches"
Write-Output "command_line: $CommandLine"
Write-Output "done/total checkpoints: $SeenCount/unknown"
Write-Output "last output timestamp: $LastWrite"
Write-Output "best_json: $BestJson"
Write-Output "seen_file: $SeenFile"
Write-Output "stdout log: $Stdout"
Write-Output "stderr log: $Stderr"
Write-Output "tracked PID tree: $($TrackedPids -join ', ')"
if ($GpuProcess) {
  Write-Output "GPU process signal for PID tree: $GpuProcess"
} else {
  Write-Output "GPU process signal for PID tree: not found"
}
if ($GpuSummary) {
  Write-Output "GPU summary util%,mem.used,mem.total: $GpuSummary"
}
if (Test-Path $BestJson) {
  Write-Output "--- best val checkpoint ---"
  Get-Content -LiteralPath $BestJson -Raw
}
if (Test-Path $Stdout) {
  Write-Output "--- stdout tail ---"
  Get-Content -LiteralPath $Stdout -Tail $Tail
}
if ((Test-Path $Stderr) -and ((Get-Item -LiteralPath $Stderr).Length -gt 0)) {
  Write-Output "--- stderr tail ---"
  Get-Content -LiteralPath $Stderr -Tail $Tail
}
