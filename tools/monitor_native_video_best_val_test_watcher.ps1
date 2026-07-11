param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$WatcherRunId = "native_video_best_val_test_watcher",
  [string]$WatcherSubdir = "best_val_test_watcher",
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
$SummaryFile = [string]$Meta.summary_file
$BestJson = [string]$Meta.best_json
$ResultJson = [string]$Meta.result_json
$AuditJson = [string]$Meta.audit_json
if ([string]::IsNullOrWhiteSpace($AuditJson)) {
  $AuditJson = Join-Path $RunDir "mvp_audit.json"
}
$Result = $null
if ($ResultJson -and (Test-Path $ResultJson)) {
  try {
    $Result = Get-Content -LiteralPath $ResultJson -Raw | ConvertFrom-Json
  } catch {
    $Result = $null
  }
}
$Audit = $null
if ($AuditJson -and (Test-Path $AuditJson)) {
  try {
    $Audit = Get-Content -LiteralPath $AuditJson -Raw | ConvertFrom-Json
  } catch {
    $Audit = $null
  }
}
$Done = 0
$Total = 5
$LastUnit = "wait_training_summary_and_best_val"
if ((Test-Path $SummaryFile) -and (Test-Path $BestJson)) {
  $Done = 1
  $LastUnit = "selection_ready"
}
if (Test-Path $ResultJson) {
  $Done = 4
  $LastUnit = "best_val_test_done"
}
if (Test-Path $AuditJson) {
  $Done = 5
  $LastUnit = "mvp_audit_done"
}

$LastWrite = $null
foreach ($Path in @($Stdout, $Stderr, $SummaryFile, $BestJson, $ResultJson, $AuditJson)) {
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
Write-Output "done/total: $Done/$Total"
Write-Output "last completed unit: $LastUnit"
Write-Output "last output timestamp: $LastWrite"
Write-Output "summary_file: $SummaryFile"
Write-Output "best_json: $BestJson"
Write-Output "result_json: $ResultJson"
Write-Output "audit_json: $AuditJson"
if ($Result) {
  Write-Output "baseline gate: status=$($Result.baseline_status) primary_metric=$($Result.baseline_primary_metric) method=$($Result.baseline_primary_method) baseline=$($Result.baseline_primary_value) delta=$($Result.baseline_primary_delta) beat=$($Result.baseline_primary_beat) test_full_split=$($Result.test_full_split) test_max_samples=$($Result.test_max_samples) test_map50=$($Result.test_map50) test_recall=$($Result.test_recall) test_precision=$($Result.test_precision)"
}
if ($Audit) {
  $FailedNames = @()
  if ($Audit.failed_checks) {
    $FailedNames = @($Audit.failed_checks | ForEach-Object { $_.name })
  }
  Write-Output "mvp audit: status=$($Audit.status) passed=$($Audit.passed) failed_checks=$($FailedNames -join ',') primary_metric=$($Audit.primary_metric)"
}
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
if (Test-Path $ResultJson) {
  Write-Output "--- best-val test result ---"
  Get-Content -LiteralPath $ResultJson -Raw
}
if (Test-Path $AuditJson) {
  Write-Output "--- mvp audit ---"
  Get-Content -LiteralPath $AuditJson -Raw
}
if (Test-Path $BestJson) {
  Write-Output "--- selected best val checkpoint ---"
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
