param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$WatcherRunId = "native_video_checkpoint_eval_watcher",
  [string]$WatcherSubdir = "checkpoint_eval_watcher",
  [string]$RunDir = "",
  [string]$RunRoot = "",
  [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
}
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = $RunDir
}
$OutputRoot = Join-Path $RunRoot $WatcherSubdir
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
$Running = ($null -ne $ProcInfo) -and ($null -ne $Proc)
$ProcessStartTime = ""
if ($Proc -and $Proc.StartTime) {
  $ProcessStartTime = $Proc.StartTime.ToString('o')
} elseif ($ProcInfo) {
  $ProcessStartTime = [System.Management.ManagementDateTimeConverter]::ToDateTime($ProcInfo.CreationDate).ToString('o')
}
$CommandLine = if ($ProcInfo) { [string]$ProcInfo.CommandLine } else { "" }
$RunnerFile = [string]$Meta.runner_file
$CommandLineMatches = $Running -and $RunnerFile -and ($CommandLine -like "*$RunnerFile*")
$TrackedPids = @()
if ($PidValue) {
  $Queue = New-Object System.Collections.Queue
  $Seen = @{}
  $Queue.Enqueue($PidValue)
  try {
    while ($Queue.Count -gt 0) {
      $CurrentPid = [int]$Queue.Dequeue()
      if ($Seen.ContainsKey($CurrentPid)) {
        continue
      }
      $Seen[$CurrentPid] = $true
      $TrackedPids += $CurrentPid
      $ChildPids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$CurrentPid" |
        Select-Object -ExpandProperty ProcessId
      foreach ($ChildPid in $ChildPids) {
        if (-not $Seen.ContainsKey([int]$ChildPid)) {
          $Queue.Enqueue([int]$ChildPid)
        }
      }
    }
  } catch {
    $TrackedPids += $PidValue
  }
  $TrackedPids = $TrackedPids | Select-Object -Unique
}
$Stdout = [string]$Meta.stdout_log
$Stderr = [string]$Meta.stderr_log
$Weights = [string]$Meta.weights
$WaitForFile = [string]$Meta.wait_for_file
$EvalName = if ($Meta.PSObject.Properties.Name -contains "eval_name") { [string]$Meta.eval_name } else { "eval_$($Meta.split)" }
$EvalJson = Join-Path $RunDir "$EvalName\eval.json"
$SweepJson = Join-Path $RunDir "$EvalName\threshold_sweep.json"
$BestEvalJson = Join-Path $RunDir "$EvalName\best_eval.json"
$CompareJson = Join-Path $RunDir "$EvalName\baseline_comparison.json"
$Pkl = Join-Path $RunDir "$EvalName\predictionsgt.pkl"

$Done = 0
$Total = 3
$LastUnit = "wait_file"
if ($WaitForFile -and (Test-Path $WaitForFile)) {
  $Done = 1
  $LastUnit = "wait_file_ready"
}
if (Test-Path $Pkl) {
  $Done = 2
  $LastUnit = "predictions_exported"
}
if (Test-Path $EvalJson) {
  $Done = 3
  $LastUnit = "eval_ready"
}
if (Test-Path $SweepJson) {
  $Done = 3
  $LastUnit = "threshold_sweep_ready"
}
if (Test-Path $BestEvalJson) {
  $Done = 3
  $LastUnit = "best_eval_ready"
}
if (Test-Path $CompareJson) {
  $Done = 3
  $LastUnit = "baseline_comparison_ready"
}

$LastWrite = $null
foreach ($Path in @($Stdout, $Stderr, $Weights, $WaitForFile, $Pkl, $EvalJson, $SweepJson, $BestEvalJson, $CompareJson)) {
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
  Write-Output "ProcessStartTime: $ProcessStartTime"
} else {
  Write-Output "Status: NOT RUNNING"
  Write-Output "PID: $PidValue"
  if ($ProcessStartTime) {
    Write-Output "ProcessStartTime: $ProcessStartTime"
  }
}
Write-Output "command_line_matches: $CommandLineMatches"
Write-Output "command_line: $CommandLine"
Write-Output "done/total: $Done/$Total"
Write-Output "last completed unit: $LastUnit"
Write-Output "last output timestamp: $LastWrite"
Write-Output "eval_name: $EvalName"
if ($Meta.PSObject.Properties.Name -contains "proposal_score_weight") {
  Write-Output "proposal_score_weight: $($Meta.proposal_score_weight)"
}
if ($Meta.PSObject.Properties.Name -contains "quality_score_weight") {
  Write-Output "quality_score_weight: $($Meta.quality_score_weight)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_motion_rerank") {
  Write-Output "samurai_motion_rerank: $($Meta.samurai_motion_rerank)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_motion_iou_weight") {
  Write-Output "samurai_motion_iou_weight: $($Meta.samurai_motion_iou_weight)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_center_weight") {
  Write-Output "samurai_center_weight: $($Meta.samurai_center_weight)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_tracklet_rerank") {
  Write-Output "samurai_tracklet_rerank: $($Meta.samurai_tracklet_rerank)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_tracklet_match_threshold") {
  Write-Output "samurai_tracklet_match_threshold: $($Meta.samurai_tracklet_match_threshold)"
}
if ($Meta.PSObject.Properties.Name -contains "samurai_tracklet_weight") {
  Write-Output "samurai_tracklet_weight: $($Meta.samurai_tracklet_weight)"
}
if ($Meta.PSObject.Properties.Name -contains "action_chunk_backfill") {
  Write-Output "action_chunk_backfill: $($Meta.action_chunk_backfill)"
}
if ($Meta.PSObject.Properties.Name -contains "action_chunk_max_step") {
  Write-Output "action_chunk_max_step: $($Meta.action_chunk_max_step)"
}
if ($Meta.PSObject.Properties.Name -contains "action_chunk_score_decay") {
  Write-Output "action_chunk_score_decay: $($Meta.action_chunk_score_decay)"
}
if ($Meta.PSObject.Properties.Name -contains "action_chunk_merge_mode") {
  Write-Output "action_chunk_merge_mode: $($Meta.action_chunk_merge_mode)"
}
Write-Output "wait_for_file: $WaitForFile"
Write-Output "weights: $Weights"
Write-Output "predictionsgt: $Pkl"
Write-Output "eval_json: $EvalJson"
Write-Output "threshold_sweep_json: $SweepJson"
Write-Output "best_eval_json: $BestEvalJson"
Write-Output "baseline_comparison_json: $CompareJson"
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

if (Test-Path $CompareJson) {
  Write-Output "--- baseline comparison json ---"
  Get-Content -LiteralPath $CompareJson -Raw
}
if (Test-Path $EvalJson) {
  Write-Output "--- eval json ---"
  Get-Content -LiteralPath $EvalJson -Raw
}
if (Test-Path $BestEvalJson) {
  Write-Output "--- best eval json ---"
  Get-Content -LiteralPath $BestEvalJson -Raw
}
if (Test-Path $Stdout) {
  Write-Output "--- stdout tail ---"
  Get-Content -LiteralPath $Stdout -Tail $Tail
}
if ((Test-Path $Stderr) -and ((Get-Item -LiteralPath $Stderr).Length -gt 0)) {
  Write-Output "--- stderr tail ---"
  Get-Content -LiteralPath $Stderr -Tail $Tail
}
