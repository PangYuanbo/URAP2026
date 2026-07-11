param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$RunRoot = "",
  [int]$Tail = 8
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = Join-Path $Repo "artifacts\native_video_detector\$RunId"
}
$PidFile = Join-Path $RunRoot "train.pid"
$MetaFile = Join-Path $RunRoot "train_meta.json"

if (-not (Test-Path $MetaFile)) {
  throw "Meta file not found: $MetaFile"
}
if (-not (Test-Path $PidFile)) {
  throw "PID file not found: $PidFile"
}

$Meta = Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json
$PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
$Proc = Get-Process -Id $PidValue -ErrorAction SilentlyContinue
$ProcInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue
$Running = ($null -ne $ProcInfo) -and ($null -ne $Proc)
$ProcessStartTime = ""
if ($Proc) {
  $ProcessStartTime = $Proc.StartTime.ToString('o')
} elseif ($ProcInfo) {
  $ProcessStartTime = [System.Management.ManagementDateTimeConverter]::ToDateTime($ProcInfo.CreationDate).ToString('o')
}
$CommandLine = if ($ProcInfo) { [string]$ProcInfo.CommandLine } else { "" }
$ExpectedCommand = if ($Meta.PSObject.Properties.Name -contains "command") { [string]$Meta.command } else { "" }
$CommandLineMatches = $Running -and ($CommandLine -like "*train_native_video_detector.py*")
$TrackedPids = @($PidValue)
try {
  $ChildPids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$PidValue" |
    Select-Object -ExpandProperty ProcessId
  if ($ChildPids) {
    $TrackedPids = @($TrackedPids + $ChildPids) | Select-Object -Unique
  }
} catch {
}
$StdoutLog = [string]$Meta.stdout_log
$StderrLog = [string]$Meta.stderr_log
$LastProgress = $null
$LastDone = $null
$LastCheckpoint = $null
$ProgressRows = @()

if (Test-Path $StdoutLog) {
  $Lines = Get-Content -LiteralPath $StdoutLog -Tail 300 -ErrorAction SilentlyContinue
  foreach ($Line in $Lines) {
    if ($Line -match '^\{') {
      try {
        $Obj = $Line | ConvertFrom-Json -ErrorAction Stop
        if ($Obj.kind -eq "native_video_train_progress") {
          $LastProgress = $Obj
          $ProgressRows += $Obj
        }
        if ($Obj.kind -eq "native_video_train_done") { $LastDone = $Obj }
        if ($Obj.kind -eq "native_video_epoch_checkpoint") { $LastCheckpoint = $Obj }
        if ($Obj.kind -eq "native_video_step_checkpoint") { $LastCheckpoint = $Obj }
      } catch {
      }
    }
  }
}

$AvgDataMs = $null
$AvgStepMs = $null
$DataWaitRatio = $null
$ThroughputFramesPerSecond = $null
$LoaderDiagnosis = "insufficient_progress_samples"
if ($ProgressRows.Count -gt 0) {
  $DataValues = @($ProgressRows | Where-Object { $_.PSObject.Properties.Name -contains "data_ms" } | ForEach-Object { [double]$_.data_ms })
  $StepValues = @($ProgressRows | Where-Object { $_.PSObject.Properties.Name -contains "step_ms" } | ForEach-Object { [double]$_.step_ms })
  $FpsValues = @($ProgressRows | Where-Object { $_.PSObject.Properties.Name -contains "frames_per_second" } | ForEach-Object { [double]$_.frames_per_second })
  if ($DataValues.Count -gt 0) {
    $AvgDataMs = [Math]::Round((($DataValues | Measure-Object -Average).Average), 3)
  }
  if ($StepValues.Count -gt 0) {
    $AvgStepMs = [Math]::Round((($StepValues | Measure-Object -Average).Average), 3)
  }
  if ($FpsValues.Count -gt 0) {
    $ThroughputFramesPerSecond = [Math]::Round((($FpsValues | Measure-Object -Average).Average), 3)
  }
  if ($null -ne $AvgDataMs -and $null -ne $AvgStepMs) {
    $Denom = [Math]::Max($AvgDataMs + $AvgStepMs, 1e-9)
    $DataWaitRatio = [Math]::Round($AvgDataMs / $Denom, 3)
    if ($DataWaitRatio -ge 0.35) {
      $LoaderDiagnosis = "data_loader_bottleneck"
    } else {
      $LoaderDiagnosis = "compute_bound_or_balanced"
    }
  }
}

$Done = 0
$Total = [int]$Meta.epochs
if ($LastDone) {
  $Done = $Total
} elseif ($LastProgress) {
  $Done = [int]$LastProgress.epoch - 1
}

$StdoutInfo = if (Test-Path $StdoutLog) { Get-Item -LiteralPath $StdoutLog } else { $null }
$StderrInfo = if (Test-Path $StderrLog) { Get-Item -LiteralPath $StderrLog } else { $null }
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
if ($ExpectedCommand) {
  Write-Output "expected_command: $ExpectedCommand"
}
Write-Output "done/total epochs: $Done/$Total"
Write-Output "tracked PID tree: $($TrackedPids -join ', ')"
if ($Meta.PSObject.Properties.Name -contains "ema") {
  Write-Output "ema_enabled: $($Meta.ema) ema_decay: $($Meta.ema_decay)"
}
if ($Meta.PSObject.Properties.Name -contains "lr_scheduler") {
  Write-Output "lr: $($Meta.lr) lr_scheduler: $($Meta.lr_scheduler) warmup_steps: $($Meta.warmup_steps) min_lr_ratio: $($Meta.min_lr_ratio)"
}
if ($Meta.PSObject.Properties.Name -contains "augment_hflip_prob") {
  Write-Output "augment_hflip_prob: $($Meta.augment_hflip_prob) augment_brightness: $($Meta.augment_brightness) augment_contrast: $($Meta.augment_contrast)"
}
if ($Meta.PSObject.Properties.Name -contains "seed") {
  Write-Output "seed: $($Meta.seed)"
}
if ($Meta.PSObject.Properties.Name -contains "memory_attention") {
  $ArchParts = @(
    "memory_mode=$($Meta.memory_mode)",
    "memory_attention=$($Meta.memory_attention)",
    "memory_slots=$($Meta.memory_slots)",
    "query_mode=$($Meta.query_mode)",
    "patch_stride=$($Meta.patch_stride)",
    "dense_obj_source=$($Meta.dense_obj_source)"
  )
  if ($Meta.PSObject.Properties.Name -contains "spatial_refine_layers") {
    $ArchParts += "spatial_refine_layers=$($Meta.spatial_refine_layers)"
    $ArchParts += "spatial_refine_kernel=$($Meta.spatial_refine_kernel)"
    $ArchParts += "spatial_refine_expansion=$($Meta.spatial_refine_expansion)"
  }
  if ($Meta.PSObject.Properties.Name -contains "memory_match_mode") {
    $ArchParts += "memory_match_mode=$($Meta.memory_match_mode)"
    $ArchParts += "memory_match_weight=$($Meta.memory_match_weight)"
    $ArchParts += "memory_match_temperature=$($Meta.memory_match_temperature)"
  }
  if ($Meta.PSObject.Properties.Name -contains "motion_score_mode") {
    $ArchParts += "motion_score_mode=$($Meta.motion_score_mode)"
    $ArchParts += "motion_score_weight=$($Meta.motion_score_weight)"
  }
  if ($Meta.PSObject.Properties.Name -contains "proposal_mode") {
    $ArchParts += "proposal_mode=$($Meta.proposal_mode)"
  }
  if ($Meta.PSObject.Properties.Name -contains "quality_score_mode") {
    $ArchParts += "quality_score_mode=$($Meta.quality_score_mode)"
  }
  Write-Output "native_video_arch: $($ArchParts -join ' ')"
}
if ($Meta.PSObject.Properties.Name -contains "dense_hard_negative_topk") {
  $DenseLossParts = @(
    "dense_positive_radius=$($Meta.dense_positive_radius)",
    "dense_positive_topk=$($Meta.dense_positive_topk)",
    "dense_hard_negative_topk=$($Meta.dense_hard_negative_topk)"
  )
  if ($Meta.PSObject.Properties.Name -contains "dense_rank_weight") {
    $DenseLossParts += "dense_rank_weight=$($Meta.dense_rank_weight)"
    $DenseLossParts += "dense_rank_margin=$($Meta.dense_rank_margin)"
    $DenseLossParts += "dense_rank_negative_topk=$($Meta.dense_rank_negative_topk)"
    if ($Meta.PSObject.Properties.Name -contains "dense_rank_positive_mode") {
      $DenseLossParts += "dense_rank_positive_mode=$($Meta.dense_rank_positive_mode)"
    }
  }
  if ($Meta.PSObject.Properties.Name -contains "action_chunk_consistency_weight") {
    $DenseLossParts += "action_chunk_consistency_weight=$($Meta.action_chunk_consistency_weight)"
  }
  if ($Meta.PSObject.Properties.Name -contains "memory_quality_weight") {
    $DenseLossParts += "memory_quality_weight=$($Meta.memory_quality_weight)"
    if ($Meta.PSObject.Properties.Name -contains "memory_quality_sigma") {
      $DenseLossParts += "memory_quality_sigma=$($Meta.memory_quality_sigma)"
    }
    if ($Meta.PSObject.Properties.Name -contains "memory_quality_recency_tau") {
      $DenseLossParts += "memory_quality_recency_tau=$($Meta.memory_quality_recency_tau)"
    }
    if ($Meta.PSObject.Properties.Name -contains "memory_quality_exclude_current") {
      $DenseLossParts += "memory_quality_exclude_current=$($Meta.memory_quality_exclude_current)"
    }
  }
  if ($Meta.PSObject.Properties.Name -contains "motion_obj_weight") {
    $DenseLossParts += "motion_obj_weight=$($Meta.motion_obj_weight)"
  }
  if ($Meta.PSObject.Properties.Name -contains "dense_heatmap_weight") {
    $DenseLossParts += "dense_heatmap_weight=$($Meta.dense_heatmap_weight)"
    $DenseLossParts += "dense_heatmap_sigma=$($Meta.dense_heatmap_sigma)"
    $DenseLossParts += "dense_heatmap_neg_weight=$($Meta.dense_heatmap_neg_weight)"
    $DenseLossParts += "dense_heatmap_focal_gamma=$($Meta.dense_heatmap_focal_gamma)"
  }
  if ($Meta.PSObject.Properties.Name -contains "memory_match_loss_weight") {
    $DenseLossParts += "memory_match_loss_weight=$($Meta.memory_match_loss_weight)"
  }
  if ($Meta.PSObject.Properties.Name -contains "quality_loss_weight") {
    $DenseLossParts += "quality_loss_weight=$($Meta.quality_loss_weight)"
    if ($Meta.PSObject.Properties.Name -contains "quality_warmup_steps") {
      $DenseLossParts += "quality_warmup_steps=$($Meta.quality_warmup_steps)"
    }
    if ($Meta.PSObject.Properties.Name -contains "quality_ramp_steps") {
      $DenseLossParts += "quality_ramp_steps=$($Meta.quality_ramp_steps)"
    }
    $DenseLossParts += "quality_positive_iou=$($Meta.quality_positive_iou)"
    $DenseLossParts += "quality_hard_negative_topk=$($Meta.quality_hard_negative_topk)"
    $DenseLossParts += "quality_focal_gamma=$($Meta.quality_focal_gamma)"
  }
  Write-Output "dense_loss: $($DenseLossParts -join ' ')"
}
if ($LastProgress) {
  $ProgressParts = @(
    "epoch=$($LastProgress.epoch)",
    "batch=$($LastProgress.batch)/$($LastProgress.batches_total)",
    "global_step=$($LastProgress.global_step)",
    "loss=$($LastProgress.loss)"
  )
  if ($LastProgress.data_ms) { $ProgressParts += "data_ms=$($LastProgress.data_ms)" }
  if ($LastProgress.lr) { $ProgressParts += "lr=$($LastProgress.lr)" }
  if ($LastProgress.step_ms) { $ProgressParts += "step_ms=$($LastProgress.step_ms)" }
  if ($LastProgress.clips_per_second) { $ProgressParts += "clips_per_second=$($LastProgress.clips_per_second)" }
  if ($LastProgress.frames_per_second) { $ProgressParts += "frames_per_second=$($LastProgress.frames_per_second)" }
  if ($LastProgress.future_obj_pos) { $ProgressParts += "future_obj_pos=$($LastProgress.future_obj_pos)" }
  if ($LastProgress.PSObject.Properties.Name -contains "action_chunk_consistency_pos") {
    $ProgressParts += "action_chunk_consistency_pos=$($LastProgress.action_chunk_consistency_pos)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "action_chunk_consistency_loss") {
    $ProgressParts += "action_chunk_consistency_loss=$($LastProgress.action_chunk_consistency_loss)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "memory_quality_samples") {
    $ProgressParts += "memory_quality_samples=$($LastProgress.memory_quality_samples)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "memory_quality_loss") {
    $ProgressParts += "memory_quality_loss=$($LastProgress.memory_quality_loss)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "memory_quality_target_entropy") {
    $ProgressParts += "memory_quality_target_entropy=$($LastProgress.memory_quality_target_entropy)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "quality_loss_weight_effective") {
    $ProgressParts += "quality_loss_weight_effective=$($LastProgress.quality_loss_weight_effective)"
  }
  if ($LastProgress.PSObject.Properties.Name -contains "memory_match_loss") {
    $ProgressParts += "memory_match_loss=$($LastProgress.memory_match_loss)"
  }
  Write-Output "last completed unit: $($ProgressParts -join ' ')"
} elseif ($LastDone) {
  Write-Output "last completed unit: training done"
} else {
  Write-Output "last completed unit: no progress JSON found yet"
}
Write-Output "throughput diagnosis: samples=$($ProgressRows.Count) avg_data_ms=$AvgDataMs avg_step_ms=$AvgStepMs data_wait_ratio=$DataWaitRatio avg_frames_per_second=$ThroughputFramesPerSecond diagnosis=$LoaderDiagnosis"
if ($LastCheckpoint) {
  $CheckpointParts = @(
    "kind=$($LastCheckpoint.kind)",
    "epoch=$($LastCheckpoint.epoch)",
    "batch=$($LastCheckpoint.batch)",
    "global_step=$($LastCheckpoint.global_step)"
  )
  if ($LastCheckpoint.batches_total) {
    $CheckpointParts += "batches_total=$($LastCheckpoint.batches_total)"
  }
  if ($LastCheckpoint.epochs_total) {
    $CheckpointParts += "epochs_total=$($LastCheckpoint.epochs_total)"
  }
  if ($LastCheckpoint.epoch_loss_so_far) {
    $CheckpointParts += "epoch_loss_so_far=$($LastCheckpoint.epoch_loss_so_far)"
  }
  if ($LastCheckpoint.latest_weights) {
    $CheckpointParts += "latest_weights=$($LastCheckpoint.latest_weights)"
  }
  if ($LastCheckpoint.weights) {
    $CheckpointParts += "weights=$($LastCheckpoint.weights)"
  }
  if ($LastCheckpoint.step_weights) {
    $CheckpointParts += "step_weights=$($LastCheckpoint.step_weights)"
  }
  if ($LastCheckpoint.epoch_weights) {
    $CheckpointParts += "epoch_weights=$($LastCheckpoint.epoch_weights)"
  }
  Write-Output "last checkpoint: $($CheckpointParts -join ' ')"
}
if ($StdoutInfo) {
  Write-Output "last output timestamp: $($StdoutInfo.LastWriteTime.ToString('o'))"
  Write-Output "stdout log: $StdoutLog"
} else {
  Write-Output "last output timestamp: stdout log missing"
  Write-Output "stdout log: $StdoutLog"
}
if ($StderrInfo) {
  Write-Output "stderr log: $StderrLog last_write=$($StderrInfo.LastWriteTime.ToString('o')) size=$($StderrInfo.Length)"
} else {
  Write-Output "stderr log: $StderrLog"
}
if ($GpuProcess) {
  Write-Output "GPU process signal for PID: $GpuProcess"
} else {
  Write-Output "GPU process signal for PID: not found"
}
if ($GpuSummary) {
  Write-Output "GPU summary util%,mem.used,mem.total: $GpuSummary"
}
if (Test-Path $StdoutLog) {
  Write-Output "--- stdout tail ---"
  Get-Content -LiteralPath $StdoutLog -Tail $Tail
}
if ((Test-Path $StderrLog) -and ((Get-Item -LiteralPath $StderrLog).Length -gt 0)) {
  Write-Output "--- stderr tail ---"
  Get-Content -LiteralPath $StderrLog -Tail $Tail
}
