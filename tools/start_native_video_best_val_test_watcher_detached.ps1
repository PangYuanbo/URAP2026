param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$DataRoot = "U:\URAP_datasets",
  [int]$MaxSamples = 0,
  [int]$BatchSize = 16,
  [double]$ScoreThreshold = 0.0,
  [int]$TopK = 32,
  [double]$NmsIouThreshold = 0.5,
  [string]$PrimaryMetric = "map50",
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [int]$PollSeconds = 120,
  [int]$StableSeconds = 10,
  [int]$TimeoutMinutes = 0,
  [string]$WatcherRunId = "native_video_best_val_test_watcher",
  [string]$WatcherSubdir = "best_val_test_watcher",
  [string]$BestJson = "",
  [string]$CacheDir = ""
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RunDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
$OutputRoot = Join-Path $RunDir $WatcherSubdir
$LogDir = Join-Path $OutputRoot "logs"
$PidFile = Join-Path $OutputRoot "$WatcherRunId.pid"
$MetaFile = Join-Path $OutputRoot "$WatcherRunId.meta.json"
$RunnerFile = Join-Path $OutputRoot "$WatcherRunId.runner.ps1"
$SummaryFile = Join-Path $RunDir "summary.json"
if ([string]::IsNullOrWhiteSpace($BestJson)) {
  $BestJson = Join-Path $RunDir "continuous_val_watcher\best_val_checkpoint.json"
}
$ResultJson = Join-Path $OutputRoot "best_val_test_result.json"
$AuditJson = Join-Path $RunDir "mvp_audit.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $PidFile) {
  $ExistingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($ExistingPid) {
    $Existing = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($Existing -and $Existing.CommandLine -like "*$WatcherRunId.runner.ps1*") {
      Write-Output "Native video best-val test watcher already running: pid=$ExistingPid"
      Get-Content -LiteralPath $MetaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$StartedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LogDir "watcher_${WatcherRunId}_${StartedAt}.out.log"
$Stderr = Join-Path $LogDir "watcher_${WatcherRunId}_${StartedAt}.err.log"

$Runner = @"
`$ErrorActionPreference = "Stop"
Set-Location "$Repo"
`$start = Get-Date
function Wait-StableFile {
  param(
    [Parameter(Mandatory = `$true)][string]`$Path,
    [int]`$StableSeconds = 10,
    [int]`$PollSeconds = 2
  )
  `$lastLength = -1L
  `$stableFor = 0
  while (`$true) {
    if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) {
      throw "timeout waiting for stable file: `$Path"
    }
    if (-not (Test-Path `$Path)) {
      `$lastLength = -1L
      `$stableFor = 0
    } else {
      `$item = Get-Item -LiteralPath `$Path
      if (`$item.Length -gt 0 -and `$item.Length -eq `$lastLength) {
        `$stableFor += `$PollSeconds
        if (`$stableFor -ge `$StableSeconds) {
          return `$item
        }
      } else {
        `$lastLength = `$item.Length
        `$stableFor = 0
      }
    }
    Start-Sleep -Seconds `$PollSeconds
  }
}
Write-Output "{`"kind`":`"native_video_best_val_test_watcher_start`",`"run_id`":`"$RunId`",`"best_json`":`"$BestJson`",`"summary_file`":`"$SummaryFile`"}"
while ((-not (Test-Path "$SummaryFile")) -or (-not (Test-Path "$BestJson"))) {
  if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) {
    throw "timeout waiting for training summary and best-val json"
  }
  `$progress = [ordered]@{
    kind = "native_video_best_val_test_watcher_progress"
    summary_ready = (Test-Path "$SummaryFile")
    best_json_ready = (Test-Path "$BestJson")
  }
  Write-Output (`$progress | ConvertTo-Json -Compress)
  Start-Sleep -Seconds $PollSeconds
}
`$best = Get-Content -LiteralPath "$BestJson" -Raw | ConvertFrom-Json
`$weights = [string]`$best.weights
if ([string]::IsNullOrWhiteSpace(`$weights) -or -not (Test-Path `$weights)) {
  throw "best-val weights not found: `$weights"
}
`$stableWeights = Wait-StableFile -Path `$weights -StableSeconds $StableSeconds -PollSeconds 2
`$selectedValEvalPath = [string]`$best.best_eval_json
if ([string]::IsNullOrWhiteSpace(`$selectedValEvalPath) -or -not (Test-Path `$selectedValEvalPath)) {
  throw "selected val best_eval_json not found: `$selectedValEvalPath"
}
`$selectedValEval = Get-Content -LiteralPath `$selectedValEvalPath -Raw | ConvertFrom-Json
`$selectedScoreThreshold = [double]`$selectedValEval.score_threshold
`$selectedTopK = [int]`$selectedValEval.top_k
`$evalName = "test_best_val_" + [System.IO.Path]::GetFileNameWithoutExtension(`$weights)
`$evalStart = [ordered]@{
  kind = "native_video_best_val_test_eval_start"
  weights = `$weights
  eval_name = `$evalName
  weights_bytes = `$stableWeights.Length
  selected_score_threshold = `$selectedScoreThreshold
  selected_top_k = `$selectedTopK
  threshold_source = "validation"
}
Write-Output (`$evalStart | ConvertTo-Json -Compress)
& "$Repo\tools\eval_native_video_checkpoint.ps1" -RunId "$RunId" -Weights "`$weights" -Split "test" -DataRoot "$DataRoot" -BatchSize $BatchSize -MaxSamples $MaxSamples -ScoreThreshold `$selectedScoreThreshold -TopK `$selectedTopK -NmsIouThreshold $NmsIouThreshold -SweepScoreThresholds `$selectedScoreThreshold -SweepTopKs `$selectedTopK -PrimaryMetric "$PrimaryMetric" -RequireFullSplitBaseline 1 -EvalName "`$evalName" -CacheDir "$CacheDir" -Device "$Device"
`$bestEvalPath = Join-Path "$RunDir" (Join-Path `$evalName "best_eval.json")
`$comparePath = Join-Path "$RunDir" (Join-Path `$evalName "baseline_comparison.json")
if (-not (Test-Path `$bestEvalPath)) {
  throw "best eval json missing after test eval: `$bestEvalPath"
}
if (-not (Test-Path `$comparePath)) {
  throw "baseline comparison json missing after test eval: `$comparePath"
}
`$bestEval = Get-Content -LiteralPath `$bestEvalPath -Raw | ConvertFrom-Json
`$comparison = Get-Content -LiteralPath `$comparePath -Raw | ConvertFrom-Json
`$result = [ordered]@{
  run_id = "$RunId"
  split = "test"
  selection = "best_val"
  selected_val_json = "$BestJson"
  selected_val_best_eval_json = `$selectedValEvalPath
  selected_val_metric = `$best.metric
  selected_val_primary_metric = `$best.primary_metric
  selected_val_score_threshold = `$selectedScoreThreshold
  selected_val_top_k = `$selectedTopK
  test_threshold_source = "validation"
  weights = `$weights
  eval_name = `$evalName
  best_eval_json = `$bestEvalPath
  baseline_comparison_json = `$comparePath
  test_images = [int]`$bestEval.images
  test_labels = [int]`$bestEval.labels
  test_detections = [int]`$bestEval.detections
  test_precision = [double]`$bestEval.precision
  test_recall = [double]`$bestEval.recall
  test_map50 = [double]`$bestEval.map50
  test_map5095 = [double]`$bestEval.map5095
  test_f1 = [double]`$bestEval.f1
  baseline_name = [string]`$comparison.baseline_name
  baseline_status = [string]`$comparison.status
  baseline_primary_metric = [string]`$comparison.primary_metric
  baseline_primary_method = [double]`$comparison.primary.method
  baseline_primary_value = [double]`$comparison.primary.baseline
  baseline_primary_delta = [double]`$comparison.primary.delta
  baseline_primary_beat = [bool]`$comparison.primary.beat
  test_max_samples = $MaxSamples
  test_full_split = [bool]($MaxSamples -le 0)
  mvp_audit_json = "$AuditJson"
  completed_at = (Get-Date).ToString("o")
}
`$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$ResultJson" -Encoding UTF8
& "$Repo\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe" "$Repo\tools\audit_native_video_mvp_run.py" --run-dir "$RunDir" --out-json "$AuditJson" --primary-metric "$PrimaryMetric"
`$auditExitCode = `$LASTEXITCODE
if (`$auditExitCode -ne 0) {
  throw "MVP audit failed with exit code `$auditExitCode. See: $AuditJson"
}
Write-Output "{`"kind`":`"native_video_best_val_test_watcher_done`",`"run_id`":`"$RunId`",`"weights`":`"`$weights`",`"eval_name`":`"`$evalName`"}"
"@
$Runner | Set-Content -LiteralPath $RunnerFile -Encoding UTF8

$Proc = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunnerFile) `
  -WorkingDirectory $Repo `
  -RedirectStandardOutput $Stdout `
  -RedirectStandardError $Stderr `
  -PassThru `
  -WindowStyle Hidden

Set-Content -LiteralPath $PidFile -Value $Proc.Id -Encoding ASCII
$Meta = [ordered]@{
  started_at = (Get-Date).ToString("o")
  pid = $Proc.Id
  watcher_run_id = $WatcherRunId
  run_id = $RunId
  run_dir = $RunDir
  data_root = $DataRoot
  split = "test"
  selection = "best_val"
  max_samples = $MaxSamples
  batch_size = $BatchSize
  score_threshold = $ScoreThreshold
  top_k = $TopK
  test_threshold_source = "validation"
  nms_iou_threshold = $NmsIouThreshold
  primary_metric = $PrimaryMetric
  device = $Device
  cache_dir = $CacheDir
  poll_seconds = $PollSeconds
  stable_seconds = $StableSeconds
  timeout_minutes = $TimeoutMinutes
  watcher_subdir = $WatcherSubdir
  runner_file = $RunnerFile
  summary_file = $SummaryFile
  best_json = $BestJson
  result_json = $ResultJson
  audit_json = $AuditJson
  stdout_log = $Stdout
  stderr_log = $Stderr
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetaFile -Encoding UTF8

Write-Output "Started native video best-val test watcher."
Write-Output "PID: $($Proc.Id)"
Write-Output "RunId: $RunId"
Write-Output "BestJson: $BestJson"
Write-Output "ResultJson: $ResultJson"
Write-Output "Stdout: $Stdout"
Write-Output "Stderr: $Stderr"
Write-Output "Meta: $MetaFile"
