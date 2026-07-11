param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$DataRoot = "U:\URAP_datasets",
  [int]$MaxSamples = 512,
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
  [string]$WatcherRunId = "native_video_continuous_val_watcher",
  [string]$WatcherSubdir = "continuous_val_watcher",
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
$BestJson = Join-Path $OutputRoot "best_val_checkpoint.json"
$SeenFile = Join-Path $OutputRoot "seen_checkpoints.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $PidFile) {
  $ExistingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($ExistingPid) {
    $Existing = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($Existing -and $Existing.CommandLine -like "*$WatcherRunId.runner.ps1*") {
      Write-Output "Native video continuous val watcher already running: pid=$ExistingPid"
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
`$seen = @{}
`$best = `$null
function Write-JsonAtomic {
  param(
    [Parameter(Mandatory = `$true)]`$Value,
    [Parameter(Mandatory = `$true)][string]`$Path,
    [int]`$Depth = 5
  )
  `$tmp = "`$Path.tmp.`$PID"
  try {
    `$Value | ConvertTo-Json -Depth `$Depth | Set-Content -LiteralPath `$tmp -Encoding UTF8
    Move-Item -LiteralPath `$tmp -Destination `$Path -Force
  } finally {
    if (Test-Path `$tmp) {
      Remove-Item -LiteralPath `$tmp -Force -ErrorAction SilentlyContinue
    }
  }
}
if (Test-Path "$SeenFile") {
  try {
    foreach (`$seenPath in @((Get-Content -LiteralPath "$SeenFile" -Raw | ConvertFrom-Json))) {
      if (-not [string]::IsNullOrWhiteSpace([string]`$seenPath)) {
        `$seen[[string]`$seenPath] = `$true
      }
    }
  } catch {
    Write-Output "{`"kind`":`"native_video_continuous_val_seen_restore_failed`",`"seen_file`":`"$SeenFile`"}"
  }
}
if (Test-Path "$BestJson") {
  try {
    `$best = Get-Content -LiteralPath "$BestJson" -Raw | ConvertFrom-Json
  } catch {
    Write-Output "{`"kind`":`"native_video_continuous_val_best_restore_failed`",`"best_json`":`"$BestJson`"}"
  }
}
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
Write-Output "{`"kind`":`"native_video_continuous_val_watcher_start`",`"run_id`":`"$RunId`",`"run_dir`":`"$RunDir`",`"restored_seen`":`$(`$seen.Count),`"restored_best`":`$(`$null -ne `$best)}"
while (`$true) {
  if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) {
    Write-Output "{`"kind`":`"native_video_continuous_val_watcher_timeout`",`"run_id`":`"$RunId`"}"
    break
  }
  `$checkpoints = @(Get-ChildItem -LiteralPath "$RunDir" -Filter "native_video_detector_epoch_*.pt" -File -ErrorAction SilentlyContinue | Sort-Object Name)
  foreach (`$ckpt in `$checkpoints) {
    if (`$seen.ContainsKey(`$ckpt.FullName)) { continue }
    `$evalName = "val_" + `$ckpt.BaseName
    `$stableWeights = Wait-StableFile -Path `$ckpt.FullName -StableSeconds $StableSeconds -PollSeconds 2
    Write-Output "{`"kind`":`"native_video_continuous_val_eval_start`",`"weights`":`"`$(`$ckpt.FullName)`",`"eval_name`":`"`$evalName`",`"bytes`":`$(`$stableWeights.Length)}"
    & "$Repo\tools\eval_native_video_checkpoint.ps1" -RunId "$RunId" -Weights "`$(`$ckpt.FullName)" -Split "val" -DataRoot "$DataRoot" -BatchSize $BatchSize -MaxSamples $MaxSamples -ScoreThreshold $ScoreThreshold -TopK $TopK -NmsIouThreshold $NmsIouThreshold -PrimaryMetric "$PrimaryMetric" -EvalName "`$evalName" -CacheDir "$CacheDir" -Device "$Device"
    `$bestEvalPath = Join-Path "$RunDir" (Join-Path "`$evalName" "best_eval.json")
    if (-not (Test-Path `$bestEvalPath)) {
      throw "best eval json missing after eval: `$bestEvalPath"
    }
    `$eval = Get-Content -LiteralPath `$bestEvalPath -Raw | ConvertFrom-Json
    `$metricProperty = `$eval.PSObject.Properties["$PrimaryMetric"]
    if (`$null -eq `$metricProperty) {
      throw "Primary metric '$PrimaryMetric' not found in `$bestEvalPath"
    }
    `$metric = [double]`$metricProperty.Value
    `$recall = [double]`$eval.recall
    `$precision = [double]`$eval.precision
    `$isBetter = `$null -eq `$best -or
      `$metric -gt [double]`$best.metric -or
      (`$metric -eq [double]`$best.metric -and `$recall -gt [double]`$best.recall) -or
      (`$metric -eq [double]`$best.metric -and `$recall -eq [double]`$best.recall -and `$precision -gt [double]`$best.precision)
    if (`$isBetter) {
      `$best = [ordered]@{
        run_id = "$RunId"
        split = "val"
        primary_metric = "$PrimaryMetric"
        metric = `$metric
        recall = `$recall
        precision = `$precision
        weights = `$ckpt.FullName
        eval_name = `$evalName
        best_eval_json = `$bestEvalPath
        updated_at = (Get-Date).ToString("o")
      }
      Write-JsonAtomic -Value `$best -Path "$BestJson" -Depth 5
      `$bestUpdate = [ordered]@{
        kind = "native_video_continuous_val_best_update"
        metric = `$metric
        recall = `$recall
        precision = `$precision
        weights = `$ckpt.FullName
        eval_name = `$evalName
      }
      Write-Output (`$bestUpdate | ConvertTo-Json -Compress)
    }
    `$seen[`$ckpt.FullName] = `$true
    Write-JsonAtomic -Value (`$seen.Keys | Sort-Object) -Path "$SeenFile" -Depth 3
  }
  if (Test-Path (Join-Path "$RunDir" "summary.json")) {
    Write-Output "{`"kind`":`"native_video_continuous_val_watcher_done`",`"run_id`":`"$RunId`"}"
    break
  }
  Start-Sleep -Seconds $PollSeconds
}
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
  split = "val"
  max_samples = $MaxSamples
  batch_size = $BatchSize
  score_threshold = $ScoreThreshold
  top_k = $TopK
  nms_iou_threshold = $NmsIouThreshold
  primary_metric = $PrimaryMetric
  device = $Device
  cache_dir = $CacheDir
  poll_seconds = $PollSeconds
  stable_seconds = $StableSeconds
  timeout_minutes = $TimeoutMinutes
  watcher_subdir = $WatcherSubdir
  runner_file = $RunnerFile
  best_json = $BestJson
  seen_file = $SeenFile
  stdout_log = $Stdout
  stderr_log = $Stderr
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetaFile -Encoding UTF8

Write-Output "Started native video continuous val watcher."
Write-Output "PID: $($Proc.Id)"
Write-Output "RunId: $RunId"
Write-Output "BestJson: $BestJson"
Write-Output "Stdout: $Stdout"
Write-Output "Stderr: $Stderr"
Write-Output "Meta: $MetaFile"
