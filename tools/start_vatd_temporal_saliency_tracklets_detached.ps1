param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$ListFiles,
  [string]$GtCsv,
  [string]$FrameRoot = '',
  [string]$FrameListOut = '',
  [int]$MaxFramesPerSeq = 0,
  [switch]$RecursiveFrameRoot,
  [string]$OutDir,
  [string]$DatasetSource = 'vatd_temporal_saliency',
  [int]$MaxImages = 0,
  [double]$Threshold = 24.0,
  [double]$MinArea = 2.0,
  [double]$MaxArea = 400.0,
  [int]$DilateIters = 1,
  [int]$MaxGap = 3,
  [double]$BaseRadius = 18.0,
  [double]$RadiusPerSide = 0.75,
  [double]$MinIou = 0.0,
  [int]$MinTrackletRows = 2,
  [double]$IouThreshold = 0.3,
  [double]$CenterThreshold = 24.0,
  [double]$HardTinySide = 24.0,
  [double]$HardLowScore = 0.25,
  [int]$ProgressEverySequences = 10,
  [string]$RunId = 'vatd_temporal_saliency_tracklets',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_temporal_saliency_tracklets_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $GtCsv -PathType Leaf)) { throw "GtCsv not found: $GtCsv" }
if (-not $OutDir) { throw 'OutDir must be provided' }
if (-not $ListFiles -or $ListFiles.Count -eq 0) {
  if (-not $FrameRoot) { throw 'Either ListFiles or FrameRoot must be provided' }
  if (-not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }
  if (-not $FrameListOut) {
    $FrameListOut = Join-Path $OutDir 'frame_list_from_gt.txt'
  }
  $frameListDir = Split-Path -Parent $FrameListOut
  if ($frameListDir) { New-Item -ItemType Directory -Force -Path $frameListDir | Out-Null }
  $frameArgs = @(
    '-m', 'qstr_dronedet.cli', 'export-frame-list-from-gt-csv',
    '--gt-csv', $GtCsv,
    '--frame-root', $FrameRoot,
    '--out', $FrameListOut
  )
  if ($MaxImages -gt 0) { $frameArgs += @('--max-frames', [string]$MaxImages) }
  if ($MaxFramesPerSeq -gt 0) { $frameArgs += @('--max-frames-per-seq', [string]$MaxFramesPerSeq) }
  if ($RecursiveFrameRoot) { $frameArgs += '--recursive' }
  $env:PYTHONPATH = $RepoRoot
  & $Python @frameArgs
  if ($LASTEXITCODE -ne 0) { throw "export-frame-list-from-gt-csv failed with exit code $LASTEXITCODE" }
  $ListFiles = @($FrameListOut)
}
foreach ($listFile in $ListFiles) {
  if (-not (Test-Path -Path $listFile -PathType Leaf)) { throw "List file not found: $listFile" }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*export-temporal-saliency-tracklets*') {
      Write-Host "VATD temporal-saliency export already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logDir "runner_${RunId}_${ts}.err.txt"

$argList = @(
  '-m', 'qstr_dronedet.cli', 'export-temporal-saliency-tracklets',
  '--list-files'
)
$argList += $ListFiles
$argList += @(
  '--gt-csv', $GtCsv,
  '--out', $OutDir,
  '--dataset-source', $DatasetSource,
  '--threshold', [string]$Threshold,
  '--min-area', [string]$MinArea,
  '--max-area', [string]$MaxArea,
  '--dilate-iters', [string]$DilateIters,
  '--max-gap', [string]$MaxGap,
  '--base-radius', [string]$BaseRadius,
  '--radius-per-side', [string]$RadiusPerSide,
  '--min-iou', [string]$MinIou,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--iou-threshold', [string]$IouThreshold,
  '--center-threshold', [string]$CenterThreshold,
  '--hard-tiny-side', [string]$HardTinySide,
  '--hard-low-score', [string]$HardLowScore,
  '--progress-every-sequences', [string]$ProgressEverySequences
)
if ($MaxImages -gt 0) { $argList += @('--max-images', [string]$MaxImages) }

$env:PYTHONPATH = $RepoRoot
$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "list_files=$($ListFiles -join ';')",
  "gt_csv=$GtCsv",
  "frame_root=$FrameRoot",
  "frame_list_out=$FrameListOut",
  "max_frames_per_seq=$MaxFramesPerSeq",
  "recursive_frame_root=$RecursiveFrameRoot",
  "out_dir=$OutDir",
  "dataset_source=$DatasetSource",
  "max_images=$MaxImages",
  "threshold=$Threshold",
  "min_area=$MinArea",
  "max_area=$MaxArea",
  "dilate_iters=$DilateIters",
  "max_gap=$MaxGap",
  "base_radius=$BaseRadius",
  "radius_per_side=$RadiusPerSide",
  "min_iou=$MinIou",
  "min_tracklet_rows=$MinTrackletRows",
  "iou_threshold=$IouThreshold",
  "center_threshold=$CenterThreshold",
  "hard_tiny_side=$HardTinySide",
  "hard_low_score=$HardLowScore",
  "progress_every_sequences=$ProgressEverySequences",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached VATD temporal-saliency tracklet export.'
Get-Content $metaFile
