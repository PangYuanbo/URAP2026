param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = 'python',
  [ValidateSet('yolomg', 'transvisdrone', 'esod', 'edtc')]
  [string]$Method = 'yolomg',
  [string]$PaperRepo = '',
  [string]$Data = '',
  [string]$Weights = '',
  [string]$Gt,
  [string]$GtFormat = 'yolo-dir',
  [string]$Out = 'runs\window_accuracy\detached_yolo_eval',
  [double]$Fps = 30,
  [double]$WindowSeconds = 3,
  [double]$MatchIou = 0.5,
  [double]$ScoreThreshold = 0.25,
  [int]$GtFrameOffset = 0,
  [int]$PredFrameOffset = 0,
  [string]$FrameManifest = '',
  [string]$FrameManifestFormat = '',
  [int]$FrameManifestOffset = 0,
  [Nullable[double]]$ImgWidth = $null,
  [Nullable[double]]$ImgHeight = $null,
  [string]$Name = 'window_accuracy_eval',
  [string]$Project = '',
  [string]$Task = 'val',
  [int]$Img = 1280,
  [int]$BatchSize = 1,
  [string]$Device = '0',
  [int]$NumFrames = 5,
  [switch]$Half,
  [switch]$Augment,
  [switch]$SkipEval,
  [string]$PredLabelsDir = '',
  [string[]]$ExtraEvalArg = @(),
  [string]$RunId = 'paper_window_accuracy'
)

$ErrorActionPreference = 'Stop'

if (-not $Gt) { throw 'Gt is required.' }
if (-not $SkipEval -and (-not $Data -or -not $Weights)) {
  throw 'Data and Weights are required unless -SkipEval is set.'
}
if ($SkipEval -and -not $PredLabelsDir) {
  throw 'PredLabelsDir is required when -SkipEval is set.'
}
if (($null -eq $ImgWidth) -ne ($null -eq $ImgHeight)) {
  throw 'ImgWidth and ImgHeight must be provided together.'
}

$repo = (Resolve-Path $RepoRoot).Path
$runner = Join-Path $repo 'tools\run_yolo_eval_window_accuracy.py'
if (-not (Test-Path -Path $runner -PathType Leaf)) { throw "Runner not found: $runner" }

$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$projectAbs = if ($Project) {
  if ([System.IO.Path]::IsPathRooted($Project)) { $Project } else { Join-Path $repo $Project }
} else {
  Join-Path $outAbs 'eval'
}
$labelsDir = if ($PredLabelsDir) {
  if ([System.IO.Path]::IsPathRooted($PredLabelsDir)) { $PredLabelsDir } else { Join-Path $repo $PredLabelsDir }
} else {
  Join-Path (Join-Path $projectAbs $Name) 'labels'
}

New-Item -ItemType Directory -Force -Path $outAbs | Out-Null
$logsDir = Join-Path $outAbs 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $outAbs ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $outAbs ("{0}_meta.json" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
      Write-Host ("RUNNING pid={0}" -f $existingPid)
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("{0}_{1}.out.log" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("{0}_{1}.err.log" -f $RunId, $ts)

$argList = @(
  $runner,
  '--method', $Method,
  '--gt', $Gt,
  '--gt-format', $GtFormat,
  '--out', $outAbs,
  '--project', $projectAbs,
  '--name', $Name,
  '--task', $Task,
  '--img', "$Img",
  '--batch-size', "$BatchSize",
  '--device', $Device,
  '--num-frames', "$NumFrames",
  '--fps', "$Fps",
  '--window-seconds', "$WindowSeconds",
  '--match-iou', "$MatchIou",
  '--score-threshold', "$ScoreThreshold",
  '--gt-frame-offset', "$GtFrameOffset",
  '--pred-frame-offset', "$PredFrameOffset",
  '--frame-manifest-offset', "$FrameManifestOffset"
)
if ($PaperRepo) { $argList += @('--repo', $PaperRepo) }
if ($Data) { $argList += @('--data', $Data) }
if ($Weights) { $argList += @('--weights', $Weights) }
if ($SkipEval) { $argList += @('--skip-eval', '--pred-labels-dir', $labelsDir) }
if ($FrameManifest) { $argList += @('--frame-manifest', $FrameManifest) }
if ($FrameManifestFormat) { $argList += @('--frame-manifest-format', $FrameManifestFormat) }
if ($Half) { $argList += '--half' }
if ($Augment) { $argList += '--augment' }
if ($null -ne $ImgWidth) { $argList += @('--img-width', "$ImgWidth", '--img-height', "$ImgHeight") }
foreach ($extra in $ExtraEvalArg) {
  if ($extra) { $argList += @('--extra-eval-arg', $extra) }
}

$proc = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argList `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Encoding ascii -Path $pidFile -Value $proc.Id
@{
  start_time = (Get-Date).ToString('o')
  pid = $proc.Id
  method = $Method
  run_id = $RunId
  stdout_log = $stdout
  stderr_log = $stderr
  output_root = $outAbs
  project = $projectAbs
  prediction_labels_dir = $labelsDir
  runner = $runner
  command = "$PythonExe $($argList -join ' ')"
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'RUNNING'
Write-Host 'done/total: 0/2'
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Write-Host 'last_completed_unit: launched paper eval/window-accuracy runner'
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
Write-Host ("output_root: {0}" -f $outAbs)
Write-Host ("prediction_labels_dir: {0}" -f $labelsDir)
