param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = 'python',
  [string]$DatasetRoot,
  [string]$TrackerModel = '',
  [string]$YoloWeights = '',
  [string]$YoloData = '',
  [string]$Out = 'runs\window_accuracy\papers\edtc_antiuav600',
  [string]$ResultsDir = '',
  [switch]$SkipTrack,
  [string]$ConfigName = 'urap_window_accuracy',
  [double]$Fps = 30,
  [double]$WindowSeconds = 3,
  [double]$Iou = 0.5,
  [int]$Threads = 32,
  [int]$NumGpus = 8,
  [string]$Device = '0',
  [double]$SearchAreaScale = 4.55,
  [string]$RunId = 'edtc_antiuav600'
)

$ErrorActionPreference = 'Stop'

if (-not $DatasetRoot) { throw 'DatasetRoot is required.' }
if (-not $SkipTrack -and (-not $TrackerModel -or -not $YoloWeights -or -not $YoloData)) {
  throw 'TrackerModel, YoloWeights, and YoloData are required unless -SkipTrack is set.'
}
if ($SkipTrack -and -not $ResultsDir) {
  throw 'ResultsDir is required when -SkipTrack is set.'
}

$repo = (Resolve-Path $RepoRoot).Path
$runner = Join-Path $repo 'tools\run_edtc_tracker_window_accuracy.py'
if (-not (Test-Path -Path $runner -PathType Leaf)) { throw "Runner not found: $runner" }

$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
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

$listFile = Join-Path $DatasetRoot 'list.txt'
$totalSeq = 0
if (Test-Path -Path $listFile -PathType Leaf) {
  $totalSeq = @((Get-Content $listFile | Where-Object { $_.Trim() })).Count
}

$argList = @(
  $runner,
  '--dataset-root', $DatasetRoot,
  '--out', $outAbs,
  '--config-name', $ConfigName,
  '--fps', "$Fps",
  '--window-seconds', "$WindowSeconds",
  '--iou', "$Iou",
  '--threads', "$Threads",
  '--num-gpus', "$NumGpus",
  '--device', $Device,
  '--search-area-scale', "$SearchAreaScale"
)
if ($SkipTrack) {
  $argList += @('--skip-track', '--results-dir', $ResultsDir)
} else {
  $argList += @(
    '--tracker-model', $TrackerModel,
    '--yolo-weights', $YoloWeights,
    '--yolo-data', $YoloData
  )
}

$resolvedResultsDir = if ($ResultsDir) {
  if ([System.IO.Path]::IsPathRooted($ResultsDir)) { $ResultsDir } else { Join-Path $repo $ResultsDir }
} else {
  Join-Path (Join-Path (Join-Path $outAbs 'tracking_results') 'uavtrack_eh') $ConfigName
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("{0}_{1}.out.log" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("{0}_{1}.err.log" -f $RunId, $ts)

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
  run_id = $RunId
  dataset_root = $DatasetRoot
  total_sequences = $totalSeq
  output_root = $outAbs
  results_dir = $resolvedResultsDir
  runner = $runner
  stdout_log = $stdout
  stderr_log = $stderr
  command = "$PythonExe $($argList -join ' ')"
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'RUNNING'
Write-Host ("done/total: 0/{0}" -f $totalSeq)
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Write-Host 'last_completed_unit: launched EDTC tracker/window-accuracy runner'
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
Write-Host ("output_root: {0}" -f $outAbs)
