param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$TransVisDroneRepo = (Join-Path $RepoRoot 'papers\TransVisDrone'),
  [string]$PythonExe = (Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'),
  [string]$DataYaml = (Join-Path $RepoRoot 'papers\TransVisDrone\data\AOTTestSplits_URAP\AOTTest_0.yaml'),
  [string]$Weights = (Join-Path $RepoRoot 'papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\AOT\image_size_1280_YOLOXL_3_frames_AOT_with_yolo_weights_end_to_end\weights\best.pt'),
  [string]$Project = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_part0_tvd_val'),
  [string]$Name = 'tvd_aot_part0_conf0p2',
  [int]$Img = 1280,
  [int]$BatchSize = 2,
  [int]$NumFrames = 3,
  [double]$ConfThres = 0.2,
  [double]$IouThres = 0.6,
  [string]$Device = '',
  [switch]$NoHalf,
  [switch]$UseIouTracker,
  [double]$TrackIouThres = 0.3,
  [int]$TrackMaxAge = 2,
  [string]$RunId = 'aot_part0_tvd_val',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_part0_tvd_val_runner')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $TransVisDroneRepo -PathType Container)) { throw "TransVisDroneRepo not found: $TransVisDroneRepo" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $DataYaml -PathType Leaf)) { throw "DataYaml not found: $DataYaml" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Weights not found: $Weights" }

$DataYaml = (Resolve-Path $DataYaml).Path
$Weights = (Resolve-Path $Weights).Path
if (-not [System.IO.Path]::IsPathRooted($Project)) { $Project = Join-Path (Get-Location) $Project }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path (Get-Location) $OutputRoot }
$Project = [System.IO.Path]::GetFullPath($Project)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $Project | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*val.py*' -and $existing.CommandLine -like '*--save-aot-predictions*') {
      Write-Host "Route B AOT TVD inference already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 160 }
      exit 0
    }
  }
}

function ConvertTo-WindowsArgumentString([string[]]$Values) {
  $quoted = foreach ($arg in $Values) {
    if ($arg -match '[\s"]') {
      $escaped = $arg -replace '"', '\"'
      '"' + $escaped + '"'
    } else {
      $arg
    }
  }
  return ($quoted -join ' ')
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '.\val.py',
  '--task', 'test',
  '--data', $DataYaml,
  '--weights', $Weights,
  '--img', [string]$Img,
  '--batch-size', [string]$BatchSize,
  '--num-frames', [string]$NumFrames,
  '--conf-thres', [string]$ConfThres,
  '--iou-thres', [string]$IouThres,
  '--save-aot-predictions',
  '--project', $Project,
  '--name', $Name,
  '--exist-ok'
)
if (-not $NoHalf) { $argList += '--half' }
if ($Device) { $argList += @('--device', $Device) }
if ($UseIouTracker) {
  $argList += @(
    '--pp-use-iou-tracker',
    '--pp-track-iou-thres', [string]$TrackIouThres,
    '--pp-track-max-age', [string]$TrackMaxAge
  )
}
$argumentString = ConvertTo-WindowsArgumentString -Values $argList

$process = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argumentString `
  -WorkingDirectory $TransVisDroneRepo `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$process.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $process.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('transvisdrone_repo={0}' -f $TransVisDroneRepo),
  ('data_yaml={0}' -f $DataYaml),
  ('weights={0}' -f $Weights),
  ('project={0}' -f $Project),
  ('name={0}' -f $Name),
  ('save_dir={0}' -f (Join-Path $Project $Name)),
  ('prediction_part={0}' -f (Join-Path (Join-Path $Project $Name) 'aotpredictions\predictions_split_0.pkl')),
  ('img={0}' -f $Img),
  ('batch_size={0}' -f $BatchSize),
  ('num_frames={0}' -f $NumFrames),
  ('conf_thres={0}' -f $ConfThres),
  ('iou_thres={0}' -f $IouThres),
  ('device={0}' -f $Device),
  ('half={0}' -f (-not [bool]$NoHalf)),
  ('use_iou_tracker={0}' -f [bool]$UseIouTracker),
  ('output_root={0}' -f $OutputRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B AOT TVD inference.'
Get-Content $metaFile
