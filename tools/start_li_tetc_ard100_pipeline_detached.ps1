param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\URAP-UAV-to-UAV-Detection-and-Tracking')).Path,
  [string]$ProjectRoot = '',
  [string]$ARDRoot = 'D:\URAP_datasets\ARD100',
  [string]$ARDSplitRoot = 'D:\URAP_datasets\TransVisDrone\ARD100\Videos',
  [int]$Epochs = 5,
  [int]$BatchSize = 2,
  [int]$FrameStride = 3,
  [int]$MaxFramesPerVideo = 0,
  [int]$EmptyStride = 10,
  [double]$LR = 0.002,
  [int]$MinSize = 1080,
  [int]$MaxSize = 1920,
  [int]$NumWorkers = 0,
  [string]$AnchorPreset = 'tiny',
  [string]$RunId = 'ard100_pipeline',
  [string]$Checkpoint = '',
  [string]$ValJson = '',
  [string]$TestJson = '',
  [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not $ProjectRoot) { $ProjectRoot = Join-Path $RepoRoot 'baselines\li_tetc_pt_pipeline' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'baselines\li_tetc_pt_pipeline\runs\detached_ard100_pipeline' }
if (-not $Checkpoint) { $Checkpoint = Join-Path $ProjectRoot 'runs\fasterrcnn_ard100_v1.pt' }
if (-not $ValJson) { $ValJson = Join-Path $ProjectRoot 'runs\eval_ard100_val.json' }
if (-not $TestJson) { $TestJson = Join-Path $ProjectRoot 'runs\eval_ard100_test.json' }
if (-not (Test-Path -Path $ProjectRoot -PathType Container)) { throw "ProjectRoot not found: $ProjectRoot" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.json" -f $RunId)
$stateFile = Join-Path $OutputRoot ("runner_{0}_state.json" -f $RunId)

if (Test-Path $pidFile) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing) { throw "Runner already active with pid=$existingPid" }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_pipeline_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_pipeline_{0}_{1}.err.txt" -f $RunId, $ts)
$script = Join-Path $PSScriptRoot 'run_li_tetc_ard100_pipeline.ps1'
$args = @(
  '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script,
  '-RepoRoot', $RepoRoot,
  '-ProjectRoot', $ProjectRoot,
  '-ARDRoot', $ARDRoot,
  '-ARDSplitRoot', $ARDSplitRoot,
  '-Epochs', "$Epochs",
  '-BatchSize', "$BatchSize",
  '-FrameStride', "$FrameStride",
  '-MaxFramesPerVideo', "$MaxFramesPerVideo",
  '-EmptyStride', "$EmptyStride",
  '-LR', "$LR",
  '-MinSize', "$MinSize",
  '-MaxSize', "$MaxSize",
  '-NumWorkers', "$NumWorkers",
  '-AnchorPreset', $AnchorPreset,
  '-Checkpoint', $Checkpoint,
  '-ValJson', $ValJson,
  '-TestJson', $TestJson,
  '-StateFile', $stateFile
)

$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
@{
  pid = $proc.Id
  start_time = (Get-Date).ToString('s')
  repo_root = $RepoRoot
  project_root = $ProjectRoot
  run_id = $RunId
  checkpoint = $Checkpoint
  val_json = $ValJson
  test_json = $TestJson
  state_file = $stateFile
  stdout_log = $stdout
  stderr_log = $stderr
  argv = ($args -join ' ')
} | ConvertTo-Json | Set-Content -Path $metaFile -Encoding UTF8

Write-Host 'RUNNING'
Write-Host 'done/total: 0/3'
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f ((Get-Date).ToString('s')))
Write-Host 'last_completed_unit: queued'
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
