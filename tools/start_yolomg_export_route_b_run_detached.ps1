param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$ImagesList,
  [string]$PredLabelDir,
  [string]$OutRunRoot,
  [Nullable[int]]$ImageWidth = 1920,
  [Nullable[int]]$ImageHeight = 1080,
  [string]$Profile = 'hard_recovery',
  [string]$DiagnosticsName = 'diagnostics_raw.jsonl',
  [string]$Source = 'yolomg_lowconf',
  [int]$MaxImages = 0,
  [string]$RunId = 'yolomg_export_route_b_run',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\yolomg_action\route_b_export_detached')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "ImagesList not found: $ImagesList" }
if (-not (Test-Path -Path $PredLabelDir -PathType Container)) { throw "PredLabelDir not found: $PredLabelDir" }
if (-not $OutRunRoot) { throw 'OutRunRoot must be provided' }
if (($null -eq $ImageWidth) -ne ($null -eq $ImageHeight)) { throw 'ImageWidth and ImageHeight must be provided together' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutRunRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*export-yolo-predictions-route-b-run*') {
      Write-Host "YOLOMG Route-B export already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$argList = @(
  '-m', 'qstr_dronedet.cli', 'export-yolo-predictions-route-b-run',
  '--list-files', $ImagesList,
  '--pred-label-dir', $PredLabelDir,
  '--out-run-root', $OutRunRoot,
  '--profile', $Profile,
  '--diagnostics-name', $DiagnosticsName,
  '--source', $Source
)
if ($null -ne $ImageWidth) {
  $argList += @('--image-width', [string]$ImageWidth, '--image-height', [string]$ImageHeight)
}
if ($MaxImages -gt 0) {
  $argList += @('--max-images', [string]$MaxImages)
}

$env:PYTHONPATH = $RepoRoot
$proc = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Encoding ascii -Path $pidFile
@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $proc.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('images_list={0}' -f $ImagesList),
  ('pred_label_dir={0}' -f $PredLabelDir),
  ('out_run_root={0}' -f $OutRunRoot),
  ('image_width={0}' -f $ImageWidth),
  ('image_height={0}' -f $ImageHeight),
  ('profile={0}' -f $Profile),
  ('diagnostics_name={0}' -f $DiagnosticsName),
  ('source={0}' -f $Source),
  ('max_images={0}' -f $MaxImages),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG Route-B prediction export.'
Get-Content $metaFile
