param(
  [string]$Out = 'artifacts\esod_nps_motion_pipeline'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$pidFile = Join-Path $outAbs 'pipeline_pid.txt'
$metaFile = Join-Path $outAbs 'pipeline_meta.json'
$stateFile = Join-Path $outAbs 'pipeline_state.json'
$imageRoot = Join-Path $outAbs 'images\nps'
$labelRoot = Join-Path $outAbs 'labels\nps'

if (-not (Test-Path $pidFile)) {
  Write-Host 'NOT RUNNING'
  exit 0
}

$procId = [int](Get-Content $pidFile | Select-Object -First 1)
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
$meta = if (Test-Path $metaFile) { Get-Content $metaFile -Raw | ConvertFrom-Json } else { $null }
$state = if (Test-Path $stateFile) { Get-Content $stateFile -Raw | ConvertFrom-Json } else { $null }

if ($proc) { Write-Host 'RUNNING' } else { Write-Host 'NOT RUNNING' }

$stage = if ($state) { [string]$state.stage } else { 'launched' }
$stageIndex = switch ($stage) {
  'build_rois' { 0 }
  'train_esod' { 1 }
  'complete' { 2 }
  default { 0 }
}
Write-Host ("done/total: {0}/2" -f $stageIndex)
Write-Host ("state: {0}" -f $stage)

$lastUnit = ''
$patchCount = 0
$positiveCount = 0
$latestFile = $null
if (Test-Path $imageRoot) {
  $imgs = Get-ChildItem $imageRoot -Recurse -File -Filter '*.jpg' -ErrorAction SilentlyContinue
  $patchCount = @($imgs).Count
  $latestFile = $imgs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if (Test-Path $labelRoot) {
  $positiveCount = @(
    Get-ChildItem $labelRoot -Recurse -File -Filter '*.txt' -ErrorAction SilentlyContinue |
      Where-Object { $_.Length -gt 0 }
  ).Count
}
if ($state -and $state.current_stderr -and (Test-Path $state.current_stderr)) {
  if ($stage -eq 'train_esod') {
    $epochLine = Select-String -Path $state.current_stderr -Pattern '^\s*\d+/\d+' | Select-Object -Last 1
    if ($epochLine) { $lastUnit = $epochLine.Line.Trim() }
  }
}
if (-not $lastUnit -and $state -and $state.train_images) {
  $lastUnit = "train_images=$($state.train_images) val_images=$($state.val_images)"
}
if (-not $lastUnit -and $latestFile) {
  $lastUnit = "latest_patch=$($latestFile.Name) patches=$patchCount positive=$positiveCount"
}
if (-not $lastUnit) { $lastUnit = 'no progress line yet' }
Write-Host ("last_completed_unit: {0}" -f $lastUnit)
Write-Host ("pid: {0}" -f $procId)
if ($meta) {
  Write-Host ("start_time: {0}" -f $meta.start_time)
}

$lastOutput = $null
if ($state -and $state.current_stderr -and (Test-Path $state.current_stderr)) {
  $lastOutput = (Get-Item $state.current_stderr).LastWriteTime
} elseif ($state -and $state.current_stdout -and (Test-Path $state.current_stdout)) {
  $lastOutput = (Get-Item $state.current_stdout).LastWriteTime
} elseif ($meta -and (Test-Path $meta.stdout_log)) {
  $lastOutput = (Get-Item $meta.stdout_log).LastWriteTime
}
if ($lastOutput) { Write-Host ("last_output: {0}" -f $lastOutput.ToString('yyyy-MM-dd HH:mm:ss')) }

if ($meta) {
  Write-Host ("stdout: {0}" -f $meta.stdout_log)
  Write-Host ("stderr: {0}" -f $meta.stderr_log)
  Write-Host ("output_root: {0}" -f $meta.output_root)
}
if ($state) {
  if ($state.current_stdout) { Write-Host ("current_stdout: {0}" -f $state.current_stdout) }
  if ($state.current_stderr) { Write-Host ("current_stderr: {0}" -f $state.current_stderr) }
  if ($state.data_yaml) { Write-Host ("data_yaml: {0}" -f $state.data_yaml) }
}
