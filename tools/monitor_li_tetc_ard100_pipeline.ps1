param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\URAP-UAV-to-UAV-Detection-and-Tracking')).Path,
  [string]$RunId = 'ard100_pipeline',
  [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'baselines\li_tetc_pt_pipeline\runs\detached_ard100_pipeline' }
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.json" -f $RunId)
$stateFile = Join-Path $OutputRoot ("runner_{0}_state.json" -f $RunId)

if (-not (Test-Path $pidFile)) {
  Write-Host 'NOT RUNNING'
  exit 0
}

$procId = [int](Get-Content $pidFile | Select-Object -First 1)
$meta = if (Test-Path $metaFile) { Get-Content $metaFile -Raw | ConvertFrom-Json } else { $null }
$state = if (Test-Path $stateFile) { Get-Content $stateFile -Raw | ConvertFrom-Json } else { $null }
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue

if (-not $proc) {
  Write-Host 'NOT RUNNING'
  if ($state) {
    Write-Host ("done/total: {0}/{1}" -f $state.stage_index, $state.stage_total)
    Write-Host ("last_completed_unit: {0}" -f $state.last_completed_unit)
    Write-Host ("state: {0}" -f $state.status)
  }
  Write-Host ("pid: {0}" -f $procId)
  if ($meta) { Write-Host ("start_time: {0}" -f $meta.start_time) }
  if ($meta -and (Test-Path $meta.stdout_log)) { Write-Host ("last_output: {0}" -f ((Get-Item $meta.stdout_log).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))) }
  if ($meta) {
    Write-Host ("stdout: {0}" -f $meta.stdout_log)
    Write-Host ("stderr: {0}" -f $meta.stderr_log)
  }
  exit 0
}

$lastUnit = ''
$activeStdout = if ($state) { [string]$state.current_stdout } else { '' }
$activeStderr = if ($state) { [string]$state.current_stderr } else { '' }
if ($state) {
  if ($activeStderr -and (Test-Path $activeStderr)) {
    if ($state.stage_name -eq 'train') {
      $epochLine = Select-String -Path $activeStderr -Pattern 'epoch\s+([0-9]+)/([0-9]+)' | Select-Object -Last 1
      if ($epochLine) { $lastUnit = $epochLine.Line.Trim() }
    } elseif ($state.stage_name -eq 'val' -or $state.stage_name -eq 'test') {
      $evalLine = Select-String -Path $activeStderr -Pattern 'eval:\s+.*?(\d+)/(\d+)' | Select-Object -Last 1
      if ($evalLine) { $lastUnit = $evalLine.Line.Trim() }
    }
  }
  if (-not $lastUnit) { $lastUnit = $state.last_completed_unit }
}

Write-Host 'RUNNING'
if ($state) {
  Write-Host ("done/total: {0}/{1}" -f $state.stage_index, $state.stage_total)
  Write-Host ("last_completed_unit: {0}" -f $lastUnit)
  Write-Host ("state: {0}" -f $state.stage_name)
  Write-Host ("status: {0}" -f $state.status)
} else {
  Write-Host 'done/total: staged/3'
}
Write-Host ("pid: {0}" -f $procId)
if ($meta) { Write-Host ("start_time: {0}" -f $meta.start_time) }
if ($activeStderr -and (Test-Path $activeStderr)) {
  Write-Host ("last_output: {0}" -f ((Get-Item $activeStderr).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))
} elseif ($activeStdout -and (Test-Path $activeStdout)) {
  Write-Host ("last_output: {0}" -f ((Get-Item $activeStdout).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))
} elseif ($meta -and (Test-Path $meta.stdout_log)) {
  Write-Host ("last_output: {0}" -f ((Get-Item $meta.stdout_log).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))
}
if ($meta) {
  Write-Host ("stdout: {0}" -f $meta.stdout_log)
  Write-Host ("stderr: {0}" -f $meta.stderr_log)
}
if ($state) {
  if ($state.checkpoint) { Write-Host ("checkpoint: {0}" -f $state.checkpoint) }
  if ($state.val_json) { Write-Host ("val_json: {0}" -f $state.val_json) }
  if ($state.test_json) { Write-Host ("test_json: {0}" -f $state.test_json) }
  if ($state.current_stdout) { Write-Host ("current_stdout: {0}" -f $state.current_stdout) }
  if ($state.current_stderr) { Write-Host ("current_stderr: {0}" -f $state.current_stderr) }
}
