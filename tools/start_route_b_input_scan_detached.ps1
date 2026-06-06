param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$ScanRoots = @(),
  [string]$Out = (Join-Path $RepoRoot 'artifacts\route_b_official\proposal_input_scan.json'),
  [string[]]$Profiles = @('hard_recovery'),
  [string[]]$DiagnosticsNames = @('diagnostics_raw.jsonl', 'diagnostics.jsonl'),
  [int]$MaxDepth = 8,
  [int]$MaxFiles = 20000,
  [int]$MaxDiagSampleFiles = 20,
  [int]$MaxRowsPerDiagFile = 200,
  [Nullable[int]]$MaxFrames = $null,
  [string]$RunId = 'route_b_input_scan',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_input_scan')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if ($ScanRoots.Count -lt 1) { throw 'ScanRoots must contain at least one root' }
foreach ($path in $ScanRoots) {
  if (-not (Test-Path -Path $path -PathType Container)) { throw "Scan root not found: $path" }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*scan-route-b-proposal-inputs*') {
      Write-Host "Route B input scan already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '-m', 'qstr_dronedet.cli', 'scan-route-b-proposal-inputs',
  '--scan-roots'
)
$argList += $ScanRoots
$argList += @(
  '--out', $Out,
  '--profiles'
)
$argList += $Profiles
$argList += '--diagnostics-names'
$argList += $DiagnosticsNames
$argList += @(
  '--max-depth', [string]$MaxDepth,
  '--max-files', [string]$MaxFiles,
  '--max-diag-sample-files', [string]$MaxDiagSampleFiles,
  '--max-rows-per-diag-file', [string]$MaxRowsPerDiagFile
)
if ($null -ne $MaxFrames) { $argList += @('--max-frames', [string]$MaxFrames) }

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

$argumentString = ConvertTo-WindowsArgumentString -Values $argList

$process = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argumentString `
  -WorkingDirectory $RepoRoot `
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
  ('output_root={0}' -f $OutputRoot),
  ('scan_roots={0}' -f ($ScanRoots -join ';')),
  ('out={0}' -f $Out),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('max_depth={0}' -f $MaxDepth),
  ('max_files={0}' -f $MaxFiles),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B input scan.'
Get-Content $metaFile
