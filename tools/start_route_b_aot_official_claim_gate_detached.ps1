param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$SummaryGlob,
  [string]$OutCsv,
  [string]$OutJson,
  [string]$RunId = 'route_b_aot_official_claim_gate',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_official_claim_gate_runner'),
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 0
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not $SummaryGlob) { throw 'SummaryGlob must be provided' }
if (-not $OutCsv) { throw 'OutCsv must be provided' }
if (-not $OutJson) { throw 'OutJson must be provided' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"
$runnerFile = Join-Path $OutputRoot "$RunId.runner.ps1"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$RunId.runner.ps1*") {
      Write-Host "AOT official claim gate watcher already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

foreach ($path in @($OutCsv, $OutJson)) {
  $parent = Split-Path -Parent $path
  if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"

$runner = @"
param()
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoRoot'
`$env:PYTHONPATH = '$RepoRoot'
`$summaryGlob = '$SummaryGlob'
`$outCsv = '$OutCsv'
`$outJson = '$OutJson'
`$pollSeconds = $PollSeconds
`$timeoutMinutes = $TimeoutMinutes
`$started = Get-Date

function Write-Event([string]`$Kind, [hashtable]`$Extra) {
  `$row = [ordered]@{ kind = `$Kind; time = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
  foreach (`$key in `$Extra.Keys) { `$row[`$key] = `$Extra[`$key] }
  Write-Host (`$row | ConvertTo-Json -Compress)
}

Write-Event 'aot_official_claim_gate_wait_start' @{ summary_glob = `$summaryGlob; out_json = `$outJson }
while (`$true) {
  `$matches = @(Get-ChildItem -Path `$summaryGlob -File -ErrorAction SilentlyContinue)
  if (`$matches.Count -gt 0) {
    Write-Event 'aot_official_claim_gate_collect' @{ matches = `$matches.Count; latest = `$matches[0].FullName }
    & '$Python' tools\collect_vatd_aot_official_results.py --summary-glob `$summaryGlob --out-csv `$outCsv --out-json `$outJson
    Write-Event 'aot_official_claim_gate_done' @{ out_json = `$outJson; claim_gate_json = ([System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath(`$outCsv)), ([System.IO.Path]::GetFileNameWithoutExtension(`$outCsv) + '_claim_gate.json'))) }
    break
  }
  if (`$timeoutMinutes -gt 0 -and ((Get-Date) - `$started).TotalMinutes -ge `$timeoutMinutes) {
    Write-Event 'aot_official_claim_gate_timeout' @{ summary_glob = `$summaryGlob; timeout_minutes = `$timeoutMinutes }
    exit 2
  }
  Write-Event 'aot_official_claim_gate_wait' @{ summary_glob = `$summaryGlob; matches = 0 }
  Start-Sleep -Seconds `$pollSeconds
}
"@
$runner | Set-Content -Path $runnerFile -Encoding utf8

$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerFile) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "python=$Python",
  "summary_glob=$SummaryGlob",
  "out_csv=$OutCsv",
  "out_json=$OutJson",
  "output_root=$OutputRoot",
  "poll_seconds=$PollSeconds",
  "timeout_minutes=$TimeoutMinutes",
  "runner_file=$runnerFile",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached AOT official claim gate watcher.'
Get-Content $metaFile
