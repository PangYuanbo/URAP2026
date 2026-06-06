param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$AotGate = 'artifacts\route_b_official\aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605\aot_official_claim_comparison_claim_gate.json',
  [string]$NpsGate = 'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full_comparison_claim_gate.json',
  [string]$OutJson = 'artifacts\vatd_claim_summary_final_20260605.json',
  [string]$OutMd = 'artifacts\vatd_claim_summary_final_20260605.md',
  [string]$RunId = 'vatd_claim_summary_final_20260605',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_claim_summary_final_runner_20260605'),
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 0
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not $AotGate) { throw 'AotGate must be provided' }
if (-not $NpsGate) { throw 'NpsGate must be provided' }
if (-not $OutJson) { throw 'OutJson must be provided' }
if (-not $OutMd) { throw 'OutMd must be provided' }

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
      Write-Host "VATD claim summary watcher already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

foreach ($path in @($OutJson, $OutMd)) {
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
`$aotGate = '$AotGate'
`$npsGate = '$NpsGate'
`$outJson = '$OutJson'
`$outMd = '$OutMd'
`$pollSeconds = $PollSeconds
`$timeoutMinutes = $TimeoutMinutes
`$started = Get-Date

function Write-Event([string]`$Kind, [hashtable]`$Extra) {
  `$row = [ordered]@{ kind = `$Kind; time = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
  foreach (`$key in `$Extra.Keys) { `$row[`$key] = `$Extra[`$key] }
  Write-Host (`$row | ConvertTo-Json -Compress)
}

Write-Event 'vatd_claim_summary_wait_start' @{ aot_gate = `$aotGate; nps_gate = `$npsGate; out_json = `$outJson }
while (`$true) {
  `$aotReady = Test-Path `$aotGate
  `$npsReady = Test-Path `$npsGate
  if (`$aotReady -and `$npsReady) {
    Write-Event 'vatd_claim_summary_collect' @{ aot_gate = `$aotGate; nps_gate = `$npsGate }
    & '$Python' tools\collect_vatd_claim_summary.py --gate aot `$aotGate --gate nps `$npsGate --required aot nps --out-json `$outJson --out-md `$outMd
    Write-Event 'vatd_claim_summary_done' @{ out_json = `$outJson; out_md = `$outMd }
    break
  }
  if (`$timeoutMinutes -gt 0 -and ((Get-Date) - `$started).TotalMinutes -ge `$timeoutMinutes) {
    Write-Event 'vatd_claim_summary_timeout' @{ aot_ready = `$aotReady; nps_ready = `$npsReady; timeout_minutes = `$timeoutMinutes }
    exit 2
  }
  Write-Event 'vatd_claim_summary_wait' @{ aot_ready = `$aotReady; nps_ready = `$npsReady }
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
  "aot_gate=$AotGate",
  "nps_gate=$NpsGate",
  "out_json=$OutJson",
  "out_md=$OutMd",
  "output_root=$OutputRoot",
  "poll_seconds=$PollSeconds",
  "timeout_minutes=$TimeoutMinutes",
  "runner_file=$runnerFile",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached VATD claim summary watcher.'
Get-Content $metaFile
