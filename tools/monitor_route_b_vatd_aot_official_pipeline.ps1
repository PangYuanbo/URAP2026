param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'route_b_vatd_aot_official',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_fulltest_vatd_motion_action_official_runner'),
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path $metaFile)) {
  Write-Host "Meta file not found: $metaFile"
  exit 0
}

$metaLines = Get-Content $metaFile
Write-Host '== VATD Pipeline Meta =='
$metaLines | Select-Object -First 160

function Get-MetaValue([string]$Key) {
  $line = ($metaLines | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1)
  if ($line) { return $line.Substring($Key.Length + 1) }
  return $null
}

$attachedTracklets = Get-MetaValue 'attached_tracklets'
$rescoreDir = Get-MetaValue 'rescore_dir'
$evalRunnerDir = Get-MetaValue 'eval_runner_dir'
$claimGateRunnerDir = Get-MetaValue 'claim_gate_runner_dir'
$attachStdout = Get-MetaValue 'attach_stdout'
$attachStderr = Get-MetaValue 'attach_stderr'
$rescoreStdout = Get-MetaValue 'rescore_stdout'
$rescoreStderr = Get-MetaValue 'rescore_stderr'

Write-Host ''
Write-Host '== Attach / Rescore Outputs =='
if ($attachedTracklets -and (Test-Path $attachedTracklets)) {
  $attachedLines = (Get-Content $attachedTracklets | Measure-Object -Line).Lines
  Write-Host "attached_tracklet_lines=$attachedLines"
  $summary = "$attachedTracklets.summary.json"
  if (Test-Path $summary) {
    Write-Host "attach_summary=$summary"
    try { Get-Content $summary -Raw | ConvertFrom-Json | ConvertTo-Json -Compress } catch {}
  }
} else {
  Write-Host "attached_tracklets_missing=$attachedTracklets"
}

if ($rescoreDir -and (Test-Path $rescoreDir)) {
  $rescoreSummary = Join-Path $rescoreDir 'aot_tracklet_rescore_summary.json'
  if (Test-Path $rescoreSummary) {
    Write-Host "rescore_summary=$rescoreSummary"
    try { Get-Content $rescoreSummary -Raw | ConvertFrom-Json | ConvertTo-Json -Compress } catch {}
  }
  $predDir = Join-Path $rescoreDir 'aotpredictions'
  if (Test-Path $predDir) {
    $parts = @(Get-ChildItem $predDir -Filter '*.pkl' -File)
    Write-Host "rescored_aot_prediction_parts=$($parts.Count)"
  }
} else {
  Write-Host "rescore_dir_missing=$rescoreDir"
}

Write-Host ''
Write-Host '== Official Eval =='
if ($evalRunnerDir -and (Test-Path $evalRunnerDir)) {
  & (Join-Path $PSScriptRoot 'monitor_route_b_aot_official_eval.ps1') -RepoRoot $RepoRoot -OutputRoot $evalRunnerDir -RunId $RunId -TailLines $TailLines
} else {
  Write-Host "eval_runner_dir_missing=$evalRunnerDir"
}

Write-Host ''
Write-Host '== Claim Gate =='
if ($claimGateRunnerDir -and (Test-Path $claimGateRunnerDir)) {
  & (Join-Path $PSScriptRoot 'monitor_route_b_aot_official_claim_gate.ps1') -OutputRoot $claimGateRunnerDir -RunId $RunId -TailLines $TailLines
} else {
  Write-Host "claim_gate_runner_dir_missing=$claimGateRunnerDir"
}

Write-Host ''
Write-Host '== Attach stdout tail =='
if ($attachStdout -and (Test-Path $attachStdout)) { Get-Content $attachStdout -Tail $TailLines }
Write-Host ''
Write-Host '== Attach stderr tail =='
if ($attachStderr -and (Test-Path $attachStderr)) { Get-Content $attachStderr -Tail $TailLines }
Write-Host ''
Write-Host '== Rescore stdout tail =='
if ($rescoreStdout -and (Test-Path $rescoreStdout)) { Get-Content $rescoreStdout -Tail $TailLines }
Write-Host ''
Write-Host '== Rescore stderr tail =='
if ($rescoreStderr -and (Test-Path $rescoreStderr)) { Get-Content $rescoreStderr -Tail $TailLines }
