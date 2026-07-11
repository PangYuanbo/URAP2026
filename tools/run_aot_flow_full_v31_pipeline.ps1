param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v31_pipeline' }
New-Item -ItemType Directory -Force -Path $OutputRoot, (Join-Path $OutputRoot 'logs') | Out-Null
$progressPath = Join-Path $OutputRoot 'progress.json'
function Write-ProgressFile($phase, $done, $total, $extra = @{}) {
  $value = @{phase=$phase; done=$done; total=$total; updated=(Get-Date).ToString('o')}
  foreach ($key in $extra.Keys) { $value[$key] = $extra[$key] }
  $value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $progressPath -Encoding utf8
}
function Run-Logged($name, $exe, $arguments, $working) {
  $stdout = Join-Path $OutputRoot "logs\${name}.out.txt"
  $stderr = Join-Path $OutputRoot "logs\${name}.err.txt"
  $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $working -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "$name failed exit=$($process.ExitCode) stderr=$stderr" }
}
$flowRoot = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v30'
$flowSummary = Join-Path $flowRoot 'flow_recovery_summary.json'
$flowRunner = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v30_runner'
$flowPidFile = Join-Path $flowRunner 'aot_flow_full_v30_pid.txt'
$flowProgress = Join-Path $flowRunner 'aot_flow_full_v30_progress.json'
Write-ProgressFile 'startup' 0 3
while (-not (Test-Path -LiteralPath $flowSummary)) {
  $pidValue = if (Test-Path -LiteralPath $flowPidFile) { Get-Content -LiteralPath $flowPidFile | Select-Object -First 1 } else { '' }
  $native = if ($pidValue -match '^\d+$') { Get-Process -Id $pidValue -ErrorAction SilentlyContinue } else { $null }
  if (-not $native) { throw 'Flow v30 is NOT RUNNING and summary is missing' }
  $flow = @{done=0; total=440; last_completed_unit='none'}
  if (Test-Path -LiteralPath $flowProgress) { $flow = Get-Content -LiteralPath $flowProgress -Raw | ConvertFrom-Json }
  Write-ProgressFile 'wait_flow_v30' ([int]$flow.done) ([int]$flow.total) @{flow_pid=$pidValue; last_completed_unit=$flow.last_completed_unit}
  Start-Sleep -Seconds 30
}
Write-ProgressFile 'apply_fixed_gate' 1 3
$python = Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'
$gateOut = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v31_fixed'
Run-Logged 'apply_fixed_gate' $python @(
  (Join-Path $RepoRoot 'tools\aot_action_bank_train_apply_fixed_gate.py'),
  '--train-predictions', (Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_part0_v27\aotpredictions\predictions_split_0.pkl'),
  '--train-candidate-match-folder', (Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_part0_v27\official_eval\result\result_metrics_min_track_len_0'),
  '--train-baseline-match-folder', (Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_chunk_transfer_v1\val_baseline_eval\result\result_metrics_min_track_len_0'),
  '--target-predictions', (Join-Path $flowRoot 'aotpredictions\predictions_split_0.pkl'),
  '--out-dir', $gateOut,
  '--candidate-threshold', '0.05',
  '--base-threshold', '0.1'
) $RepoRoot
Write-ProgressFile 'official_eval' 2 3
$tvdRepo = Join-Path $RepoRoot 'papers\TransVisDrone'
$evalOut = Join-Path $gateOut 'official_eval'
Run-Logged 'official_eval' $python @(
  '.\evaluate_aot.py',
  '--results_folder', (Join-Path $gateOut 'aotpredictions'),
  '--evaluation_folder', $evalOut,
  '--detection_threshold', '0.2',
  '--dataset-path', 'U:\URAP_datasets\AOT\part1'
) $tvdRepo
$summaryFile = Get-ChildItem -LiteralPath (Join-Path $evalOut 'summaries') -Filter 'result_metrics*_summary*.json' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $summaryFile) { throw 'Official summary missing' }
$metrics = Get-Content -LiteralPath $summaryFile.FullName -Raw | ConvertFrom-Json
$baseline = @{afdr=0.8685312193818473; fppi=0.262303510022747; edr300=0.9257142857142857}
$gain = 100 * ([double]$metrics.fl_dr_in_range - $baseline.afdr)
$result = @{
  protocol='part0-selected fixed full AOT camera-compensated Action Bank'
  baseline=$baseline
  full=@{afdr=[double]$metrics.fl_dr_in_range; fppi=[double]$metrics.fppi; far=[double]$metrics.far; summary=$summaryFile.FullName}
  afdr_gain_points=$gain
  fppi_change=[double]$metrics.fppi - $baseline.fppi
  target_met=($gain -ge 3 -and $gain -le 5 -and [double]$metrics.fppi -le $baseline.fppi)
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $gateOut 'official_comparison.json') -Encoding utf8
Write-ProgressFile 'done' 3 3 @{result=$result}
