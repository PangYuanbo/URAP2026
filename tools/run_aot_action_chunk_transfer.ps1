param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'aot_action_chunk_transfer_v1'
)

$ErrorActionPreference = 'Stop'
$outRoot = Join-Path $RepoRoot "artifacts\route_b_official\$RunId"
$progressPath = Join-Path $outRoot 'progress.json'
$source = Join-Path $RepoRoot 'papers\TransVisDrone\runs\val\AOT_URAP\fulltest_conf0p2_wport_baseline\aotpredictions'
$tracklets = Join-Path $outRoot 'tracklets_with_action_chunk_scores.jsonl'
$qstrPython = Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$tvdRepo = Join-Path $RepoRoot 'papers\TransVisDrone'
$tvdPython = Join-Path $tvdRepo '.venv\Scripts\python.exe'
$dataset = 'D:\URAP_datasets\AOT\part1'
$betas = @(0.08, 0.12, 0.18)

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
function Progress([string]$phase, [int]$done, [int]$total, [hashtable]$extra = @{}) {
  $value = @{phase=$phase; done=$done; total=$total; updated=(Get-Date).ToString('o')}
  foreach ($key in $extra.Keys) { $value[$key] = $extra[$key] }
  $value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $progressPath -Encoding UTF8
}
function Run-Logged([string]$name, [string]$exe, [string[]]$arguments, [string]$workingDirectory) {
  $stdout = Join-Path $outRoot "$name.out.txt"
  $stderr = Join-Path $outRoot "$name.err.txt"
  $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $workingDirectory -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "$name failed with exit code $($process.ExitCode); stderr=$stderr" }
}
function Read-Metrics([string]$evaluationFolder) {
  $summary = Get-ChildItem -LiteralPath (Join-Path $evaluationFolder 'summaries') -Filter 'result_metrics*_summary*.json' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $summary) { throw "No official summary in $evaluationFolder" }
  $value = Get-Content -Raw -LiteralPath $summary.FullName | ConvertFrom-Json
  return [ordered]@{summary=$summary.FullName; fppi=[double]$value.fppi; hfar=[double]$value.far; afdr=[double]$value.fl_dr_in_range; edr300=[double]$value.Detection.Encounters.'300'.All.dr}
}
function Eval-Aot([string]$name, [string]$resultsFolder, [string]$evaluationFolder) {
  Run-Logged $name $tvdPython @('.\evaluate_aot.py','--results_folder',$resultsFolder,'--evaluation_folder',$evaluationFolder,'--detection_threshold','0.2','--dataset-path',$dataset) $tvdRepo
  return Read-Metrics $evaluationFolder
}

Progress 'prepare_validation' 0 6
$part0Source = Join-Path $outRoot 'validation_source'
New-Item -ItemType Directory -Force -Path $part0Source | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'predictions_split_0.pkl') -Destination (Join-Path $part0Source 'predictions_split_0.pkl') -Force
$baselineVal = Eval-Aot 'val_baseline_eval' $part0Source (Join-Path $outRoot 'val_baseline_eval')
Progress 'validation_sweep' 1 6 @{baseline_validation=$baselineVal}

$rows = @()
$index = 1
foreach ($beta in $betas) {
  $tag = ('beta_{0}' -f ([string]$beta).Replace('.','p'))
  $rescore = Join-Path $outRoot "val_$tag"
  Run-Logged "val_${tag}_rescore" $qstrPython @('-m','qstr_dronedet.cli','rescore-aot-prediction-parts-by-tracklets','--results-folder',$part0Source,'--tracklet-jsonl',$tracklets,'--out-dir',$rescore,'--score-field','vatd_score','--center','0.5','--beta',[string]$beta,'--mode','suppress-only','--min-tracklet-rows','1','--missing-score-behavior','keep') $RepoRoot
  $metrics = Eval-Aot "val_${tag}_eval" (Join-Path $rescore 'aotpredictions') (Join-Path $outRoot "val_${tag}_eval")
  $rows += [pscustomobject]@{beta=$beta; fppi=$metrics.fppi; hfar=$metrics.hfar; afdr=$metrics.afdr; edr300=$metrics.edr300; summary=$metrics.summary}
  $index++
  Progress 'validation_sweep' $index 6 @{baseline_validation=$baselineVal; candidates=$rows}
}

$eligible = @($rows | Where-Object { $_.edr300 -ge ($baselineVal.edr300 - 1e-9) -and $_.hfar -le ($baselineVal.hfar + 1e-6) })
if ($eligible.Count -eq 0) { $eligible = @($rows | Sort-Object @{Expression='edr300';Descending=$true}, fppi) }
$best = $eligible | Sort-Object fppi, @{Expression='edr300';Descending=$true} | Select-Object -First 1
Progress 'full_rescore' 4 6 @{baseline_validation=$baselineVal; candidates=$rows; selected=$best}

$fullRescore = Join-Path $outRoot 'full_fixed'
Run-Logged 'full_rescore' $qstrPython @('-m','qstr_dronedet.cli','rescore-aot-prediction-parts-by-tracklets','--results-folder',$source,'--tracklet-jsonl',$tracklets,'--out-dir',$fullRescore,'--score-field','vatd_score','--center','0.5','--beta',[string]$best.beta,'--mode','suppress-only','--min-tracklet-rows','1','--missing-score-behavior','keep') $RepoRoot
Progress 'full_official_eval' 5 6 @{selected=$best}
$full = Eval-Aot 'full_official_eval' (Join-Path $fullRescore 'aotpredictions') (Join-Path $outRoot 'full_official_eval')
$baselineFull = @{fppi=0.262303510022747; hfar=89.47674418604652; afdr=0.8685312193818473; edr300=0.9257142857142857}
$summary = @{protocol='part0 validation-selected, fixed full AOT zero-shot Action Chunk transfer'; baseline_full=$baselineFull; validation_baseline=$baselineVal; validation_candidates=$rows; selected=$best; full_fixed=$full; relative_fppi_reduction=($baselineFull.fppi-$full.fppi)/$baselineFull.fppi; absolute_afdr_gain=$full.afdr-$baselineFull.afdr; absolute_edr300_gain=$full.edr300-$baselineFull.edr300}
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outRoot 'official_summary.json') -Encoding UTF8
Progress 'done' 6 6 @{summary=$summary}
