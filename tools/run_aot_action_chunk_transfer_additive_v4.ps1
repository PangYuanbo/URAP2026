param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'aot_action_chunk_transfer_additive_v4'
)
$ErrorActionPreference = 'Stop'
$outRoot = Join-Path $RepoRoot "artifacts\route_b_official\$RunId"
$progressPath = Join-Path $outRoot 'progress.json'
$source = Join-Path $RepoRoot 'papers\TransVisDrone\runs\val\AOT_URAP\fulltest_conf0p2_wport_baseline\aotpredictions'
$tracklets = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_chunk_transfer_v1\tracklets_with_action_chunk_scores.jsonl'
$qstrPython = Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$tvdRepo = Join-Path $RepoRoot 'papers\TransVisDrone'
$tvdPython = Join-Path $tvdRepo '.venv\Scripts\python.exe'
$dataset = 'D:\URAP_datasets\AOT\part1'
$configs = @(
  @{name='b0p04_add'; beta=0.04; protect=-1},
  @{name='b0p08_add'; beta=0.08; protect=-1},
  @{name='b0p12_add'; beta=0.12; protect=-1},
  @{name='b0p18_add'; beta=0.18; protect=-1}
)
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
function Progress($phase, $done, $total, $extra=@{}) { $value=@{phase=$phase;done=$done;total=$total;updated=(Get-Date).ToString('o')}; foreach($key in $extra.Keys){$value[$key]=$extra[$key]}; $value|ConvertTo-Json -Depth 8|Set-Content $progressPath -Encoding UTF8 }
function Run-Logged($name,$exe,$arguments,$workingDirectory){$stdout=Join-Path $outRoot "$name.out.txt";$stderr=Join-Path $outRoot "$name.err.txt";$process=Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $workingDirectory -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru;if($process.ExitCode -ne 0){throw "$name failed: $stderr"}}
function Read-Metrics($folder){$summary=Get-ChildItem (Join-Path $folder 'summaries') -Filter 'result_metrics*_summary*.json' -File|Sort-Object LastWriteTime -Descending|Select-Object -First 1;$value=Get-Content -Raw $summary.FullName|ConvertFrom-Json;[ordered]@{summary=$summary.FullName;fppi=[double]$value.fppi;hfar=[double]$value.far;afdr=[double]$value.fl_dr_in_range;edr300=[double]$value.Detection.Encounters.'300'.All.dr}}
function Eval-Aot($name,$results,$evaluation){Run-Logged $name $tvdPython @('.\evaluate_aot.py','--results_folder',$results,'--evaluation_folder',$evaluation,'--detection_threshold','0.2','--dataset-path',$dataset) $tvdRepo;Read-Metrics $evaluation}

Progress 'prepare_validation' 0 7
$part0=Join-Path $outRoot 'validation_source';New-Item -ItemType Directory -Force -Path $part0|Out-Null;Copy-Item (Join-Path $source 'predictions_split_0.pkl') (Join-Path $part0 'predictions_split_0.pkl') -Force
$baseline=Eval-Aot 'val_baseline_eval' $part0 (Join-Path $outRoot 'val_baseline_eval');Progress 'validation_sweep' 1 7 @{baseline_validation=$baseline}
$rows=@();$done=1
foreach($config in $configs){$rescore=Join-Path $outRoot "val_$($config.name)";Run-Logged "val_$($config.name)_rescore" $qstrPython @('-m','qstr_dronedet.cli','rescore-aot-prediction-parts-by-tracklets','--results-folder',$part0,'--tracklet-jsonl',$tracklets,'--out-dir',$rescore,'--score-field','vatd_score','--center','0.5','--beta',[string]$config.beta,'--mode','additive','--min-tracklet-rows','1','--missing-score-behavior','keep') $RepoRoot;$metrics=Eval-Aot "val_$($config.name)_eval" (Join-Path $rescore 'aotpredictions') (Join-Path $outRoot "val_$($config.name)_eval");$rows+=[pscustomobject]@{name=$config.name;beta=$config.beta;fppi=$metrics.fppi;hfar=$metrics.hfar;afdr=$metrics.afdr;edr300=$metrics.edr300;summary=$metrics.summary};$done++;Progress 'validation_sweep' $done 7 @{baseline_validation=$baseline;candidates=$rows}}
$eligible=@($rows|Where-Object{$_.edr300 -ge ($baseline.edr300-1e-9) -and $_.hfar -le ($baseline.hfar+1e-6)});if($eligible.Count -eq 0){$eligible=@($rows|Sort-Object @{Expression='edr300';Descending=$true},fppi)};foreach($row in $eligible){$row|Add-Member -NotePropertyName target_distance -NotePropertyValue ([math]::Abs((($baseline.fppi-$row.fppi)/$baseline.fppi)-0.04))};$best=$eligible|Sort-Object target_distance,fppi|Select-Object -First 1
Progress 'full_rescore' 5 7 @{selected=$best;candidates=$rows}
$fullRescore=Join-Path $outRoot 'full_fixed';Run-Logged 'full_rescore' $qstrPython @('-m','qstr_dronedet.cli','rescore-aot-prediction-parts-by-tracklets','--results-folder',$source,'--tracklet-jsonl',$tracklets,'--out-dir',$fullRescore,'--score-field','vatd_score','--center','0.5','--beta',[string]$best.beta,'--mode','additive','--min-tracklet-rows','1','--missing-score-behavior','keep') $RepoRoot
Progress 'full_official_eval' 6 7 @{selected=$best}
$full=Eval-Aot 'full_official_eval' (Join-Path $fullRescore 'aotpredictions') (Join-Path $outRoot 'full_official_eval');$base=@{fppi=0.262303510022747;hfar=89.47674418604652;afdr=0.8685312193818473;edr300=0.9257142857142857};$summary=@{protocol='part0 validation-selected additive zero-shot Action Chunk residual; fixed full AOT';baseline_full=$base;validation_baseline=$baseline;validation_candidates=$rows;selected=$best;full_fixed=$full;relative_fppi_reduction=($base.fppi-$full.fppi)/$base.fppi;absolute_afdr_gain=$full.afdr-$base.afdr;absolute_edr300_gain=$full.edr300-$base.edr300;strict_target_met=((($base.fppi-$full.fppi)/$base.fppi)-ge 0.03 -and (($base.fppi-$full.fppi)/$base.fppi)-le 0.05 -and $full.edr300-ge $base.edr300 -and $full.hfar-le $base.hfar)};$summary|ConvertTo-Json -Depth 10|Set-Content (Join-Path $outRoot 'official_summary.json') -Encoding UTF8;Progress 'done' 7 7 @{summary=$summary}
