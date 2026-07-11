param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'papers\TransVisDrone\.venv\Scripts\python.exe'),
  [string]$RunId = 'nps_vatd_row_gbdt_trainval_v1',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_sota_research\nps_vatd_row_gbdt_trainval_v1_runner')
)
$ErrorActionPreference = 'Stop'; Set-Location $RepoRoot
$train = 'artifacts\nps_sota_research\tvd_nps_trainval_tracklets\tracklets_with_vatd_scores_nps_trainval_nocrop.jsonl'
$test = 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\tracklets_with_vatd_scores_nps_traintrain_nocrop.jsonl'
$trainGt = @('artifacts\nps_sota_research\tvd_nps_train_route_b_v3\gt.csv','artifacts\nps_sota_research\tvd_nps_val_route_b_v2\gt.csv')
$metaScores = 'artifacts\nps_sota_research\nps_vatd_gbdt_trainval_v1\test_tracklets_scored.jsonl'
$predictions = 'papers\TransVisDrone\runs\val\NPS_URAP_D\nps_test_best_aug_bs8_half\predictionsgt\predictionsgt_split_0.pkl'
$outDir = 'artifacts\nps_sota_research\nps_vatd_row_gbdt_trainval_v1'
foreach($path in @($Python,$train,$test,$metaScores,$predictions)+$trainGt){if(-not(Test-Path $path -PathType Leaf)){throw "Required file not found: $path"}}
New-Item -ItemType Directory -Force -Path $OutputRoot,$outDir,(Join-Path $OutputRoot 'logs')|Out-Null
$pidFile=Join-Path $OutputRoot "$RunId.pid"; if(Test-Path $pidFile){$oldPid=Get-Content $pidFile|Select-Object -First 1;$old=Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue;if($old -and $old.CommandLine -like "*$RunId*"){Write-Host "already running pid=$oldPid";exit 0}}
$ts=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $OutputRoot "logs\${RunId}_${ts}.out.txt";$stderr=Join-Path $OutputRoot "logs\${RunId}_${ts}.err.txt";$cmdFile=Join-Path $OutputRoot "$RunId.cmd.ps1";$progress=Join-Path $OutputRoot 'progress.json'
$script=@"
`$ErrorActionPreference='Stop';Set-Location '$RepoRoot'
@{stage='train';done=0;total=2;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content '$progress'
& '$Python' tools\train_detection_row_gbdt_score_head.py --train-tracklets '$train' --train-gt-csv '$($trainGt[0])' '$($trainGt[1])' --test-tracklets '$test' --out-test-tracklets '$outDir\test_tracklets_scored.jsonl' --out-model '$outDir\model.joblib' --out-summary '$outDir\train_summary.json' --score-field row_vatd_gbdt_score --label-policy unique-iou --negative-min-score 0.005 --feature-groups all --max-iter 400 --max-leaf-nodes 63 --min-samples-leaf 30
@{stage='evaluate';done=1;total=2;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content '$progress'
& '$Python' tools\sweep_tvd_predictionsgt_two_score_fusion.py --predictionsgt-pkl '$predictions' --meta-tracklet-jsonl '$metaScores' --meta-score-field vatd_gbdt_score --row-tracklet-jsonl '$outDir\test_tracklets_scored.jsonl' --row-score-field row_vatd_gbdt_score --modes logit-3mix meta-logit-row-geom meta-logit-row-boost --alphas '0.00 0.02 0.04 0.06' --betas '0.05 0.10 0.18 0.30' --out-json '$outDir\fusion_sweep.json' --write-best-pkl '$outDir\best_predictionsgt.pkl'
@{stage='done';done=2;total=2;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content '$progress'
"@;Set-Content -LiteralPath $cmdFile -Value $script -Encoding UTF8
$process=Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$cmdFile) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id;@("pid=$($process.Id)","started=$(Get-Date -Format o)","stdout=$stdout","stderr=$stderr","progress=$progress")|Set-Content -LiteralPath (Join-Path $OutputRoot "$RunId.meta.txt");Write-Host "started pid=$($process.Id) stdout=$stdout stderr=$stderr"
