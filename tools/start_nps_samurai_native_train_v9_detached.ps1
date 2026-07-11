$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_nps_samurai_native_train_v9'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$StatePath=Join-Path $Run 'state.json'
if(Test-Path $StatePath){$old=Get-Content $StatePath -Raw|ConvertFrom-Json;$oldProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)" -ErrorAction SilentlyContinue;if($oldProcess -and $oldProcess.CommandLine -like '*score_predictionsgt_samurai_native.py*'){throw "run already active pid=$($old.pid)"}}
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout=Join-Path $Logs "nps_samurai_native_train_v9_$Stamp.out.txt"
$Stderr=Join-Path $Logs "nps_samurai_native_train_v9_$Stamp.err.txt"
$Out='D:\URAP_vatd_rank_results\nps_samurai_native_train_v9'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Arguments=@(
 (Join-Path $Repo 'tools\score_predictionsgt_samurai_native.py'),
 '--predictionsgt-pkl','D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl',
 '--frame-root','U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train',
 '--output-jsonl',(Join-Path $Out 'train_scores.jsonl'),
 '--output-summary',(Join-Path $Out 'train_score_summary.json'),
 '--frame-cache','U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_train_frames',
 '--progress-json',(Join-Path $Run 'progress.json'),
 '--start-gate','0.55','--reset-gate','0.70','--reset-iou','0.05','--object-gate','0.20',
 '--reset-policy','any','--reset-patience','1','--disagreement-reset-gate','0.70'
)
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');stdout=$Stdout;stderr=$Stderr;progress=(Join-Path $Run 'progress.json');output=$Out;total=51933} | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
$Process.Id | Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
Write-Host "started pid=$($Process.Id) stdout=$Stdout stderr=$Stderr"
