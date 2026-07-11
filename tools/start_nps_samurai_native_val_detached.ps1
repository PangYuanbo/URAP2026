$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_nps_samurai_native_val_v7'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout=Join-Path $Logs "nps_samurai_native_val_v7_$Stamp.out.txt"
$Stderr=Join-Path $Logs "nps_samurai_native_val_v7_$Stamp.err.txt"
$Out='D:\URAP_vatd_rank_results\nps_samurai_native_v7'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Arguments=@(
 (Join-Path $Repo 'tools\score_predictionsgt_samurai_native.py'),
 '--predictionsgt-pkl','D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl',
 '--frame-root','U:\URAP_datasets\TransVisDrone\NPS\AllFrames\val',
 '--output-jsonl',(Join-Path $Out 'val_scores.jsonl'),
 '--output-summary',(Join-Path $Out 'val_score_summary.json'),
 '--frame-cache','U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_val_frames',
 '--progress-json',(Join-Path $Run 'progress.json'),
 '--start-gate','0.55','--reset-gate','0.70','--reset-iou','0.05','--object-gate','0.20'
)
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');stdout=$Stdout;stderr=$Stderr;progress=(Join-Path $Run 'progress.json');output=$Out;total=5944} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Run 'state.json')
$Process.Id | Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
Write-Host "started pid=$($Process.Id) stdout=$Stdout stderr=$Stderr"
