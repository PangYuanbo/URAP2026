$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Existing=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*score_predictionsgt_samurai_native.py*' })
if ($Existing.Count -gt 0) { throw ('SAMURAI scorer already running: ' + (($Existing | ForEach-Object { $_.ProcessId }) -join ',')) }
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_nps_samurai_native_clip38_v8'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout=Join-Path $Logs "nps_samurai_native_clip38_v8_$Stamp.out.txt"
$Stderr=Join-Path $Logs "nps_samurai_native_clip38_v8_$Stamp.err.txt"
$Out='D:\URAP_vatd_rank_results\nps_samurai_native_clip38_v8'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Arguments=@(
 (Join-Path $Repo 'tools\score_predictionsgt_samurai_native.py'),
 '--predictionsgt-pkl','D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl',
 '--frame-root','U:\URAP_datasets\TransVisDrone\NPS\AllFrames\val',
 '--output-jsonl',(Join-Path $Out 'val_scores.jsonl'),
 '--output-summary',(Join-Path $Out 'val_score_summary.json'),
 '--frame-cache','U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_val_frames',
 '--progress-json',(Join-Path $Run 'progress.json'),
 '--sequences','Clip_38',
 '--start-gate','0.55','--reset-gate','0.85','--reset-iou','0.05','--object-gate','0.20',
 '--reset-policy','quality-only','--reset-patience','3','--disagreement-reset-gate','0.90'
)
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');stdout=$Stdout;stderr=$Stderr;progress=(Join-Path $Run 'progress.json');output=$Out;total=1800} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Run 'state.json')
$Process.Id | Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
Write-Host "started pid=$($Process.Id) stdout=$Stdout stderr=$Stderr"
