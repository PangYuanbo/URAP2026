$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_nps_motion_heuristic_v17';$Logs=Join-Path $Run 'logs';New-Item -ItemType Directory -Force $Logs|Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss';$Out=Join-Path $Logs "nps_motion_heuristic_v17_$Stamp.out.txt";$Err=Join-Path $Logs "nps_motion_heuristic_v17_$Stamp.err.txt"
$Args=@((Join-Path $Repo 'tools\sweep_nps_action_bank_motion_heuristics.py'),'--val-pkl','D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--test-pkl','D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--val-jsonl','D:\URAP_vatd_rank_results\nps_online_action_bank_v14\val_scores.jsonl','--test-jsonl','D:\URAP_vatd_rank_results\nps_online_action_bank_v14\test_scores.jsonl','--tvd-root','D:\urap_modal_stage\TransVisDrone','--out-json','D:\URAP_vatd_rank_results\nps_motion_heuristic_v17\official_summary.json')
$P=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Args -WorkingDirectory $Repo -RedirectStandardOutput $Out -RedirectStandardError $Err -PassThru
@{pid=$P.Id;start_time=(Get-Date).ToString('o');stdout=$Out;stderr=$Err;summary='D:\URAP_vatd_rank_results\nps_motion_heuristic_v17\official_summary.json';total=1}|ConvertTo-Json|Set-Content -Encoding UTF8 (Join-Path $Run 'state.json');$P.Id|Set-Content (Join-Path $Run 'pid.txt');Write-Host "started pid=$($P.Id) stdout=$Out stderr=$Err"
