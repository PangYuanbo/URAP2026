param([int]$BatchSize=16,[int]$ImgSz=1280)
$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG'
$Root='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Root 'artifacts\detached_ard100_yolomg_train_candidates_v86'
$Logs=Join-Path $Run 'logs';New-Item -ItemType Directory -Force $Logs|Out-Null
$State=Join-Path $Run 'state.json'
if(Test-Path $State){$old=Get-Content $State -Raw|ConvertFrom-Json;$active=Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)" -ErrorAction SilentlyContinue;if($active -and $active.CommandLine -like '*yolomg_ard100_train_candidates_v86*'){throw "run already active pid=$($old.pid)"}}
$Python=Join-Path $Repo '.venv\Scripts\python.exe'
$Weights=Join-Path $Repo 'runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt'
$Data='D:\URAP_datasets\ARD100_YOLOMG\ARD100_mask32_local.yaml'
$Project='D:\URAP_vatd_rank_results\ard100_action_memory_target_v86'
$Name='yolomg_ard100_train_candidates_v86'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss';$OutLog=Join-Path $Logs "train_candidates_$Stamp.out.txt";$ErrLog=Join-Path $Logs "train_candidates_$Stamp.err.txt"
$Args=@('val.py','--data',$Data,'--weights',$Weights,'--task','train','--task2','train2','--imgsz',"$ImgSz",'--batch-size',"$BatchSize",'--device','0','--workers','4','--half','--conf-thres','0.001','--iou-thres','0.45','--save-txt','--save-conf','--project',$Project,'--name',$Name,'--exist-ok')
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Args -WorkingDirectory $Repo -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');command="$Python $($Args -join ' ')";stdout=$OutLog;stderr=$ErrLog;output_dir=(Join-Path $Project $Name);total_images=106734;batch_size=$BatchSize;total_batches=[math]::Ceiling(106734/$BatchSize)}|ConvertTo-Json|Set-Content -Encoding UTF8 $State
$Process.Id|Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt');Write-Host "started pid=$($Process.Id) stdout=$OutLog stderr=$ErrLog"
