$ErrorActionPreference='Stop'
$Repo='U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone'
$Python=Join-Path $Repo '.venv\Scripts\python.exe'
$Run='C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_detector_finetune_v102'
$Project='D:\URAP_vatd_rank_results\tvd_detector_finetune_v102'
$Weights=Join-Path $Repo 'pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt'
$Data=Join-Path $Repo 'data\NPS_URAP_D.yaml'
$Hyp='C:\Users\aaron\Desktop\URAP\data_templates\hyp.nps_finetune_v102.yaml'
New-Item -ItemType Directory -Force $Run|Out-Null
$PidFile=Join-Path $Run 'pid.txt';$Stdout=Join-Path $Run 'stdout.log';$Stderr=Join-Path $Run 'stderr.log';$Meta=Join-Path $Run 'start_meta.json'
if(Test-Path $PidFile){$OldPid=[int](Get-Content $PidFile -Raw).Trim();$Old=Get-CimInstance Win32_Process -Filter "ProcessId=$OldPid" -ErrorAction SilentlyContinue;if($Old){Write-Output "ALREADY RUNNING PID=$OldPid";exit 0}}
$env:WANDB_MODE='disabled'
$env:WANDB_SILENT='true'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$Cfg=Join-Path $Repo 'models\yolov5l.yaml'
$Args=@('.\train.py','--weights',$Weights,'--cfg',$Cfg,'--data',$Data,'--hyp',$Hyp,'--epochs','8','--batch-size','2','--img','1280','--adam','--freeze','10','--workers','6','--patience','8','--save-period','1','--project',$Project,'--name','finetune_lowaug_e8','--exist-ok')
Set-Content $Stdout '' -Encoding utf8;Set-Content $Stderr '' -Encoding utf8;$Started=Get-Date
$Process=Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
Set-Content $PidFile $Process.Id -Encoding ascii
@{pid=$Process.Id;start_time=$Started.ToString('o');command_line="$Python $($Args -join ' ')";stdout=$Stdout;stderr=$Stderr;project=$Project;total_epochs=8}|ConvertTo-Json|Set-Content $Meta -Encoding utf8
Write-Output "STARTED PID=$($Process.Id) START=$($Started.ToString('o'))"
