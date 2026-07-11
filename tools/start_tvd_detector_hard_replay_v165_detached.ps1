$ErrorActionPreference='Stop'
$repo='U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone'
$python=Join-Path $repo '.venv\Scripts\python.exe'
$run='C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_detector_hard_replay_v165'
$project='D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165'
New-Item -ItemType Directory -Force -Path $run|Out-Null
$stdout=Join-Path $run 'stdout.log';$stderr=Join-Path $run 'stderr.log'
$name='hard8_replay12k_img1280_noval_e2'
$args=@('.\train.py','--weights','D:\URAP_models\TransVisDrone_NPS_official\best.pt','--cfg',(Join-Path $repo 'models\yolov5l.yaml'),'--data','C:\Users\aaron\Desktop\URAP\data_templates\NPS_hard_replay_v165.yaml','--hyp','C:\Users\aaron\Desktop\URAP\data_templates\hyp.nps_hard_replay_v165.yaml','--epochs','3','--batch-size','2','--img','1280','--adam','--freeze','10','--workers','2','--patience','3','--save-period','1','--noval','--project',$project,'--name',$name,'--exist-ok')
$env:WANDB_MODE='disabled';$env:WANDB_SILENT='true';$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$process=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id|Set-Content (Join-Path $run 'pid.txt') -Encoding ASCII
@{pid=$process.Id;start_time=(Get-Date).ToString('o');stdout_log=$stdout;stderr_log=$stderr;command_line=($python+' '+($args-join ' '));expected_epochs=2;requested_epochs_argument=3;actual_epoch_indices=@('18/19','19/19');expected_train_images=21861;results_csv=(Join-Path $project "$name\results.csv");weights_dir=(Join-Path $project "$name\weights");validation_protocol='skip pathological in-train validation; select epoch18/epoch19 with detached official val.py'}|ConvertTo-Json|Set-Content (Join-Path $run 'meta.json') -Encoding UTF8
Write-Host "RUNNING PID: $($process.Id)"

