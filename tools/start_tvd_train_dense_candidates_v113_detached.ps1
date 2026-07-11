$ErrorActionPreference = 'Stop'
$Repo = 'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$Weights = Join-Path $Repo 'pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt'
$Data = Join-Path $Repo 'data\NPS_URAP_D.yaml'
$Run = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_train_dense_candidates_v113'
$Project = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113'
$Name = 'official_train_dense'
foreach ($Path in @($Python, $Weights, $Data)) { if (-not (Test-Path -LiteralPath $Path)) { throw "Required path missing: $Path" } }
New-Item -ItemType Directory -Force -Path $Run | Out-Null
New-Item -ItemType Directory -Force -Path $Project | Out-Null
$PidFile = Join-Path $Run 'pid.txt'; $Stdout = Join-Path $Run 'stdout_batch16.log'; $Stderr = Join-Path $Run 'stderr_batch16.log'; $Meta = Join-Path $Run 'start_meta.json'
if (Test-Path -LiteralPath $PidFile) { $OldPid = [int](Get-Content -LiteralPath $PidFile -Raw).Trim(); $OldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$OldPid" -ErrorAction SilentlyContinue; if ($OldProcess) { Write-Output "ALREADY RUNNING PID=$OldPid COMMAND=$($OldProcess.CommandLine)"; exit 0 } }
$Arguments = @('.\val.py','--data',$Data,'--weights',$Weights,'--batch-size','16','--imgsz','1280','--conf-thres','0.001','--iou-thres','0.6','--task','train','--device','0','--half','--num-frames','5','--save-json-gt','--project',$Project,'--name',$Name,'--exist-ok')
Set-Content -LiteralPath $Stdout -Value ''; Set-Content -LiteralPath $Stderr -Value ''; $Started = Get-Date
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $PidFile -Value $Process.Id
@{pid=$Process.Id;start_time=$Started.ToString('o');command_line="$Python $($Arguments -join ' ')";stdout=$Stdout;stderr=$Stderr;output_dir=(Join-Path $Project $Name);total_frames=51951} | ConvertTo-Json | Set-Content -LiteralPath $Meta
Write-Output "STARTED PID=$($Process.Id) START=$($Started.ToString('o'))"; Write-Output "STDOUT=$Stdout"; Write-Output "STDERR=$Stderr"; Write-Output "OUTPUT=$(Join-Path $Project $Name)"




