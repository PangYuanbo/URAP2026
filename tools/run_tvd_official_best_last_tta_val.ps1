$ErrorActionPreference='Stop'
$repo='U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone'
$python=Join-Path $repo '.venv\Scripts\python.exe'
$args=@('.\val.py','--task','val','--data',(Join-Path $repo 'data\NPS_URAP_D.yaml'),'--weights','D:\URAP_models\TransVisDrone_NPS_official\best.pt','D:\URAP_models\TransVisDrone_NPS_official\last.pt','--img','1280','--batch-size','4','--half','--num-frames','5','--conf-thres','0.001','--iou-thres','0.6','--augment','--save-json-gt','--project',(Join-Path $repo 'runs\val\NPS_URAP'),'--name','official_best_last_tta_val','--exist-ok')
Push-Location $repo
try { & $python @args; if($LASTEXITCODE -ne 0){throw "val.py failed: $LASTEXITCODE"} } finally { Pop-Location }
