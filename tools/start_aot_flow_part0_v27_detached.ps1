param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'aot_flow_part0_v27',
  [string]$OutputRoot = ''
)
$ErrorActionPreference='Stop'
if(-not $OutputRoot){$OutputRoot=Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_part0_v27_runner'}
$python=Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'
$out=Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_part0_v27'
New-Item -ItemType Directory -Force -Path $OutputRoot,(Join-Path $OutputRoot 'logs'),$out|Out-Null
$pidFile=Join-Path $OutputRoot "${RunId}_pid.txt";$metaFile=Join-Path $OutputRoot "${RunId}_meta.txt";$ts=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $OutputRoot "logs\${RunId}_${ts}.out.txt";$stderr=Join-Path $OutputRoot "logs\${RunId}_${ts}.err.txt"
$args=@((Join-Path $RepoRoot 'tools\aot_action_bank_flow_recovery.py'),'--results-folder',(Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_chunk_transfer_v1\validation_source'),'--tracklets',(Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_chunk_transfer_v1\tracklets_with_action_chunk_scores.jsonl'),'--frames-root','D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test\part0\frames','--out-dir',$out,'--min-action-score','0.6','--max-gap','30','--promotion-iou','0.3')
$process=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$process.Id|Set-Content -LiteralPath $pidFile -Encoding ascii
@("started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')","pid=$($process.Id)","output=$out","stdout=$stdout","stderr=$stderr")|Set-Content -LiteralPath $metaFile -Encoding ascii
Get-Content -LiteralPath $metaFile
