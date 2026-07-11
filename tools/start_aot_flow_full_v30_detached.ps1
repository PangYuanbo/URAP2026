param(
  [string]$RepoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId='aot_flow_full_v30',
  [string]$OutputRoot=''
)
$ErrorActionPreference='Stop'
if(-not $OutputRoot){$OutputRoot=Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v30_runner'}
$python=Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe';$out=Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_bank_flow_full_v30';$progress=Join-Path $OutputRoot "${RunId}_progress.json";New-Item -ItemType Directory -Force -Path $OutputRoot,(Join-Path $OutputRoot 'logs'),$out|Out-Null
$pidFile=Join-Path $OutputRoot "${RunId}_pid.txt";$metaFile=Join-Path $OutputRoot "${RunId}_meta.txt"
if(Test-Path -LiteralPath $pidFile){$old=Get-Content -LiteralPath $pidFile|Select-Object -First 1;if($old-match'^\d+$'){$existing=Get-CimInstance Win32_Process -Filter "ProcessId = $old" -ErrorAction SilentlyContinue;if($existing-and$existing.CommandLine-like'*aot_action_bank_flow_recovery_sharded.py*'){Write-Host "Already running PID=$old";exit 0}}}
$ts=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $OutputRoot "logs\${RunId}_${ts}.out.txt";$stderr=Join-Path $OutputRoot "logs\${RunId}_${ts}.err.txt"
$args=@((Join-Path $RepoRoot 'tools\aot_action_bank_flow_recovery_sharded.py'),'--repo-root',$RepoRoot,'--out-dir',$out,'--progress',$progress)
$process=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru;$process.Id|Set-Content -LiteralPath $pidFile -Encoding ascii
@("started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')","pid=$($process.Id)","output=$out","progress=$progress","stdout=$stdout","stderr=$stderr")|Set-Content -LiteralPath $metaFile -Encoding ascii;Get-Content -LiteralPath $metaFile

