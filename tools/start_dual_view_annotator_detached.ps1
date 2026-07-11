param([int]$Port = 8787)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo 'artifacts\venvs\nps_flow\Scripts\python.exe'
$runRoot = Join-Path $repo 'artifacts\runs\dual_view_annotator'
$runId = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $runRoot $runId
$stdout = Join-Path $runDir 'stdout.log'
$stderr = Join-Path $runDir 'stderr.log'
$latestFile = Join-Path $repo 'artifacts\runs\dual_view_annotator_latest.json'
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
$existing = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($existing) { throw "Port $Port is already listening (PID $($existing.OwningProcess))." }
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$process = Start-Process -FilePath $python -ArgumentList @('-m','http.server',$Port.ToString(),'--bind','127.0.0.1','--directory',$repo) -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $runDir 'pid.txt') -Encoding ascii
[ordered]@{run_id=$runId;run_dir=$runDir;pid=$process.Id;port=$Port;started_at=(Get-Date).ToString('o');url="http://127.0.0.1:$Port/annotation_tool/annotator.html";stdout_log=$stdout;stderr_log=$stderr} | ConvertTo-Json | Set-Content -LiteralPath $latestFile -Encoding utf8
Write-Output "Started dual-view annotator PID=$($process.Id)"
Write-Output "URL: http://127.0.0.1:$Port/annotation_tool/annotator.html"
Write-Output "Logs: $stdout ; $stderr"
