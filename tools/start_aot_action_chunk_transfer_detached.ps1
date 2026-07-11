param([string]$RunId = 'aot_action_chunk_transfer_v1')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$run = Join-Path $repo "artifacts\route_b_official\${RunId}_runner"
New-Item -ItemType Directory -Force -Path (Join-Path $run 'logs') | Out-Null
$pidFile = Join-Path $run 'pid.txt'
if (Test-Path $pidFile) {
  $oldPid = [int](Get-Content -Raw $pidFile)
  $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
  if ($old -and $old.CommandLine -like '*run_aot_action_chunk_transfer.ps1*') { Write-Host "ALREADY RUNNING PID=$oldPid"; exit 0 }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $run "logs\$stamp.out.txt"
$stderr = Join-Path $run "logs\$stamp.err.txt"
$worker = Join-Path $repo 'tools\run_aot_action_chunk_transfer.ps1'
$started = Get-Date
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$worker,'-RepoRoot',$repo,'-RunId',$RunId) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ASCII
@{pid=$process.Id; start_time=$started.ToString('o'); command_line="powershell -File $worker -RunId $RunId"; stdout_log=$stdout; stderr_log=$stderr; progress=(Join-Path $repo "artifacts\route_b_official\$RunId\progress.json")} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'meta.json') -Encoding UTF8
Write-Host "STARTED PID=$($process.Id) START=$($started.ToString('o'))"
