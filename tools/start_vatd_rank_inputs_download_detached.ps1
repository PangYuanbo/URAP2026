param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='vatd_rank_inputs_v1')
$ErrorActionPreference='Stop';$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8';$env:TERM='dumb'
$root=Join-Path $RepoRoot 'artifacts\detached_vatd_rank_download';$logs=Join-Path $root 'logs';New-Item -ItemType Directory -Force -Path $logs|Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt');$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$process=Start-Process -FilePath 'C:\Program Files\dotnet\dotnet.exe' -ArgumentList @('C:\Users\aaron\Desktop\URAP\tmp\pwsh\.store\powershell\7.4.6\powershell\7.4.6\tools\net8.0\any\win\pwsh.dll','-NoProfile','-File',(Join-Path $RepoRoot 'tools\download_vatd_rank_inputs.ps1')) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $root ($RunId+'.pid')) -Value $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;progress='D:\URAP_vatd_rank_inputs\download_progress.json'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"
