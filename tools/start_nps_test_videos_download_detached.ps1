param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_test_videos_v1')

$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_nps_test_videos_download'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$process=Start-Process -FilePath 'C:\Program Files\dotnet\dotnet.exe' -ArgumentList @('C:\Users\aaron\Desktop\URAP\tmp\pwsh\.store\powershell\7.4.6\powershell\7.4.6\tools\net8.0\any\win\pwsh.dll','-NoProfile','-File',(Join-Path $RepoRoot 'tools\download_nps_test_videos.ps1')) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;progress='D:\URAP_nps_test_tvd\download_progress.json'} | ConvertTo-Json | Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"
