param(
    [string]$RepoRoot = 'C:\Users\aaron\Desktop\URAP',
    [string]$RunId = 'vatd_aot_to_nps_cuda_rank_v1'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TERM = 'dumb'
$root = Join-Path $RepoRoot 'artifacts\detached_vatd_aot_to_nps_cuda_rank'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logs ($RunId + '_' + $timestamp + '.out.txt')
$stderr = Join-Path $logs ($RunId + '_' + $timestamp + '.err.txt')
$pidFile = Join-Path $root ($RunId + '.pid')
$metaFile = Join-Path $root ($RunId + '.meta.json')
$process = Start-Process -FilePath 'C:\Users\aaron\.local\bin\modal.exe' -ArgumentList @('run', 'tools\modal_train_aot_to_nps_cuda_rank.py') -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;modal_app='urap-vatd-aot-to-nps-rank-v1';result_volume='vatd-rank-results-v1';progress_path='/aot_to_nps_cuda_rank_v1/progress.json'} | ConvertTo-Json | Set-Content -LiteralPath $metaFile
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"


