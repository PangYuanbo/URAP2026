$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\aaron\Desktop\URAP'
$python = 'C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe'
$orchestrator = 'C:\Users\aaron\Desktop\URAP\tools\run_nps_vatd_cuda_rank_v4.py'
$outputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner'
$logs = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner\logs'
$stdout = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner\logs\run.out.txt'
$stderr = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner\logs\run.err.txt'
$pidFile = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner\nps_vatd_cuda_rank_v4.pid'
$metaFile = 'C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\nps_vatd_cuda_rank_v4_runner\nps_vatd_cuda_rank_v4.meta.txt'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$process = Start-Process -FilePath $python -ArgumentList @($orchestrator) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id
@("pid=$($process.Id)", "started=$(Get-Date -Format o)", "stdout=$stdout", "stderr=$stderr", "progress=$outputRoot\progress.json") | Set-Content -LiteralPath $metaFile
Write-Output $process.Id
