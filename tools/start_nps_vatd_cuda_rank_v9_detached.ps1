$ErrorActionPreference = 'Stop'
$process = Start-Process `
    -FilePath 'C:\Users\aaron\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe' `
    -ArgumentList @('C:\Users\aaron\Desktop\URAP\tools\run_nps_vatd_cuda_rank_v9.py') `
    -WorkingDirectory 'C:\Users\aaron\Desktop\URAP' `
    -RedirectStandardOutput 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.out.txt' `
    -RedirectStandardError 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.err.txt' `
    -PassThru
Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.pid' -Value $process.Id
@("pid=$($process.Id)", "started=$(Get-Date -Format o)", 'stdout=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.out.txt', 'stderr=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.err.txt', 'progress=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9_runner\progress.json') | Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v9.meta.txt'
Write-Output $process.Id
