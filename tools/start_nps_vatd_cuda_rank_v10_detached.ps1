$ErrorActionPreference = 'Stop'
$python = 'C:\Users\aaron\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @('C:\Users\aaron\Desktop\URAP\tools\run_nps_vatd_cuda_rank_v10.py') `
    -WorkingDirectory 'C:\Users\aaron\Desktop\URAP' `
    -RedirectStandardOutput 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.out.txt' `
    -RedirectStandardError 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.err.txt' `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.pid' -Value $process.Id
@("pid=$($process.Id)", "started=$(Get-Date -Format o)", 'stdout=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.out.txt', 'stderr=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.err.txt', 'progress=C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10_runner\progress.json') | Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v10.meta.txt'
Write-Output $process.Id
