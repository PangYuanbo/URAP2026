$ErrorActionPreference = 'Stop'
$process = Start-Process `
    -FilePath 'C:\Users\aaron\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe' `
    -ArgumentList @('C:\Users\aaron\Desktop\URAP\tools\run_nps_vatd_cuda_rank_v4.py') `
    -WorkingDirectory 'C:\Users\aaron\Desktop\URAP' `
    -RedirectStandardOutput 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v7.out.txt' `
    -RedirectStandardError 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v7.err.txt' `
    -PassThru
Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v7.pid' -Value $process.Id
Write-Output $process.Id
