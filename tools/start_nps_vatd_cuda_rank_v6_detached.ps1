$ErrorActionPreference = 'Stop'
$process = Start-Process `
    -FilePath 'C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe' `
    -ArgumentList @('C:\Users\aaron\Desktop\URAP\tools\run_nps_vatd_cuda_rank_v4.py') `
    -WorkingDirectory 'C:\Users\aaron\Desktop\URAP' `
    -RedirectStandardOutput 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v6.out.txt' `
    -RedirectStandardError 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v6.err.txt' `
    -PassThru
Set-Content -LiteralPath 'C:\Users\aaron\Desktop\URAP\tmp\nps_vatd_cuda_rank_v6.pid' -Value $process.Id
Write-Output $process.Id
