param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$runnerDir = Join-Path $RepoRoot "artifacts\modal_aot_s3_sync"
$logDir = Join-Path $runnerDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidPath = Join-Path $runnerDir "sync.pid"
$oldPid = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
if ($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) { throw "Sync already running PID=$oldPid" }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "sync_$stamp.out.txt"
$stderr = Join-Path $logDir "sync_$stamp.err.txt"
$script = Join-Path $RepoRoot "tools\modal_sync_aot_part1_from_s3.py"
$arguments = @("run", "--detach", $script, "--workers", "64", "--commit-every", "5000")
$process = Start-Process modal.exe -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
@(
    "started=$((Get-Date).ToString('o'))"
    "pid=$($process.Id)"
    "command=modal.exe $($arguments -join ' ')"
    "stdout=$stdout"
    "stderr=$stderr"
) | Set-Content (Join-Path $runnerDir "sync.meta.txt") -Encoding utf8
Get-Content (Join-Path $runnerDir "sync.meta.txt")
