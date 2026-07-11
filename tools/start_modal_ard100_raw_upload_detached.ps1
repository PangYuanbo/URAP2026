param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$ErrorActionPreference = "Stop"
$runnerDir = Join-Path $RepoRoot "artifacts\modal_ard100_raw_upload"
$logDir = Join-Path $runnerDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidPath = Join-Path $runnerDir "upload.pid"
$oldPid = if(Test-Path $pidPath){(Get-Content $pidPath -Raw).Trim()}else{""}
if($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)){throw "Upload already running PID=$oldPid"}
$stamp=Get-Date -Format "yyyyMMdd_HHmmss"
$stdout=Join-Path $logDir "upload_$stamp.out.txt"
$stderr=Join-Path $logDir "upload_$stamp.err.txt"
$script=Join-Path $RepoRoot "tools\upload_ard100_raw_to_modal.ps1"
$arguments=@("-NoProfile","-ExecutionPolicy","Bypass","-File",$script)
$process=Start-Process powershell.exe -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id|Set-Content $pidPath -Encoding ascii
@("started=$((Get-Date).ToString('o'))","pid=$($process.Id)","command=powershell.exe $($arguments -join ' ')","stdout=$stdout","stderr=$stderr")|
    Set-Content (Join-Path $runnerDir "upload.meta.txt") -Encoding utf8
Get-Content (Join-Path $runnerDir "upload.meta.txt")
