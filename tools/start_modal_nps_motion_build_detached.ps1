param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_nps_motion_build"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$LogDir = Join-Path $RunnerDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$pidPath = Join-Path $RunnerDir "modal_nps_motion_build.pid"
$oldPid = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
if ($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) { throw "Already running PID=$oldPid" }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $LogDir "modal_nps_motion_build_$stamp.out.txt"
$stderr = Join-Path $LogDir "modal_nps_motion_build_$stamp.err.txt"
$script = Join-Path $RepoRoot "tools\modal_build_nps_motion_interventions.py"
$arguments = @("run", $script, "--interventions", "slow_0p5,accelerate_g2,decelerate_g2")
$modalExe = (Get-Command modal).Source
$process = Start-Process -FilePath $modalExe -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
@("started=$((Get-Date).ToString('o'))", "pid=$($process.Id)", "stdout=$stdout", "stderr=$stderr") | Set-Content (Join-Path $RunnerDir "meta.txt") -Encoding utf8
Get-Content (Join-Path $RunnerDir "meta.txt")
