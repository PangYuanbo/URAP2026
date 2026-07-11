param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_nps_original_revalidate"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$LogDir = Join-Path $RunnerDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$pidPath = Join-Path $RunnerDir "modal_nps_original_revalidate.pid"
$oldPid = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
if ($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
    throw "Modal NPS original revalidation already running PID=$oldPid"
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $LogDir "modal_nps_original_revalidate_$stamp.out.txt"
$stderr = Join-Path $LogDir "modal_nps_original_revalidate_$stamp.err.txt"
$script = Join-Path $RepoRoot "tools\modal_build_nps_motion_interventions.py"
$arguments = @("run", $script, "--interventions", "original")
$modalExe = (Get-Command modal).Source
$process = Start-Process -FilePath $modalExe -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
@("started=$((Get-Date).ToString('o'))", "pid=$($process.Id)",
  "command=$modalExe $($arguments -join ' ')", "stdout=$stdout", "stderr=$stderr") |
    Set-Content -LiteralPath (Join-Path $RunnerDir "modal_nps_original_revalidate.meta.txt") -Encoding utf8
Get-Content (Join-Path $RunnerDir "modal_nps_original_revalidate.meta.txt")
