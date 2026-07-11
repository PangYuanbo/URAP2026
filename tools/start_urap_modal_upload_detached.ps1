param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_urap_upload"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$LogDir = Join-Path $RunnerDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $LogDir "modal_upload_$stamp.out.txt"
$stderr = Join-Path $LogDir "modal_upload_$stamp.err.txt"
$pidPath = Join-Path $RunnerDir "modal_upload.pid"
$metaPath = Join-Path $RunnerDir "modal_upload.meta.txt"
$script = Join-Path $RepoRoot "tools\upload_urap_assets_to_modal.ps1"

$existing = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
if ($existing -match "^\d+$" -and (Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue)) {
    throw "Upload already running with PID=$existing"
}

$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "-RepoRoot", $RepoRoot, "-StateDir", "artifacts\modal_urap_upload")
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
@(
    "started=$((Get-Date).ToString('o'))"
    "pid=$($process.Id)"
    "command=powershell.exe $($arguments -join ' ')"
    "stdout=$stdout"
    "stderr=$stderr"
    "progress=$(Join-Path $RunnerDir 'progress.json')"
) | Set-Content -LiteralPath $metaPath -Encoding utf8
Get-Content $metaPath

