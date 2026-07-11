param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunnerDir = "artifacts\modal_urap_upload"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RunnerDir = Join-Path $RepoRoot $RunnerDir
$LogDir = Join-Path $RunnerDir "logs"
$PidDir = Join-Path $RunnerDir "pids"
New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
$script = Join-Path $RepoRoot "tools\upload_urap_assets_to_modal.ps1"
$volumes = @(
    "urap-nps-formatted-v1",
    "urap-nps-motion-original-v1",
    "urap-nps-motion-variants-v1",
    "urap-ard100-raw-v1",
    "urap-ard100-yolomg-train-v1",
    "urap-ard100-yolomg-eval-v1",
    "urap-ard100-yolomg-annotations-v1",
    "urap-yolomg-eval-extras-v1"
)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
foreach ($volume in $volumes) {
    $safe = $volume -replace "[^A-Za-z0-9_.-]", "_"
    $pidPath = Join-Path $PidDir ($safe + ".pid")
    $oldPid = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
    if ($oldPid -match "^\d+$" -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) {
        Write-Host "ALREADY RUNNING volume=$volume PID=$oldPid"
        continue
    }
    $stdout = Join-Path $LogDir ("$safe`_$stamp.out.txt")
    $stderr = Join-Path $LogDir ("$safe`_$stamp.err.txt")
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "-RepoRoot", $RepoRoot, "-StateDir", "artifacts\modal_urap_upload", "-OnlyVolume", $volume)
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
    @(
        "started=$((Get-Date).ToString('o'))"
        "pid=$($process.Id)"
        "volume=$volume"
        "command=powershell.exe $($arguments -join ' ')"
        "stdout=$stdout"
        "stderr=$stderr"
        "progress=$(Join-Path $RunnerDir ("progress_$safe.json"))"
    ) | Set-Content -LiteralPath (Join-Path $PidDir ($safe + ".meta.txt")) -Encoding utf8
    Write-Host "STARTED volume=$volume PID=$($process.Id) stdout=$stdout stderr=$stderr"
}

