param([string]$RunName = "ard100_short166_modal_build_v1")

$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
if (-not (Test-Path -LiteralPath $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*modal_build_ard100_short_tracklets.py*"
$progressLines = @()
if (Test-Path -LiteralPath $meta.stdout_log) {
    $progressLines = @(Select-String -LiteralPath $meta.stdout_log -Pattern 'tracklet_build_progress|"sequence_count"|App completed' | Select-Object -Last 12 | ForEach-Object { $_.Line })
}
$stdoutItem = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$stderrItem = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
$readySplits = @()
$manifestRoot = Join-Path $controlRoot "manifest_probe"
New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$previousUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = "1"
foreach ($split in @("train", "val", "test")) {
    $manifestPath = Join-Path $manifestRoot "$split.json"
    & $modal volume get --force urap-ard100-samurai-short166-v1 "ARD100_SAMURAI_SHORT166/$($split)_v1/manifest.json" $manifestPath 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $manifestPath)) { $readySplits += $split }
}
$env:PYTHONUTF8 = $previousUtf8
$completedByLog = $stdoutItem -and (Select-String -LiteralPath $meta.stdout_log -Pattern 'App completed' -Quiet)
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } elseif ($completedByLog -and $readySplits.Count -eq 3) { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    pid = [int]$meta.pid
    started_at = $meta.started_at
    command_line_matches = [bool]$commandMatches
    done_splits = $readySplits.Count
    total_splits = 3
    completed_splits = $readySplits
    last_output_timestamp = if ($stdoutItem) { $stdoutItem.LastWriteTime.ToString("o") } else { $null }
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    stderr_bytes = if ($stderrItem) { $stderrItem.Length } else { 0 }
    recent_progress = $progressLines
} | ConvertTo-Json -Depth 5
