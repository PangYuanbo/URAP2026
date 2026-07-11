param(
    [ValidateSet("train", "val", "test")][string]$Split = "train",
    [string]$OutputRoot = "U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\train_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $repoRoot "tools\build_nps_samurai_dataset.py"
$gtBySplit = @{
    train = Join-Path $repoRoot "artifacts\nps_sota_research\tvd_nps_train_route_b_v3\gt.csv"
    val = Join-Path $repoRoot "artifacts\nps_sota_research\tvd_nps_val_route_b_v2\gt.csv"
    test = Join-Path $repoRoot "artifacts\nps_sota_research\tvd_nps_test_route_b_v2\gt.csv"
}
$framesRoot = "U:\URAP_datasets\TransVisDrone\NPS\AllFrames\$Split"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot, $OutputRoot | Out-Null
$runName = "dataset_build_${Split}_v1"
$pidPath = Join-Path $controlRoot "$runName.pid"
$metaPath = Join-Path $controlRoot "$runName.meta.json"
$stdoutPath = Join-Path $controlRoot "$runName.stdout.log"
$stderrPath = Join-Path $controlRoot "$runName.stderr.log"
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*build_nps_samurai_dataset.py*") { throw "Dataset build is already running with PID $oldPid" }
}
$arguments = @(
    $runner,
    "--gt-csv", $gtBySplit[$Split],
    "--frames-root", $framesRoot,
    "--output-root", $OutputRoot,
    "--split", $Split,
    "--min-visible-frames", "8",
    "--min-visibility", "0.5",
    "--max-gap", "2",
    "--image-mode", "hardlink"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "$python $($arguments -join ' ')"
    output_root = $OutputRoot
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    progress_file = (Join-Path $OutputRoot "progress.json")
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4
