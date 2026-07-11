param(
    [ValidateSet("train", "val", "test")][string]$Split,
    [string]$SourceRoot = "U:\URAP_datasets\TransVisDrone\ARD100",
    [string]$RawVideoRoot = "U:\URAP_datasets\ARD100",
    [string]$AnnotationsZip = "U:\URAP_datasets\ARD100\annotations.zip",
    [string]$OutputRoot = "",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $repoRoot "tools\build_ard100_samurai_dataset.py"
if (-not $OutputRoot) { $OutputRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI\${Split}_v1" }
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot, $OutputRoot | Out-Null
$runName = "ard100_samurai_dataset_${Split}_v1"
$pidPath = Join-Path $controlRoot "$runName.pid"
$metaPath = Join-Path $controlRoot "$runName.meta.json"
$stdoutPath = Join-Path $controlRoot "$runName.stdout.log"
$stderrPath = Join-Path $controlRoot "$runName.stderr.log"
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.Name -eq "python.exe" -and $old.CommandLine -like "*build_ard100_samurai_dataset.py*") { throw "Already running PID $oldPid" }
}
$arguments = @($runner, "--source-root", $SourceRoot, "--raw-video-root", $RawVideoRoot, "--annotations-zip", $AnnotationsZip, "--split", $Split, "--output-root", $OutputRoot, "--image-mode", "hardlink")
if ($Resume) { $arguments += "--resume" }
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
[ordered]@{pid=$process.Id;started_at=(Get-Date).ToString("o");command="$python $($arguments -join ' ')";split=$Split;output_root=$OutputRoot;progress_file=(Join-Path $OutputRoot "progress.json");stdout_log=$stdoutPath;stderr_log=$stderrPath} | ConvertTo-Json -Depth 4 | Set-Content $metaPath -Encoding utf8
Get-Content $metaPath
