param([string]$RunName = "ard100_short166_local_build_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
$statePath = Join-Path $controlRoot "state.json"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$builder = Join-Path $repoRoot "tools\build_ard100_short_tracklets.py"
$validator = Join-Path $repoRoot "tools\materialize_ard100_short166_local.py"
$sourceRoot = "U:\URAP_datasets\TransVisDrone\ARD100"
$rawRoot = "U:\URAP_datasets\ARD100"
$annotations = Join-Path $rawRoot "annotations.zip"
$outputRoot = Join-Path $sourceRoot "SAMURAI_SHORT166"
New-Item -ItemType Directory -Force -Path $controlRoot, $logRoot, $outputRoot | Out-Null

function Write-State([string]$Stage, [int]$Done, [int]$Total, [string]$LastUnit, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{ stage = $Stage; done = $Done; total = $Total; last_completed_unit = $LastUnit; updated_at = (Get-Date).ToString("o") }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $temporary = "$statePath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
    $payload | ConvertTo-Json -Compress -Depth 8 | Write-Output
}

$expected = [ordered]@{ val = 10; test = 35; train = 55 }
$done = 0
foreach ($split in $expected.Keys) {
    $stdout = Join-Path $logRoot "$split.stdout.log"
    $stderr = Join-Path $logRoot "$split.stderr.log"
    $arguments = @(
        $builder,
        "--source-root", $sourceRoot,
        "--raw-video-root", $rawRoot,
        "--annotations-zip", $annotations,
        "--split", $split,
        "--output-root", (Join-Path $outputRoot "${split}_v1"),
        "--max-gap", "2",
        "--max-frames", "166",
        "--min-visible-frames", "8",
        "--min-visibility", "0.5",
        "--image-mode", "hardlink",
        "--resume"
    )
    $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-State "building" $done 3 $split @{ child_pid = $child.Id; split = $split; stdout_log = $stdout; stderr_log = $stderr }
    $child.WaitForExit()
    $child.Refresh()
    $exitCode = $child.ExitCode
    if ($null -ne $exitCode -and $exitCode -ne 0) { throw "Local short166 build failed for $split with exit code $exitCode" }
    $manifestPath = Join-Path $outputRoot "${split}_v1\manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Local short166 build ended without $manifestPath" }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ([int]$manifest.source_video_count -ne [int]$expected[$split]) { throw "Incomplete $split manifest" }
    $done += 1
    Write-State "split_completed" $done 3 $split @{ manifest = $manifestPath; sequences = [int]$manifest.sequence_count; frames = [int]$manifest.frame_count }
}

Write-State "validating" 3 3 "all_files" @{ output_root = $outputRoot }
& $python $validator --root $outputRoot
if ($LASTEXITCODE -ne 0) { throw "Local materialization validation failed" }
Write-State "completed" 3 3 "LOCAL_MATERIALIZE_COMPLETE.json" @{ output_root = $outputRoot; marker = (Join-Path $outputRoot "LOCAL_MATERIALIZE_COMPLETE.json") }
