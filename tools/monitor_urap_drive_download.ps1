$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$statusDir = Join-Path $repoRoot "artifacts\urap_drive_download"
$pidPath = Join-Path $statusDir "download.pid"
$startedPath = Join-Path $statusDir "download.started.txt"
$stdout = Join-Path $statusDir "download.stdout.log"
$stderr = Join-Path $statusDir "download.stderr.log"
$datasetRoot = Join-Path $repoRoot "datasets\urap_drive"
$expected = @(
    "videos\dji_fly_20260527_122540_15_1779921105591_hdrvideo.MP4",
    "videos\dji_fly_20260527_121932_14_1779921254906_hdrvideo.MP4",
    "videos\dji_fly_20260527_121806_13_1779921757607_hdrvideo.MP4",
    "videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4",
    "annotation_workspace\annotations\qstr_real_boxes_manual.csv",
    "annotation_workspace\annotations\recording_manifest.csv",
    "annotation_workspace\annotations\frame_index.csv",
    "annotation_workspace\cvat_exports\task1_labels.json",
    "annotation_workspace\cvat_exports\task1_annotations_raw.json",
    "annotation_workspace\cvat_exports\task1_data_meta.json",
    "annotation_workspace\cvat_upload\CVAT_LABELS_AND_NOTES.md",
    "annotation_workspace\cvat_upload\dji_fly_frames_stride60.zip"
)

$pidText = if (Test-Path -LiteralPath $pidPath) { (Get-Content -LiteralPath $pidPath -Raw).Trim() } else { "" }
$process = $null
if ($pidText) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}

$running = $false
if ($process -and $process.CommandLine -like "*download_urap_drive_files_worker.ps1*") {
    $running = $true
}

$done = 0
$lastCompleted = ""
foreach ($rel in $expected) {
    $path = Join-Path $datasetRoot $rel
    if ((Test-Path -LiteralPath $path) -and ((Get-Item -LiteralPath $path).Length -gt 0)) {
        $done += 1
        $lastCompleted = $rel
    }
}

$lastOutput = "missing"
if (Test-Path -LiteralPath $stdout) {
    $lastOutput = (Get-Item -LiteralPath $stdout).LastWriteTime.ToString("o")
}

$startTime = "unknown"
if (Test-Path -LiteralPath $startedPath) {
    $startTime = (Get-Content -LiteralPath $startedPath -Raw).Trim()
}

Write-Output "done/total: $done/$($expected.Count)"
if ($running) {
    Write-Output "PID: $pidText"
    Write-Output "start time: $startTime"
    Write-Output "process command: $($process.CommandLine)"
} else {
    Write-Output "NOT RUNNING"
    if ($pidText) {
        Write-Output "last PID file value: $pidText"
    }
    Write-Output "start time: $startTime"
}
Write-Output "last output timestamp: $lastOutput"
Write-Output "last completed unit: $lastCompleted"
Write-Output "stdout log: $stdout"
Write-Output "stderr log: $stderr"
Write-Output "dataset root: $datasetRoot"

if (Test-Path -LiteralPath $stdout) {
    Write-Output "stdout tail:"
    Get-Content -LiteralPath $stdout -Tail 20
}
if (Test-Path -LiteralPath $stderr) {
    $err = Get-Content -LiteralPath $stderr -Tail 20
    if ($err) {
        Write-Output "stderr tail:"
        $err
    }
}
