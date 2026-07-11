param(
    [int]$DetectorPid = 56336
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\aaron\Desktop\URAP"
$runDir = Join-Path $repo "artifacts\detached_ard100_yolomg_corrected_v2"
$meta = Get-Content (Join-Path $runDir "meta.json") -Raw | ConvertFrom-Json
$stderrLog = $meta.stderr_log
$outputDir = $meta.output_dir
$marker = Join-Path $outputDir "results.txt"
$progress = Join-Path $runDir "completion_watcher.json"

while (Get-Process -Id $DetectorPid -ErrorAction SilentlyContinue) {
    @{
        status = "waiting_for_detector"
        detector_pid = $DetectorPid
        updated = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content $progress -Encoding UTF8
    Start-Sleep -Seconds 20
}

$logText = Get-Content $stderrLog -Raw
$completed = $logText -match "Results saved to"
$labels = if (Test-Path (Join-Path $outputDir "labels")) {
    (Get-ChildItem (Join-Path $outputDir "labels") -File -Filter "*.txt").Count
} else { 0 }

if (-not $completed) {
    @{
        status = "detector_stopped_without_completion"
        detector_pid = $DetectorPid
        label_files = $labels
        updated = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content $progress -Encoding UTF8
    exit 1
}

@{
    status = "verified_complete"
    detector_pid = $DetectorPid
    label_files = $labels
    verified_at = (Get-Date).ToString("o")
    stderr_log = $stderrLog
} | ConvertTo-Json | Set-Content $marker -Encoding UTF8

@{
    status = "done"
    detector_pid = $DetectorPid
    label_files = $labels
    marker = $marker
    updated = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content $progress -Encoding UTF8
