param(
    [string]$RunId = "nps_motion_boundary_full"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_nps_motion_boundary\" + $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

function Read-Meta($path) {
    $meta = @{}
    if (Test-Path $path) {
        foreach ($line in Get-Content $path) {
            if ($line -match "^(.*?)=(.*)$") {
                $meta[$matches[1]] = $matches[2]
            }
        }
    }
    return $meta
}

if (-not (Test-Path $pidFile)) {
    Write-Output "status=NOT RUNNING"
    Write-Output "done/total=0/0"
    Write-Output "pid="
    Write-Output "started="
    Write-Output "last_output_timestamp="
    Write-Output "last_completed_unit="
    Write-Output "stdout="
    Write-Output "stderr="
    exit 0
}

$meta = Read-Meta $metaFile
$runPid = [int](Get-Content $pidFile -Raw).Trim()
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $runPid" -ErrorAction SilentlyContinue
$status = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }
$outDir = $meta["output_dir"]
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]

$progressPath = if ($outDir) { Join-Path $outDir "progress.json" } else { "" }
$progress = $null
if ($progressPath -and (Test-Path $progressPath)) {
    $progress = Get-Content $progressPath -Raw | ConvertFrom-Json
}

$done = 0
$total = 0
$lastUnit = ""
if ($progress) {
    $done = [int]$progress.clip_index
    $total = [int]$progress.total_clips
    $lastUnit = [string]$progress.last_completed_unit
}

$lastOutTs = ""
if ($stdout -and (Test-Path $stdout)) {
    $lastOutTs = (Get-Item $stdout).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}

$mp4Count = 0
$mp4Bytes = 0
if ($outDir -and (Test-Path $outDir)) {
    $mp4s = Get-ChildItem -LiteralPath $outDir -Recurse -File -Filter *.mp4 -ErrorAction SilentlyContinue
    $mp4Count = @($mp4s).Count
    $sum = ($mp4s | Measure-Object -Property Length -Sum).Sum
    if ($null -ne $sum) { $mp4Bytes = $sum }
}

Write-Output "status=$status"
Write-Output "done/total=$done/$total"
Write-Output "pid=$runPid"
Write-Output "started=$($meta["started"])"
Write-Output "last_output_timestamp=$lastOutTs"
Write-Output "last_completed_unit=$lastUnit"
Write-Output "output_dir=$outDir"
Write-Output "mp4_count=$mp4Count"
Write-Output "mp4_bytes=$mp4Bytes"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

if ($progress) {
    Write-Output ""
    Write-Output "== progress =="
    $progress | ConvertTo-Json -Depth 5
}

if ($stdout -and (Test-Path $stdout)) {
    Write-Output ""
    Write-Output "== stdout tail =="
    Get-Content $stdout -Tail 20
}

if ($stderr -and (Test-Path $stderr)) {
    Write-Output ""
    Write-Output "== stderr tail =="
    Get-Content $stderr -Tail 20
}
