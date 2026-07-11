param([string]$RunId = "seqtrack_b384_weights")

$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\ata_reproduction\$RunId"
$metaPath = Join-Path $runRoot "download.meta.json"
$pidPath = Join-Path $runRoot "download.pid"
if (!(Test-Path -LiteralPath $metaPath) -or !(Test-Path -LiteralPath $pidPath)) {
    throw "Missing download metadata for $RunId"
}
$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$pidValue = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$target = [string]$meta.target
$partial = Get-ChildItem -LiteralPath (Split-Path -Parent $target) -Filter "$(Split-Path -Leaf $target)*.part" `
    -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$currentPath = if (Test-Path -LiteralPath $target) { $target } elseif ($partial) { $partial.FullName } else { $null }
$size = if ($currentPath) { (Get-Item -LiteralPath $currentPath).Length } else { 0 }
$timestamp = if ($currentPath) { (Get-Item -LiteralPath $currentPath).LastWriteTime.ToString("o") } else { $null }

[pscustomobject]@{
    run_id = $RunId
    running = [bool]$process
    pid = $pidValue
    start_time = $meta.start_time
    downloaded_bytes = $size
    downloaded_mib = [math]::Round($size / 1MB, 2)
    last_output_timestamp = $timestamp
    target_complete = Test-Path -LiteralPath $target
    target = $target
    stdout = $meta.stdout
    stderr = $meta.stderr
} | ConvertTo-Json
