param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_build_proposal_tracklets_runner',
  [string]$RunId = 'yolomg_build_proposal_tracklets'
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "NOT RUNNING: pid file missing: $pidFile"
  exit 0
}
$pidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
$proc = $null
if ($pidText -match '^\d+$') {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}
$meta = @{}
if (Test-Path -Path $metaFile -PathType Leaf) {
  foreach ($line in Get-Content $metaFile) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) { $meta[$parts[0]] = $parts[1] }
  }
}
$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$outDir = if ($meta.ContainsKey('out_dir')) { [string]$meta['out_dir'] } else { '' }
$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) { $lastOutput = $item.LastWriteTime }
  }
}
$summary = if ($outDir) { Join-Path $outDir 'summary.json' } else { '' }
$jsonl = if ($outDir) { Join-Path $outDir 'proposal_tracklets.jsonl' } else { '' }
$done = 0
$total = 1
$lastUnit = ''
if ($summary -and (Test-Path -Path $summary -PathType Leaf)) {
  $done = 1
  $lastUnit = 'summary.json'
} elseif ($jsonl -and (Test-Path -Path $jsonl -PathType Leaf)) {
  $lastUnit = 'proposal_tracklets.jsonl'
}
$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $total)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastUnit)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_dir={0}" -f $outDir)
if ($summary -and (Test-Path -Path $summary -PathType Leaf)) {
  Write-Host 'summary_head:'
  Get-Content $summary -TotalCount 80
}
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 30
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 30
}
