param(
  [string]$RunId = 'li_tetc_nps_proposal_export',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_li_tetc_compare\proposal_export_runner'),
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "status=NOT RUNNING"
  Write-Host "done/total=0/1"
  Write-Host "pid="
  Write-Host "start_time="
  Write-Host "last_output_timestamp="
  Write-Host "last_completed_unit="
  Write-Host "pid_file_missing=$pidFile"
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
$outSummary = if ($meta.ContainsKey('out_summary')) { [string]$meta['out_summary'] } else { '' }
$outRunRoot = if ($meta.ContainsKey('out_run_root')) { [string]$meta['out_run_root'] } else { '' }
$lastOutput = $null
foreach ($path in @($stdout, $stderr, $outSummary)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) {
    $t = (Get-Item $path).LastWriteTime
    if ($null -eq $lastOutput -or $t -gt $lastOutput) { $lastOutput = $t }
  }
}
$done = 0
$lastUnit = ''
if ($outSummary -and (Test-Path -Path $outSummary -PathType Leaf)) {
  $done = 1
  $lastUnit = 'export_summary.json'
}
$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/1" -f $done)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastUnit)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_run_root={0}" -f $outRunRoot)
Write-Host ("out_summary={0}" -f $outSummary)
if ($null -ne $proc) {
  Write-Host ("process_command={0}" -f $proc.CommandLine)
}
Write-Host 'gpu_signal:'
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
} else {
  Write-Host 'nvidia-smi not found'
}
if ($outSummary -and (Test-Path -Path $outSummary -PathType Leaf)) {
  Write-Host 'summary_head:'
  Get-Content $outSummary -TotalCount 80
}
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail $TailLines
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail $TailLines
}
