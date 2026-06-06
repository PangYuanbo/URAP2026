param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\route_b_oracle_action_chunks',
  [string]$RunId = 'route_b_oracle_action_chunks',
  [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "NOT RUNNING: pid file missing: $pidFile"
  exit 0
}

$pidText = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
$proc = $null
if ($pidText -match '^\d+$') {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}

$meta = @{}
if (Test-Path -Path $metaFile -PathType Leaf) {
  foreach ($line in Get-Content $metaFile) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
      $meta[$parts[0]] = $parts[1]
    }
  }
}
if (-not $OutDir -and $meta.ContainsKey('out_dir')) {
  $OutDir = [string]$meta['out_dir']
}

$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) {
      $lastOutput = $item.LastWriteTime
    }
  }
}

$sourceProgress = @()
$doneSources = 0
$totalSources = 0
if ($meta.ContainsKey('source_names')) {
  $sourceNames = ([string]$meta['source_names']).Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
  $totalSources = $sourceNames.Count
  foreach ($source in $sourceNames) {
    $sourceDir = Join-Path $OutDir $source
    $summary = Join-Path $sourceDir 'summary.json'
    $action = Join-Path $sourceDir 'action_chunk_samples.jsonl'
    $oracleRows = 0
    $actionRows = 0
    if (Test-Path -Path $summary -PathType Leaf) {
      $doneSources += 1
      try {
        $json = Get-Content $summary -Raw | ConvertFrom-Json
        $oracleRows = [int]$json.num_tracklets
      } catch {}
    }
    if (Test-Path -Path $action -PathType Leaf) {
      Get-Content $action | ForEach-Object { $actionRows += 1 }
    }
    $sourceProgress += ("{0}:oracle_tracklets={1},action_chunks={2}" -f $source, $oracleRows, $actionRows)
  }
}

$pipelineSummary = if ($OutDir) { Join-Path $OutDir 'oracle_action_chunk_pipeline_summary.json' } else { '' }
$splitManifest = if ($OutDir) { Join-Path $OutDir 'split\action_chunk_split_manifest.json' } else { '' }
$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }

Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $doneSources, $totalSources)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } elseif ($null -ne $proc) { $proc.CreationDate } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $(if (Test-Path -Path $pipelineSummary -PathType Leaf) { 'pipeline_summary' } elseif (Test-Path -Path $splitManifest -PathType Leaf) { 'split_manifest' } else { ($sourceProgress -join '; ') }))
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_dir={0}" -f $OutDir)

if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 30
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 30
}
