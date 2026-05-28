param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_curve_player_server',
  [string]$RunId = 'yolomg_curve_player_server',
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "Meta file not found: $metaFile"
  exit 1
}

$meta = Get-Content $metaFile
Write-Host '== Meta =='
$meta | Select-Object -First 120

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
}

Write-Host ''
if ($null -ne $proc) {
  Write-Host ('RUNNING=true PID={0}' -f $pidValue)
  Write-Host ('PID_START={0}' -f $proc.StartTime.ToString('yyyy-MM-dd HH:mm:ss'))
} else {
  Write-Host ('NOT RUNNING PID={0}' -f $pidValue)
}

$stdoutLine = ($meta | Where-Object { $_ -like 'stdout=*' } | Select-Object -First 1)
$stderrLine = ($meta | Where-Object { $_ -like 'stderr=*' } | Select-Object -First 1)
$stdoutPath = if ($stdoutLine) { $stdoutLine.Substring(7) } else { $null }
$stderrPath = if ($stderrLine) { $stderrLine.Substring(7) } else { $null }

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) {
  $stdoutItem = Get-Item $stdoutPath
  Write-Host ('STDOUT_LOG={0}' -f $stdoutPath)
  Write-Host ('STDOUT_LAST_WRITE={0}' -f $stdoutItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Write-Host '== stdout tail =='
  Get-Content -Path $stdoutPath -Tail $TailLines -ErrorAction SilentlyContinue
}

if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $stderrItem = Get-Item $stderrPath
  Write-Host ''
  Write-Host ('STDERR_LOG={0}' -f $stderrPath)
  Write-Host ('STDERR_LAST_WRITE={0}' -f $stderrItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Write-Host '== stderr tail =='
  Get-Content -Path $stderrPath -Tail $TailLines -ErrorAction SilentlyContinue
}
