param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'fl_drones_official_download',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\benchmarks\fl_drones_download'),
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("{0}_meta.txt" -f $RunId)
if (-not (Test-Path -LiteralPath $metaFile)) { throw "Meta file missing: $metaFile" }
$meta = Get-Content -LiteralPath $metaFile
function Get-MetaValue([string]$Key) {
  $line = $meta | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1
  if ($line) { return $line.Substring($Key.Length + 1) }
  return $null
}

$pidValue = if (Test-Path -LiteralPath $pidFile) { Get-Content -LiteralPath $pidFile | Select-Object -First 1 } else { $null }
$process = if ($pidValue -match '^\d+$') { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$outputDir = Get-MetaValue 'output_dir'
$outputFile = Get-MetaValue 'output_file'
$stdout = Get-MetaValue 'stdout'
$stderr = Get-MetaValue 'stderr'
$files = if ($outputDir -and (Test-Path -LiteralPath $outputDir)) { @(Get-ChildItem -LiteralPath $outputDir -File -Recurse -ErrorAction SilentlyContinue) } else { @() }
$videos = @($files | Where-Object Extension -in '.avi', '.mp4', '.mov', '.mkv')
$annotations = @($files | Where-Object Extension -in '.txt', '.xml', '.json', '.csv')
$bytes = ($files | Measure-Object Length -Sum).Sum
$lastOutput = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($process -and $process.CommandLine -like '*gdown*18CoTpjMs80dfanYNpbznjL4e-KB_Diel*') {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$($process.CreationDate)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host "NOT RUNNING PID=$pidValue"
}
Write-Host "done/total: videos=$($videos.Count)/14 files=$($files.Count)/unknown"
Write-Host "annotations=$($annotations.Count) bytes=$bytes"
if ($outputFile) { Write-Host "archive=$outputFile archive_exists=$(Test-Path -LiteralPath $outputFile)" }
Write-Host "last output timestamp: $(if($lastOutput){$lastOutput.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')}else{'missing'})"
Write-Host "last completed unit: $(if($lastOutput){$lastOutput.FullName}else{'none'})"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"
if ($stdout -and (Test-Path -LiteralPath $stdout)) { Write-Host '== stdout tail =='; Get-Content -LiteralPath $stdout -Tail $TailLines }
if ($stderr -and (Test-Path -LiteralPath $stderr)) { Write-Host '== stderr tail =='; Get-Content -LiteralPath $stderr -Tail $TailLines }
