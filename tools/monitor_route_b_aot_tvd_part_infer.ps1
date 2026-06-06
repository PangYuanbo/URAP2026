param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_part0_tvd_val_runner'),
  [string]$RunId = 'aot_part0_tvd_val'
)

$ErrorActionPreference = 'Stop'

function Read-Meta([string]$Path) {
  $meta = @{}
  if (Test-Path -Path $Path -PathType Leaf) {
    foreach ($line in Get-Content $Path) {
      $idx = $line.IndexOf('=')
      if ($idx -gt 0) {
        $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1)
      }
    }
  }
  return $meta
}

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not (Test-Path -Path $metaFile -PathType Leaf)) { throw "Meta file not found: $metaFile" }

Write-Host '== Meta =='
Get-Content $metaFile | Select-Object -First 160
Write-Host ''

$meta = Read-Meta $metaFile
$pidText = if (Test-Path -Path $pidFile -PathType Leaf) { (Get-Content $pidFile | Select-Object -First 1) } else { $meta['pid'] }
$process = $null
if ($pidText -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}
if ($process -and $process.CommandLine -like '*val.py*' -and $process.CommandLine -like '*--save-aot-predictions*') {
  Write-Host ("RUNNING=true PID={0}" -f $pidText)
  try {
    $p = Get-Process -Id ([int]$pidText) -ErrorAction Stop
    Write-Host ("PID_START={0}" -f $p.StartTime.ToString('yyyy-MM-dd HH:mm:ss'))
  } catch {}
  Write-Host ("PROCESS_COMMAND={0}" -f $process.CommandLine)
} else {
  Write-Host ("NOT RUNNING PID={0}" -f $pidText)
}
Write-Host ''

$predictionPart = $meta['prediction_part']
$saveDir = $meta['save_dir']
$stdout = $meta['stdout']
$stderr = $meta['stderr']

$done = 0
$total = 1
$lastUnit = ''
$times = @()
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) { $times += (Get-Item $stdout).LastWriteTime }
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) { $times += (Get-Item $stderr).LastWriteTime }
if ($predictionPart -and (Test-Path -Path $predictionPart -PathType Leaf)) {
  $done = 1
  $lastUnit = 'aot_prediction_part'
  $times += (Get-Item $predictionPart).LastWriteTime
  Write-Host ("AOT_PREDICTION_PART={0}" -f $predictionPart)
  try {
    $summary = & (Join-Path $meta['repo_root'] 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe') -c "import pickle,sys,json; rows=pickle.load(open(sys.argv[1],'rb')); print(json.dumps({'records':len(rows),'detections':sum(len(r.get('detections',[])) for r in rows)}, indent=2))" $predictionPart
    Write-Host 'AOT_PREDICTION_SUMMARY='
    Write-Host $summary
  } catch {
    Write-Host ("AOT_PREDICTION_SUMMARY_ERROR={0}" -f $_.Exception.Message)
  }
}
if ($saveDir -and (Test-Path -Path $saveDir -PathType Container)) {
  $times += (Get-Item $saveDir).LastWriteTime
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try {
    Write-Host ''
    Write-Host '== GPU =='
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
  } catch {
    Write-Host ("GPU_QUERY_ERROR={0}" -f $_.Exception.Message)
  }
}

$lastTs = if ($times.Count -gt 0) { ($times | Sort-Object | Select-Object -Last 1).ToString('yyyy-MM-dd HH:mm:ss') } else { '' }
Write-Host ''
Write-Host ("done/total: {0}/{1}" -f $done, $total)
Write-Host ("last output timestamp: {0}" -f $lastTs)
Write-Host ("last completed unit: {0}" -f $lastUnit)
Write-Host ("stdout log: {0}" -f $stdout)
Write-Host ("stderr log: {0}" -f $stderr)

if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host ''
  Write-Host '== stdout tail =='
  Get-Content $stdout -Tail 40
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host ''
  Write-Host '== stderr tail =='
  Get-Content $stderr -Tail 80
}
