param(
  [string]$RunId = "aot_fulltest_conf0p1_wport_baseline",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\route_b_official\aot_fulltest_conf0p1_runner",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"

function Read-Meta([string]$Path) {
  $meta = @{}
  if (Test-Path -Path $Path -PathType Leaf) {
    foreach ($line in Get-Content $Path) {
      $idx = $line.IndexOf("=")
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

Write-Host "== Meta =="
Get-Content $metaFile | Select-Object -First 180
Write-Host ""

$meta = Read-Meta $metaFile
$pidText = if (Test-Path -Path $pidFile -PathType Leaf) { Get-Content $pidFile | Select-Object -First 1 } else { $meta["pid"] }
$process = $null
if ($pidText -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}
if ($process -and $process.CommandLine -like "*run_aot_full_test.ps1*") {
  Write-Host ("RUNNING=true PID={0}" -f $pidText)
  try {
    $p = Get-Process -Id ([int]$pidText) -ErrorAction Stop
    Write-Host ("PID_START={0}" -f $p.StartTime.ToString("yyyy-MM-dd HH:mm:ss"))
  } catch {}
  Write-Host ("PROCESS_COMMAND={0}" -f $process.CommandLine)
} else {
  Write-Host ("NOT RUNNING PID={0}" -f $pidText)
}
Write-Host ""

$predDir = $meta["prediction_dir"]
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$times = @()
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) { $times += (Get-Item $stdout).LastWriteTime }
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) { $times += (Get-Item $stderr).LastWriteTime }

$done = 0
$total = 0
$latestUnit = ""
$predMb = 0.0
if ($predDir) {
  $yamlDir = Join-Path (Join-Path $meta["cwd"] "papers\TransVisDrone") "data\AOTTestSplits_URAP"
  if (Test-Path -Path $yamlDir -PathType Container) {
    $total = (Get-ChildItem -Path $yamlDir -Filter "AOTTest_*.yaml" -File -ErrorAction SilentlyContinue | Measure-Object).Count
  }
  if (Test-Path -Path $predDir -PathType Container) {
    $parts = @(Get-ChildItem -Path $predDir -Filter "predictions_split_*.pkl" -File -ErrorAction SilentlyContinue)
    $done = $parts.Count
    if ($parts.Count -gt 0) {
      $latest = $parts | Sort-Object LastWriteTime -Descending | Select-Object -First 1
      $latestUnit = $latest.Name
      $times += $latest.LastWriteTime
      $predMb = [math]::Round((($parts | Measure-Object Length -Sum).Sum / 1MB), 3)
      Write-Host "== Latest Prediction Parts =="
      $parts | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,LastWriteTime,@{Name="MB";Expression={[math]::Round($_.Length/1MB,3)}} | Format-Table -AutoSize
      Write-Host ""
    }
  }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try {
    Write-Host "== GPU signal =="
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    Write-Host ""
  } catch {
    Write-Host ("GPU_QUERY_ERROR={0}" -f $_.Exception.Message)
  }
}

$lastTs = if ($times.Count -gt 0) { ($times | Sort-Object | Select-Object -Last 1).ToString("yyyy-MM-dd HH:mm:ss") } else { "" }
Write-Host ("done/total: {0}/{1}" -f $done, $total)
Write-Host ("prediction parts MB: {0}" -f $predMb)
Write-Host ("last output timestamp: {0}" -f $lastTs)
Write-Host ("last completed unit: {0}" -f $latestUnit)
Write-Host ("prediction dir: {0}" -f $predDir)
Write-Host ("stdout log: {0}" -f $stdout)
Write-Host ("stderr log: {0}" -f $stderr)

if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host ""
  Write-Host "== stdout tail =="
  Get-Content $stdout -Tail $TailLines
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host ""
  Write-Host "== stderr tail =="
  Get-Content $stderr -Tail $TailLines
}
