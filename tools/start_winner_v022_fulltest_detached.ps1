param(
  [string]$DatasetPath = "D:\URAP_datasets\AOT\part1\Images",
  [string]$FlightIdsJson = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\aot_flight_ids\testflightidsfull1.json",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\submission-v022\airborne-detection-starter-kit-submission-v022",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_fulltest",
  [string]$RunId = "fulltest"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $DatasetPath -PathType Container)) { throw "DatasetPath not found: $DatasetPath" }
if (-not (Test-Path -Path $FlightIdsJson -PathType Leaf)) { throw "FlightIdsJson not found: $FlightIdsJson" }
if (-not (Test-Path -Path $RepoDir -PathType Container)) { throw "RepoDir not found: $RepoDir" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$runOut = Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Force -Path $runOut | Out-Null

$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

# Prevent duplicate concurrent runs.
if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) {
      Write-Host "Already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 50 }
      exit 0
    }
  }
}

$ids = Get-Content $FlightIdsJson | ConvertFrom-Json
if ($null -eq $ids -or $ids.Count -eq 0) { throw "No flight ids loaded from $FlightIdsJson" }

$done = @{}
Get-ChildItem -Path $runOut -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $rid = Join-Path $_.FullName "result.json"
  if (Test-Path -Path $rid -PathType Leaf) { $done[$_.Name] = $true }
}

$remaining = @()
foreach ($id in $ids) {
  if (-not $done.ContainsKey($id)) { $remaining += $id }
}

Write-Host ("Total test flights: {0}" -f $ids.Count)
Write-Host ("Already done:       {0}" -f $done.Keys.Count)
Write-Host ("Remaining:          {0}" -f $remaining.Count)

if ($remaining.Count -eq 0) {
  Write-Host "Nothing to do."
  exit 0
}

# Env vars are inherited by the detached process.
$env:TEST_DATASET_PATH = $DatasetPath
$env:INFERENCE_OUTPUT_PATH = $OutputRoot
$env:DATASET_ENV = $RunId
$env:PARTIAL_RUN_FLIGHTS = ($remaining -join ",")

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$p = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList @(".\seg_test.py") `
  -WorkingDirectory $RepoDir `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$p.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  ("pid={0}" -f $p.Id)
  ("repo_dir={0}" -f $RepoDir)
  ("python={0}" -f $PythonExe)
  ("dataset_path={0}" -f $DatasetPath)
  ("run_id={0}" -f $RunId)
  ("stdout={0}" -f $stdout)
  ("stderr={0}" -f $stderr)
  ("done_at_start={0}" -f $done.Keys.Count)
  ("remaining_at_start={0}" -f $remaining.Count)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached runner."
Get-Content $metaFile
