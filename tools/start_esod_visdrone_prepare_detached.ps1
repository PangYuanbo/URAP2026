param(
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\ESOD",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\ESOD\.venv\Scripts\python.exe",
  [string]$DatasetDir = "C:\Users\aaron\Desktop\URAP\papers\ESOD\VisDrone",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\ESOD\runs\visdrone_prepare",
  [string]$RunId = "visdrone_prepare"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $RepoDir -PathType Container)) { throw "RepoDir not found: $RepoDir" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $DatasetDir -PathType Container)) { throw "DatasetDir not found: $DatasetDir" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
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
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 80 }
      exit 0
    }
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$p = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList @(".\scripts\data_prepare.py", "--dataset", $DatasetDir) `
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
  ("dataset_dir={0}" -f $DatasetDir)
  ("run_id={0}" -f $RunId)
  ("stdout={0}" -f $stdout)
  ("stderr={0}" -f $stderr)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached VisDrone prepare runner."
Get-Content $metaFile

