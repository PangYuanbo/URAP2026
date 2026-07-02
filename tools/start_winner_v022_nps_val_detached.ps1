param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$DatasetPath = "D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\submission-v022\airborne-detection-starter-kit-submission-v022",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val",
  [string]$RunId = "nps_val",
  [string]$PreparedDatasetPath = "",
  [string]$JobId = "winner_v022_nps_val",
  [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $URAPRoot -PathType Container)) { throw "URAPRoot not found: $URAPRoot" }
if (-not (Test-Path -Path $DatasetPath -PathType Container)) { throw "DatasetPath not found: $DatasetPath" }
if (-not (Test-Path -Path $RepoDir -PathType Container)) { throw "RepoDir not found: $RepoDir" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }

$runner = Join-Path $URAPRoot "tools\run_winner_v022_nps_val.ps1"
if (-not (Test-Path -Path $runner -PathType Leaf)) { throw "Runner not found: $runner" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$runOut = Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Force -Path $runOut | Out-Null
$logsDir = Join-Path $runOut "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $runOut ("{0}_pid.txt" -f $JobId)
$metaFile = Join-Path $runOut ("{0}_meta.json" -f $JobId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
      Write-Host ("RUNNING pid={0}" -f $existingPid)
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$allClips = @(Get-ChildItem -Path $DatasetPath -Filter "*.png" -File -ErrorAction SilentlyContinue | ForEach-Object {
  $parts = $_.BaseName -split "_"
  if ($parts.Length -ge 3 -and $parts[0] -eq "Clip") { ($parts[0..1] -join "_") }
} | Where-Object { $_ } | Sort-Object -Unique)
if ($allClips.Count -eq 0 -and $SkipPrepare) {
  $allClips = @(Get-ChildItem -Path $DatasetPath -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name | Sort-Object -Unique)
}

$done = @{}
Get-ChildItem -Path $runOut -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $rid = Join-Path $_.FullName "result.json"
  if (Test-Path -Path $rid -PathType Leaf) { $done[$_.Name] = $true }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir ("{0}_{1}.out.log" -f $JobId, $ts)
$stderr = Join-Path $logsDir ("{0}_{1}.err.log" -f $JobId, $ts)

$argList = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $runner,
  "-URAPRoot", $URAPRoot,
  "-DatasetPath", $DatasetPath,
  "-RepoDir", $RepoDir,
  "-PythonExe", $PythonExe,
  "-OutputRoot", $OutputRoot,
  "-RunId", $RunId
)
if ($PreparedDatasetPath) { $argList += @("-PreparedDatasetPath", $PreparedDatasetPath) }
if ($SkipPrepare) { $argList += "-SkipPrepare" }

$proc = Start-Process `
  -FilePath "powershell" `
  -ArgumentList $argList `
  -WorkingDirectory $URAPRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Encoding ascii -Path $pidFile -Value $proc.Id
@{
  start_time = (Get-Date).ToString("o")
  pid = $proc.Id
  job_id = $JobId
  run_id = $RunId
  dataset_path = $DatasetPath
  prepared_dataset_path = $PreparedDatasetPath
  output_root = $OutputRoot
  run_output = $runOut
  repo_dir = $RepoDir
  python = $PythonExe
  stdout_log = $stdout
  stderr_log = $stderr
  total_clips = $allClips.Count
  done_at_start = $done.Keys.Count
  command = "powershell $($argList -join ' ')"
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii -Path $metaFile

Write-Host "RUNNING"
Write-Host ("done/total: {0}/{1}" -f $done.Keys.Count, $allClips.Count)
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
Write-Host "last_completed_unit: launched AICrowd winner NPS runner"
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
Write-Host ("output_root: {0}" -f $OutputRoot)
Write-Host ("run_output: {0}" -f $runOut)
