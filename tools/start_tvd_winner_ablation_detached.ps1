param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\runs\ablation\winner_port_v1"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot "runner_pid.txt"
$metaFile = Join-Path $OutputRoot "runner_meta.txt"

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
$stdout = Join-Path $logsDir ("runner_{0}.out.txt" -f $ts)
$stderr = Join-Path $logsDir ("runner_{0}.err.txt" -f $ts)

$runner = Join-Path $URAPRoot "tools\\run_tvd_winner_ablation.ps1"
if (-not (Test-Path -Path $runner -PathType Leaf)) { throw "Missing runner script: $runner" }

$argList = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $runner,
  "-URAPRoot", $URAPRoot,
  "-OutputRoot", $OutputRoot
)

$p = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList $argList `
  -WorkingDirectory $URAPRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$p.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  ("pid={0}" -f $p.Id)
  ("cwd={0}" -f $URAPRoot)
  ("output_root={0}" -f $OutputRoot)
  ("stdout={0}" -f $stdout)
  ("stderr={0}" -f $stderr)
  ("runner={0}" -f $runner)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached ablation runner."
Get-Content $metaFile

