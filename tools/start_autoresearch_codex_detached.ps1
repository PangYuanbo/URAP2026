param(
  [string]$AutoresearchDir = "C:\Users\aaron\Desktop\URAP\autoresearch",
  [string]$MainRepoDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$RunTag = "",
  [int]$MaxRounds = 6,
  [switch]$LocalOnly,
  [string]$AgendaFile = "",
  [string]$Model = "gpt-5.4",
  [string]$Reasoning = "xhigh",
  [string]$CodexExe = "codex.cmd"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $AutoresearchDir -PathType Container)) { throw "AutoresearchDir not found: $AutoresearchDir" }
if (-not (Test-Path -Path $MainRepoDir -PathType Container)) { throw "MainRepoDir not found: $MainRepoDir" }
if (-not [string]::IsNullOrWhiteSpace($AgendaFile)) {
  if (-not (Test-Path -Path $AgendaFile -PathType Leaf)) { throw "AgendaFile not found: $AgendaFile" }
}

$runsRoot = Join-Path $AutoresearchDir "runs"
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $RunTag = "urban-tiny-uav-{0}" -f (Get-Date -Format "yyyyMMdd")
}

$runDir = Join-Path $runsRoot $RunTag
if (Test-Path -Path $runDir -PathType Container) {
  $RunTag = "{0}-{1}" -f $RunTag, (Get-Date -Format "HHmmss")
  $runDir = Join-Path $runsRoot $RunTag
}

$mirrorDir = Join-Path (Join-Path $MainRepoDir "doc\autoresearch") $RunTag
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $mirrorDir | Out-Null

$pidFile = Join-Path $runDir "runner_pid.txt"
$metaFile = Join-Path $runDir "runner_meta.txt"
$statusFile = Join-Path $runDir "run_status.json"
$stdout = Join-Path $runDir "runner_stdout.txt"
$stderr = Join-Path $runDir "runner_stderr.txt"
$latestRunFile = Join-Path $runsRoot "latest_run.txt"
$launcherCmd = Join-Path $runDir "launch_helper.cmd"

$helperScript = Join-Path $PSScriptRoot "run_autoresearch_codex_rounds.ps1"
if (-not (Test-Path -Path $helperScript -PathType Leaf)) { throw "Helper script not found: $helperScript" }

function Quote-CmdArg {
  param([string]$Value)
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '[\s"]') { return $Value }
  return '"' + ($Value -replace '"', '\"') + '"'
}

$argList = @(
  "-NoLogo",
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $helperScript,
  "-AutoresearchDir", $AutoresearchDir,
  "-MainRepoDir", $MainRepoDir,
  "-RunTag", $RunTag,
  "-RunDir", $runDir,
  "-MirrorDir", $mirrorDir,
  "-MaxRounds", $MaxRounds,
  "-Model", $Model,
  "-Reasoning", $Reasoning,
  "-CodexExe", $CodexExe
)
if ($LocalOnly) { $argList += "-LocalOnly" }
if (-not [string]::IsNullOrWhiteSpace($AgendaFile)) { $argList += @("-AgendaFile", $AgendaFile) }

$quotedPsArgs = ($argList | ForEach-Object { Quote-CmdArg $_ }) -join " "
$cmdContent = @(
  "@echo off",
  ('cd /d "{0}"' -f $AutoresearchDir),
  ('"powershell.exe" {0} 1>>"{1}" 2>>"{2}"' -f $quotedPsArgs, $stdout, $stderr),
  "exit /b %errorlevel%"
)
$cmdContent | Set-Content -Encoding ascii -Path $launcherCmd

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = ('/d /c ""{0}""' -f $launcherCmd)
$psi.WorkingDirectory = $AutoresearchDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$null = $proc.Start()

$proc.Id | Set-Content -Encoding ascii -Path $pidFile
$RunTag | Set-Content -Encoding ascii -Path $latestRunFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
  ("pid={0}" -f $proc.Id),
  ("run_tag={0}" -f $RunTag),
  ("run_dir={0}" -f $runDir),
  ("mirror_dir={0}" -f $mirrorDir),
  ("status_file={0}" -f $statusFile),
  ("stdout={0}" -f $stdout),
  ("stderr={0}" -f $stderr),
  ("launcher_cmd={0}" -f $launcherCmd),
  ("max_rounds={0}" -f $MaxRounds),
  ("local_only={0}" -f $LocalOnly.IsPresent),
  ("agenda_file={0}" -f $AgendaFile),
  ("model={0}" -f $Model),
  ("reasoning={0}" -f $Reasoning),
  ("codex_exe={0}" -f $CodexExe),
  ("cmd_args={0}" -f ($argList -join " "))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached Codex autoresearch run."
Get-Content $metaFile
