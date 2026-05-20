param(
  [string]$AutoresearchDir = "C:\Users\aaron\Desktop\URAP\autoresearch",
  [string]$MainRepoDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$RunTag = "",
  [int]$MaxRounds = 12,
  [switch]$LocalOnly,
  [string]$Model = "gpt-5.4",
  [string]$Reasoning = "xhigh",
  [string]$CodexExe = "codex.cmd"
)

$ErrorActionPreference = "Stop"

$agendaFile = Join-Path $AutoresearchDir "prompts\section_reinforcement_learning.md"
if (-not (Test-Path -Path $agendaFile -PathType Leaf)) {
  throw "RL agenda file not found: $agendaFile"
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $RunTag = "urban-tiny-uav-rl-{0}" -f (Get-Date -Format "yyyyMMdd")
}

$args = @(
  "-NoLogo",
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $PSScriptRoot "start_autoresearch_codex_detached.ps1"),
  "-AutoresearchDir", $AutoresearchDir,
  "-MainRepoDir", $MainRepoDir,
  "-RunTag", $RunTag,
  "-MaxRounds", $MaxRounds,
  "-AgendaFile", $agendaFile,
  "-Model", $Model,
  "-Reasoning", $Reasoning,
  "-CodexExe", $CodexExe
)
if ($LocalOnly) { $args += "-LocalOnly" }

& powershell.exe @args
exit $LASTEXITCODE
