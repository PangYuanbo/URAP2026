param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_eval_runner",
  [string]$RunId = "samurai_memory_gate_yolomg_val_eval",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "monitor_temporal_recovery_yolomg_val_eval.ps1") -RunRoot $RunRoot -RunId $RunId -TailLines $TailLines
