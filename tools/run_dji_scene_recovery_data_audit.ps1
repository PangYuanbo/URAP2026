param(
  [string]$AnnotationsDir = "D:\datasets\my_video\final_annotations",
  [string]$Out = "reports\dji_scene_recovery_data_audit"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python tools\audit_dji_scene_recovery_data.py `
  --annotations-dir $AnnotationsDir `
  --out $Out
