param(
    [ValidateSet("train20", "val5", "adapt5")]
    [string]$Split = "train20",
    [string]$Out = "",
    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"

if ($Out -eq "") {
    $Out = "D:\datasets\Anti-UAV300\qstr_scene_recovery_profiles\${Split}_v2_broad_20260531"
}

$monitor = Join-Path $PSScriptRoot "monitor_qstr_stage_b_source_scene_profile.ps1"
if (-not (Test-Path -LiteralPath $monitor)) {
    throw "Missing generic monitor: $monitor"
}

& $monitor `
    -Out $Out `
    -ProfileName "antiuav_scene_recovery_select" `
    -Tail $Tail
