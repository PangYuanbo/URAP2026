param(
    [ValidateSet("train20", "val5", "adapt5")]
    [string]$Split = "train20",
    [string]$Out = "",
    [string]$StageAWeights = "D:\datasets\stage_a_mixed\runs\ard100_dji_aot_yolo_p2_v2\yolo_p2_candidate\weights\best.pt",
    [string]$RecallCropWeights = "C:\Users\pc\Desktop\tiny object detection\runs\dji_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt",
    [string]$RecallTemporalWeights = "C:\Users\pc\Desktop\tiny object detection\runs\dji_stage_b_hardneg_repair_20260530\models\temporal_binary_dji_hardneg.pt",
    [string]$StrictCropWeights = "C:\Users\pc\Desktop\tiny object detection\runs\dji_dense_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt",
    [string]$StrictTemporalWeights = "C:\Users\pc\Desktop\tiny object detection\runs\dji_dense_stage_b_hardneg_repair_20260530\models\temporal_binary_dji_hardneg.pt",
    [string]$TrackletClassifierWeights = "D:\datasets\stage_a_mixed\ura18_temporal_only\models\tracklet_mlp_temporal_only_v3_dji90.pt",
    [string]$Device = "0",
    [int]$MaxFramesPerVideo = 0,
    [int]$FrameStride = 5,
    [int]$MaxYoloCandidatesPerFrame = 20,
    [int]$MaxCandidatesPerFrame = 20,
    [double]$YoloConf = 0.05
)

$ErrorActionPreference = "Stop"

$annotationsBySplit = @{
    train20 = "D:\datasets\Anti-UAV300\qstr_train_visible_20seq\annotations\qstr_real_boxes.csv"
    val5 = "D:\datasets\Anti-UAV300\qstr_train_visible_val_5seq\annotations\qstr_real_boxes.csv"
    adapt5 = "D:\datasets\Anti-UAV300\qstr_adapt_test_visible_5seq\annotations\qstr_real_boxes.csv"
}

if ($Out -eq "") {
    $Out = "D:\datasets\Anti-UAV300\qstr_scene_recovery_profiles\${Split}_v2_broad_20260531"
}

$launcher = Join-Path $PSScriptRoot "start_qstr_stage_b_source_scene_profile_detached.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Missing generic source/scene launcher: $launcher"
}

Write-Host "Anti-UAV scene-recovery profile run"
Write-Host "Split:       $Split"
Write-Host "Annotations: $($annotationsBySplit[$Split])"
Write-Host "Out:         $Out"
Write-Host "Stage A:     $StageAWeights"

& $launcher `
    -Out $Out `
    -Annotations $annotationsBySplit[$Split] `
    -BalancedWeights $StageAWeights `
    -RecallCropWeights $RecallCropWeights `
    -RecallTemporalWeights $RecallTemporalWeights `
    -StrictCropWeights $StrictCropWeights `
    -StrictTemporalWeights $StrictTemporalWeights `
    -TrackletClassifierWeights $TrackletClassifierWeights `
    -Device $Device `
    -YoloConf $YoloConf `
    -TileSize 256 `
    -TileStride 128 `
    -MaxYoloCandidatesPerFrame $MaxYoloCandidatesPerFrame `
    -MaxCandidatesPerFrame $MaxCandidatesPerFrame `
    -MaxFramesPerVideo $MaxFramesPerVideo `
    -FrameStride $FrameStride `
    -ProfileName "antiuav_scene_recovery_select" `
    -HardTinyMaxSide 48 `
    -RecallMinScore 0.18 `
    -RecallMinProb 0.55 `
    -RecallMaxBackground 0.60
