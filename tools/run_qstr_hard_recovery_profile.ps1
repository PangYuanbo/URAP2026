param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [Parameter(Mandatory = $true)]
    [string]$Out,
    [string]$PrimaryYoloWeights = "runs\detect\runs\detect\anti_uav300_visible_5seq_yolo_p2_gpu\yolo_p2_candidate\weights\best.pt",
    [string]$FallbackYoloWeights = "runs\detect\runs\detect\anti_uav300_stage_a_hardneg_v2_train20_adapt5_yolo_p2_gpu\yolo_p2_candidate\weights\best.pt",
    [string]$CropWeights = "D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\crop_drone_binary_e12.pt",
    [string]$TemporalWeights = "D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\temporal_drone_binary_e12.pt",
    [string]$Device = "0",
    [double]$YoloConf = 0.05,
    [double]$FallbackYoloConf = 0.15,
    [int]$TileSize = 256,
    [int]$TileStride = 128,
    [int]$MaxYoloCandidatesPerFrame = 10,
    [int]$MaxFallbackYoloCandidatesPerFrame = 5,
    [int]$MaxCandidatesPerFrame = 10,
    [double]$FallbackTriggerFinalScore = 0.50,
    [double]$FallbackPostTriggerMaxPrimaryObjectness = 0.35,
    [double]$FallbackMaxBoxSide = 128.0,
    [double]$FallbackGateMinBranchDrone = 0.45,
    [double]$FallbackGateMinCropTemporalMean = 0.48,
    [double]$FallbackGateMaxNegativeEvidence = 0.62,
    [double]$VerifiedMinBranchDrone = 0.45,
    [double]$VerifiedMinCropTemporalMean = 0.48,
    [double]$VerifiedMaxNegativeEvidence = 0.62,
    [double]$VerifiedObjectnessFloor = 0.55,
    [int]$MaxFrames = 0,
    [switch]$EnableMotionCandidates,
    [switch]$SaveVideo
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

foreach ($PathToCheck in @($Video, $PrimaryYoloWeights, $FallbackYoloWeights, $CropWeights, $TemporalWeights)) {
    if (-not (Test-Path $PathToCheck)) {
        throw "Missing required path: $PathToCheck"
    }
}

$ArgsList = @(
    "-m", "qstr_dronedet.cli", "infer",
    "--video", $Video,
    "--out", $Out,
    "--yolo-weights", $PrimaryYoloWeights,
    "--yolo-conf", "$YoloConf",
    "--yolo-tile-size", "$TileSize",
    "--yolo-tile-stride", "$TileStride",
    "--yolo-device", $Device,
    "--max-yolo-candidates-per-frame", "$MaxYoloCandidatesPerFrame",
    "--max-candidates-per-frame", "$MaxCandidatesPerFrame",
    "--fallback-yolo-weights", $FallbackYoloWeights,
    "--fallback-yolo-conf", "$FallbackYoloConf",
    "--fallback-yolo-tile-size", "$TileSize",
    "--fallback-yolo-tile-stride", "$TileStride",
    "--fallback-trigger-final-score", "$FallbackTriggerFinalScore",
    "--fallback-post-trigger-max-primary-objectness", "$FallbackPostTriggerMaxPrimaryObjectness",
    "--max-fallback-yolo-candidates-per-frame", "$MaxFallbackYoloCandidatesPerFrame",
    "--fallback-max-box-side", "$FallbackMaxBoxSide",
    "--fallback-gate-min-branch-drone", "$FallbackGateMinBranchDrone",
    "--fallback-gate-min-crop-temporal-mean", "$FallbackGateMinCropTemporalMean",
    "--fallback-gate-max-negative-evidence", "$FallbackGateMaxNegativeEvidence",
    "--crop-weights", $CropWeights,
    "--temporal-weights", $TemporalWeights,
    "--recognition-crop-scale", "2.0",
    "--recognition-tube-scale", "2.0",
    "--verified-objectness-mode", "hard_recovery",
    "--verified-min-branch-drone", "$VerifiedMinBranchDrone",
    "--verified-min-crop-temporal-mean", "$VerifiedMinCropTemporalMean",
    "--verified-max-negative-evidence", "$VerifiedMaxNegativeEvidence",
    "--verified-objectness-floor", "$VerifiedObjectnessFloor"
)

if (-not $EnableMotionCandidates) {
    $ArgsList += "--disable-motion-candidates"
}
if ($MaxFrames -gt 0) {
    $ArgsList += @("--max-frames", "$MaxFrames")
}
if ($SaveVideo) {
    $ArgsList += "--save-video"
}

Write-Host "=== QSTR hard-recovery profile ==="
Write-Host "Video: $Video"
Write-Host "Out:   $Out"
python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Hard-recovery profile failed with exit code $LASTEXITCODE"
}
