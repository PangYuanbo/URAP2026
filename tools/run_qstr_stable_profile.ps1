param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [Parameter(Mandatory = $true)]
    [string]$Out,
    [string]$PrimaryYoloWeights = "runs\detect\runs\detect\anti_uav300_visible_5seq_yolo_p2_gpu\yolo_p2_candidate\weights\best.pt",
    [string]$CropWeights = "D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\crop_drone_binary_e12.pt",
    [string]$TemporalWeights = "D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\temporal_drone_binary_e12.pt",
    [string]$Device = "0",
    [double]$YoloConf = 0.05,
    [int]$TileSize = 256,
    [int]$TileStride = 128,
    [int]$MaxYoloCandidatesPerFrame = 10,
    [int]$MaxCandidatesPerFrame = 10,
    [string]$TrackletClassifierWeights = "",
    [double]$TrackletClassifierThreshold = 0.5,
    [ValidateSet("keep", "suppress")]
    [string]$TrackletFilterUntracked = "keep",
    [switch]$DisableTrackletPromotion,
    [double]$TrackletPromotionScoreFloor = 0.22,
    [double]$TrackletPromotionMinBranchDrone = 0.40,
    [double]$TrackletPromotionMaxBackground = 0.68,
    [int]$MaxFrames = 0,
    [switch]$EnableMotionCandidates,
    [switch]$SaveVideo
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

foreach ($PathToCheck in @($Video, $PrimaryYoloWeights, $CropWeights, $TemporalWeights)) {
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
    "--crop-weights", $CropWeights,
    "--temporal-weights", $TemporalWeights,
    "--recognition-crop-scale", "2.0",
    "--recognition-tube-scale", "2.0",
    "--disable-verified-objectness"
)

if (-not $EnableMotionCandidates) {
    $ArgsList += "--disable-motion-candidates"
}
if ($MaxFrames -gt 0) {
    $ArgsList += @("--max-frames", "$MaxFrames")
}
if ($TrackletClassifierWeights -ne "") {
    if (-not (Test-Path $TrackletClassifierWeights)) {
        throw "Missing tracklet classifier weights: $TrackletClassifierWeights"
    }
    $ArgsList += @(
        "--tracklet-classifier-weights", $TrackletClassifierWeights,
        "--tracklet-classifier-threshold", "$TrackletClassifierThreshold",
        "--tracklet-filter-untracked", $TrackletFilterUntracked,
        "--tracklet-promotion-score-floor", "$TrackletPromotionScoreFloor",
        "--tracklet-promotion-min-branch-drone", "$TrackletPromotionMinBranchDrone",
        "--tracklet-promotion-max-background", "$TrackletPromotionMaxBackground"
    )
    if ($DisableTrackletPromotion) {
        $ArgsList += "--disable-tracklet-promotion"
    }
}
if ($SaveVideo) {
    $ArgsList += "--save-video"
}

Write-Host "=== QSTR stable/default profile ==="
Write-Host "Video: $Video"
Write-Host "Out:   $Out"
python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Stable profile failed with exit code $LASTEXITCODE"
}
