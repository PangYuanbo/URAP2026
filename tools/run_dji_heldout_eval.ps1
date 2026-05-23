param(
    [string]$Annotations = "D:\datasets\my_video\heldout_annotation_workspace\annotations\qstr_real_boxes_heldout.csv",
    [string]$OutRoot = "D:\datasets\my_video\qstr_heldout_eval",
    [string]$ArdYoloWeights = "C:\Users\pc\Desktop\tiny object detection\runs\detect\runs\detect\ard100_yolo_p2_sample_tiled256_v1\yolo_p2_candidate\weights\best.pt",
    [string]$DjiAdaptedYoloWeights = "D:\datasets\my_video\qstr_stage_a_adapt\runs\dji_ard100_mixed_yolo_p2_v1\yolo_p2_candidate\weights\best.pt",
    [string]$ArdCropWeights = "D:\datasets\ARD100\qstr_stage_b\models\crop_detector_proposal_train_sample_stride128_conf0025_top20_1000.pt",
    [string]$DjiCropWeights = "D:\datasets\my_video\qstr_stage_b\models\crop_dji_adapted_stagea_finetune_v1.pt",
    [string]$MixedCropWeights = "D:\datasets\my_video\qstr_stage_b\models\crop_mixed_ard100_dji_finetune_v1.pt",
    [string]$Device = "0",
    [double]$YoloConf = 0.025,
    [int]$TileSize = 256,
    [int]$TileStride = 128,
    [double]$ProposalNmsIou = 0.5,
    [int]$ProposalTopK = 20
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path $Annotations)) {
    throw "Held-out annotations not found: $Annotations"
}
if (-not (Test-Path $ArdYoloWeights)) {
    throw "ARD YOLO weights not found: $ArdYoloWeights"
}
if (-not (Test-Path $DjiAdaptedYoloWeights)) {
    throw "DJI-adapted YOLO weights not found: $DjiAdaptedYoloWeights"
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Run-StageARecall([string]$Name, [string]$Weights) {
    $out = Join-Path $OutRoot $Name
    python -m qstr_dronedet.cli stage-a-real-yolo-recall `
        --annotations $Annotations `
        --out $out `
        --yolo-weights $Weights `
        --yolo-conf $YoloConf `
        --yolo-tile-size $TileSize `
        --yolo-tile-stride $TileStride `
        --device $Device `
        --max-det 300 `
        --frame-stride 1 `
        --keep-top $ProposalTopK `
        --match-iou 0.3 `
        --match-center-px 16 `
        --proposal-nms-iou $ProposalNmsIou `
        --proposal-top-k $ProposalTopK `
        --class-name drone
}

Run-StageARecall "stage_a_ard100_baseline" $ArdYoloWeights
Run-StageARecall "stage_a_dji_adapted" $DjiAdaptedYoloWeights

$proposalOut = Join-Path $OutRoot "detector_proposals_dji_adapted"
python -m qstr_dronedet.cli build-real-detector-proposal-stage-b `
    --annotations $Annotations `
    --out $proposalOut `
    --yolo-weights $DjiAdaptedYoloWeights `
    --yolo-conf $YoloConf `
    --yolo-tile-size $TileSize `
    --yolo-tile-stride $TileStride `
    --device $Device `
    --max-proposals-per-frame $ProposalTopK `
    --proposal-nms-iou $ProposalNmsIou `
    --max-negatives-per-frame 8 `
    --match-iou 0.1 `
    --match-center-px 24 `
    --non-drone-label-mode binary_buckets `
    --hard-positive-repeat 1

$manifest = Join-Path $proposalOut "proposal_manifest.jsonl"
if (-not (Test-Path $manifest)) {
    throw "Proposal manifest was not created: $manifest"
}

$evals = @(
    @{ Name = "stage_b_ard_crop"; Weights = $ArdCropWeights },
    @{ Name = "stage_b_dji_crop"; Weights = $DjiCropWeights },
    @{ Name = "stage_b_mixed_crop"; Weights = $MixedCropWeights }
)

foreach ($eval in $evals) {
    if (Test-Path $eval.Weights) {
        python -m qstr_dronedet.cli eval-proposal-stage-b `
            --manifest $manifest `
            --crop-weights $eval.Weights `
            --out (Join-Path $OutRoot $eval.Name) `
            --threshold 0.5 `
            --batch-size 64
    }
    else {
        Write-Warning "Skipping $($eval.Name); weights not found: $($eval.Weights)"
    }
}

Write-Host "DJI held-out evaluation complete: $OutRoot"
