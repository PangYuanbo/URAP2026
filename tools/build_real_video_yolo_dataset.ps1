param(
    [string]$Annotations = "data\real\annotations\qstr_real_boxes.csv",
    [string]$Out = "data\real\yolo_candidate\real_tiled_v1",
    [string]$VideoRoot = "",
    [switch]$FullFrame,
    [int]$TileSize = 256,
    [int]$PositivesPerBox = 2,
    [int]$NegativesPerImage = 2,
    [double]$ValFraction = 0.2,
    [double]$MinBoxPx = 8.0
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$ArgsList = @(
    "-m", "qstr_dronedet.cli", "prepare-real-yolo-dataset",
    "--annotations", $Annotations,
    "--out", $Out,
    "--tile-size", "$TileSize",
    "--positives-per-box", "$PositivesPerBox",
    "--negatives-per-image", "$NegativesPerImage",
    "--val-fraction", "$ValFraction",
    "--min-box-px", "$MinBoxPx"
)

if ($VideoRoot -ne "") {
    $ArgsList += @("--video-root", $VideoRoot)
}

if ($FullFrame) {
    $ArgsList += "--full-frame"
}

python @ArgsList
