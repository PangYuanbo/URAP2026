param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$datasetRoot = Join-Path $RepoRoot "datasets\urap_drive"
$videoDir = Join-Path $datasetRoot "videos"
$annotationDir = Join-Path $datasetRoot "annotation_workspace"
$statusDir = Join-Path $RepoRoot "artifacts\urap_drive_download"
$manifestPath = Join-Path $statusDir "manifest.tsv"

New-Item -ItemType Directory -Force -Path $videoDir, $annotationDir, $statusDir | Out-Null

$files = @(
    @{ Id = "1-H7ltDQA5kdXUvn7Nc_2ntpom7r430A5"; Name = "dji_fly_20260527_122540_15_1779921105591_hdrvideo.MP4"; Subdir = "videos" },
    @{ Id = "1AC-yNqd1CGMDJonxm_yW_zaM5iGUUhwc"; Name = "dji_fly_20260527_121932_14_1779921254906_hdrvideo.MP4"; Subdir = "videos" },
    @{ Id = "1O9VX1FZcaheVCD_-0Zfv6v_skAy6FlbS"; Name = "dji_fly_20260527_121806_13_1779921757607_hdrvideo.MP4"; Subdir = "videos" },
    @{ Id = "1eCx26e4NJhgiD3ySojC1Lga8dzbpbnHG"; Name = "dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4"; Subdir = "videos" },
    @{ Id = "1S-BOpTF3B0L4Aht1ariAqppyqNrIcfA5"; Name = "qstr_real_boxes_manual.csv"; Subdir = "annotation_workspace\annotations" },
    @{ Id = "1Uaa6_PkSydHi1fnC3U5uaGZJOKQoOrCg"; Name = "recording_manifest.csv"; Subdir = "annotation_workspace\annotations" },
    @{ Id = "118PzcGaQ44eZjrOww80zjH9LYYuhHmdK"; Name = "frame_index.csv"; Subdir = "annotation_workspace\annotations" },
    @{ Id = "1cfZjFfANIVrrvgNm8_DX4jq_QomEz0_9"; Name = "task1_labels.json"; Subdir = "annotation_workspace\cvat_exports" },
    @{ Id = "1E2ME3mTBhfSN_-e8gtTcvUY8br7Y451T"; Name = "task1_annotations_raw.json"; Subdir = "annotation_workspace\cvat_exports" },
    @{ Id = "1-5X-zixHn2vEkYwJ7mhBIeFz6-dGmILJ"; Name = "task1_data_meta.json"; Subdir = "annotation_workspace\cvat_exports" },
    @{ Id = "1ME8xjQmSNGtWf5mXQAg_jTduCN0Wn61G"; Name = "CVAT_LABELS_AND_NOTES.md"; Subdir = "annotation_workspace\cvat_upload" },
    @{ Id = "1cWlvNSs9vEdNvncblOZDOzJOHEGfKbXz"; Name = "dji_fly_frames_stride60.zip"; Subdir = "annotation_workspace\cvat_upload" }
)

"id`tname`tsubdir" | Set-Content -LiteralPath $manifestPath -Encoding UTF8
foreach ($file in $files) {
    "$($file.Id)`t$($file.Name)`t$($file.Subdir)" | Add-Content -LiteralPath $manifestPath -Encoding UTF8
}

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    throw "Python launcher 'py' was not found."
}

$gdownCheck = & py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('gdown') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Output "Installing gdown for Google Drive downloads..."
    & py -m pip install --user gdown
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install gdown."
    }
}

$total = $files.Count
$index = 0
foreach ($file in $files) {
    $index += 1
    $outDir = Join-Path $datasetRoot $file.Subdir
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $outPath = Join-Path $outDir $file.Name
    $partialPath = "$outPath.part"
    if ((Test-Path -LiteralPath $outPath) -and ((Get-Item -LiteralPath $outPath).Length -gt 0)) {
        Write-Output "[$index/$total] SKIP existing $outPath"
        continue
    }

    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }

    Write-Output "[$index/$total] Downloading $($file.Name) -> $outPath"
    & py -m gdown --id $file.Id --output $partialPath
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($file.Name) ($($file.Id))."
    }

    Move-Item -LiteralPath $partialPath -Destination $outPath -Force
    $bytes = (Get-Item -LiteralPath $outPath).Length
    Write-Output "[$index/$total] DONE $($file.Name) bytes=$bytes"
}

Write-Output "ALL_DONE datasetRoot=$datasetRoot"
