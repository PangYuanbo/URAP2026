param(
    [string]$Video = "",
    [string]$SearchRoot = "D:\datasets\my_video",
    [string]$OutRoot = "D:\datasets\my_video\heldout_annotation_workspace",
    [string]$ExistingAnnotations = "D:\datasets\my_video\annotation_workspace\annotations\qstr_real_boxes_manual.csv",
    [int]$FrameStride = 30,
    [int]$MaxFrames = 0,
    [switch]$AllowAlreadyAnnotated
)

$ErrorActionPreference = "Stop"

function Normalize-PathString([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }
    try {
        return ([System.IO.Path]::GetFullPath($PathValue)).ToLowerInvariant()
    }
    catch {
        return $PathValue.ToLowerInvariant()
    }
}

if ($FrameStride -lt 1) {
    throw "FrameStride must be >= 1."
}

$annotated = @{}
if (Test-Path $ExistingAnnotations) {
    Import-Csv $ExistingAnnotations | ForEach-Object {
        $path = ""
        if ($_.video_path) {
            $path = $_.video_path
        }
        elseif ($_.source_video) {
            $path = $_.source_video
        }
        $norm = Normalize-PathString $path
        if ($norm -ne "") {
            $annotated[$norm] = $true
        }
    }
}

if ($Video -eq "") {
    if (-not (Test-Path $SearchRoot)) {
        throw "SearchRoot does not exist: $SearchRoot"
    }
    $videos = Get-ChildItem -Path $SearchRoot -Recurse -File -Include *.mp4,*.MP4,*.mov,*.MOV |
        Where-Object { $_.Name -like "*dji_fly*" } |
        Sort-Object FullName
    $unannotated = @($videos | Where-Object {
        -not $annotated.ContainsKey((Normalize-PathString $_.FullName))
    })
    if ($unannotated.Count -eq 0) {
        Write-Host "No unannotated dji_fly video found under $SearchRoot."
        Write-Host "Already annotated videos in ${ExistingAnnotations}: $($annotated.Count)"
        Write-Host "Put the second DJI held-out video under D:\datasets\my_video\heldout_raw or pass -Video explicitly."
        exit 0
    }
    $Video = $unannotated[0].FullName
}

if (-not (Test-Path $Video)) {
    throw "Video does not exist: $Video"
}

$videoFull = (Resolve-Path $Video).Path
$videoNorm = Normalize-PathString $videoFull
if ($annotated.ContainsKey($videoNorm) -and -not $AllowAlreadyAnnotated) {
    throw "This video is already present in $ExistingAnnotations. Use a different held-out video, or pass -AllowAlreadyAnnotated only for debugging."
}

$clipName = [System.IO.Path]::GetFileNameWithoutExtension($videoFull)
$framesDir = Join-Path (Join-Path $OutRoot "frames") $clipName
$uploadDir = Join-Path $OutRoot "cvat_upload"
$notesDir = Join-Path $OutRoot "annotations"
New-Item -ItemType Directory -Force -Path $framesDir, $uploadDir, $notesDir | Out-Null

$frameIndex = Join-Path $framesDir "frame_index.csv"
$zipPath = Join-Path $uploadDir ("{0}_frames_stride{1}.zip" -f $clipName, $FrameStride)
$notesPath = Join-Path $notesDir "CVAT_LABELS_AND_NOTES.md"

$py = @'
import argparse
import csv
import os
import sys
import zipfile

try:
    import cv2
except Exception as exc:
    print(f"OpenCV import failed: {exc}", file=sys.stderr)
    print("Install opencv-python in the active environment, then rerun this script.", file=sys.stderr)
    raise SystemExit(10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-index", required=True)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 2

    os.makedirs(args.frames_dir, exist_ok=True)
    rows = []
    frame_id = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % args.stride == 0:
            name = f"frame_{frame_id:06d}.jpg"
            out_path = os.path.join(args.frames_dir, name)
            if not cv2.imwrite(out_path, frame):
                print(f"Failed to write frame: {out_path}", file=sys.stderr)
                return 3
            rows.append({
                "video_path": args.video,
                "frame_id": frame_id,
                "frame_path": out_path,
            })
            kept += 1
            if args.max_frames > 0 and kept >= args.max_frames:
                break
        frame_id += 1
    cap.release()

    with open(args.frame_index, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "frame_id", "frame_path"])
        writer.writeheader()
        writer.writerows(rows)

    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            zf.write(row["frame_path"], arcname=os.path.basename(row["frame_path"]))

    print(f"Extracted {len(rows)} frames from {args.video}")
    print(f"Frame index: {args.frame_index}")
    print(f"CVAT image zip: {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$py | python - `
    --video $videoFull `
    --frames-dir $framesDir `
    --frame-index $frameIndex `
    --zip-path $zipPath `
    --stride $FrameStride `
    --max-frames $MaxFrames

@"
# CVAT labels and held-out notes

Import `$zipPath` into CVAT as an image task.

Use labels:
- drone
- bird
- airplane
- insect
- ground_object
- alignment_artifact
- background
- unknown

After export, convert annotations to QSTR CSV with columns:

video_path,frame_id,x1,y1,x2,y2,class,tag

For this held-out video, use the original video path from frame_index.csv:
$videoFull

Use tags:
- static_hovering
- fast_target
- bad_alignment
- tiny
- hard_negative

Do not use this held-out CSV for training or threshold selection. Use it only for final evaluation.
"@ | Set-Content -Path $notesPath -Encoding UTF8

Write-Host "Prepared held-out CVAT import."
Write-Host "Video: $videoFull"
Write-Host "Frames: $framesDir"
Write-Host "Frame index: $frameIndex"
Write-Host "Upload zip: $zipPath"
Write-Host "Notes: $notesPath"
