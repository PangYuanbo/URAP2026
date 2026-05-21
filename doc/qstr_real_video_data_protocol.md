# Real Video Data Protocol

This is the data protocol to use before and after recording real QSTR-DroneDet videos.

## Directory Layout

Use this local-only structure:

```text
data/real/
  raw_videos/
    static_hovering/
    fast_target/
    bad_alignment/
    hard_negative/
  annotations/
    qstr_real_boxes.csv
  motion_debug/
  frames/
  crops/
  yolo_candidate/
  stage_b/
```

`data/real/` is intentionally ignored by git. Keep only templates and documentation in the repository.

## Recording Manifest

Before labeling boxes, record one row per video in:

```text
data_templates/recording_manifest_template.csv
```

Columns:

```text
video_path,scenario,camera_motion,target_motion,notes
```

Allowed `scenario` values:

- `static_hovering`
- `fast_target`
- `bad_alignment`
- `hard_negative`

## Box Annotation Format

Use:

```text
video_path,frame_id,x1,y1,x2,y2,class,tag
```

Template:

```text
data_templates/qstr_real_boxes_template.csv
```

Allowed `class` values:

- `drone`
- `bird`
- `airplane`
- `insect`
- `ground_object`
- `alignment_artifact`
- `background`
- `unknown`

Allowed `tag` values:

- `static_hovering`
- `fast_target`
- `bad_alignment`
- `tiny`
- `hard_negative`

Use pixel coordinates in the original video frame. `frame_id` is zero-based.

## Initialize Local Folders

```powershell
python -m qstr_dronedet.cli init-real-data-layout --root data/real
```

## First Pass After Recording

Run motion/debug logging before model training:

```powershell
tools/run_real_video_motion_debug.ps1 -Video data/real/raw_videos/static_hovering/clip001.mp4 -Scenario static_hovering
```

Inspect:

```text
data/real/motion_debug/<scenario>/<clip>/diagnostics.jsonl
```

Key fields:

- `best_quality`
- `best_k`
- `inlier_ratio`
- `photometric_residual`
- `blur_score`

Expected pattern:

- static/hovering: low motion score but usable temporal/tracker evidence
- fast target: larger motion response and track speed
- bad alignment: low `q_H`, high residual, or blur

## Build YOLO Candidate Data From Real Videos

After annotating `data/real/annotations/qstr_real_boxes.csv`:

```powershell
tools/build_real_video_yolo_dataset.ps1 `
  -Annotations data/real/annotations/qstr_real_boxes.csv `
  -Out data/real/yolo_candidate/real_tiled_v1
```

This extracts labeled frames, writes a frame-level CSV, and builds a class-agnostic YOLO dataset.

Outputs:

```text
data/real/yolo_candidate/real_tiled_v1/
  frames/
  frame_annotations.csv
  yolo_tiled/
    data.yaml
    images/
    labels/
  summary.json
```

## Train Stage A

```powershell
python -m qstr_dronedet.cli train-yolo-p2 `
  --data data/real/yolo_candidate/real_tiled_v1/yolo_tiled/data.yaml `
  --out runs/detect/yolo_p2_real_tiled_v1 `
  --epochs 50 `
  --imgsz 256 `
  --batch 8
```

For final training, pass local pretrained weights with `--pretrained`.

## Full Pipeline Smoke

```powershell
python -m qstr_dronedet.cli infer `
  --video data/real/raw_videos/static_hovering/clip001.mp4 `
  --out runs/infer/clip001_real `
  --yolo-weights runs/detect/yolo_p2_real_tiled_v1/yolo_p2_candidate/weights/best.pt `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128
```

## What To Report

Keep these separate:

- motion/q_H sanity results
- Stage A candidate recall
- Stage B recognition with oracle boxes
- Stage B recognition with detector proposals
- full-pipeline stress test
