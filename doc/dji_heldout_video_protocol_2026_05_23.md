# DJI held-out video protocol

This note freezes how the second DJI video should be used.

## Current state

Only one DJI video has been found locally so far:

`D:\datasets\my_video\dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4`

It is already used by:

`D:\datasets\my_video\annotation_workspace\annotations\qstr_real_boxes_manual.csv`

Therefore its current metrics are same-video sanity checks, not real held-out generalization.

## Rule

The next DJI video must be treated as held-out:

- Do not use it for YOLO-P2 training.
- Do not use it for crop, temporal, feature, tracker, fusion, or threshold tuning.
- Use it only after the current ARD100 + DJI-adapted system is fixed.
- Report it separately from ARD100 and from the first DJI adaptation video.

## Recommended location

Put the second video here:

`D:\datasets\my_video\heldout_raw\`

The filename should keep `dji_fly` in the name so the helper script can find it.

## Prepare frames for CVAT

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\prepare_dji_heldout_cvat_import.ps1 `
  -SearchRoot D:\datasets\my_video `
  -FrameStride 30
```

Use `-FrameStride 10` for denser annotation, or `-FrameStride 1` only if you really want every frame.

Outputs:

- frames under `D:\datasets\my_video\heldout_annotation_workspace\frames\...`
- CVAT upload zip under `D:\datasets\my_video\heldout_annotation_workspace\cvat_upload\...`
- `frame_index.csv` mapping extracted image names back to original `video_path,frame_id`

## CVAT labels

Use exactly these labels:

- `drone`
- `bird`
- `airplane`
- `insect`
- `ground_object`
- `alignment_artifact`
- `background`
- `unknown`

Use tags:

- `static_hovering`
- `fast_target`
- `bad_alignment`
- `tiny`
- `hard_negative`

## Final CSV

After export, convert the annotations to:

`D:\datasets\my_video\heldout_annotation_workspace\annotations\qstr_real_boxes_heldout.csv`

Format:

```csv
video_path,frame_id,x1,y1,x2,y2,class,tag
D:\datasets\my_video\heldout_raw\dji_fly_example.MP4,35,612,288,620,296,drone,static_hovering
```

Use the original video path and frame id from `frame_index.csv`, not the temporary image path.

## Evaluation

Run the held-out evaluation detached:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_dji_heldout_eval_detached.ps1 `
  -Annotations D:\datasets\my_video\heldout_annotation_workspace\annotations\qstr_real_boxes_heldout.csv `
  -OutRoot D:\datasets\my_video\qstr_heldout_eval `
  -Device 0
```

Monitor:

```powershell
powershell -ExecutionPolicy Bypass -File tools\monitor_dji_heldout_eval.ps1 `
  -LogDir <log-dir-printed-by-start-script>
```

The evaluation runs:

1. ARD100 YOLO-P2 baseline Stage A recall.
2. DJI-adapted YOLO-P2 Stage A recall.
3. DJI-adapted detector proposal dataset build.
4. Stage B crop recognizer comparison:
   - ARD-only crop recognizer
   - DJI-only recovery recognizer
   - mixed ARD100+DJI recognizer

This separates Stage A candidate failure from Stage B recognition failure on a true held-out DJI video.
