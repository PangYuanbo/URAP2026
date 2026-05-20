# Winner submission-v022 on NPS (Generalization Test)

Goal: run the **AICrowd AOT Challenge winner** (`submission-v022`) inference code on the **TransVisDrone NPS** dataset (val split) and evaluate detection quality to see cross-dataset generalization.

## Why We Need A Wrapper

`submission-v022` was built for:

- AOT layout: `TEST_DATASET_PATH/<flight_id>/<frame>.png`
- AOT frame size: `2048x2448` (thermal-like grayscale), then it pads width by 112 internally.

NPS (TransVisDrone) val is:

- Flat frames folder: `.../AllFrames/val/Clip_37_00010.png ...`
- Frame size: `960x1280` (visible/RGB)

So we add a **separate entrypoint** (does not touch the running AOT fulltest script):

- `papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022/seg_test_nps.py`

It:

1. Treats each `Clip_<id>` as a “flight”.
2. Groups frames by filename prefix.
3. Resizes frames to `2048x2448` for inference, then scales predicted bboxes back to `960x1280`.

## Run Inference (Resumable)

This will create per-clip outputs under `results_nps_val/nps_val/Clip_xx/result.json` and can be resumed (skips clips with existing `result.json`).

```powershell
tools\run_winner_v022_nps_val.ps1
```

Default inputs:

- Frames: `D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val`
- Output: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/nps_val`

## Evaluate (AP@0.5 IoU)

We evaluate against NPS YOLO labels:

- `D:\URAP_datasets\TransVisDrone\NPS\NPSvisdroneStyle\val\labels`

Script:

- `tools/eval_winner_v022_nps_val.py` (computes AP@IoU=0.5, and precision/recall at the chosen `--min-score`)

Runner:

```powershell
tools\eval_winner_v022_nps_val.ps1
```

Outputs:

- Merged predictions:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/nps_val/result.json`
- Metrics summary JSON:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json`

## Results (Winner v022 -> NPS val)

Winner inference output characteristics:

- Clips: `Clip_37..Clip_40` (4 clips)
- Total frames: `5944`
- Total GT boxes: `4656` (from YOLO labels)
- Total predictions produced by winner pipeline: `3` boxes

AP evaluation (single-class, IoU=0.5):

- `AP@0.5`: `0.000286`
- Precision (at `min_score=0.0`): `0.6667`
- Recall (at `min_score=0.0`): `0.000430`

Source:

- Summary JSON: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json`

## Baseline Reference (TransVisDrone NPS-val)

For context, the TransVisDrone NPS model (trained on NPS) reaches:

- Precision: `0.901`
- Recall: `0.881`
- `mAP@0.5`: `0.948`

Source:

- `artifacts/logs/transvisdrone_nps_val_bs8_half_aug.log` (line containing `all 5944 4657 ... mAP@.5 0.948`)

## Conclusion (Generalization)

`submission-v022` (trained/tuned for AOT) does **not** generalize to NPS visible frames under the same pipeline settings: it outputs almost no detections and recall collapses.

## Analysis: Why Winner Works On AOT But Fails On NPS

This looks like a **real domain / task mismatch**, not just an evaluation bug.

What makes `submission-v022` strong on AOT:

- **Two-frame motion-aligned input**: it explicitly estimates a global transform between `prev_image` and `image`, warps the previous frame, and feeds a 2-frame tensor to the detector. This is well-matched to AOT where the target is tiny and motion cues are strong.
- **Thermal/grayscale assumption**: the whole stack reads frames with `cv2.IMREAD_GRAYSCALE` and the models are trained on AOT distribution.
- **Hard suppression of false alarms** through multiple gates:
  - Mask thresholding (`full_res_threshold=0.35`), then tracking gating.
  - `SimpleOffsetTracker` requires a track to accumulate (`min_track_size=8`) before *any* outputs are emitted.
  - `SimpleOffsetTracker` also filters by predicted **distance** (`cur_item.distance > min_distance` is dropped). In our v022 code path the tracker is created with `min_distance=1000`.
- **Everything is tuned for AOT metrics** (low HFAR/FPPI is rewarded heavily), so high thresholds and long-track requirements are rational.

Why that same design collapses on NPS:

- **Massive domain shift** (AOT thermal-style grayscale vs NPS visible frames). The segmentation/objectness mask becomes unreliable, so confidences tend to fall below the tracker thresholds.
- **Distance head becomes meaningless** out-of-domain. Distance is decoded as `2 ** pred_distance`; if the head outputs slightly larger values, it can explode and be filtered by `min_distance`, resulting in *zero* final outputs.
- **Track-length gate amplifies weak detection**: even if there are sporadic correct detections, if they do not form a stable track longer than 8, the tracker suppresses them and we end up with near-empty output.

Engineering factors that also hurt (but are likely secondary):

- **Aspect ratio distortion**: NPS is ~`1280x960` while AOT is `2448x2048`. Our adapter resizes by stretching to AOT size, which changes geometry. A letterbox-based adapter would be more faithful.
- **Color to grayscale**: NPS is RGB; converting to grayscale removes color cues (but the winner models were not trained to use color anyway).

## How To Verify It’s Not An Adapter Bug (Suggested Ablations)

If we want to test whether the *core detector* has any transferable signal, we should run these ablations on NPS:

1. **Disable the distance gate** (set `min_distance` to a very large value) and re-run.
2. **Disable the long-track gate** (`min_track_size=0`) to measure detection-only recall/precision.
3. **Use letterbox instead of stretching** when mapping NPS frames into AOT-resolution space.
4. Log internal stats for a few clips: distribution of `conf` and `distance` before filtering.

If even after (1)-(3) the detector still produces near-zero objectness, then the conclusion is: **v022 is genuinely over-specialized to AOT and not designed/trained for cross-domain generalization**.

## Threshold Ablation (Clip_37 Diagnostic, 2026-02-10)

To validate whether "thresholds are too high" is the main reason for near-zero output, we added runtime knobs to the NPS wrapper:

- `WINNER_FULL_RES_THRESHOLD`
- `WINNER_MIN_TRACK_SIZE`
- `WINNER_THRESHOLD_TO_FIND`
- `WINNER_THRESHOLD_TO_CONTINUE`
- `WINNER_THRESHOLD_DISTANCE`
- `WINNER_MIN_DISTANCE`

Code updates:

- `papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022/seg_tracker/seg_tracker.py`
- `papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022/seg_test_nps.py`

Experiment setup:

- Split: `Clip_37` only (`1532` images, `1260` GT boxes)
- IoU: `0.5`
- Comparison target: isolate detector threshold vs tracker gates

Results:

1. `base` (default v022)
   - Pred: `0`, TP: `0`, FP: `0`, Recall: `0.000000`, Precision: `0.000000`
2. `lowdet_only` (`WINNER_FULL_RES_THRESHOLD=0.20`, tracker defaults)
   - Pred: `0`, TP: `0`, FP: `0`, Recall: `0.000000`, Precision: `0.000000`
3. `trackrelaxed_only` (detector default; `min_track_size=0`, `threshold_to_find=0.30`, `threshold_to_continue=0.30`, `threshold_distance=60`, `min_distance=1000000`)
   - Pred: `65`, TP: `31`, FP: `34`, Recall: `0.024603`, Precision: `0.476923`
4. `relaxed` (combine 2+3)
   - Pred: `149`, TP: `63`, FP: `86`, Recall: `0.050000`, Precision: `0.422819`

Key finding:

- Lowering detector threshold alone does **nothing** (`0 -> 0` predictions).
- The dominant bottleneck is the **tracking gate chain** (track length + confidence continuation + distance gate).
- This supports your hypothesis that hard gating, not only model confidence, is suppressing NPS outputs.

Practical implication for "align to TransVisDrone-style generalization":

- For cross-domain testing, first run with weak/disabled tracker gates (detection-priority mode), then add tracking constraints back gradually.
- Keep AOT-optimized strict gates only for AOT leaderboard-style evaluation.
