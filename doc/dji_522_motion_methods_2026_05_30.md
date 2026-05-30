# DJI 522 Motion-Method Test on Manual Annotations

Date: 2026-05-30

## Input

- Video: `data/raw_videos/dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4`
- Manual annotations: `/Users/aaron/Downloads/dji_fly_20260522_113924_10_1779475848691_hdrvideo_step5_annotations.json`
- Evaluated frames: 1600 manually annotated frames, frame step 5
- Processing resolution: width 1280, preserving aspect ratio

## Methods Tested

1. `yolomg_diff_k1`
   - YOLOMG-style raw frame-difference map from frame `t-1` to frame `t`.
   - This follows the YOLOMG paper's high-level idea: create a motion difference map, then fuse it with RGB. Here we test the motion map alone, without the YOLO detector.

2. `yolomg_diff_k5`
   - Same as above, but uses frame `t-5` because the manual labels are every 5 frames.

3. `nps_sparse_flow`
   - Lightweight approximation of the NPS/Purdue U2U-D&T paper's motion stage.
   - Tracks sparse points with Lucas-Kanade optical flow, fits a global homography with RANSAC, subtracts the stabilized previous frame, and boosts points whose flow differs from the background median.
   - This tests the paper idea of removing camera/background motion before looking for moving UAV candidates.

## Metrics

For each annotated box and motion map:

- `target_mean`: mean motion inside the annotated box.
- `context_mean`: mean motion in a local annulus around the annotated box.
- `contrast`: `target_mean / context_mean`; higher means the drone stands out from local background.
- `hit_rate_iou_0p1`: whether a simple contour proposal from the motion map overlaps the GT with IoU >= 0.1.
- `hit_rate_gt_overlap_0p25`: whether the best proposal covers at least 25% of the GT area.

## Full-Run Results

| Method | Median contrast | Mean target motion | Mean context motion | IoU>=0.1 hit rate | GT-overlap>=25% hit rate | Mean proposals/frame |
|---|---:|---:|---:|---:|---:|---:|
| `yolomg_diff_k1` | 1.118 | 0.181 | 0.155 | 2.06% | 1.13% | 87.8 |
| `yolomg_diff_k5` | 1.029 | 0.243 | 0.230 | 1.31% | 1.13% | 51.9 |
| `nps_sparse_flow` | 1.703 | 0.107 | 0.060 | 5.69% | 2.88% | 73.8 |

Artifacts:

- Summary JSON: `artifacts/dji_522_motion_methods_full/motion_method_summary.json`
- Per-frame CSV: `artifacts/dji_522_motion_methods_full/motion_method_metrics.csv`
- Contact sheet: `artifacts/dji_522_motion_methods_full/contact_sheet.jpg`
- Sample panels: `artifacts/dji_522_motion_methods_full/sample_frame_*.jpg`

## Takeaways

1. Raw YOLOMG-style frame differencing is not enough by itself on this DJI clip.
   - The drone moves, but the camera/background motion also lights up grass, trees, benches, shadows, and people.
   - Using `k=5` increases absolute target motion, but also increases context motion, so the target-to-background contrast does not improve.

2. The NPS-style optical-flow/background-compensation idea is better.
   - It has the highest median contrast and the best simple-proposal hit rate.
   - It visually suppresses broad background motion more than raw differencing.

3. But none of these motion maps is strong enough as a standalone detector.
   - Even the best method only hits IoU >= 0.1 on 5.69% of annotated frames with the current simple contour proposal rule.
   - The correct role is likely a motion prior / auxiliary cue, not the only detector.

4. The useful next experiment is ROI scoring, not full-frame proposal generation.
   - Since we already have human labels, the motion evidence inside the labeled box is measurable.
   - For a real model, use this motion map to boost or validate detector proposals near candidate boxes, instead of expecting it to generate clean boxes from scratch.

## Source Notes

- YOLOMG describes generating a motion difference map and fusing it with RGB before YOLOv5-style detection: https://arxiv.org/abs/2503.07115
- The NPS/Purdue U2U-D&T method estimates sparse optical flow, fits a background perspective transform, subtracts stabilized background, extracts salient points, and uses optical-flow motion traits plus tracking: https://engineering.purdue.edu/~bouman/UAV_Dataset/pubs/tetc01.pdf

