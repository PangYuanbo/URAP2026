# SAMURAI Motion Mask Inspection

Date: 2026-06-25

## Setup

The same NPS-fine-tuned base-plus checkpoint was run twice on selected NPS test tracks:

- stock SAM2.1 video configuration;
- SAMURAI configuration with Kalman/motion-aware mask selection and memory gates.

The tool stores the complete selected binary mask for every frame, mask-derived boxes, per-frame metrics, enlarged overlays and an MP4 comparison.

## Catastrophic Case: Clip_43__track_002

This is the largest fine-tuned SAMURAI regression in the 99-track test set.

| Metric | Stock SAM2 | SAMURAI motion |
|---|---:|---:|
| Mean box IoU from exported masks | 0.5152 | 0.0408 |
| Median center error | 1.5 px | 135.4 px |
| Median mask area | 45 px | 81 px |
| Mean fraction of mask inside GT box | 96.4% | 6.8% |

The mean stock/SAMURAI mask IoU is only 0.0597. The first frame where stock exceeds SAMURAI by at least 0.3 IoU is frame index 18 (display frame 19).

Temporal progression:

- Frame 19: SAMURAI selects a small response adjacent to the true target; the mask is not merely a wider boundary.
- From frame index 40: center error is at least 10 px for 10 consecutive frames.
- From frame index 53: center error is at least 20 px for 10 consecutive frames.
- From frame index 82: center error is at least 50 px for 10 consecutive frames.
- Frames 50-99 already have zero mean SAMURAI box IoU, while stock remains at 0.431.
- At frame 210, stock IoU is 0.900 and SAMURAI IoU is 0.000; the SAMURAI mask center is 185.5 px away.
- SAMURAI exceeds 100 px center error on 196 of 324 frames.

The erroneous SAMURAI masks remain small and target-like. This is a wrong-candidate/identity selection failure, not mask-area explosion. The selected wrong candidate is then reinforced by the Kalman trajectory and video memory, producing a stable but incorrect track.

## Positive Control: Clip_43__track_001

| Metric | Stock SAM2 | SAMURAI motion |
|---|---:|---:|
| Mean box IoU from exported masks | 0.6121 | 0.6630 |
| Median center error | 1.12 px | 1.41 px |
| Mean stock/SAMURAI mask IoU | 0.783 |  |

There is no major divergence on this 53-frame track. Motion-aware selection can improve the chosen candidate when its trajectory prior remains aligned with the true target.

## Dataset-Level Interpretation

The catastrophic track strongly affects the aggregate result, but it does not fully explain the negative motion ablation:

- Mean paired sequence AUC delta over all 99 tracks: -0.00944.
- Excluding Clip_43__track_002: -0.00515.
- Excluding it, 30 tracks improve and 64 still worsen.

Therefore, the default motion selection has both a rare catastrophic failure mode and a broader small negative bias on NPS.

## Mechanism

SAMURAI uses a 15-frame stability threshold, then combines the learned mask-quality score with Kalman overlap. On the catastrophic track, visible divergence begins around frames 16-19, close to this transition. The camera is rolling and translating, so image-plane target motion does not obey a stable object-only constant-velocity model. A nearby weak candidate can be more consistent with the stale image-plane trajectory than the visually correct mask. Once selected, it updates the motion state and video memory, making recovery unlikely.

## Artifacts

- Catastrophic comparison video: `U:\URAP_runs\samurai\motion_mask_inspection\Clip_43__track_002\comparison.mp4`
- Catastrophic full masks: `U:\URAP_runs\samurai\motion_mask_inspection\Clip_43__track_002\masks.npz`
- Catastrophic frame metrics: `U:\URAP_runs\samurai\motion_mask_inspection\Clip_43__track_002\frame_metrics.csv`
- Positive-control video: `U:\URAP_runs\samurai\motion_mask_inspection\Clip_43__track_001\comparison.mp4`
- Export tool: `tools/compare_samurai_motion_masks.py`
