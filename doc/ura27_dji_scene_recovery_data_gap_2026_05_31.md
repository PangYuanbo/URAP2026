# URA-27 DJI Scene-Recovery Data Gap

## Context

The current Stage B scene-recovery gate is not limited by model complexity. V7 added MLP/objective variants, and V8 added GT-suppressed candidate mining. Neither beat the v3 feature gate on the non-held-out calibration split.

The blocker is now data coverage: there are not enough non-held-out DJI positive scene-recovery tracklets that look like the held-out failures.

## Current DJI Footage

Audit command:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_dji_scene_recovery_data_audit.ps1
```

Audit output:

```json
{
  "role_counts": {
    "calibration": 1,
    "heldout": 1,
    "train": 2
  },
  "unique_annotated_frames_by_role": {
    "calibration": 138,
    "heldout": 99,
    "train": 2012
  }
}
```

Known roles:

| video | current role | reason |
| --- | --- | --- |
| `dji_fly_20260527_121932_14_1779921254906_hdrvideo.MP4` | train | dense non-held-out labels, already used |
| `dji_fly_20260527_122540_15_1779921105591_hdrvideo.MP4` | train | dense non-held-out labels, already used |
| `dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4` | calibration | old 5 segment calibration source |
| `dji_fly_20260527_121806_13_1779921757607_hdrvideo.MP4` | held-out | must not be used for training or calibration |

There is no additional annotated DJI source video available at this point.

## Why More Data Is Needed

V8 confirmed that previous mining missed true positives:

| sequence | strict-suppressed GT-hit rows | rows already predicted drone by recall |
| --- | ---: | ---: |
| `121932_14_1779921254906` | 528 | 59 |
| `122540_15_1779921105591` | 434 | 148 |

After adding GT-suppressed candidate mining, the training pool increased:

| pool | tracklets | positives | negatives |
| --- | ---: | ---: | ---: |
| v6 `suppressed_recall_drone` | 7396 | 121 | 7275 |
| v8 `gt_suppressed_candidate` | 7585 | 324 | 7261 |

But calibration still did not improve:

| gate | calibration recall | calibration FP |
| --- | ---: | ---: |
| v3 feature gate | 0.4634 | 2140 |
| v8 p98 | 0.4634 | 2337 |

This means the same two training videos can produce more positives, but those positives do not transfer well enough.

## Required Next Data

Before training another scene-recovery gate, add at least:

- one new non-held-out DJI training clip with dense labels;
- one separate non-held-out DJI calibration clip with dense labels;
- preferably 500 to 1000 annotated frames per new clip;
- static/hovering or slow apparent target motion where Stage B tends to suppress true candidates;
- include frames where the target is tiny and high-background evidence is likely.

Do not use `121806` for this. It remains the held-out check.

## New Tooling

Added:

- `tools/audit_dji_scene_recovery_data.py`
- `tools/run_dji_scene_recovery_data_audit.ps1`

Purpose:

After new labels are exported into `D:\datasets\my_video\final_annotations`, run the audit script to verify whether the new video is detected as a new train/calibration candidate and whether the frame count is sufficient.

Default role patterns:

- train: `121932`, `122540`
- calibration: `20260522`
- held-out: `121806`
- anything else: `candidate_new_train_or_calibration`

## Decision

Do not continue tuning gate thresholds or model structure on the current two-video train pool. The next productive step is to add a new non-held-out DJI clip, assign it to train or calibration before mining, and then rebuild paired recall/strict outputs for scene-recovery mining.
