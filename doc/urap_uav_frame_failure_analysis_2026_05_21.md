# URAP-UAV Frame Failure Analysis

Date: 2026-05-21

Goal:

- visualize per-frame success/failure for the current QSTR frozen10 run;
- identify where the algorithm fails after tracklet filtering/promotion;
- produce a Linear-ready report for project `URAP-UAV`.

## Command

```powershell
python -m qstr_dronedet.cli analyze-frame-failures `
  --run-root runs\profiles\frozen10_tracklet_filter_eval_20260521 `
  --gt D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv `
  --out reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521 `
  --max-frames 60 `
  --score-threshold 0.2 `
  --iou-threshold 0.3
```

## Artifacts

```text
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\raw_frame_timeline.png
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\filtered_frame_timeline.png
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\per_frame_raw.csv
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\per_frame_filtered.csv
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\summary.json
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\URAP-UAV_linear_issue.md
```

The PNG timelines visualize each sequence/frame as:

- green: success;
- yellow: partial hit;
- red: miss/localization failure;
- purple: GT frame with no drone prediction;
- orange: false-positive frame with no GT.

## Result

| output | TP | FP | FN | precision | recall | frame success |
|---|---:|---:|---:|---:|---:|---:|
| raw hard-recovery | 37 | 682 | 17 | 0.051 | 0.685 | 0.685 |
| tracklet filtered/promoted | 43 | 761 | 11 | 0.053 | 0.796 | 0.796 |

## Failure Buckets

Raw:

```json
{
  "false_positive_no_gt": 393,
  "no_drone_prediction": 10,
  "proposal_or_localization_failure": 7,
  "success": 37
}
```

Filtered/promoted:

```json
{
  "false_positive_no_gt": 482,
  "no_drone_prediction": 4,
  "proposal_or_localization_failure": 6,
  "success": 43,
  "weak_localization": 1
}
```

## Interpretation

Tracklet filtering/promotion improves the success-frame rate:

```text
0.685 -> 0.796
```

But it also increases false-positive-only frames:

```text
393 -> 482
```

So the current profile is useful for hard-case recovery, but not yet suitable as the default low-FP profile.

## Linear Project: URAP-UAV

A Linear-ready issue body was generated at:

```text
reports\URAP-UAV\frozen10_tracklet_frame_analysis_20260521\URAP-UAV_linear_issue.md
```

Recommended Linear issues:

1. Tighten tracklet promotion on sequences where recall gain creates many FP.
2. Add per-frame UI overlays for `tracklet_promoted`, `tracklet_confirmed`, and `tracklet_rejected`.
3. Build train/adapt threshold sweep for reject-only vs promotion profiles without tuning on frozen10.

Current environment note:

- The Linear plugin install flow completed, but no callable Linear API/CLI was exposed in this session.
- The generated markdown can be pasted into the `URAP-UAV` Linear project, or synced once the Linear tool becomes available.

## Validation

```text
pytest tests -q
47 passed
```

Generated reports remain local-only and should not be committed as experiment artifacts.
