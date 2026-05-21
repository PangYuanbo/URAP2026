# Tracklet Classifier V2 Hard-Tiny Repair

Date: 2026-05-21

Goal:

- fix the `1_7` tracklet miss without loosening fallback globally;
- add features that represent temporal-only hard-tiny evidence;
- train only from train/adapt tracklets;
- evaluate on frozen10.

## Code Change

`qstr_dronedet/tracking/tracklet_classifier.py` now adds tracklet features beyond the original objectness/probability summary:

- `temporal_minus_crop_mean`
- `temporal_minus_background_mean`
- `final_minus_background_mean`
- `std_box_side`
- `mean_center_step`
- `max_center_step`
- `std_center_step`
- `track_span_frames`
- `frame_density`
- `weak_detector_temporal_signal`

The training CLI also has a new augmentation option:

```text
--hard-tiny-positive-augments N
```

This creates synthetic positive variants with:

- low objectness;
- short duration;
- weak or zero detector update / validation;
- high temporal-over-crop gain;
- moderate background competition.

This directly covers the pattern found in the `1_7` feature audit, where true positives were short tracker-only temporal-support tracklets.

## Training

Input:

```text
runs\profiles\tracklet_train_eval_20260521_154002\tracklets_train_adapt_60f_v2_features\tracklets.csv
```

Dataset:

```text
num_tracklets=253
positives=43
negatives=210
```

Training command:

```powershell
python -m qstr_dronedet.cli train-tracklet-classifier `
  --csv runs\profiles\tracklet_train_eval_20260521_154002\tracklets_train_adapt_60f_v2_features\tracklets.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt `
  --epochs 100 `
  --hidden 48 `
  --hard-tiny-positive-augments 4
```

The checkpoint records:

- feature list;
- normalization mean/std;
- hidden size;
- number of synthetic hard-tiny positives.

## Frozen10 Result

Evaluation input:

```text
runs\profiles\tracklet_train_eval_20260521_154002\tracklets_frozen10_60f_eval_once_v2_features\tracklets.csv
```

Frozen10 result at threshold `0.5`:

| model | TP | FP | FN | TN | precision | recall | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 baseline | 17 | 20 | 11 | 200 | 0.459 | 0.607 | 0.875 |
| v2 hard-tiny | 20 | 17 | 8 | 203 | 0.541 | 0.714 | 0.899 |

The repair improved both precision and recall on the same frozen10 tracklet set.

## 1_7 Result

Before v2:

```text
TP=0
FP=2
FN=2
TN=36
recall=0.000
```

After v2:

```text
TP=2
FP=2
FN=0
TN=36
precision=0.500
recall=1.000
```

The two formerly missed true `1_7` tracklets are now recovered:

| track_id | label | IoU | temporal_gain | detector_update | validated | weak_detector_temporal_signal | prob |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1 | 0.593 | 1.000 | 0.000 | 0.000 | 1.000 | 0.981 |
| 40 | 1 | 0.595 | 0.857 | 0.000 | 0.000 | 0.857 | 0.993 |

Remaining `1_7` false positives:

| track_id | label | temporal_gain | detector_update | validated | prob |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.305 | 0.729 | 0.881 | 1.000 |
| 37 | 0 | 1.000 | 0.000 | 0.000 | 1.000 |

Interpretation:

- the original missed-target failure is fixed at the tracklet level;
- one stable detector-supported false track remains too drone-like;
- one weak-detector temporal false track is nearly indistinguishable from the true `1_7` positives using the current aggregate features.

## Conclusion

This repair succeeds at the specific goal:

- `1_7` true tracklets are no longer suppressed;
- frozen10 tracklet-level precision and recall both improve;
- no fallback threshold was globally loosened.

Next integration step:

- wire the tracklet classifier into full inference as a post-track filter, so final frame detections can use `tracklet_is_drone` instead of frame-only hard-recovery decisions.

Large artifacts and checkpoints remain local-only and should not be committed.
