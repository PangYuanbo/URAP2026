# Tracklet Classifier Train/Adapt to Frozen10 Evaluation

Date: 2026-05-21

Goal:

- train the new tracklet-level classifier only on non-frozen train/adaptation sequences;
- evaluate once on frozen10;
- keep frozen10 out of threshold tuning and model selection.

## Inputs

Train/adaptation diagnostics were regenerated with current QSTR hard-recovery code so every candidate row includes tracker metadata and `track_id`.

Train split:

```text
D:\datasets\Anti-UAV300\qstr_train_visible_20seq
first 5 sequences
first 60 frames per sequence
```

Adaptation split:

```text
D:\datasets\Anti-UAV300\qstr_adapt_test_visible_5seq
all 5 sequences
first 60 frames per sequence
```

Frozen evaluation split:

```text
D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq
all 10 sequences
first 60 frames per sequence
```

Output root:

```text
runs\profiles\tracklet_train_eval_20260521_154002
```

## Frame-Level Hard-Recovery Baseline

These are the raw hard-recovery profile outputs before applying the tracklet classifier.

Train5:

```text
gt=60
pred_drone=627
TP=60
FP=567
FN=0
precision=0.096
recall=1.000
```

Adapt5:

```text
gt=60
pred_drone=346
TP=49
FP=297
FN=11
precision=0.142
recall=0.817
```

Frozen10:

```text
gt=54
pred_drone=719
TP=37
FP=682
FN=17
precision=0.051
recall=0.685
```

Interpretation:

- hard-recovery still produces many frame-level false positives;
- this is exactly the setting where a tracklet-level filter is useful.

## Tracklet Dataset

Train/adapt tracklets:

```text
num_tracklets=253
positives=43
negatives=210
```

Frozen10 evaluation tracklets:

```text
num_tracklets=248
positives=28
negatives=220
```

Tracklet labels use:

- positive if any row in the tracklet matches GT by IoU `>= 0.30`;
- or by center distance `<= 24px`.

## Frozen10 Tracklet Classifier Result

The MLP was trained on train/adapt only and evaluated once on frozen10 at threshold `0.5`.

```text
TP=17
FP=20
FN=11
TN=200
precision=0.459
recall=0.607
accuracy=0.875
```

This is a strong reduction in false-positive density relative to raw frame-level hard-recovery output, but it still loses some true tracklets.

## Per-Sequence Frozen10 Result

| sequence | pos | neg | pred_pos | TP | FP | FN | TN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20190925_111757_1_5 | 5 | 22 | 10 | 3 | 7 | 2 | 15 | 0.300 | 0.600 |
| 20190925_111757_1_6 | 1 | 3 | 2 | 1 | 1 | 0 | 2 | 0.500 | 1.000 |
| 20190925_111757_1_7 | 2 | 38 | 2 | 0 | 2 | 2 | 36 | 0.000 | 0.000 |
| 20190925_111757_1_8 | 0 | 38 | 0 | 0 | 0 | 0 | 38 | 0.000 | 0.000 |
| 20190925_111757_1_9 | 2 | 20 | 2 | 1 | 1 | 1 | 19 | 0.500 | 0.500 |
| 20190925_124000_1_1 | 4 | 9 | 2 | 2 | 0 | 2 | 9 | 1.000 | 0.500 |
| 20190925_124000_1_10 | 3 | 20 | 2 | 2 | 0 | 1 | 20 | 1.000 | 0.667 |
| 20190925_124000_1_2 | 2 | 20 | 4 | 2 | 2 | 0 | 18 | 0.500 | 1.000 |
| 20190925_124000_1_3 | 4 | 25 | 9 | 3 | 6 | 1 | 19 | 0.333 | 0.750 |
| 20190925_124000_1_4 | 5 | 25 | 4 | 3 | 1 | 2 | 24 | 0.750 | 0.600 |

## 1_7 Diagnosis

The hardest `1_7` sequence is no longer a pure no-tracklet case in this evaluation:

```text
positive_tracklets=2
negative_tracklets=38
predicted_positive_tracklets=2
TP=0
FP=2
FN=2
max_tracklet_best_iou=0.595
```

So the remaining problem changed:

- the system can form GT-matching tracklets for `1_7`;
- the first MVP classifier still ranks the wrong `1_7` tracklets above the true ones.

That means the next fix should not be another broad fallback threshold change. It should inspect positive-vs-false-positive `1_7` tracklet features, then improve the training features or add harder `1_7`-like positives from train/adapt.

## Artifacts

```text
runs\profiles\tracklet_train_eval_20260521_154002\tracklets_train_adapt_60f\tracklets.csv
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_train_adapt_60f.pt
runs\profiles\tracklet_train_eval_20260521_154002\frozen10_60f_eval_once\profile_benchmark_summary.json
runs\profiles\tracklet_train_eval_20260521_154002\tracklets_frozen10_60f_eval_once\tracklets.csv
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_eval_frozen10_60f_once\metrics.json
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_eval_frozen10_60f_once\per_sequence_metrics.json
```

Weights and generated diagnostics remain local-only and should not be committed.
