# QSTR 1_7 Tracklet Feature Audit

Date: 2026-05-21

Scope:

- frozen10 evaluation artifact: `runs\profiles\tracklet_train_eval_20260521_154002`
- sequence: `20190925_111757_1_7`
- first 60 frames
- classifier threshold: `0.5`

This audit compares the true `1_7` positive tracklets that the MVP classifier missed against the false-positive tracklets that it accepted.

## Summary

`1_7` is no longer a pure "no tracklet exists" case.

Frozen eval produced:

```text
TP tracklets: 0
FP tracklets: 2
FN tracklets: 2
TN tracklets: 36
max positive best_iou: 0.595
```

So the remaining error is ranking/calibration:

- real `1_7` tracklets exist and overlap GT;
- the classifier scores them near zero;
- two wrong tracklets are scored above threshold.

## FP vs FN Tracklets

| track_id | group | prob | best_iou | rows | mean_obj | max_obj | crop | temporal | background | temporal_gain | detector_update | validated | mean_drift | box_side |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | FP | 0.999 | 0.000 | 59 | 0.387 | 0.729 | 0.557 | 0.586 | 0.453 | 0.305 | 0.729 | 0.881 | 3.23 | 137.0 |
| 27 | FP | 0.600 | 0.000 | 3 | 0.218 | 0.284 | 0.519 | 0.420 | 0.601 | 0.000 | 0.667 | 0.667 | 18.57 | 146.9 |
| 40 | FN | 0.006 | 0.595 | 7 | 0.221 | 0.280 | 0.454 | 0.547 | 0.546 | 0.857 | 0.000 | 0.000 | 0.00 | 108.6 |
| 19 | FN | 0.000 | 0.593 | 8 | 0.129 | 0.170 | 0.440 | 0.593 | 0.594 | 1.000 | 0.000 | 0.000 | 0.00 | 106.3 |

Key difference:

- false positives look like "stable detector-supported tracks":
  - longer tracks;
  - higher objectness;
  - high detector update rate;
  - high validation rate;
  - larger boxes.
- true `1_7` positives look like "short tracker-only temporal-support tracks":
  - low objectness;
  - no detector update support;
  - no validation support;
  - strong temporal-over-crop gain;
  - moderate crop/temporal evidence, but background is still competitive.

## Train/Adapt Distribution Check

Train/adapt tracklet dataset:

```text
positive_total=43
negative_total=210
```

Positive tracklet coverage:

```text
detector_update_zero positives: 1 / 43
validated_zero positives: 1 / 43
both detector_update_zero and validated_zero positives: 0 / 43
temporal_gain >= 0.8 positives: 7 / 43
1_7-like positives:
  detector_update_rate == 0
  validated_rate == 0
  temporal_gain_rate >= 0.8
  count = 0 / 43
```

Mean feature comparison:

| group | rows | mean_obj | max_obj | crop | temporal | background | temporal_gain | detector_update | validated | box_side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train positive | 22.3 | 0.435 | 0.817 | 0.517 | 0.594 | 0.513 | 0.350 | 0.455 | 0.628 | 108.7 |
| train negative | 6.1 | 0.187 | 0.293 | 0.481 | 0.583 | 0.607 | 0.350 | 0.118 | 0.245 | 81.0 |
| 1_7 FN | 7.5 | 0.175 | 0.225 | 0.447 | 0.570 | 0.570 | 0.929 | 0.000 | 0.000 | 107.5 |
| 1_7 FP | 31.0 | 0.303 | 0.506 | 0.538 | 0.503 | 0.527 | 0.153 | 0.698 | 0.774 | 142.0 |

The model is behaving consistently with its training distribution: it learned that detector-supported, validated, long tracks are more likely to be drone. `1_7` true positives are out-of-distribution because they have no detector support but high temporal gain.

## Diagnosis

The current MVP classifier fails `1_7` for a data/feature coverage reason, not because the tracklet framework is useless.

What the model over-trusts:

- detector update rate;
- validation rate;
- long track age / number of rows;
- high objectness.

What the model under-trusts:

- high temporal-over-crop gain on short tracks;
- tracker-only temporal consistency;
- low-objectness tiny positives with competitive background.

The hardest part is that `1_7` positives look like negatives under the current feature distribution:

- low objectness;
- short duration;
- no detector update;
- no validation;
- background around `0.55-0.59`.

## Next Fix

Do not loosen fallback globally.

The next useful implementation should add `hard-tiny positive mining` for tracklets:

1. Mine train/adapt tracklets that are GT-matching but have:
   - low objectness;
   - short track length;
   - weak or zero detector update rate;
   - high temporal gain.
2. If train/adapt does not naturally contain enough, synthesize this condition by:
   - dropping detector update evidence on some positive tiny tracklets during training;
   - augmenting labels with short temporal-only slices from GT-positive tracks.
3. Add features that separate stable false tracks from short temporal tiny tracks:
   - `temporal_minus_crop_mean`;
   - `temporal_minus_background_mean`;
   - `early_vs_late_center_stability`;
   - `box_side_variance`;
   - detector-overlap recency instead of generic detector update rate.
4. Retrain on train/adapt only, then evaluate once on frozen10.

Expected outcome:

- recover the two `1_7` positive tracklets without accepting every long detector-supported false track.

## Local Artifacts

```text
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_1_7_feature_audit\summary.json
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_1_7_feature_audit\tracklets_1_7_ranked.csv
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_1_7_feature_audit\tracklets_1_7_fp_fn_rows.csv
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_1_7_feature_audit\train_vs_1_7_summary.json
```

Generated CSV/JSON artifacts remain local-only and should not be committed.
