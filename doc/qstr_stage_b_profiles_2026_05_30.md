# QSTR Stage B profiles, 2026-05-30

## Purpose

The DJI experiments showed that a single Stage B model cannot currently optimize both goals:

- recover hard tiny true drones,
- suppress high-score `yolo_tile` false positives.

We now keep two Stage B profiles and a deterministic source/scene-aware selector.

## Profiles

### `recall_oriented`

Scripts:

- `tools/run_qstr_stage_b_recall_profile.ps1`

Weights:

- `runs\dji_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt`
- `runs\dji_stage_b_hardneg_repair_20260530\models\temporal_binary_dji_hardneg.pt`

Use when the priority is not missing hard tiny DJI targets. This model keeps more positives but also lets more high-score detector FP survive.

### `strict_fp_control`

Scripts:

- `tools/run_qstr_stage_b_strict_fp_profile.ps1`

Weights:

- `runs\dji_dense_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt`
- `runs\dji_dense_stage_b_hardneg_repair_20260530\models\temporal_binary_dji_hardneg.pt`

Use when the priority is reducing false positives. This model suppresses `high_score_detector_fp` strongly, but also suppresses some hard tiny positives.

### `source_scene_stageb_select`

Scripts:

- `tools/select_stage_b_profile_outputs.py`
- `tools/select_qstr_stage_b_source_scene_profile.ps1`
- `tools/start_qstr_stage_b_source_scene_profile_detached.ps1`
- `tools/monitor_qstr_stage_b_source_scene_profile.ps1`

Rule:

- strict profile is the default for normal `yolo_tile` candidates;
- recall profile may override strict only when:
  - recall predicts drone,
  - recall score is at least `0.18`,
  - recall drone probability is at least `0.55`,
  - recall background probability is at most `0.60`,
  - and tracker/tracklet support exists.

Scene-only hard-tiny recovery is no longer enabled by default. It can still be reproduced with `--scene-recovery-allow-untracked`, but calibration showed that "tiny + hard mode + recall score" without persistence admits too many false positives.

The conservative default requires persistence before scene hard-tiny recovery:

- `track_history_len >= 2`
- `track_detector_updates >= 2`
- `track_frames_since_detector_update <= 1`
- `track_score >= 0.10`

In current DJI outputs, candidates satisfying those persistence checks are already covered by tracker/tracklet recovery, so the effective default is track-supported recovery only.

## DJI validation result

Validation set:

`D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\annotations\qstr_real_boxes_segments.csv`

Stage A is fixed to `balanced_v2`; tracklet promotion remains disabled.

| profile | candidate recall | final recall | final/frame | approx FP | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_v2 baseline, no promotion | 0.9024 | 0.3415 | 3.2412 | 8157 | 0.0017 |
| `recall_oriented` | 0.9024 | 0.4878 | 2.9952 | 7531 | 0.0026 |
| `strict_fp_control` | 0.9024 | 0.3659 | 0.7560 | 1891 | 0.0079 |
| old `source_scene_stageb_select`, untracked scene recovery | 0.9024 | 0.4878 | 1.6335 | 4098 | 0.0049 |
| conservative `source_scene_stageb_select`, track-supported only | 0.9024 | 0.4146 | 0.8322 | 2081 | 0.0081 |
| learned scene-tracklet gate | 0.9024 | 0.4634 | 0.8520 | 2129 | 0.0088 |

Selection counts for old untracked scene recovery:

| selection reason | rows |
| --- | ---: |
| strict default | 22973 |
| recall track-supported recovery | 192 |
| recall scene hard-tiny recovery | 2020 |
| recall-only recovery | 0 |

Selection counts for conservative default:

| selection reason | rows |
| --- | ---: |
| strict default | 24993 |
| recall track-supported recovery | 192 |
| recall scene hard-tiny recovery | 0 |
| recall-only recovery | 0 |

## Scene-Recovery Audit

Audit command:

```powershell
python tools\audit_stage_b_scene_recovery_gate.py `
  --recall-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --strict-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_dense_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --out D:\datasets\my_video\full_infer_compare\dji_segments_scene_recovery_gate_audit_20260530
```

Calibration result:

- old untracked scene rule kept `2020` scene hard-tiny rows;
- only `3` rows matched GT;
- `2017` rows were false positives;
- persistent gates kept `0` rows because these scene-only candidates had no track history or detector-update persistence.

This is why the default changed from scene-only recovery to persistence-gated recovery.

## Learned Scene-Tracklet Gate

The weak part of the old selector was `recall_scene_hard_tiny_recovery`: a single tiny candidate in a hard mode could override the strict Stage B profile. To replace that with sequence evidence, we added a narrow learned gate that only decides whether a scene hard-tiny recovery tracklet is reliable.

Training command:

```powershell
python tools\train_stage_b_recovery_tracklet_gate.py `
  --recall-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --strict-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_dense_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --out D:\datasets\my_video\full_infer_compare\dji_segments_scene_tracklet_gate_train_20260530 `
  --min-precision 0.20 `
  --min-tp 1
```

Gate artifact:

`D:\datasets\my_video\full_infer_compare\dji_segments_scene_tracklet_gate_train_20260530\scene_tracklet_gate.json`

Training set:

- `1505` scene-recovery tracklets
- `3` positive tracklets
- `1502` negative tracklets

Selected calibration threshold:

- threshold `0.9957199693`
- tracklet-level calibration `tp=2`, `fp=0`, `fn=1`
- precision `1.0`, recall `0.6667`

Selector command with learned gate:

```powershell
python tools\select_stage_b_profile_outputs.py `
  --recall-root <recall_oriented\yolo_only> `
  --strict-root <strict_fp_control\yolo_only> `
  --out <out> `
  --profile-name source_scene_stageb_select_learned_gate `
  --scene-tracklet-gate D:\datasets\my_video\full_infer_compare\dji_segments_scene_tracklet_gate_train_20260530\scene_tracklet_gate.json `
  --evaluate
```

Calibration selector result:

- conservative selector: final recall `0.4146`, approx FP `2081`
- learned scene gate: final recall `0.4634`, approx FP `2129`

The learned gate adds a small amount of recall back while keeping FP close to the conservative profile. It is not a large recovery yet because the calibration data only contains `3` positive scene-recovery tracklets.

### Dense-Positive Expansion

We then added non-held-out dense DJI positive sources to the scene-tracklet gate training pool:

- `D:\datasets\my_video\full_infer_compare\dji_dense_task3_task4_balanced_v2_stageb_repaired_pool_20260530\balanced_v2_dji_stable_no_promotion\yolo_only`
- `D:\datasets\my_video\full_infer_compare\dji_dense_task4_balanced_v2_stageb_repaired_pool_900f_20260530\balanced_v2_dji_stable_no_promotion\yolo_only`

The first source contains a completed `121932` dense run. The attempted `122540` portion in that folder did not finish and has no `.done` marker, so the completed 900-frame `122540` source above is used instead.

Training command:

```powershell
python tools\train_stage_b_recovery_tracklet_gate.py `
  --recall-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --strict-root D:\datasets\my_video\full_infer_compare\dji_segments_balanced_v2_stageb_dense_hardneg_repaired_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --extra-recall-roots `
    D:\datasets\my_video\full_infer_compare\dji_dense_task3_task4_balanced_v2_stageb_repaired_pool_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
    D:\datasets\my_video\full_infer_compare\dji_dense_task4_balanced_v2_stageb_repaired_pool_900f_20260530\balanced_v2_dji_stable_no_promotion\yolo_only `
  --out D:\datasets\my_video\full_infer_compare\dji_segments_scene_tracklet_gate_v2_strict_densepos_train_20260530 `
  --min-precision 0.50 `
  --min-tp 1
```

Dense-positive v2-strict training pool:

- `3534` scene-recovery tracklets
- `37` positive tracklets
- `3497` negative tracklets
- selected threshold `0.9542292356`
- tracklet-level calibration `tp=14`, `fp=12`, `fn=23`

Selector result:

| gate | calibration final recall | calibration approx FP | held-out 121806 final recall | held-out 121806 approx FP |
| --- | ---: | ---: | ---: | ---: |
| learned gate v1 | 0.4634 | 2129 | 0.0909 | 2391 |
| dense-positive v2 loose | 0.4634 | 2307 | 0.1010 | 2780 |
| dense-positive v2 strict | 0.4634 | 2168 | 0.1010 | 2530 |

The dense-positive pool increases positive tracklets from `3` to `37`. However, on calibration v2-strict does not improve final recall over v1 and still adds FP, so it should not replace v1 as default yet. It is useful as a recovery-profile experiment and shows that more positive scene tracklets can move held-out recall without returning to the old FP explosion.

## Decision

The current default `source_scene_stageb_select` is the conservative profile:

- it keeps Stage A candidate recall unchanged;
- it keeps tracker/tracklet-supported recall recovery;
- it removes the weak scene-only hard-tiny recovery rule;
- on calibration it reduces approx FP from `4098` to `2081`, while final recall changes from `0.4878` to `0.4146`.

It should not be treated as final deployment performance. It is a deterministic profile-selection rule validated on the current DJI validation segments.

## Commands

Run recall-oriented full pipeline:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_qstr_stage_b_recall_profile.ps1
```

Run strict FP-control full pipeline:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_qstr_stage_b_strict_fp_profile.ps1
```

Combine existing recall/strict outputs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\select_qstr_stage_b_source_scene_profile.ps1
```

Reproduce the old, looser scene-only rule:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\select_qstr_stage_b_source_scene_profile.ps1 -SceneRecoveryAllowUntracked
```

Run the full detached source/scene profile chain on a new annotation CSV:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\start_qstr_stage_b_source_scene_profile_detached.ps1 `
  -Annotations D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\annotations\qstr_real_boxes_segments.csv `
  -Out D:\datasets\my_video\full_infer_compare\dji_stage_b_source_scene_profile_20260530
```

Monitor it:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\monitor_qstr_stage_b_source_scene_profile.ps1 `
  -Out D:\datasets\my_video\full_infer_compare\dji_stage_b_source_scene_profile_20260530
```

The selector writes:

- `selection_summary.json`
- `selection_summary.csv`
- `summary.csv`
- `frame_timeline.csv`
- selected `predictions.jsonl` / `diagnostics.jsonl` per sequence.

The detached full-chain runner writes:

- `recall_oriented\summary.csv`
- `strict_fp_control\summary.csv`
- `selected\summary.csv`
- `selected_summary.csv`
- `status.txt`, `stdout.log`, `stderr.log`, `pid.txt`
- `.done` or `.failed`

## Runner smoke check

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\start_qstr_stage_b_source_scene_profile_detached.ps1 `
  -Out D:\datasets\my_video\full_infer_compare\dji_stage_b_source_scene_profile_smoke_20260530 `
  -MaxFramesPerVideo 1
```

Monitor result:

- PID `25220` exited.
- `.done=True`, `.failed=False`.
- recall-oriented, strict FP-control, and selected outputs were all written.
- Smoke selected output: `5` frames, `22` final drone rows, `4.4` final rows/frame.
- Selection counts: `30` strict-default rows, `15` scene hard-tiny recovery rows.

This smoke check only validates execution and output shape. It uses the first frame of each validation segment, so `gt=0` and should not be interpreted as model quality.

## Held-Out Sanity

The `121806` DJI clip was used as a held-out sanity check after the calibration decision. It was not used to select thresholds.

Annotation CSV:

`D:\datasets\my_video\final_annotations\dji_121806_heldout_99_20260530_8col.csv`

Full-chain output:

`D:\datasets\my_video\full_infer_compare\dji_121806_heldout_source_scene_20260530`

Conservative selector re-run:

`D:\datasets\my_video\full_infer_compare\dji_121806_heldout_source_scene_conservative_20260530`

| profile | candidate recall | final recall | final/frame | approx FP | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recall_oriented` | 0.9091 | 0.2222 | 2.5910 | 16926 | 0.0013 |
| `strict_fp_control` | 0.9091 | 0.0707 | 0.3102 | 2022 | 0.0034 |
| old untracked scene selector | 0.9091 | 0.1212 | 1.1533 | 7532 | 0.0016 |
| conservative selector | 0.9091 | 0.0808 | 0.3619 | 2359 | 0.0034 |
| learned scene-tracklet gate | 0.9091 | 0.0909 | 0.3669 | 2391 | 0.0038 |
| dense-positive v2 strict gate | 0.9091 | 0.1010 | 0.3883 | 2530 | 0.0039 |
| paired recall/strict gate | 0.9091 | 0.1010 | 0.4051 | 2640 | 0.0038 |
| paired v3 feature gate | 0.9091 | 0.1010 | 0.3765 | 2453 | 0.0041 |

The learned scene-tracklet gate improves held-out recall slightly over the conservative selector while keeping FP near the strict/conservative regime. It does not solve recall fully; the bottleneck is now lack of positive scene-recovery tracklet data.

## Paired Recall/Strict Gate

The next offline experiment paired recall-oriented and strict FP-control outputs before training the scene-tracklet gate. This avoids learning only from recall-only outputs and gives the gate both versions of the same proposal stream.

Pairing strategy:

- `121932`: recall full output paired with strict full output.
- `122540`: recall 900-frame output paired with strict full output, using only the overlap available from the recall side.
- Existing 5 DJI calibration segments were included with their recall/strict paired outputs.
- `121806` remained held out; no threshold was selected from it.

Paired roots:

- recall: `D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_20260531\recall_yolo_only`
- strict: `D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_20260531\strict_yolo_only`
- manifest: `D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_20260531\paired_roots_manifest.json`

Gate output:

`D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_tracklet_gate_train_20260531\scene_tracklet_gate.json`

Training summary:

| tracklets | positives | negatives | selected threshold | train precision | train recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3181 | 21 | 3160 | 0.985431 | 0.5385 | 0.3333 |

Fixed-gate evaluation:

| split | GT | candidate recall | final recall | final/frame | approx FP | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired train-fit | 1165 | 0.8970 | 0.1897 | 0.5914 | 3844 | 0.0544 |
| non-held-out calibration | 41 | 0.9024 | 0.4634 | 0.8723 | 2180 | 0.0086 |
| `121806` held-out | 99 | 0.9091 | 0.1010 | 0.4051 | 2640 | 0.0038 |

Decision:

The paired gate is useful as an experiment but should not replace the current default. It matches the dense-positive v2 strict gate on `121806` final recall, but carries slightly more FP. The failure mode is still that the learned gate cannot reliably separate persistent DJI positives from persistent detector-supported false positives using the current features and data volume.

## V3 Feature Gate

The v3 feature gate keeps the paired recall/strict training discipline but changes the tracklet representation. It adds:

- detector persistence: detector support count, longest detector streak, detector persistence;
- objectness persistence: mean/max objectness and objectness streak;
- geometry stability: normalized center-step and box-size coefficient of variation;
- contradiction terms: high-background detector-supported rows and high-background/high-drone rows;
- tracker metadata when present: max track history, max detector updates, and frames since detector update.

The gate trainer and selector now enrich prediction rows with diagnostics rows before feature extraction, so the gate can see crop/temporal probabilities instead of only `final_probs`.

Gate output:

`D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_tracklet_gate_v3_features_train_20260531\scene_tracklet_gate.json`

Training summary:

| tracklets | positives | negatives | selected threshold | train precision | train recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3181 | 21 | 3160 | 0.991370 | 0.5833 | 0.3333 |

Fixed-gate evaluation:

| split | GT | candidate recall | final recall | final/frame | approx FP | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired train-fit | 1165 | 0.8970 | 0.1820 | 0.5845 | 3806 | 0.0528 |
| non-held-out calibration | 41 | 0.9024 | 0.4634 | 0.8564 | 2140 | 0.0088 |
| `121806` held-out | 99 | 0.9091 | 0.1010 | 0.3765 | 2453 | 0.0041 |

Decision:

The v3 feature gate is the best recovery-profile variant so far for `121806`: it keeps the same held-out recall as the dense-positive v2 strict and paired gates while reducing held-out FP. It still should not replace the conservative default because the absolute precision is low and it does not recover additional held-out positives. The next blocker is data, not thresholding: add more true positive scene-recovery tracklets from non-held-out DJI clips and hard negatives with the same persistent detector support pattern.

## V4 Suppressed-Drone Pool Trial

URA-27 starts from the observation that the previous training set was too narrow: it only sampled rows that already passed the old `recall_scene_hard_tiny_recovery` rule. The v4 trial adds:

- `--sample-mode suppressed_recall_drone` in `tools/train_stage_b_recovery_tracklet_gate.py`;
- a larger training pool made from recall rows that predict `drone` while the strict profile suppresses the same candidate;
- `--scene-tracklet-gate-override-background` in `tools/select_stage_b_profile_outputs.py`, so a passing gate can recover a hard-tiny row even when branch/final background is high;
- hard-tiny cap `48 px` selected on the non-held-out training pool, not on `121806`.

Gate output:

`D:\datasets\my_video\full_infer_compare\dji_stageb_scene_tracklet_gate_v4_suppressed_drone48_train_20260531\scene_tracklet_gate.json`

Training summary:

| tracklets | positives | negatives | selected threshold | train precision | train recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4897 | 35 | 4862 | 0.982449 | 0.5143 | 0.5143 |

Fixed-gate evaluation:

| split | GT | candidate recall | final recall | final/frame | approx FP | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired train-fit | 1165 | 0.8970 | 0.1991 | 0.5895 | 3820 | 0.0573 |
| non-held-out calibration | 41 | 0.9024 | 0.4634 | 0.8631 | 2157 | 0.0087 |
| `121806` held-out | 99 | 0.9091 | 0.1010 | 0.3776 | 2460 | 0.0040 |

Decision:

V4 expands the positive pool from `21` to `35` tracklets and enables gate-controlled background override, but it does not beat v3 on `121806`: recall is unchanged and FP is slightly higher (`2460` vs `2453`). Keep the code path because it is the right mechanism for future data, but do not make this gate the default. The remaining blocker is still data density: we need many more non-held-out positive scene-recovery tracklets, not another held-out threshold tweak.

## V5 Full-122540 Pairing Trial

URA-27 then completed the missing `122540` recall-oriented full run so it could be paired with the existing strict full run instead of using only the old 900-frame overlap.

New recall output:

`D:\datasets\my_video\full_infer_compare\dji_dense_122540_balanced_v2_stageb_repaired_pool_full_stride2_20260531\balanced_v2_dji_stable_no_promotion\yolo_only\122540_15_1779921105591`

Run summary:

| sequence | frames | GT | candidate recall | final recall | final/frame | approx FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `122540_15_1779921105591` | 3377 | 560 | 0.8679 | 0.4696 | 4.1312 | 13688 |

New paired roots:

- recall: `D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_full122540_20260531\recall_yolo_only`
- strict: `D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_full122540_20260531\strict_yolo_only`

The full pairing increased the gate training pool substantially:

| gate | tracklets | positives | negatives | selected threshold | train precision | train recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 suppressed-drone48 | 4897 | 35 | 4862 | 0.982449 | 0.5143 | 0.5143 |
| v5 full122540 p50 | 9742 | 126 | 9616 | 0.979480 | 0.5055 | 0.3651 |
| v5 full122540 p70 | 9742 | 126 | 9616 | 0.993289 | 0.7556 | 0.2698 |
| v5 full122540 p80 | 9742 | 126 | 9616 | 0.998129 | 0.8000 | 0.1587 |
| v5 full122540 p90 | 9742 | 126 | 9616 | 0.999369 | 0.9333 | 0.1111 |

Calibration-only selection:

| gate | calibration recall | calibration FP | final/frame |
| --- | ---: | ---: | ---: |
| v5 p50 | 0.4634 | 2232 | 0.8929 |
| v5 p70 | 0.4634 | 2183 | 0.8735 |
| v5 p80 | 0.4634 | 2164 | 0.8659 |
| v5 p90 | 0.4634 | 2149 | 0.8600 |

The p90 gate was selected from calibration because it had the lowest v5 calibration FP while keeping calibration recall unchanged. It was then evaluated once on `121806` held-out:

| gate | held-out recall | held-out FP | final/frame | approx precision |
| --- | ---: | ---: | ---: | ---: |
| v3 feature gate | 0.1010 | 2453 | 0.3765 | 0.0041 |
| v5 full122540 p50 | 0.0909 | 2616 | 0.4013 | 0.0034 |
| v5 full122540 p90 | 0.0909 | 2511 | 0.3853 | 0.0036 |

Decision:

Full `122540` pairing fixed the sample-count problem but not the generalization problem. The larger pool contains more positive scene-recovery tracklets, but it also teaches the gate a recovery pattern that does not transfer to `121806`. Keep v3 feature gate as the best current recovery-profile experiment. URA-27 remains open for the next data step: add another non-held-out DJI clip or split current dense clips into a real train/calibration partition before touching held-out again.

## Next step

Use the detached full-chain runner on data not used to choose the selector rule, then move the rule into live inference if it still improves the recall/FP tradeoff. The live version should expose explicit Stage B profile names and write `stage_b_profile_selected` plus `stage_b_profile_selection_reason` for every candidate.

## V6 Clean Train/Calibration Split Trial

The previous full `122540` trial still mixed the old 5-segment DJI calibration clips into the gate training roots. V6 fixes the experiment discipline:

- train only: full `121932_14_1779921254906` plus full `122540_15_1779921105591`;
- calibration only: the old 5 DJI segments from `dji_fly_20260522_113924_10_1779475848691`;
- held-out once: `121806_13_1779921757607`, after calibration selection only.

Train-only paired root:

`D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_dense_train_only_20260531`

Gate training used `--sample-mode suppressed_recall_drone`, `--hard-tiny-max-side 48`, and min-precision variants from `0.50` to `0.90`.

| gate | tracklets | positives | negatives | selected threshold | train precision | train recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v6 p50 | 7396 | 121 | 7275 | 0.982068 | 0.5000 | 0.3802 |
| v6 p70 | 7396 | 121 | 7275 | 0.993919 | 0.7000 | 0.2893 |
| v6 p80 | 7396 | 121 | 7275 | 0.997294 | 0.8000 | 0.2314 |
| v6 p90 | 7396 | 121 | 7275 | 0.998625 | 0.9200 | 0.1901 |

Calibration-only selection:

| gate | calibration recall | calibration FP | final/frame | approx precision |
| --- | ---: | ---: | ---: | ---: |
| v3 feature gate | 0.4634 | 2140 | 0.8564 | 0.0088 |
| v5 full122540 p90 | 0.4634 | 2149 | 0.8600 | 0.0088 |
| v6 p50 | 0.4634 | 2297 | 0.9187 | 0.0082 |
| v6 p70 | 0.4634 | 2262 | 0.9048 | 0.0083 |
| v6 p80 | 0.4634 | 2243 | 0.8973 | 0.0084 |
| v6 p90 | 0.4634 | 2211 | 0.8846 | 0.0085 |

V6 p90 was the best V6 variant on calibration, so it was evaluated once on `121806` held-out without any threshold adjustment:

| gate | held-out candidate recall | held-out final recall | held-out FP | final/frame | approx precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 feature gate | 0.9091 | 0.1010 | 2453 | 0.3765 | 0.0041 |
| v6 p90 clean split | 0.9091 | 0.1010 | 2616 | 0.4015 | 0.0038 |

Decision:

V6 fixed the train/calibration leakage issue, but it did not improve the system. On calibration, every V6 gate preserved recall but added more FP than v3. On held-out, v6 p90 matched v3 recall but increased FP. Keep v3 as the best current recovery-profile experiment, and do not continue tuning `121806`. The next useful step is to build more positive scene-recovery tracklets from additional non-held-out DJI footage, or change the gate objective/model so persistent true DJI targets are not confused with persistent detector-supported false positives.

## V7 Recall-Guard Objective / MLP Trial

V7 tests whether the blocker can be solved by model/objective changes rather than more held-out tuning.

Code changes:

- `qstr_dronedet/tracking/sequence_gate.py` now adds explicit detector-plus-background continuity features:
  - `longest_detector_high_background_streak`;
  - `detector_high_background_persistence`;
  - `longest_detector_high_background_drone_streak`;
  - `detector_high_background_drone_persistence`;
  - `mean_detector_objectness`;
  - `mean_detector_background`;
  - `mean_detector_drone`.
- `tools/train_stage_b_recovery_tracklet_gate.py` supports:
  - `--model-type logistic|mlp`;
  - `--objective balanced|recall_preserving`;
  - weighted recall-preserving positive samples for high-background, detector-persistent true tracklets.
- `tools/select_stage_b_profile_outputs.py` can score both old logistic JSON gates and new MLP JSON gates.

Training used the same clean V6 train-only root:

`D:\datasets\my_video\full_infer_compare\dji_stageb_paired_scene_gate_train_roots_dense_train_only_20260531`

V7 training summary:

| gate | model | objective | train precision | train recall |
| --- | --- | --- | ---: | ---: |
| v7 mlp p50 | MLP | recall_preserving | 0.5000 | 0.5372 |
| v7 mlp p70 | MLP | recall_preserving | 0.7200 | 0.2975 |
| v7 mlp p90 | MLP | recall_preserving | 0.9310 | 0.2231 |
| v7 logistic p70 | logistic | balanced | 0.7317 | 0.2479 |
| v7 logistic p90 | logistic | balanced | 0.9200 | 0.1901 |

Calibration-only result on the old 5 DJI segments:

| gate | calibration recall | calibration FP | final/frame | approx precision |
| --- | ---: | ---: | ---: | ---: |
| v3 feature gate | 0.4634 | 2140 | 0.8564 | 0.0088 |
| v7 mlp p50 | 0.4634 | 2455 | 0.9814 | 0.0077 |
| v7 mlp p70 | 0.4634 | 2377 | 0.9504 | 0.0079 |
| v7 mlp p90 | 0.4634 | 2332 | 0.9326 | 0.0081 |
| v7 logistic p70 | 0.4634 | 2262 | 0.9048 | 0.0083 |
| v7 logistic p90 | 0.4634 | 2217 | 0.8869 | 0.0085 |

Decision:

No V7 variant beats the v3 calibration baseline, so V7 was not evaluated on `121806`. This is intentional: held-out should only be touched after a non-held-out calibration improvement. The model/objective change alone is not enough with the current data. The next step is data-side: add more non-held-out DJI positive scene-recovery tracklets, ideally from clips where strict FP-control suppresses true persistent drone tracks but recall-oriented output keeps them.
