# Static/Fast Sanity Tests 2026-05-20

## Models

- YOLO-P2 candidate detector: `runs/detect/runs/detect/qstr_two_source_yolo_p2_gpu_v1/yolo_p2_candidate/weights/best.pt`
- Crop recognizer: `runs/qstr_two_source_gpu_v1/crop_singleframe/crop_model.pt`
- Temporal recognizer: `runs/qstr_two_source_gpu_v1/temporal_speed_tube/temporal_model.pt`
- Feature ROI recognizer: `runs/qstr_two_source_gpu_v1/feature_roi/feature_model.pt`

## Frozen10 Profile Benchmark

Two runnable inference profiles are now fixed as PowerShell scripts:

- `tools/run_qstr_stable_profile.ps1`
  - Primary use: default QSTR run with low false-positive pressure.
  - Detector: old Anti-UAV YOLO-P2 primary detector only.
  - Stage B: hard-positive binary crop + temporal recognizers.
  - Verified objectness: disabled.
  - Motion candidates: disabled by default for clean YOLO/tracker/Stage-B evaluation.
- `tools/run_qstr_hard_recovery_profile.ps1`
  - Primary use: hard-case recovery for tiny / low-objectness / fallback-only targets.
  - Detector: old YOLO-P2 primary plus hard-neg v2 fallback.
  - Stage B: same hard-positive binary crop + temporal recognizers.
  - Fallback gate: enabled, requiring crop/temporal support before fallback proposals can become drone detections.
  - Verified objectness: enabled in `hard_recovery` mode.

The benchmark runner is:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -OutRoot runs\profiles\frozen10_profile_eval `
  -Device 0
```

Useful smoke-test form:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -MaxVideos 1 `
  -MaxFrames 5 `
  -OutRoot runs\profiles\frozen10_profile_smoke `
  -Device 0
```

The benchmark writes:

- `profile_benchmark_summary.json`
- `profile_benchmark_summary.csv`
- per-profile inference folders under `stable/` and `hard_recovery/`

Current smoke check on `20190925_111757_1_5`, first 5 frames:

- stable: `TP=1`, `FP=5`, `precision=0.1667`, `recall=1.0`
- hard_recovery: `TP=1`, `FP=9`, `precision=0.1000`, `recall=1.0`

This smoke result only verifies the profile wiring and metric aggregation. It is not the final frozen10 result.

Frozen10 first-60-frame profile check:

```text
runs/profiles/frozen10_profile_60f/profile_benchmark_summary.json
runs/profiles/frozen10_profile_60f/profile_threshold_sweep.csv
```

At final score threshold `0.20`, IoU `0.30`:

- stable: `TP=32`, `FP=355`, `precision=0.0827`, `recall=0.5926`
- hard_recovery: `TP=24`, `FP=604`, `precision=0.0382`, `recall=0.4444`

Threshold sweep from the same predictions:

| profile | threshold | TP | FP | precision | recall |
|---|---:|---:|---:|---:|---:|
| stable | 0.20 | 32 | 355 | 0.0827 | 0.5926 |
| hard_recovery | 0.20 | 24 | 604 | 0.0382 | 0.4444 |
| stable | 0.30 | 32 | 297 | 0.0973 | 0.5926 |
| hard_recovery | 0.30 | 23 | 241 | 0.0871 | 0.4259 |
| stable | 0.40 | 22 | 236 | 0.0853 | 0.4074 |
| hard_recovery | 0.40 | 22 | 204 | 0.0973 | 0.4074 |
| stable | 0.50 | 8 | 94 | 0.0784 | 0.1481 |
| hard_recovery | 0.50 | 12 | 109 | 0.0992 | 0.2222 |

Interpretation:

- The current stable profile is still the better default at low threshold because recall is higher and FP is lower.
- The current hard-recovery profile is not ready as a global default. It only becomes competitive at stricter thresholds, where both profiles lose substantial recall.
- Next profile work should focus on making fallback more selective, not on more training epochs.

## Selective Hard-Recovery Gate Update

The hard-recovery script now adds two selectivity constraints:

- post-fusion fallback only runs when the primary detector is also weak:
  - `--fallback-post-trigger-max-primary-objectness 0.35`
- fallback proposals are filtered by size:
  - `--fallback-max-box-side 128`

This keeps fallback focused on low-objectness / tiny recovery instead of running v2 whenever Stage B is uncertain.

Frozen10 first-60-frame rerun:

```text
runs/profiles/frozen10_profile_60f_selective/profile_benchmark_summary.json
runs/profiles/frozen10_profile_60f_selective/profile_threshold_sweep_compare.csv
```

Threshold sweep comparison:

| profile | threshold | TP | FP | precision | recall |
|---|---:|---:|---:|---:|---:|
| stable_old | 0.20 | 32 | 355 | 0.0827 | 0.5926 |
| hard_old | 0.20 | 24 | 604 | 0.0382 | 0.4444 |
| hard_selective | 0.20 | 33 | 568 | 0.0549 | 0.6111 |
| stable_old | 0.30 | 32 | 297 | 0.0973 | 0.5926 |
| hard_old | 0.30 | 23 | 241 | 0.0871 | 0.4259 |
| hard_selective | 0.30 | 33 | 311 | 0.0959 | 0.6111 |
| stable_old | 0.40 | 22 | 236 | 0.0853 | 0.4074 |
| hard_old | 0.40 | 22 | 204 | 0.0973 | 0.4074 |
| hard_selective | 0.40 | 23 | 240 | 0.0875 | 0.4259 |

Interpretation:

- Selective hard-recovery is materially better than the previous hard-recovery profile.
- At threshold `0.30`, it gives slightly higher recall than stable (`0.6111` vs `0.5926`) with similar precision (`0.0959` vs `0.0973`), but still more FP (`311` vs `297`).
- `20190925_111757_1_7` is still not recovered in the first-60-frame check, so this does not yet solve the hardest tiny case.
- Current practical setting:
  - default paper/system profile: stable
  - optional recovery profile: selective hard-recovery with score threshold around `0.30`

## Static / Hovering Sanity

Input:

```text
data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_black.mp4
```

Run type:

- oracle seed box at `[1054, 451, 1058, 455]`
- motion candidates disabled
- recognizer/fusion/tracker enabled

Output:

```text
runs/sanity_static_hover/fixed_sky_black_seed
```

Summary:

- frames: `80`
- predictions: `80`
- predicted drone: `80`
- max final drone score: `0.6828`
- mean final drone score: `0.6653`
- modes:
  - `static_or_hovering`: `79`
  - `normal`: `1`
- average alignment quality: `0.7405`
- average motion score: `0.0025`

Interpretation:

The static/hovering path works under oracle localization: the target is retained even though motion evidence is nearly zero. This supports the design goal that static targets should be preserved by crop/temporal/tracker evidence rather than discarded because motion is weak.

## Static Stage A Candidate Recall

Command:

```powershell
python -m qstr_dronedet.cli stage-a-yolo-recall `
  --metadata data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_black.json `
  --out runs/sanity_static_hover/stage_a_yolo_recall_fixed_sky_black `
  --yolo-weights runs/detect/runs/detect/qstr_two_source_yolo_p2_gpu_v1/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --device 0 `
  --max-frames 80 `
  --frame-stride 5
```

Single static black-dot sample:

- sampled GT frames: `16`
- IoU recall at `0.1`: `1.0`
- center recall at `24 px`: `1.0`

Four static variants:

```text
Clip_1_fixed_sky_black
Clip_1_fixed_sky_lowcontrast
Clip_1_jitter1_sky_black
Clip_1_jitter2_sky_black
```

Aggregate:

- sampled GT frames: `64`
- IoU recall at `0.1`: `0.75`
- center recall at `24 px`: `0.75`

Per variant:

| Variant | Sampled frames | Center recall |
| --- | ---: | ---: |
| fixed_sky_black | 16 | 1.0 |
| fixed_sky_lowcontrast | 16 | 0.0 |
| jitter1_sky_black | 16 | 1.0 |
| jitter2_sky_black | 16 | 1.0 |

Lowering YOLO confidence to `0.01` on the low-contrast variant still produced `0.0` recall.

Interpretation:

Stage A can localize black static/jitter targets, but it currently fails on low-contrast static targets. This is a detector training/data issue, not a Stage B recognition issue and not simply a confidence-threshold issue.

## Fast Moving Dot Sanity

Input:

```text
data/experiment_samples/nps_tracker/Clip_1_fast_moving_dot.mp4
```

Run type:

- full pipeline with YOLO-P2 tiled proposals
- motion candidates enabled
- recognizer/fusion/tracker enabled

Output:

```text
runs/sanity_fast_target/fast_moving_dot
```

Summary:

- frames: `80`
- candidate diagnostics: `3240`
- predicted drone candidates: `159`
- max final drone score: `0.8395`
- mean final drone score across all candidates: `0.0453`
- average alignment quality: `0.7497`
- average motion score: `0.0256`

Interpretation:

The fast moving synthetic target can produce high-confidence drone detections in the full pipeline. The high background count is expected because the detector/tracker emits many proposals and Stage B filters most of them.

## Speedx2 / Speedx4 Motion Sanity

Inputs:

```text
data/experiment_samples/nps_speed_subset/Clip_1/Clip_1_speedx2.mp4
data/experiment_samples/nps_speed_subset/Clip_1/Clip_1_speedx4.mp4
```

Run type:

- `motion-debug`
- k values: `1 2 4`

Outputs:

```text
runs/sanity_fast_target/motion_debug_speedx2
runs/sanity_fast_target/motion_debug_speedx4
```

Summary:

| Input | Frames | Mean q_H | Min q_H | Max q_H | Best k |
| --- | ---: | ---: | ---: | ---: | --- |
| speedx2 | 154 | 0.6483 | 0.5528 | 0.6907 | mostly 1 |
| speedx4 | 77 | 0.5769 | 0.4669 | 0.6772 | 1 |

Interpretation:

Alignment quality drops from speedx2 to speedx4. This is the expected direction: higher apparent speed makes temporal alignment harder and should reduce confidence in motion cues.

## Known Limitations

- The static oracle-seed test validates Stage B/tracker/fusion behavior. Stage A candidate recall is now measured separately with `stage-a-yolo-recall`.
- Full-pipeline infer on speedx2/speedx4 was too slow in the interactive run and was stopped before JSON output. For those clips, this report uses `motion-debug` as the speed/alignment sanity check.
- The feature recognizer is likely overfit on current synthetic feature data; treat feature-branch confidence cautiously.

## Next Detector Fix

Add low-contrast static positives and photometric augmentations to the tiled YOLO-P2 training data, then retrain Stage A and rerun `stage-a-yolo-recall` on the four static variants.

## YOLO-P2 Low-Contrast Tiled v2

Dataset build:

```powershell
python -m qstr_dronedet.cli build-static-hover-yolo-dataset `
  --metadata data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_black.json data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_lowcontrast.json data/experiment_samples/nps_static_hover_realistic/Clip_1_jitter1_sky_black.json data/experiment_samples/nps_static_hover_realistic/Clip_1_jitter2_sky_black.json `
  --out data/yolo_candidate/static_hover_lowcontrast_tiled_v2 `
  --frame-stride 1 `
  --max-frames-per-video 80 `
  --val-fraction 0.2 `
  --seed 31 `
  --min-box-px 8 `
  --tiled `
  --tile-size 256 `
  --positives-per-box 3 `
  --negatives-per-image 2 `
  --negative-pad-px 32 `
  --photometric-augmentations 4
```

Dataset summary:

- images/labels: `5440`
- positive tiles: `4800`
- negative tiles: `640`
- split: `4352` train, `1088` val
- image variants: `orig=960`, `photo0=960`, `photo1=960`, `photo2=960`, `photo3=960`, negatives=`640`

Training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/start_yolo_p2_tiled_train_detached.ps1 `
  -Data data/yolo_candidate/static_hover_lowcontrast_tiled_v2/yolo_tiled/data.yaml `
  -Out runs/detect/yolo_p2_static_hover_lowcontrast_tiled_v2_gpu `
  -Pretrained runs/detect/runs/detect/qstr_two_source_yolo_p2_gpu_v1/yolo_p2_candidate/weights/best.pt `
  -Epochs 12 `
  -ImgSize 256 `
  -Batch 16 `
  -Device 0 `
  -JobDir runs/detached/yolo_p2_lowcontrast_v2
```

Final training metrics at epoch 12:

- precision: `0.99991`
- recall: `0.79688`
- mAP50: `0.795`
- mAP50-95: `0.795`

Stage A recall with v2 weights:

| Variant | Sampled frames | IoU recall @ 0.1 | Center recall @ 24 px |
| --- | ---: | ---: | ---: |
| fixed_sky_black | 16 | 1.0 | 1.0 |
| fixed_sky_lowcontrast | 16 | 0.0 | 0.0 |
| jitter1_sky_black | 16 | 1.0 | 1.0 |
| jitter2_sky_black | 16 | 1.0 | 1.0 |

Aggregate:

- sampled GT frames: `64`
- IoU recall at `0.1`: `0.75`
- center recall at `24 px`: `0.75`

Low-contrast focused result:

- sampled GT frames: `16`
- IoU recall at `0.1`: `0.0`
- center recall at `24 px`: `0.0`
- first miss: frame `0`, GT `[1054, 451, 1058, 455]`, `num_candidates=0`

Interpretation:

Photometric augmentation on the tiled dataset did not improve low-contrast static candidate recall. The low-contrast failure remains a Stage A proposal failure: the detector emits no candidates near the target, so Stage B never receives a box. The next fix should change the positive generation itself, not only copy photometric variants. Useful next attempts are stronger low-contrast target injection, oversampling the low-contrast metadata clip, training with larger target render size before scaling back down, or adding a dedicated high-resolution low-contrast tile curriculum.

## YOLO-P2 Low-Contrast Tiled v3

Code change:

- Added low-contrast target injection for positive tiles.
- Added source-name based positive oversampling with `--positive-repeat-pattern` and `--positive-repeat-factor`.
- Fixed repeat matching to use the source frame filename instead of the full output path, so `lowcontrast` does not accidentally match the dataset directory name.

Dataset build:

```powershell
python -m qstr_dronedet.cli build-static-hover-yolo-dataset `
  --metadata data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_black.json data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_lowcontrast.json data/experiment_samples/nps_static_hover_realistic/Clip_1_jitter1_sky_black.json data/experiment_samples/nps_static_hover_realistic/Clip_1_jitter2_sky_black.json `
  --out data/yolo_candidate/static_hover_lowcontrast_tiled_v3 `
  --frame-stride 1 `
  --max-frames-per-video 80 `
  --val-fraction 0.2 `
  --seed 41 `
  --min-box-px 8 `
  --tiled `
  --tile-size 256 `
  --positives-per-box 3 `
  --negatives-per-image 2 `
  --negative-pad-px 32 `
  --photometric-augmentations 2 `
  --low-contrast-injections 8 `
  --positive-repeat-pattern lowcontrast `
  --positive-repeat-factor 4
```

Dataset summary:

- total tiles: `19120`
- positive tiles: `18480`
- negative tiles: `640`
- split: `15395` train, `3725` val
- lowcontrast positives: `10560`
- other static positives: `7920`
- per positive tile variants: `orig`, `photo0`, `photo1`, `lowcontrast0` through `lowcontrast7`

Quick GPU fine-tune:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/start_yolo_p2_tiled_train_detached.ps1 `
  -Data data/yolo_candidate/static_hover_lowcontrast_tiled_v3/yolo_tiled/data.yaml `
  -Out runs/detect/yolo_p2_static_hover_lowcontrast_tiled_v3_quick_gpu `
  -Pretrained runs/detect/runs/detect/qstr_two_source_yolo_p2_gpu_v1/yolo_p2_candidate/weights/best.pt `
  -Epochs 3 `
  -ImgSize 256 `
  -Batch 64 `
  -Device 0 `
  -JobDir runs/detached/yolo_p2_lowcontrast_v3_quick
```

Final training metrics at epoch 3:

- precision: `0.99996`
- recall: `0.99277`
- mAP50: `0.995`
- mAP50-95: `0.9355`

Stage A recall with v3 quick weights:

| Variant | Sampled frames | IoU recall @ 0.1 | Center recall @ 24 px | Avg candidates |
| --- | ---: | ---: | ---: | ---: |
| fixed_sky_black | 16 | 1.0 | 1.0 | 10.0 |
| fixed_sky_lowcontrast | 16 | 1.0 | 1.0 | 10.0 |
| jitter1_sky_black | 16 | 1.0 | 1.0 | 10.0 |
| jitter2_sky_black | 16 | 1.0 | 1.0 | 10.0 |

Aggregate:

- sampled GT frames: `64`
- IoU recall at `0.1`: `1.0`
- center recall at `24 px`: `1.0`

Interpretation:

Low-contrast static candidate recall improved from `0.0` to `1.0` on the sampled synthetic static-hover benchmark. The useful change was not generic photometric augmentation; it was explicit low-contrast target injection plus targeted oversampling of the lowcontrast clip. This supports the next experimental path: keep Stage A as a class-agnostic candidate detector, but train it with low-contrast positive curricula and hard empty tiles rather than relying on motion or recognizer recovery.

## Full Infer FP Check With v3 YOLO-P2

Input:

```text
data/experiment_samples/nps_static_hover_realistic/Clip_1_fixed_sky_lowcontrast.mp4
```

Recognizers:

```text
runs/qstr_two_source_gpu_v1/crop_singleframe/crop_model.pt
runs/qstr_two_source_gpu_v1/feature_roi/feature_model.pt
runs/qstr_two_source_gpu_v1/temporal_speed_tube/temporal_model.pt
```

YOLO-only comparison:

- motion candidates disabled
- max frames: `80`
- old YOLO: `runs/detect/runs/detect/qstr_two_source_yolo_p2_gpu_v1/yolo_p2_candidate/weights/best.pt`
- v3 YOLO: `runs/detect/runs/detect/yolo_p2_static_hover_lowcontrast_tiled_v3_quick_gpu/yolo_p2_candidate/weights/best.pt`

| Run | Frames with candidates | Diagnostics | GT high-drone frames | High-drone near GT | High-drone far from GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| old YOLO only | 0 | 0 | 0 | 0 | 0 |
| v3 YOLO only | 80 | 480 | 80 | 80 | 0 |

The v3 detector produced `6` YOLO/tracker candidates per frame on this clip. It produced one high-confidence drone candidate per frame, and all high-confidence drone candidates were near the GT box. No high-confidence far-from-GT YOLO false positives appeared in this YOLO-only check.

Partial full-pipeline comparison:

- motion candidates enabled
- both runs were stopped by the interactive timeout after partial output
- old output covered `14` frames with candidates
- v3 output covered `15` frames with candidates

| Run | Diagnostics | Avg candidates/frame | GT high-drone frames | High-drone near GT | High-drone far from GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| old full partial | 3900 | 278.57 | 0 | 0 | 1 |
| v3 full partial | 3822 | 254.80 | 15 | 15 | 1 |

The one far-from-GT high-drone candidate in both full-pipeline partial runs was the same motion-source candidate around frame `3`, bbox approximately `[1473, 11, 1479, 15]`, with score `0.563`. That false positive was not introduced by v3 YOLO; it comes from the motion branch.

Interpretation:

On the low-contrast static-hover clip, v3 YOLO fixes the Stage A proposal failure without increasing high-confidence YOLO false positives in the YOLO-only check. With motion enabled, the dominant false-positive risk remains the motion candidate branch rather than the new low-contrast YOLO-P2 detector.

## Motion-Only Artifact Gate

Change:

- Added a fusion-time gate for isolated motion artifacts.
- Trigger condition: `source == motion`, low tracker support, high motion score, high crop/feature disagreement, and feature branch strongly favors background.
- Effect: add background/alignment-artifact mass and set `diagnostic_cause="isolated_motion_artifact"` when the candidate is no longer a credible drone.
- Supported motion candidates such as `motion+tracker` are not suppressed by this gate.

Regression tests:

```text
pytest tests -q
19 passed
```

YOLO-only check after the gate:

| Run | Frames | Diagnostics | GT high-drone frames | High-drone near GT | High-drone far from GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 YOLO only + gate | 80 | 480 | 80 | 80 | 0 |

Motion-enabled check after the gate:

```text
runs/full_infer_fp_check/lowcontrast_v3_full_15f_motion_gate
```

| Run | Frames | Diagnostics | GT high-drone frames | High-drone near GT | High-drone far from GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 full partial before gate | 15 | 3822 | 15 | 15 | 1 |
| v3 full 15f after gate | 15 | 5369 | 15 | 15 | 0 |

The previous motion-source false positive at frame `3`, bbox approximately `[1473, 11, 1479, 15]`, changed from:

```text
predicted_class=drone, final_drone_score=0.563, diagnostic_cause=None
```

to:

```text
predicted_class=background, final_drone_score=0.329, diagnostic_cause=isolated_motion_artifact
```

Interpretation:

The gate removes the observed high-confidence motion-only artifact without hurting the low-contrast GT detections in this sanity test. This is a diagnostic/fusion fix, not a detector change: it keeps the candidate in the logs but prevents an unsupported motion artifact from becoming a final drone detection.

## Anti-UAV300 Second-Source Smoke

Dataset:

```text
D:/datasets/Anti-UAV300/Anti-UAV300.zip
```

Subset export:

```powershell
python -m qstr_dronedet.cli export-anti-uav300-subset `
  --zip D:/datasets/Anti-UAV300/Anti-UAV300.zip `
  --out D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq `
  --split test `
  --modality visible `
  --max-sequences 5 `
  --frame-stride 10 `
  --max-frames-per-sequence 80
```

Subset summary:

- sequences: `5`
- boxes/frames: `333`
- tags:
  - `static_hovering`: `240`
  - `fast_target`: `80`
  - `tiny`: `13`

YOLO tiled candidate dataset:

```text
D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_yolo_candidate
```

- tiles: `1332`
- positive: `666`
- negative: `666`
- train: `1064`
- val: `268`

Fine-tune:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/start_yolo_p2_tiled_train_detached.ps1 `
  -Data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_yolo_candidate/yolo_tiled/data.yaml `
  -Out runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu `
  -Pretrained runs/detect/runs/detect/yolo_p2_static_hover_lowcontrast_tiled_v3_quick_gpu/yolo_p2_candidate/weights/best.pt `
  -Epochs 5 `
  -ImgSize 256 `
  -Batch 32 `
  -Device 0 `
  -JobDir runs/detached/anti_uav300_visible_5seq_yolo_p2
```

Final val metrics:

- precision: `0.99708`
- recall: `0.97761`
- mAP50: `0.99463`
- mAP50-95: `0.70527`

Stage A recall on 93 sampled Anti-UAV frames:

| Weights | IoU recall @ 0.1 | Center recall @ 24 px |
| --- | ---: | ---: |
| NPS-derived v3 lowcontrast YOLO | 0.0323 | 0.0108 |
| Anti-UAV fine-tuned YOLO | 0.9892 | 0.9892 |

By tag after Anti-UAV fine-tune:

| Tag | Frames | Center recall |
| --- | ---: | ---: |
| tiny | 13 | 1.000 |
| static_hovering | 40 | 1.000 |
| fast_target | 40 | 0.975 |

Full infer smoke:

```powershell
python -m qstr_dronedet.cli infer `
  --video D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/raw_videos/test/visible/20190925_111757_1_2/visible.mp4 `
  --out runs/anti_uav300_visible_5seq_infer_smoke/seq_1_2_anti_uav_ft `
  --yolo-weights runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --crop-weights runs/qstr_two_source_gpu_v1/crop_singleframe/crop_model.pt `
  --feature-weights runs/qstr_two_source_gpu_v1/feature_roi/feature_model.pt `
  --temporal-weights runs/qstr_two_source_gpu_v1/temporal_speed_tube/temporal_model.pt `
  --max-frames 20 `
  --disable-motion-candidates
```

Smoke summary:

- diagnostics rows: `60`
- frames: `20`
- predicted classes:
  - `drone`: `41`
  - `background`: `19`
- high-drone detections: `20`
- frames with high-drone detections: `20`
- max final drone score: `0.6837`

Interpretation:

The NPS-derived v3 detector does not transfer directly to Anti-UAV visible clips. A small Anti-UAV fine-tune is enough to make Stage A candidate recall high on this small subset, and the resulting weights plug into the existing QSTR full-infer path. This validates Anti-UAV300 as a usable second data source, but the current result is still a small-subset smoke test, not a held-out benchmark.

## Anti-UAV300 Held-Out Visible 10-Seq

Held-out export:

```powershell
python -m qstr_dronedet.cli export-anti-uav300-subset `
  --zip D:/datasets/Anti-UAV300/Anti-UAV300.zip `
  --out D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq `
  --split test `
  --modality visible `
  --start-index 5 `
  --max-sequences 10 `
  --frame-stride 10 `
  --max-frames-per-sequence 80
```

Held-out summary:

- sequences: `10`
- boxes/frames: `737`
- tags:
  - `static_hovering`: `497`
  - `tiny`: `160`
  - `fast_target`: `80`

Stage A recall with the 5-seq Anti-UAV fine-tuned YOLO:

```text
runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt
```

| Split | Frames | IoU recall @ 0.1 | Center recall @ 24 px |
| --- | ---: | ---: | ---: |
| held-out visible 10-seq | 737 | 0.8915 | 0.8860 |

By tag:

| Tag | Frames | Center recall |
| --- | ---: | ---: |
| static_hovering | 497 | 0.9457 |
| tiny | 160 | 0.7438 |
| fast_target | 80 | 0.8000 |

Weakest sequence:

```text
20190925_111757_1_7
```

- frames: `80`
- center recall: `0.5375`
- tag: `tiny`

Full infer smoke on weakest sequence:

```text
D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq/raw_videos/test/visible/20190925_111757_1_7/visible.mp4
```

YOLO-only, 30 frames:

- diagnostics rows: `208`
- avg candidates/frame: `6.93`
- predicted classes:
  - `alignment_artifact`: `16`
  - `background`: `175`
  - `drone`: `17`
- high-drone detections: `0`
- max final drone score: `0.2925`

Motion enabled, partial 8 frames:

- diagnostics rows: `11083`
- avg candidates/frame: `1385.38`
- predicted classes:
  - `alignment_artifact`: `468`
  - `background`: `9833`
  - `drone`: `782`
- high-drone detections: `0`
- max final drone score: `0.3012`

Interpretation:

The held-out Anti-UAV check shows two separate gaps. Stage A generalizes moderately but not enough on tiny and fast-target sequences, especially `20190925_111757_1_7`. Stage B does not yet recognize Anti-UAV crops as drone with high confidence: even when YOLO/tracker candidates are present, final drone scores stay below `0.5`. Motion enabled also creates too many candidates on this held-out sequence, so any full-pipeline benchmark needs candidate caps or stronger motion filtering before long runs.

Next action:

Build Anti-UAV Stage B crop/temporal datasets from the 5-seq training subset and hard negatives from held-out motion candidates, then retrain crop/temporal recognizers before judging full-pipeline Anti-UAV performance.

## Anti-UAV Stage B Crop/Temporal Adaptation

Added real-video Stage B dataset export:

```powershell
python -m qstr_dronedet.cli build-real-stage-b-dataset `
  --annotations D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/annotations/qstr_real_boxes.csv `
  --out D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b `
  --negative-per-positive 1 `
  --crop-scale 4.0 `
  --crop-size 128 `
  --tube-t 5 `
  --tube-size 96
```

Scale-4 dataset:

- crop/tube drone: `333`
- crop/tube background: `333`

Training:

```powershell
python -m qstr_dronedet.cli train-recognizer --type crop `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b/crop `
  --out runs/anti_uav300_stage_b/crop_model.pt `
  --epochs 8 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type temporal `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b/temporal `
  --out runs/anti_uav300_stage_b/temporal_model.pt `
  --epochs 8 `
  --balance sampler
```

Held-out weakest sequence, YOLO-only 30-frame infer:

```text
20190925_111757_1_7
```

| Stage B weights | Predicted drone rows | High-drone rows | Max final drone score |
| --- | ---: | ---: | ---: |
| old NPS/synthetic Stage B | 17 | 0 | 0.2925 |
| Anti-UAV scale-4 crop/temporal | 76 | 0 | 0.4215 |

Tighter crop/tube dataset:

```powershell
python -m qstr_dronedet.cli build-real-stage-b-dataset `
  --annotations D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/annotations/qstr_real_boxes.csv `
  --out D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_tight `
  --negative-per-positive 2 `
  --crop-scale 2.0 `
  --crop-size 128 `
  --tube-t 5 `
  --tube-size 96
```

Tight dataset:

- crop/tube drone: `333`
- crop/tube background: `666`

Training:

```powershell
python -m qstr_dronedet.cli train-recognizer --type crop `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_tight/crop `
  --out runs/anti_uav300_stage_b_tight/crop_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type temporal `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_tight/temporal `
  --out runs/anti_uav300_stage_b_tight/temporal_model.pt `
  --epochs 12 `
  --balance sampler
```

Held-out weakest sequence result:

| Stage B weights | Predicted drone rows | High-drone rows | Max final drone score |
| --- | ---: | ---: | ---: |
| old NPS/synthetic Stage B | 17 | 0 | 0.2925 |
| Anti-UAV scale-4 crop/temporal | 76 | 0 | 0.4215 |
| Anti-UAV tight crop/temporal | 57 | 0 | 0.4489 |

Top tight-Stage-B candidate:

```text
frame=11, source=tracker+yolo_tile, final_drone_score=0.449,
crop_drone=0.637, temporal_drone=0.543
```

Interpretation:

Anti-UAV Stage B adaptation moves the held-out sequence in the right direction: final drone score rises from `0.2925` to `0.4489`, and the branch probabilities for the best candidate become drone-favoring. It still does not cross the `0.5` final detection threshold. The current crop/temporal training set is too small and only uses random background negatives; the next useful change is to add detector-proposal positives and hard negatives from held-out motion/YOLO proposals, then calibrate fusion weights on a validation split instead of lowering thresholds blindly.

## Anti-UAV Detector-Proposal Stage B

Added real detector-proposal Stage B export:

```powershell
python -m qstr_dronedet.cli build-real-detector-proposal-stage-b `
  --annotations D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/annotations/qstr_real_boxes.csv `
  --out D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_proposals `
  --yolo-weights runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --device 0 `
  --max-proposals-per-frame 8 `
  --max-negatives-per-frame 4 `
  --match-iou 0.1 `
  --match-center-px 24 `
  --crop-scale 2.0 `
  --crop-size 128 `
  --tube-t 5 `
  --tube-size 96
```

Proposal dataset:

- proposal drone samples: `353`
- proposal hard background samples: `118`
- missed drone rows: `1`
- average positive proposal IoU: `0.7983`

Training:

```powershell
python -m qstr_dronedet.cli train-recognizer --type crop `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_proposals/crop `
  --out runs/anti_uav300_stage_b_proposals/crop_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type temporal `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_proposals/temporal `
  --out runs/anti_uav300_stage_b_proposals/temporal_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type feature `
  --data D:/datasets/Anti-UAV300/qstr_subset_test_visible_5seq/qstr_stage_b_proposals/feature/annotations.csv `
  --out runs/anti_uav300_stage_b_proposals/feature_model.pt `
  --epochs 8
```

Training behavior:

- crop loss: `1.1884 -> 0.5492`
- temporal loss: `1.5570 -> 0.5994`
- feature loss: `0.1440 -> 0.0150`

Held-out weakest sequence, YOLO-only 30-frame infer:

| Stage B weights | Predicted drone rows | High-drone rows | Max final drone score |
| --- | ---: | ---: | ---: |
| old NPS/synthetic Stage B | 17 | 0 | 0.2925 |
| Anti-UAV oracle tight Stage B | 57 | 0 | 0.4489 |
| Anti-UAV proposal crop/temporal + proposal feature | 0 | 0 | 0.3168 |
| Anti-UAV proposal crop/temporal, no proposal feature | 30 | 0 | 0.3715 |

Top proposal no-feature candidate:

```text
frame=27, final_drone_score=0.372,
crop_drone=0.668, feature_drone=0.286, temporal_drone=0.431
```

Top proposal with-feature candidate:

```text
frame=27, final_drone_score=0.317,
crop_drone=0.668, feature_drone=0.001, temporal_drone=0.431
```

Interpretation:

Detector-proposal training makes the crop branch more drone-favoring on held-out proposals, but temporal remains weak and the proposal feature branch overfits or shifts badly: it predicts near-zero drone probability on held-out positives and suppresses the fused result. The best current held-out behavior is still the oracle-tight Stage B with the older feature recognizer. This confirms that the next step should be validation-calibrated fusion and harder temporal/feature training, not simply adding more proposal epochs.

## Validation-Calibrated Fusion

Added `calibrate-fusion` and `infer --fusion-calibration` so validation-set branch reliability can be measured and then applied during inference.

Calibration sequence:

- Held-out Anti-UAV sequence: `20190925_111757_1_5`
- Frames: first `30`
- Candidate source: YOLO-P2 tiled only
- GT matching: IoU `>=0.1` or center distance `<=24 px`
- Frame tolerance: `5`

Commands:

```powershell
python -m qstr_dronedet.cli calibrate-fusion `
  --diagnostics runs/anti_uav300_heldout_infer_smoke/seq_1_5_yolo_only_stageb_tight/diagnostics.jsonl `
  --gt D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq/annotations/qstr_real_boxes.csv `
  --video D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq/raw_videos/test/visible/20190925_111757_1_5/visible.mp4 `
  --frame-tolerance 5 `
  --out runs/anti_uav300_fusion_calibration/seq_1_5_tight_tol5

python -m qstr_dronedet.cli infer `
  --video D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq/raw_videos/test/visible/20190925_111757_1_5/visible.mp4 `
  --out runs/anti_uav300_heldout_infer_smoke/seq_1_5_yolo_only_stageb_tight_calibrated `
  --yolo-weights runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --crop-weights runs/anti_uav300_stage_b_tight/crop_model.pt `
  --temporal-weights runs/anti_uav300_stage_b_tight/temporal_model.pt `
  --feature-weights runs/qstr_two_source_gpu_v1/feature_roi/feature_model.pt `
  --fusion-calibration runs/anti_uav300_fusion_calibration/seq_1_5_tight_tol5/fusion_calibration_summary.json `
  --max-frames 30 `
  --disable-motion-candidates
```

Best tight-Stage-B calibration weights:

```json
{
  "crop": 0.3205,
  "feature": 0.3205,
  "temporal": 0.2564,
  "tracker": 0.0641,
  "motion": 0.0385
}
```

Held-out `1_5` result at final score threshold `0.5`:

| Setup | Rows | Positive proposal rows | TP rows | FP rows | Frame recall | Max score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tight Stage B, default fusion | 94 | 53 | 16 | 3 | 0.500 | 0.6096 |
| tight Stage B, calibrated fusion | 94 | 53 | 15 | 3 | 0.467 | 0.6193 |
| proposal crop/temporal, no feature | 94 | 53 | 0 | 0 | 0.000 | 0.3835 |

Interpretation:

Calibration is now wired into the runnable pipeline, but the single-sequence calibration result is not yet a stable improvement. It slightly raises the maximum score while reducing recall on this same 30-frame slice. More importantly, calibration confirms the proposal no-feature model is under-confident: its positive scores stay below `0.4`, so weight tuning alone cannot recover detections. The next data step is to train temporal/feature with harder Anti-UAV positives and hard negatives across more sequences, then calibrate on a separate validation set instead of one short clip.

## Anti-UAV 20-Sequence Train Split Stage B and Detector Check

Built a larger Anti-UAV visible train split subset:

```powershell
python -m qstr_dronedet.cli export-anti-uav300-subset `
  --zip D:/datasets/Anti-UAV300/Anti-UAV300.zip `
  --out D:/datasets/Anti-UAV300/qstr_train_visible_20seq `
  --split train `
  --modality visible `
  --max-sequences 20 `
  --frame-stride 5 `
  --max-frames-per-sequence 80
```

Subset summary:

- sequences: `20`
- GT boxes: `1600`
- tags: `fast_target=800`, `static_hovering=480`, `tiny=320`

Built detector-proposal Stage B data from the 20seq subset:

```powershell
python -m qstr_dronedet.cli build-real-detector-proposal-stage-b `
  --annotations D:/datasets/Anti-UAV300/qstr_train_visible_20seq/annotations/qstr_real_boxes.csv `
  --out D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_stage_b_proposals `
  --yolo-weights runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --device 0 `
  --max-proposals-per-frame 8 `
  --max-negatives-per-frame 4 `
  --match-iou 0.1 `
  --match-center-px 24 `
  --crop-scale 2.0 `
  --crop-size 128 `
  --tube-t 5 `
  --tube-size 96
```

Proposal dataset:

- proposal drone samples: `848`
- proposal hard background samples: `882`
- missed drone rows: `778`
- feature rows: `1730`

Trained 20seq Stage B branches:

```powershell
python -m qstr_dronedet.cli train-recognizer --type crop `
  --data D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_stage_b_proposals/crop `
  --out runs/anti_uav300_stage_b_proposals_20seq/crop_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type temporal `
  --data D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_stage_b_proposals/temporal `
  --out runs/anti_uav300_stage_b_proposals_20seq/temporal_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type feature `
  --data D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_stage_b_proposals/feature/annotations.csv `
  --out runs/anti_uav300_stage_b_proposals_20seq/feature_model.pt `
  --epochs 8
```

Training behavior:

- crop loss: `0.8194 -> 0.1676`
- temporal loss: `0.9373 -> 0.2717`
- feature loss: `0.2118 -> 0.0222`

Sanity check on a training-sequence clip `20190925_101846_1_1`:

- rows: `30`
- TP rows: `30`
- FP rows: `0`
- mean final score: `0.8365`
- mean branch drone probabilities: crop `0.9009`, feature `0.9388`, temporal `0.9851`

This confirms the 20seq Stage B models learned the training-domain proposal appearance.

### Detector/Stage B Coupling

The old 5seq detector produced very low objectness on a held-out train-split validation sequence even when Stage B branches were confident:

| Setup | Candidate rows | TP rows | FP rows | Mean objectness | Mean positive final score |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-val, old YOLO + 20seq Stage B | 23 | 0 | 0 | 0.1145 | 0.0716 |

Trained a matching 20seq YOLO-P2 detector:

```powershell
python -m qstr_dronedet.cli prepare-real-yolo-dataset `
  --annotations D:/datasets/Anti-UAV300/qstr_train_visible_20seq/annotations/qstr_real_boxes.csv `
  --out D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_yolo_candidate `
  --tile-size 256 `
  --positives-per-box 4 `
  --negatives-per-image 2 `
  --val-fraction 0.2 `
  --seed 7 `
  --min-box-px 1

python -m qstr_dronedet.cli train-yolo-p2 `
  --data D:/datasets/Anti-UAV300/qstr_train_visible_20seq/qstr_yolo_candidate/yolo_tiled/data.yaml `
  --out runs/detect/anti_uav300_train_visible_20seq_yolo_p2_gpu `
  --write-model-yaml runs/detect/anti_uav300_train_visible_20seq_yolo_p2_gpu/yolov8_p2_qstr.yaml `
  --pretrained runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --epochs 8 `
  --imgsz 256 `
  --batch 32 `
  --device 0
```

YOLO validation metrics:

- precision: `0.988`
- recall: `0.993`
- mAP50: `0.993`
- mAP50-95: `0.684`

With the matching 20seq detector, same train-val sequence recovered:

| Setup | Candidate rows | TP rows | FP rows | Mean objectness | Mean positive final score |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-val, 20seq YOLO + 20seq Stage B | 63 | 30 | 0 | 0.4377 | 0.7842 |

Held-out Anti-UAV test sequence `20190925_111757_1_5`:

| Setup | Candidate rows | Positive proposal rows | TP rows | FP rows | Mean positive final score | Max positive final score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test `1_5`, tight baseline | 94 | 53 | 16 | 3 | 0.3127 | 0.6096 |
| test `1_5`, old YOLO + 20seq Stage B | 94 | 53 | 0 | 0 | 0.1463 | 0.3713 |
| test `1_5`, 20seq YOLO + 20seq Stage B | 239 | 68 | 0 | 0 | 0.0722 | 0.3170 |

Interpretation:

The 20seq train-split pipeline is internally consistent: detector objectness and all Stage B branches work on train-domain validation. It does not transfer to the Anti-UAV test held-out sequence. On `test/visible/20190925_111757_1_5`, the 20seq crop and temporal branches assign low drone probability to matched positive proposals, so fusion cannot recover detections. This is a real domain-shift signal, not a calibration-only issue. The next useful experiment is a mixed-domain Stage B set: keep the train split as the main source, add a small validation-only/test-like adaptation pool from non-held-out test sequences, and continue keeping `20190925_111757_1_5` and the 10seq held-out set frozen for reporting.

## Mixed-Domain Stage B Adaptation

Frozen held-out discipline:

- Frozen held-out test set remains `D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq`.
- The adaptation pool excludes the frozen 10seq and the earlier 5seq smoke subset.
- Frozen reporting sequence `20190925_111757_1_5` was not used in training.

Exported non-held-out test-like adaptation pool:

```powershell
python -m qstr_dronedet.cli export-anti-uav300-subset `
  --zip D:/datasets/Anti-UAV300/Anti-UAV300.zip `
  --out D:/datasets/Anti-UAV300/qstr_adapt_test_visible_5seq `
  --split test `
  --modality visible `
  --start-index 15 `
  --max-sequences 5 `
  --frame-stride 5 `
  --max-frames-per-sequence 80
```

Adaptation sequences:

- `20190925_124000_1_5`: tiny
- `20190925_124000_1_6`: fast_target
- `20190925_124000_1_7`: fast_target
- `20190925_124000_1_8`: tiny
- `20190925_124000_1_9`: tiny

Adaptation proposal dataset:

- proposal drone samples: `370`
- proposal hard background samples: `464`
- missed drone rows: `37`
- feature rows: `834`

Mixed-domain dataset:

- source A: `qstr_train_visible_20seq/qstr_stage_b_proposals`
- source B: `qstr_adapt_test_visible_5seq/qstr_stage_b_proposals`
- mixed crop/tube drone samples: `1218`
- mixed crop/tube background samples: `1346`
- mixed feature rows: `2564`

Mixed-domain Stage B training:

```powershell
python -m qstr_dronedet.cli train-recognizer --type crop `
  --data D:/datasets/Anti-UAV300/qstr_mixed_train20_adapt5_stage_b/crop `
  --out runs/anti_uav300_stage_b_mixed_train20_adapt5/crop_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type temporal `
  --data D:/datasets/Anti-UAV300/qstr_mixed_train20_adapt5_stage_b/temporal `
  --out runs/anti_uav300_stage_b_mixed_train20_adapt5/temporal_model.pt `
  --epochs 12 `
  --balance sampler

python -m qstr_dronedet.cli train-recognizer --type feature `
  --data D:/datasets/Anti-UAV300/qstr_mixed_train20_adapt5_stage_b/feature/annotations.csv `
  --out runs/anti_uav300_stage_b_mixed_train20_adapt5/feature_model.pt `
  --epochs 8
```

Training behavior:

- crop loss: `0.8025 -> 0.4199`
- temporal loss: `0.8571 -> 0.3775`
- feature loss: `0.2281 -> 0.0235`

Frozen held-out `20190925_111757_1_5`, first 30 frames, YOLO-5seq detector:

| Setup | Candidate rows | Positive proposal rows | TP rows | FP rows | Frame recall | Mean positive score | Max positive score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tight baseline | 94 | 53 | 16 | 3 | 0.500 | 0.3127 | 0.6096 |
| pure train20 Stage B | 94 | 53 | 0 | 0 | 0.000 | 0.1463 | 0.3713 |
| mixed train20+adapt5 full Stage B | 94 | 53 | 5 | 3 | 0.133 | 0.2520 | 0.5749 |
| mixed train20+adapt5 no feature | 94 | 53 | 0 | 0 | 0.000 | 0.2285 | 0.4329 |

Branch behavior on frozen positives:

| Setup | crop drone mean | feature drone mean | temporal drone mean | final prob drone mean |
| --- | ---: | ---: | ---: | ---: |
| tight baseline | 0.4073 | 0.4751 | 0.4933 | 0.4835 |
| pure train20 Stage B | 0.0088 | 0.2466 | 0.1442 | 0.2011 |
| mixed train20+adapt5 full Stage B | 0.4074 | 0.2412 | 0.3470 | 0.3825 |
| mixed train20+adapt5 no feature | 0.4074 | 0.1592 | 0.3470 | 0.3609 |

Interpretation:

The mixed-domain adaptation partially fixes the cross-domain Stage B failure. Crop probability on frozen positives recovers from `0.0088` to `0.4074`, and full fusion recovers from `0` TP to `5` TP on the frozen `1_5` slice. It still underperforms the tight baseline because temporal remains lower than baseline and the mixed feature branch is high-variance: it helps some positives enough to cross threshold, but also creates `3` FP rows. The next step should not be blind training. It should be validation calibration using the adaptation pool and then a frozen 10seq evaluation, with separate reporting for crop-only/crop+temporal/full fusion.

## Adaptation-Pool Fusion Calibration And Frozen 10seq Profile Eval

Ran mixed full inference on the 5-sequence adaptation pool, merged diagnostics with `video_path`, and calibrated fusion weights against the adaptation annotations.

Implementation note:

- `calibrate-fusion` now honors a `video_path` field in each diagnostics row.
- This prevents frame-id collisions when multiple videos are merged into one diagnostics JSONL.

Adaptation calibration input:

- diagnostics: `runs/anti_uav300_adapt_infer/mixed_full_combined_diagnostics.jsonl`
- rows: `2256`
- labeled positive candidate rows: `422`
- GT: `D:/datasets/Anti-UAV300/qstr_adapt_test_visible_5seq/annotations/qstr_real_boxes.csv`
- frame tolerance: `5`

Best adaptation-pool calibrated full weights:

```json
{
  "crop": 0.4369,
  "feature": 0.2427,
  "temporal": 0.1942,
  "tracker": 0.0971,
  "motion": 0.0291
}
```

Calibration-pool score:

- TP rows: `89`
- FP rows: `3`
- frame recall: `0.9167`
- row precision: `0.9674`
- F1: `0.9413`

Then ran mixed full inference once on frozen held-out 10seq, first 30 frames per sequence, using the 5seq YOLO detector and mixed Stage B weights. The same branch probabilities were scored offline under multiple fusion profiles:

- `crop_only`
- `crop_temporal_70_30`
- `adapt_calibrated_full`
- `default_existing_final_score`

Frozen 10seq summary:

| Profile | TP rows | FP rows | Positive rows | Precision | Frame recall | Mean positive score | Max positive score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crop_only | 137 | 34 | 392 | 0.8012 | 0.5229 | 0.3634 | 0.8075 |
| crop_temporal_70_30 | 123 | 29 | 392 | 0.8092 | 0.4695 | 0.3595 | 0.8220 |
| adapt_calibrated_full | 125 | 29 | 392 | 0.8117 | 0.4733 | 0.3508 | 0.8458 |
| default_existing_final_score | 130 | 31 | 392 | 0.8075 | 0.4924 | 0.3590 | 0.8598 |

Per-sequence observations:

- `20190925_111757_1_6`: full/calibrated is strong, frame recall `0.9333`.
- `20190925_111757_1_7`: all profiles still have `0.0` frame recall, so this remains a hard Stage A/appearance failure.
- `20190925_111757_1_5`: default full gives `5` TP rows, calibrated full gives `3`, crop-only gives `1`.
- `20190925_124000_1_10`: crop-only gives better recall than temporal/full, so temporal is suppressing some positives.
- `20190925_124000_1_1/1_2/1_3/1_4`: all profiles are usable, but FP rows remain non-trivial.

Interpretation:

Adaptation-pool calibration works on the adaptation pool, but it does not transfer cleanly to frozen 10seq. On frozen held-out, `crop_only` has the best frame recall, while full/calibrated fusion has slightly better precision but lower recall. This means temporal and feature are not yet reliable enough to be universal positive evidence. The immediate research conclusion is useful: the system can now diagnose that Stage B branch reliability is domain-dependent, and the safest frozen-10seq setting is currently crop-heavy fusion with temporal/feature treated as optional evidence rather than mandatory confirmation.

## Frozen Failure Audit: `20190925_111757_1_7`

Reason for audit:

- In frozen 10seq evaluation, `20190925_111757_1_7` had `0.0` frame recall under all tested fusion profiles.
- This sequence is tagged `tiny` and has `TC,OV,TC-MID` in Anti-UAV metadata.

Audit command:

```powershell
python -m qstr_dronedet.cli infer `
  --video D:/datasets/Anti-UAV300/qstr_heldout_test_visible_10seq/raw_videos/test/visible/20190925_111757_1_7/visible.mp4 `
  --out runs/anti_uav300_failure_audit/seq_1_7_mixed_full_100f `
  --yolo-weights runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --yolo-conf 0.05 `
  --yolo-tile-size 256 `
  --yolo-tile-stride 128 `
  --crop-weights runs/anti_uav300_stage_b_mixed_train20_adapt5/crop_model.pt `
  --temporal-weights runs/anti_uav300_stage_b_mixed_train20_adapt5/temporal_model.pt `
  --feature-weights runs/anti_uav300_stage_b_mixed_train20_adapt5/feature_model.pt `
  --max-frames 100 `
  --disable-motion-candidates
```

Summary artifact:

- `runs/anti_uav300_failure_audit/seq_1_7_failure_audit_summary.json`
- visualization: `runs/anti_uav300_failure_audit/seq_1_7_frame0_audit.jpg`

Audit results over the first 100 frames:

| Quantity | Value |
| --- | ---: |
| candidate rows | 839 |
| GT boxes under 100 frames | 10 |
| GT boxes with any nearby candidate | 10 |
| Stage A positive matches | 7 |
| Stage A recall | 0.700 |

Matched positive candidate statistics:

| Field | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| objectness | 0.1020 | 0.0722 | 0.1492 |
| final drone score | 0.0368 | 0.0263 | 0.0468 |
| crop drone prob | 0.8158 | 0.7687 | 0.8389 |
| feature drone prob | 0.1715 | 0.0044 | 0.4577 |
| temporal drone prob | 0.6526 | 0.5853 | 0.7477 |
| final prob `P(drone|object)` | 0.3676 | 0.3139 | 0.4284 |
| IoU | 0.4237 | 0.1049 | 0.5960 |

Counterfactual scoring on matched positives:

| Profile | Mean score | Max score | Hits >= 0.5 |
| --- | ---: | ---: | ---: |
| actual final score | 0.0368 | 0.0468 | 0 |
| objectness * crop only | 0.0835 | 0.1229 | 0 |
| objectness * crop/temporal | 0.0780 | 0.1122 | 0 |
| no objectness, final prob only | 0.3676 | 0.4284 | 0 |
| sqrt(objectness) * final prob | 0.1154 | 0.1327 | 0 |
| max(objectness, 0.5) * final prob | 0.1838 | 0.2142 | 0 |

Key observation:

The GT-adjacent proposals are not absent, and crop/temporal recognition often supports `drone`. The decisive failure is low Stage A objectness: no matched positive candidate has objectness above `0.15`.

At the same time, there are high-objectness non-GT distractors. On frame 0:

- GT-adjacent candidate: objectness `0.149`, final score `0.047`
- top nonmatched distractor: objectness `0.652`, final score `0.449`

The top distractor is a building/window texture region, not the annotated drone. This creates an objectness ranking error: the detector assigns much higher objectness to structured background than to the drone-like target.

Interpretation:

`20190925_111757_1_7` is primarily a Stage A objectness-ranking and hard-negative failure, not a fusion failure. Fusion cannot fix it because `P(object)` suppresses matched positives by about an order of magnitude, while the detector promotes building texture distractors. The next useful training step is not more Stage B tuning. It is detector hard-negative mining and objectness calibration focused on building/window distractors and tiny low-contrast positives, while keeping `1_7` frozen for evaluation.

## Stage A Hard-Negative Mining V1

Goal:

- Improve detector objectness ranking without training on frozen `1_7`.
- Mine hard negatives from `train20 + adapt5`, especially high-objectness nonmatched proposals.
- Keep frozen `1_7` only for evaluation.

Built combined Stage A training annotations:

- source A: `D:/datasets/Anti-UAV300/qstr_train_visible_20seq/annotations/qstr_real_boxes.csv`
- source B: `D:/datasets/Anti-UAV300/qstr_adapt_test_visible_5seq/annotations/qstr_real_boxes.csv`
- output: `D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/annotations/qstr_real_boxes.csv`
- rows: `2000`
- videos: `25`

Extracted frames:

```powershell
python -m qstr_dronedet.cli extract-real-annotated-frames `
  --annotations D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/annotations/qstr_real_boxes.csv `
  --frames-dir D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/frames `
  --out-csv D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/frame_annotations.csv
```

Built tiled YOLO positives with photometric and low-contrast variants:

```powershell
python -m qstr_dronedet.cli build-tiled-yolo-candidate-dataset `
  --annotations D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/frame_annotations.csv `
  --images-root D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/frames `
  --out D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/yolo_tiled_hardneg_v1 `
  --tile-size 256 `
  --positives-per-box 4 `
  --negatives-per-image 2 `
  --val-fraction 0.2 `
  --seed 19 `
  --min-box-px 1 `
  --negative-pad-px 32 `
  --photometric-augmentations 1 `
  --low-contrast-injections 1
```

Appended proposal-mined hard negatives:

- mined from train20/adapt5 proposal manifests only
- source proposals: nonmatched `matched_positive=false`
- score threshold: `proposal_score >= 0.2`
- added hard-negative empty-label tiles: `523`
- hard-negative score range: `0.2005` to `0.8263`
- manifest: `D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/yolo_tiled_hardneg_v1/hard_negative_manifest.jsonl`

Trained hard-negative YOLO-P2:

```powershell
python -m qstr_dronedet.cli train-yolo-p2 `
  --data D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/yolo_tiled_hardneg_v1/data.yaml `
  --out runs/detect/anti_uav300_stage_a_hardneg_train20_adapt5_yolo_p2_gpu `
  --write-model-yaml runs/detect/anti_uav300_stage_a_hardneg_train20_adapt5_yolo_p2_gpu/yolov8_p2_qstr.yaml `
  --pretrained runs/detect/runs/detect/anti_uav300_visible_5seq_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --epochs 6 `
  --imgsz 256 `
  --batch 32 `
  --device 0
```

Training validation metrics:

- precision: `0.993`
- recall: `0.987`
- mAP50: `0.994`
- mAP50-95: `0.678`

Frozen `1_7` comparison:

- old detector diagnostics: `runs/anti_uav300_failure_audit/seq_1_7_mixed_full_100f/diagnostics.jsonl`
- hard-negative detector diagnostics: `runs/anti_uav300_failure_audit/seq_1_7_hardneg_yolo_mixed_full_100f/diagnostics.jsonl`
- comparison JSON: `runs/anti_uav300_failure_audit/seq_1_7_hardneg_comparison.json`

| Metric on frozen `1_7` | Old 5seq YOLO | Hard-neg YOLO |
| --- | ---: | ---: |
| candidate rows | 839 | 4730 |
| GT boxes evaluated | 9 | 9 |
| Stage A positive matches | 7 | 9 |
| Stage A recall | 0.7778 | 1.0000 |
| matched positive objectness mean | 0.1020 | 0.2110 |
| matched positive objectness max | 0.1492 | 0.8123 |
| matched positive final score mean | 0.0368 | 0.1349 |
| matched positive final score max | 0.0468 | 0.5847 |
| positives with final score >= 0.5 | 0 | 1 |
| matched positive IoU mean | 0.4237 | 0.6128 |

Top nonmatched distractors:

| Metric | Old 5seq YOLO | Hard-neg YOLO |
| --- | ---: | ---: |
| top-20 nonmatched score mean | 0.3355 | 0.4191 |
| top-20 nonmatched score max | 0.4492 | 0.4903 |
| top-20 nonmatched objectness mean | 0.5976 | 0.6802 |

Interpretation:

Hard-negative mining fixed part of the `1_7` failure: GT-adjacent candidate matching improved from `7/9` to `9/9`, mean matched objectness doubled, and one positive crossed the final `0.5` score threshold. The original building/window distractor seen in the frame-0 audit is no longer the top failure mode. However, the detector now emits many more candidates (`4730` vs `839`) and still has high-objectness nonmatched proposals, now mostly around other small structured regions. This is progress, but not a final detector. The next detector step should control proposal volume and ranking: add harder empty-label mining from the new detector's own FPs, raise or calibrate YOLO conf for inference, and measure frozen 10seq precision/recall before accepting the detector globally.

## Stage A Hard-Negative Mining V2 And Threshold Sweep

Second-round mining:

- Used hard-neg YOLO v1 itself on the train20+adapt5 pool.
- Built proposal mining output:
  `D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/hardneg_yolo_v2_mining_proposals`
- Limited mining to `500` annotation rows for runtime.

Mining proposal summary:

| Class | Count | Mean score | Max score | Score >= 0.2 | Score >= 0.4 | Score >= 0.6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matched positive | 546 | 0.7494 | 0.9212 | 522 | 513 | 504 |
| nonmatched negative | 960 | 0.1506 | 0.8475 | 214 | 45 | 11 |

Built `yolo_tiled_hardneg_v2` by copying v1 and adding:

- v2 hard-negative empty-label tiles: `212`
- v2 hard-negative score range: `0.2000` to `0.8475`

Trained hard-negative YOLO v2:

```powershell
python -m qstr_dronedet.cli train-yolo-p2 `
  --data D:/datasets/Anti-UAV300/qstr_stage_a_hardneg_train20_adapt5/yolo_tiled_hardneg_v2/data.yaml `
  --out runs/detect/anti_uav300_stage_a_hardneg_v2_train20_adapt5_yolo_p2_gpu `
  --write-model-yaml runs/detect/anti_uav300_stage_a_hardneg_v2_train20_adapt5_yolo_p2_gpu/yolov8_p2_qstr.yaml `
  --pretrained runs/detect/runs/detect/anti_uav300_stage_a_hardneg_train20_adapt5_yolo_p2_gpu/yolo_p2_candidate/weights/best.pt `
  --epochs 4 `
  --imgsz 256 `
  --batch 32 `
  --device 0
```

YOLO v2 validation metrics:

- precision: `0.987`
- recall: `0.982`
- mAP50: `0.993`
- mAP50-95: `0.674`

### Frozen `1_7` Threshold Sweep

Compared old 5seq detector, hard-neg v1, and hard-neg v2 on frozen `1_7`, first 60 frames. This is a detector operating-point sweep over already-generated diagnostics, not training.

Key operating points:

| Detector/profile | Rows/frame | Stage A recall | Matched objectness mean | Matched score max | High FP score >= 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| old, conf 0.05 | 7.45 | 1.000 | 0.1070 | 0.0468 | 0 |
| old, conf 0.15 | 3.58 | 0.000 | 0.0000 | 0.0000 | 0 |
| v1, conf 0.05 | 36.15 | 1.000 | 0.2282 | 0.5847 | 1 |
| v1, conf 0.15 | 4.45 | 0.667 | 0.4622 | 0.5847 | 1 |
| v1, top10/frame | 9.83 | 0.833 | 0.3853 | 0.5847 | 1 |
| v2, conf 0.05 | 39.22 | 1.000 | 0.4642 | 0.5723 | 2 |
| v2, conf 0.15 | 6.23 | 1.000 | 0.4979 | 0.5723 | 2 |
| v2, conf 0.20 | 3.58 | 0.833 | 0.5599 | 0.5723 | 2 |
| v2, top10/frame | 9.95 | 1.000 | 0.4649 | 0.5723 | 2 |

Interpretation on `1_7`:

- v2 improves objectness ranking on the hard frozen sequence.
- `conf=0.15` is a much better `1_7` operating point than `conf=0.05`: it keeps Stage A recall at `1.0` while reducing candidates from `39.22/frame` to `6.23/frame`.
- Raising to `conf=0.20` starts losing recall.

### Frozen 10seq Check With V2

Ran frozen 10seq, first 30 frames/sequence, with:

- detector: hard-neg v2
- YOLO conf: `0.15`
- Stage B: mixed full
- motion candidates disabled

Compared against the previous frozen 10seq mixed-full run with old 5seq detector at `conf=0.05`.

| Frozen 10seq setup | Candidate rows | TP rows | FP rows | Positive rows | Precision | Frame recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old 5seq detector, conf 0.05 | 1164 | 115 | 26 | 344 | 0.8156 | 0.8148 |
| hard-neg v2, conf 0.15 | 4665 | 129 | 43 | 361 | 0.7500 | 0.8148 |

Per-sequence v2 notes:

- `20190925_111757_1_7`: frame recall improved to `0.3333` with `1` TP row and `0` FP rows.
- `20190925_111757_1_6`: frame recall `1.0`, `27` TP, `0` FP.
- Most `20190925_124000_*` sequences reach frame recall `1.0`, but FP rows increase.

Interpretation:

v2 helps the hard `1_7` case, but it is not a better global detector yet. Frozen 10seq frame recall is unchanged, precision drops, and candidate volume increases by about 4x. The practical next design is not to globally replace the old detector with v2. Instead, use v2 as a targeted recovery detector for hard tiny/low-objectness mode, or add a post-detector proposal budget/ranking layer before Stage B. The current best global default remains the old 5seq detector; v2 is useful diagnostic evidence and a candidate for fallback mode.

## Proposal Budget And Fallback Detector

Implemented a proposal-budget/fallback path in `qstr_dronedet.cli infer`.

New controls:

- `--max-yolo-candidates-per-frame`: top-N primary YOLO candidates before merge.
- `--max-fallback-yolo-candidates-per-frame`: top-N fallback YOLO candidates before merge.
- `--max-candidates-per-frame`: final merged proposal budget before Stage B.
- `--fallback-yolo-weights`: optional second detector, currently intended for hard-neg v2.
- `--fallback-trigger-objectness`: pre-recognition fallback when primary YOLO has weak/no objectness.
- `--fallback-trigger-final-score`: post-recognition fallback when primary proposals do not produce a reliable final drone score.

The fallback detector is source-tagged as `*_fallback`, and diagnostics include `fallback_yolo_ran`.

Smoke-tested on frozen `1_7`, first 60 frames, old detector as primary and hard-neg v2 as fallback, motion disabled, final candidate budget `10/frame`.

| Profile | Rows/frame | Fallback frames | Fallback rows | Score threshold | TP rows | FP rows | Frame recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| post-fusion trigger `0.20` | 9.87 | 29 | 26 | 0.5 | 0 | 1 | 0.000 |
| post-fusion trigger `0.50` | 9.95 | 60 | 60 | 0.2 | 2 | 43 | 0.333 |
| post-fusion trigger `0.50` | 9.95 | 60 | 60 | 0.3 | 1 | 21 | 0.167 |
| post-fusion trigger `0.50` | 9.95 | 60 | 60 | 0.5 | 0 | 1 | 0.000 |

Interpretation:

- The budget layer works mechanically and keeps proposal volume near `10/frame`.
- Objectness-only fallback is insufficient for `1_7` because primary YOLO can produce high-objectness distractors.
- Post-fusion fallback is the right trigger family, but the current Stage B/fusion still does not rank recovered `1_7` positives high enough at a useful precision point.
- Next step should be a source-aware fallback gate: allow fallback proposals to enter Stage B, but accept final fallback detections only when crop/temporal confidence is strong and artifact/background probability is low.

## Source-Aware Fallback Gate

Implemented source-aware fallback gating in rule fusion.

Behavior:

- Fallback proposals still run through crop, feature, temporal, and normal fusion.
- If a fallback-source proposal has weak crop/temporal drone support, or strong background/alignment-artifact evidence, it is suppressed to background.
- Rejected fallback proposals keep diagnostic cause `fallback_rejected`.

New `infer` controls:

- `--disable-fallback-gate`
- `--fallback-gate-min-branch-drone`
- `--fallback-gate-min-crop-temporal-mean`
- `--fallback-gate-max-negative-evidence`

Frozen `1_7`, first 60 frames, old primary detector + hard-neg v2 fallback, post-fusion fallback trigger `0.50`, proposal budget `10/frame`:

| Profile | Fallback rejected | Score threshold | TP rows | FP rows | Frame recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| fallback gate off | 0 | 0.2 | 2 | 43 | 0.333 |
| fallback gate off | 0 | 0.3 | 1 | 21 | 0.167 |
| fallback gate off | 0 | 0.5 | 0 | 1 | 0.000 |
| fallback gate on | 56 | 0.2 | 2 | 33 | 0.333 |
| fallback gate on | 56 | 0.3 | 1 | 16 | 0.167 |
| fallback gate on | 56 | 0.5 | 0 | 1 | 0.000 |

Interpretation:

- Gate reduces fallback-induced false positives without losing the recovered low-threshold positives.
- It is not enough yet for a clean high-confidence operating point on `1_7`.
- The remaining bottleneck is Stage B score calibration/ranking for tiny recovered fallback boxes, not Stage A recall.

## Frozen 10seq Gate Sweep

Ran frozen 10seq, first 30 frames per sequence, with:

- primary detector: old 5seq YOLO-P2
- fallback detector: hard-neg v2 YOLO-P2
- motion candidates disabled
- proposal budget: top `10/frame`
- Stage B: mixed crop/feature/temporal weights

Two inference profiles were generated:

- `old_only_budget10`
- `fallback_gate_off_budget10`

Additional gate profiles were evaluated offline from the fallback diagnostics:

- loose: branch drone `0.35`, crop-temporal mean `0.25`, max negative evidence `0.65`
- default: branch drone `0.45`, crop-temporal mean `0.35`, max negative evidence `0.55`
- strict: branch drone `0.55`, crop-temporal mean `0.45`, max negative evidence `0.45`

Annotated-frame summary:

| Profile | Threshold | TP | FP | Precision | Frame recall | Fallback rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old only | 0.2 | 26 | 5 | 0.839 | 0.875 | 0 |
| fallback gate off | 0.2 | 27 | 6 | 0.818 | 0.917 | 0 |
| fallback loose | 0.2 | 23 | 4 | 0.852 | 0.833 | 15 |
| fallback default | 0.2 | 22 | 4 | 0.846 | 0.792 | 16 |
| fallback strict | 0.2 | 22 | 4 | 0.846 | 0.792 | 16 |
| old only | 0.3 | 21 | 2 | 0.913 | 0.833 | 0 |
| fallback gate off | 0.3 | 20 | 2 | 0.909 | 0.833 | 0 |
| fallback loose | 0.3 | 17 | 1 | 0.944 | 0.708 | 15 |
| fallback default | 0.3 | 16 | 1 | 0.941 | 0.667 | 16 |
| old only | 0.5 | 14 | 0 | 1.000 | 0.583 | 0 |
| fallback gate off | 0.5 | 15 | 0 | 1.000 | 0.625 | 0 |
| fallback default | 0.5 | 15 | 0 | 1.000 | 0.625 | 16 |

Per-sequence notes:

- `20190925_111757_1_7`: fallback improves recall from `0.0` to `0.667` at threshold `0.2`, but adds FP. The default gate keeps the recall gain and reduces FP by one on annotated frames.
- `20190925_111757_1_9`: fallback and the current gate hurt recall. This is the main reason the global gate profiles underperform old-only at lower thresholds.
- `20190925_111757_1_6`: fallback improves high-threshold recall from `0.667` to `1.0` with no FP at threshold `0.5`.

Interpretation:

- Hard-neg v2 fallback is useful, but not as a default always-on detector.
- The current gate is too blunt: it suppresses some useful fallback positives on non-`1_7` sequences.
- The best global default remains old-only for now.
- Fallback should be limited to a diagnostic/hard-case recovery mode until Stage B calibration is improved with more detector-proposal positives and hard negatives.

## Detector-Proposal Stage B Dataset

Extended `build-real-detector-proposal-stage-b` to support a primary detector plus fallback detector.

New behavior:

- primary YOLO proposals are kept as normal detector proposals.
- fallback YOLO proposals are source-tagged as fallback proposals.
- nonmatched high-score fallback proposals can be labeled as `alignment_artifact`.
- lower-score nonmatched proposals remain `background`.

New CLI options:

- `--fallback-yolo-weights`
- `--fallback-yolo-conf`
- `--max-fallback-proposals-per-frame`
- `--artifact-score-threshold`

Smoke datasets were written to the external drive:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_train20_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_adapt5_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_combined_smoke`

Initial threshold `artifact_score_threshold=0.25` produced:

| Split | Drone | Background | Alignment artifact |
| --- | ---: | ---: | ---: |
| train20 smoke | 101 | 27 | 42 |
| adapt5 smoke | 84 | 166 | 101 |
| combined smoke | 185 | 193 | 143 |

Short GPU training ran successfully:

- crop e4: `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_combined_smoke\models\crop_model.pt`
- temporal e4: `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_combined_smoke\models\temporal_model.pt`

However, smoke evaluation showed background/artifact confusion. Raising the artifact label threshold to `0.5` produced a cleaner but more imbalanced dataset:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_art05_combined_smoke`

Counts:

| Class | Count |
| --- | ---: |
| drone | 272 |
| background | 335 |
| alignment_artifact | 80 |

Training diagnostics:

- sampler training over-predicts `alignment_artifact`.
- class-weight training over-predicts `background`.
- The data pipeline is ready, but the current three-way auto-label scheme is not ready for large-scale Stage B training.

Interpretation:

The right next modeling target is likely:

1. train Stage B first as `drone` vs `non-drone`;
2. keep `alignment_artifact` as a diagnostic/auxiliary label;
3. only promote artifact to a full class after adding cleaner artifact negatives from bad-alignment video/motion-debug.

## Drone-vs-Non-Drone Stage B

Added `--target-mode drone_binary` to `train-recognizer`.

Implementation detail:

- The models still keep the existing 8-class output head.
- During training, `drone` is mapped to the drone logit and every non-drone class is mapped to the background logit.
- During inference, checkpoints with `target_mode=drone_binary` use a two-logit softmax over `drone/background`; untrained class logits are not included in the probability normalization.

Smoke training on:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_art05_combined_smoke`

Produced:

- `models\crop_drone_binary_e12.pt`
- `models\temporal_drone_binary_e12.pt`
- `models\crop_drone_binary_e12_class_weight.pt`
- `models\drone_binary_e12_smoke_train_eval.json`

Training-set sanity after fixing the binary sampler:

| Branch | Precision | Recall | Notes |
| --- | ---: | ---: | --- |
| crop binary sampler | 0.433 | 0.917 | high recall, many hard-negative FP |
| temporal binary sampler | 0.416 | 0.883 | similar high-recall/low-precision behavior |
| crop binary class_weight | 0.572 | 0.849 at threshold 0.5 | better precision, but scores are compressed |

Frozen `1_7` 60-frame smoke with binary crop/temporal and fallback:

| Threshold | TP | FP | Frame recall |
| ---: | ---: | ---: | ---: |
| 0.2 | 1 | 51 | 0.167 |
| 0.3 | 1 | 27 | 0.167 |
| 0.5 | 0 | 0 | 0.000 |

Interpretation:

- The binary training/inference path is now implemented and runnable.
- It does not yet improve frozen `1_7`.
- Current proposal labels are still too noisy for direct large-scale Stage B training.
- Before scaling, the next dataset change should separate `non-drone` into cleaner buckets: easy background, high-score detector false positives, and motion/alignment artifacts from motion-debug.

## Binary-Bucket Non-Drone Labels

Updated `build-real-detector-proposal-stage-b` so training labels and diagnostic labels can be separated.

New mode:

```powershell
--non-drone-label-mode binary_buckets
```

In this mode:

- training folders contain only `drone` and `background`;
- every non-drone proposal trains as `background`;
- `proposal_manifest.jsonl` and feature CSV preserve `diagnostic_bucket`;
- buckets are `easy_background`, `high_score_detector_fp`, and `alignment_artifact`.

Additional control:

```powershell
--high-score-fp-threshold 0.5
```

Smoke datasets on the external drive:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_train20_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_adapt5_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_combined_smoke`

Combined counts:

| Training class / diagnostic bucket | Count |
| --- | ---: |
| drone | 272 |
| background training label | 415 |
| easy_background | 322 |
| high_score_detector_fp | 13 |
| alignment_artifact | 80 |

Trained binary crop recognizer:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_combined_smoke\models\crop_drone_binary_e12.pt`

Training sanity:

| Threshold | Precision | Recall | Accuracy |
| ---: | ---: | ---: | ---: |
| 0.4 | 0.506 | 0.941 | 0.613 |
| 0.5 | 0.532 | 0.868 | 0.645 |
| 0.6 | 0.591 | 0.695 | 0.689 |

Frozen `1_7` 60-frame smoke with binary-bucket crop + fallback:

| Threshold | TP | FP | Frame recall |
| ---: | ---: | ---: | ---: |
| 0.2 | 0 | 31 | 0.000 |
| 0.3 | 0 | 2 | 0.000 |
| 0.5 | 0 | 0 | 0.000 |

Interpretation:

- The cleaner binary-bucket data improves default non-drone suppression.
- It does not recover the hard `1_7` tiny positives.
- To recover `1_7`, the next data step must add more hard tiny positive proposals to the Stage B training set, not just more negatives.

## Hard Tiny Positive Proposals

Added hard-positive controls to `build-real-detector-proposal-stage-b`:

```powershell
--hard-positive-max-size-px
--hard-positive-max-score
--hard-positive-repeat
```

Matched positive proposals are tagged as `hard_tiny_positive` and repeated when:

- proposal max side is below `--hard-positive-max-size-px`; and
- proposal score is low enough or the source is fallback.

First attempt used `--hard-positive-max-size-px 32`, but Anti-UAV detector proposal boxes were usually much larger because they include context. Positive proposal size statistics showed medians around `117-127 px`, so the hard-positive run used `128 px`.

Smoke datasets:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_train20_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_adapt5_smoke`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke`

Combined counts:

| Bucket | Count |
| --- | ---: |
| drone training label | 847 |
| background training label | 499 |
| hard_tiny_positive | 729 |
| easy_background | 389 |
| alignment_artifact | 97 |
| high_score_detector_fp | 13 |

Trained:

- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\crop_drone_binary_e12.pt`
- `D:\datasets\Anti-UAV300\qstr_stage_b_detector_proposals_old_v2_binary_buckets_hardpos128_combined_smoke\models\temporal_drone_binary_e12.pt`

Training-set sanity for crop:

| Threshold | Precision | Recall | Accuracy |
| ---: | ---: | ---: | ---: |
| 0.4 | 0.769 | 0.880 | 0.758 |
| 0.5 | 0.785 | 0.798 | 0.736 |
| 0.6 | 0.799 | 0.576 | 0.642 |

Also fixed an inference/data mismatch:

- Stage B proposal crops were trained with `crop_scale=2.0`.
- `infer` previously used the default crop/tube scale `4.0`.
- Added `--recognition-crop-scale` and `--recognition-tube-scale` to align inference with training.

Frozen `1_7` 60-frame smoke:

| Setup | Threshold | TP | FP | Frame recall |
| --- | ---: | ---: | ---: | ---: |
| hardpos128 crop, scale 2.0 | 0.2 | 0 | 26 | 0.000 |
| hardpos128 crop, scale 2.0 | 0.3 | 0 | 1 | 0.000 |
| hardpos128 crop+temporal, scale 2.0 | 0.2 | 0 | 45 | 0.000 |
| hardpos128 crop+temporal, scale 2.0 | 0.3 | 0 | 17 | 0.000 |

Diagnostic observation:

- hard-positive training raised branch probabilities near GT boxes into the `0.44-0.60` range;
- final scores remain low because tracker candidates have low objectness and fallback candidates are often suppressed by the fallback gate;
- the next bottleneck is score fusion/objectness handling for tracker/fallback-verified candidates, not just branch recognition.

## Verified Objectness For Tracker/Fallback

Added verified-objectness support in rule fusion.

Rationale:

- `objectness` remains the raw detector/tracker localization score for diagnostics.
- `final_drone_score` can now use an `effective_objectness` floor when a tracker/fallback candidate is verified by Stage B.
- Plain YOLO candidates are not boosted.

New `infer` controls:

```powershell
--disable-verified-objectness
--verified-min-branch-drone
--verified-min-crop-temporal-mean
--verified-max-negative-evidence
--verified-objectness-floor
```

Tests added:

- tracker/fallback candidates with crop+temporal support get an effective objectness floor;
- plain YOLO candidates do not get this boost.

Frozen `1_7` 60-frame smoke with hardpos128 crop+temporal:

| Verified profile | Threshold | TP | FP | Frame recall |
| --- | ---: | ---: | ---: | ---: |
| default | 0.2 | 2 | 177 | 0.333 |
| default | 0.3 | 0 | 20 | 0.000 |
| strict | 0.2 | 0 | 131 | 0.000 |
| strict | 0.3 | 0 | 17 | 0.000 |

Interpretation:

- verified objectness can recover low-threshold `1_7` positives, so the mechanism is useful;
- the current branch scores are still too weak to separate positives cleanly from hard negatives;
- strict verified thresholds remove the recovered positives;
- this now needs a frozen10 verified-objectness sweep/calibration, not more detector training.

## Frozen 10seq Verified-Objectness Sweep

Ran frozen 10seq, first 30 frames per sequence, using:

- old 5seq YOLO-P2 primary detector
- hard-neg v2 fallback detector
- hardpos128 binary crop/temporal recognizers
- recognition crop/tube scale `2.0`
- motion candidates disabled
- proposal budget `10/frame`

Inference was run once with `--disable-verified-objectness`, then verified-objectness profiles were swept offline from diagnostics.

Annotated-frame summary:

| Profile | Threshold | TP | FP | Precision | Frame recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw no verified | 0.20 | 16 | 6 | 0.727 | 0.583 |
| raw no verified | 0.30 | 14 | 2 | 0.875 | 0.583 |
| raw no verified | 0.40 | 12 | 0 | 1.000 | 0.500 |
| loose floor 0.45 | 0.20 | 21 | 21 | 0.500 | 0.583 |
| loose floor 0.45 | 0.30 | 14 | 2 | 0.875 | 0.583 |
| default floor 0.55 | 0.20 | 23 | 27 | 0.460 | 0.667 |
| default floor 0.55 | 0.30 | 18 | 3 | 0.857 | 0.583 |
| medium floor 0.55 | 0.20 | 22 | 25 | 0.468 | 0.625 |
| medium floor 0.55 | 0.30 | 18 | 3 | 0.857 | 0.583 |
| strict floor 0.50 | 0.20 | 21 | 23 | 0.477 | 0.583 |
| strict floor 0.50 | 0.30 | 16 | 3 | 0.842 | 0.583 |

Key per-sequence finding:

- At threshold `0.20`, default verified objectness improves `20190925_111757_1_7` from `0/3` frame recall to `2/3`, but increases FP on several other sequences.
- At threshold `0.30`, verified objectness does not improve global frame recall over raw no-verified; it mostly increases duplicate TP rows on already-detected frames.

Interpretation:

- verified objectness is useful as a hard-case recovery mechanism;
- it is not a clean global default yet;
- the current best global operating point remains raw/no-verified or a high threshold verified profile;
- to use verified objectness in the final system, it should be mode-conditional, e.g. enabled only for diagnostic hard tiny/fallback mode rather than always-on.

## Mode-Conditional Verified Objectness

Changed verified objectness from a global tracker/fallback boost into a mode-conditioned policy.

New CLI option:

```powershell
--verified-objectness-mode always|hard_recovery
```

Behavior:

- `always`: previous behavior, boost verified tracker or fallback candidates.
- `hard_recovery`: boost fallback candidates and only low-objectness tracker recovery candidates.
- plain YOLO candidates are never boosted.

Frozen10 offline sweep from the same raw diagnostics:

| Profile | Threshold | TP | FP | Precision | Frame recall | Verified rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0.20 | 16 | 6 | 0.727 | 0.583 | 0 |
| always default | 0.20 | 23 | 27 | 0.460 | 0.667 | 101 |
| hard_recovery default | 0.20 | 19 | 23 | 0.452 | 0.667 | 87 |
| hard_recovery lowobj0.15 | 0.20 | 17 | 12 | 0.586 | 0.625 | 74 |
| off | 0.30 | 14 | 2 | 0.875 | 0.583 | 0 |
| hard_recovery default | 0.30 | 14 | 2 | 0.875 | 0.583 | 87 |

`1_7` at threshold `0.20`:

| Profile | TP | FP | Frame recall |
| --- | ---: | ---: | ---: |
| off | 0 | 3 | 0.000 |
| always default | 2 | 9 | 0.667 |
| hard_recovery default | 2 | 8 | 0.667 |
| hard_recovery lowobj0.15 | 1 | 4 | 0.333 |

Interpretation:

- `hard_recovery` is better than always-on, but still not clean enough as a global low-threshold mode.
- `hard_recovery lowobj0.15` is the best compromise in this sweep, but its recall gain is small.
- At threshold `0.30`, mode-conditional verified objectness no longer improves frame recall over raw scoring.
- The current reliable default should keep verified objectness effectively off for normal inference; verified should be exposed as a hard-case diagnostic/recovery profile.
