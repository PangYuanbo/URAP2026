# QSTR 1_7 Failure Audit

Date: 2026-05-21

Sequence:

```text
D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_7\visible.mp4
```

Scope:

- first 60 frames
- GT source: `D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv`
- GT frames in this window: `6`
- match criterion for detection summary: IoU `>= 0.30`

Audit outputs:

```text
runs/profiles/1_7_failure_audit/audit_summary.json
runs/profiles/1_7_failure_audit/audit_frames.csv
runs/profiles/1_7_failure_audit/hard_recovery_obj070/
```

## Profile-Level Candidate Coverage

| profile | rows | frames IoU>=0.1 | frames IoU>=0.3 | frames center<=24px | max IoU | mean best IoU |
|---|---:|---:|---:|---:|---:|---:|
| stable | 379 | 0 | 0 | 0 | 0.000 | 0.000 |
| hard_selective | 587 | 4 | 2 | 1 | 0.522 | 0.239 |
| hard_old | 597 | 6 | 4 | 2 | 0.772 | 0.445 |

Interpretation:

- `stable` is a Stage A proposal failure on `1_7`: it never produces a candidate close enough to GT in the first 60 frames.
- `hard_old` proves the fallback detector can produce correct candidates for this sequence.
- `hard_selective` improves global frozen10 behavior but partly suppresses useful fallback behavior on `1_7`.

## Why Selective Hard-Recovery Misses

The new selective profile uses:

```text
--fallback-post-trigger-max-primary-objectness 0.35
--fallback-max-box-side 128
```

The correct old fallback candidates are not oversized:

| frame | old fallback bbox | max side |
|---:|---|---:|
| 0 | `[729.7, 582.5, 825.8, 646.1]` | `96.0` |
| 20 | `[713.4, 592.0, 838.4, 662.8]` | `125.0` |

So the size filter is not the main problem. The issue is the primary-objectness guard. On frame 0, the primary detector has a high-objectness false positive far from GT, so post-fusion fallback is blocked even though the primary final drone score is weak.

## Stage B / Fusion Behavior On Correct Candidates

Old hard-recovery found several correct or near-correct candidates:

| frame | IoU | source | objectness | predicted | final score | crop drone | temporal drone | final P(drone) | final P(bg) |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.772 | yolo_tile_fallback | 0.589 | background | 0.241 | 0.438 | 0.547 | 0.409 | 0.444 |
| 10 | 0.680 | tracker | 0.193 | drone | 0.227 | 0.406 | 0.585 | 0.413 | 0.406 |
| 20 | 0.620 | tracker+yolo_tile_fallback | 0.326 | background | 0.211 | 0.477 | 0.570 | 0.384 | 0.384 |
| 30 | 0.342 | tracker | 0.115 | background | 0.147 | 0.481 | 0.602 | 0.267 | 0.599 |

This is not just a localization problem. For correct candidates, crop and temporal are only moderately confident, while background remains competitive. The fusion rule often chooses `background` even when the final drone score is above `0.20`.

## Objectness Guard Ablation

A one-off rerun on `1_7` relaxed the post-fusion fallback guard:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_hard_recovery_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_7\visible.mp4 `
  -Out runs\profiles\1_7_failure_audit\hard_recovery_obj070 `
  -Device 0 `
  -MaxFrames 60 `
  -FallbackPostTriggerMaxPrimaryObjectness 0.70 `
  -FallbackMaxBoxSide 128
```

Result at score threshold `0.20`:

```text
TP=1, FP=136, FN=5
```

This recovered the frame-0 fallback candidate but caused too many false positives. Relaxing the primary-objectness guard alone is not a good global fix.

## Diagnosis

`1_7` has two separate failures:

1. **Stable profile: Stage A proposal failure.**
   The primary detector does not generate GT-covering candidates.

2. **Hard-recovery profile: Stage B / fusion calibration failure after fallback.**
   Fallback can produce correct candidates, but crop/temporal evidence is not strong enough and background remains too high. Verified objectness can lift score, but it does not change the predicted class when `P(background) >= P(drone)`.

## Next Fix

Do not simply loosen fallback globally.

The next targeted change should be a hard-tiny recovery fusion rule:

- only applies to fallback/tracker candidates,
- candidate has IoU-like temporal consistency from tracker or repeated fallback support,
- crop and temporal are both moderately pro-drone, e.g. `crop_drone >= 0.40` and `temporal_drone >= 0.55`,
- background is not overwhelmingly high, e.g. `background - drone <= 0.08`,
- then allow predicted class `drone` under a diagnostic cause such as `hard_tiny_recovery`.

This would recover frame 0 / 10 / 20 style candidates without simply accepting every fallback proposal.

## Hard-Tiny Recovery Implementation Check

Implemented an experimental recovery rule with CLI controls:

```text
--enable-hard-tiny-recovery
--hard-tiny-min-crop-drone
--hard-tiny-min-temporal-drone
--hard-tiny-min-temporal-crop-delta
--hard-tiny-max-bg-minus-drone
--hard-tiny-min-support
--hard-tiny-score-floor
--hard-tiny-allow-tracker-only
```

Important implementation choice:

- tracker-only recovery is disabled by default;
- fallback or fallback+tracker recovery is allowed;
- tracker-only can be enabled only for diagnostic runs with `--hard-tiny-allow-tracker-only`.

Reason:

An initial run that allowed tracker-only recovery did recover the correct frame-10 tracker candidate, but it also recovered many stale false tracks. The pattern was:

```text
thr=0.20: TP=1, FP=121, FN=5
```

Many false tracks had the same crop/temporal pattern as the true tiny target, so crop/temporal margin alone was not enough to safely recover tracker-only candidates.

The safer default fallback-only recovery run was:

```text
runs/profiles/1_7_failure_audit/hard_tiny_recovery_fallback_only
```

Result:

```text
thr=0.20: TP=0, FP=76, FN=6
thr=0.22: TP=0, FP=69, FN=6
thr=0.30: TP=0, FP=12, FN=6
```

This confirms the current bottleneck:

- fallback-only recovery is safe but does not rescue `1_7` under the selective fallback trigger;
- tracker-only recovery can rescue a true box but is too noisy without stronger track validation;
- the next real fix should be tracker validation, not a looser fusion rule.

Concrete next step:

- add track metadata to diagnostics and fusion decisions:
  - `track_id`
  - track age/history length
  - number of detector updates
  - last detector source
  - track drift from last detector update
- only allow tracker-only hard-tiny recovery when the track has recent detector support or stable low-drift temporal consistency.
