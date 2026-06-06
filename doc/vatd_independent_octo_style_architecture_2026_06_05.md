# VATD: Independent Octo-Style Video-Action Tiny Drone Detector

Date: 2026-06-05

## Goal

Build an independent tiny-UAV detector that can beat TransVisDrone under matched evaluation. TransVisDrone is a baseline, dataset source, and evaluator target. It is not the architecture that this method depends on.

The method is:

```text
video episode
  -> observation tokens
  -> state/action tokens
  -> temporal transformer
  -> motion-action drone score
  -> detection outputs
```

This follows the Octo-style vision-action pattern without language. Robot trajectories become drone video episodes, and robot action chunks become short-horizon bbox motion chunks.

## Naming

Working name:

```text
VATD: Video-Action Tiny Drone Detector
```

Paper positioning:

```text
An independent vision-action transformer for tiny UAV detection.
```

Avoid describing the method as:

```text
TransVisDrone + action chunk
YOLOMG + action chunk
```

Existing detectors may be used as optional proposal sources during early training and ablation, but the model definition is independent.

## Architecture

### Inputs

```text
RGB video clip: T frames
candidate/tube tokens: optional, from dense anchors or weak proposals
state tokens: bbox, score, visibility, optional camera-motion quality
```

Initial short-horizon setting:

```text
past_len = 3 to 5
future_len = 1 to 3
action = [dx, dy, dlogw, dlogh]
```

### Tokenizers

```text
frame/patch tokenizer
ROI tokenizer for tiny candidate regions
state tokenizer for normalized bbox + score + visibility
readout tokens for detection, motion-action, and action chunk
```

The current implementation can start from ROI crops because that is already wired in `VideoActionTrackletDataset`. The later fully independent version should replace external proposal crops with a dense tiny-object proposal tokenizer.

### Backbone

```text
temporal transformer over video/action/state tokens
```

No language branch is used. This is a VA model, not a VLA model.

### Heads

Primary head:

```text
motion_action_head:
  output = P(short clip/tube has drone-like motion)
  loss = BCEWithLogitsLoss(label)
```

This is the main "action" interpretation in the current VATD branch. The action
token is not being used as a pure future-coordinate regressor. It is used as a
short video-action readout that learns whether the observed few-frame motion is
drone-like.

Auxiliary head:

```text
short_action_head:
  output = future action chunk
  target = future_actions - constant_velocity_actions
  loss = SmoothL1Loss
```

Detection head:

```text
detection_head:
  output = frame bbox + drone confidence
```

In the proposal-conditioned MVP, frame boxes can be inherited from the candidate tube and rescored by the motion-action head. In the fully independent model, a dense proposal/detection head must generate boxes directly.

## Training Objective

MVP objective:

```text
L = L_motion_action + lambda_action * L_action_residual
```

Full detector objective:

```text
L = L_detection + lambda_motion * L_motion_action + lambda_action * L_action_residual
```

The action target should be short-horizon residual motion, not direct future boxes as the primary task. Direct action-only prediction is too sensitive to a few-pixel error on tiny boxes and does not answer whether the clip contains a UAV-like motion pattern.

Therefore the paper-facing method should be described as:

```text
video/action transformer learns a drone-like motion-action score,
with short-horizon action residual prediction as an auxiliary dynamics prior
```

not:

```text
predict the next box, then call that detection
```

## Data Format

Unified episode row:

```text
dataset_source, seq, frame_id, image_width, image_height,
x1, y1, x2, y2,
score, visible, label, bucket
```

Unified short video-action sample:

```text
dataset_source, seq, track_id, anchor_frame,
crops[T, C, H, W],
state[T, 6],
label,
future_actions[H, 4],
future_boxes[H, 4]
```

Positive samples:

```text
GT drone tube or proposal tube matched to GT
```

Negative samples:

```text
weak background tube
bird/insect/static speck/compression artifact
wrong proposal tube near hard motion
```

Formal runs must train on the full available mixture. Early stopping is allowed, but reporting should preserve the mixture manifest and checkpoint summary.

## Inference

Proposal-conditioned MVP:

```text
low-threshold candidate tubes
  -> VATD motion-action score
  -> optional action residual consistency
  -> rescore candidate frame detections
  -> official evaluator
```

Fully independent target:

```text
video clip
  -> VATD-owned weak candidates (temporal saliency now, dense tiny-object head later)
  -> VATD transformer
  -> frame detections + motion-action confidence
  -> official evaluator
```

Score fusion for the MVP:

```text
final_score = learned_fusion(base_detector_score, motion_action_score, action_residual_consistency)
```

or a simple first ablation:

```text
final_score = base_detector_score * motion_action_score
```

## Baseline Comparison

TransVisDrone should appear as:

```text
baseline: TransVisDrone official/pretrained
evaluation: same split, same evaluator, matched FPPI/FAR or confidence operating point
```

The first target is not only higher raw recall. The target is:

```text
same or lower false alarms than TransVisDrone,
higher recall / detection encounters,
especially for tiny and weak-motion targets.
```

## Implementation Milestones

1. Add `VATDMotionActionTransformer`: video/state transformer with a motion-action classification head and residual action head.
2. Train it from `VideoActionTrackletDataset` labels, not proposal confidence mean/max.
3. Add scoring output fields:

```text
motion_action_score
action_residual_error
vatd_score
```

4. Run proposal-conditioned VATD against current YOLOMG/TVD proposal tracklets.
5. Replace optional external proposals with an internal dense tiny-object proposal head.
6. Evaluate through the same official pipeline used for TransVisDrone.

Current implementation status:

- Done: `VATDMotionActionTransformer` has a `motion_action_head` and an `action_residual_head`.
- Done: training target for the primary head is the tracklet label, i.e. drone-like motion vs non-drone-like motion.
- Done: scoring emits `motion_action_score`, `vatd_action_residual_center_error`, and fused `vatd_score`.
- Done: `export-temporal-saliency-tracklets` adds a VATD-owned weak-candidate source from frame differencing, so the independent branch is no longer defined only as YOLOMG/TVD proposal rescoring.
- In progress: stronger full-data crop/video runs for AOT/YOLOMG shuffle and NPS.
- Not done: fully independent learned dense proposal/detection head; the current official MVP still evaluates by rescoring proposal/prediction tubes.

## Current CLI / Runner Entry Points

Train directly:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli train-vatd-motion-action-policy `
  --tracklet-jsonl artifacts\route_b_official\aot_part0_tvd_val\tvd_aot_part0_conf0p2_ioutrack\route_b_tracklets_min2_gap2.jsonl `
  --out artifacts\route_b_official\aot_part0_vatd_motion_action_train\vatd_motion_action.pt `
  --frame-root D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test\part0\frames `
  --past-len 3 `
  --future-len 1 `
  --crop-size 48 `
  --image-width 2432 `
  --image-height 2048 `
  --epochs 10 `
  --batch-size 128 `
  --d-model 128 `
  --num-layers 3 `
  --num-workers 4 `
  --frame-cache-size 16
```

Detached training:

```powershell
.\tools\start_route_b_vatd_motion_action_train_detached.ps1
.\tools\monitor_route_b_vatd_motion_action_train.ps1
```

The detached VATD trainer defaults to a 5090-oriented profile:

```text
batch_size = 512
crop_size = 96
d_model = 256
num_layers = 6
nhead = 8
num_workers = 4
frame_cache_size = 32
```

This does not create a large on-disk crop cache. It keeps the Windows DataLoader pressure below the crashy regime seen with `batch_size=1024`, `num_workers=12`, `frame_cache_size=256`, while still using a larger model and batch than the smoke profile. Do not launch it while another video-action training process is still running on the same GPU.

Detached scoring:

```powershell
.\tools\start_route_b_vatd_motion_action_score_detached.ps1 `
  -TrackletJsonl artifacts\route_b_official\aot_fulltest_wport_baseline_tracklets\route_b_tracklets_min2_gap2_with_paths.jsonl `
  -Weights artifacts\route_b_official\aot_part0_vatd_motion_action_train\vatd_motion_action.pt `
  -Out artifacts\route_b_official\aot_fulltest_vatd_motion_action\vatd_scores.jsonl `
  -BatchSize 256 `
  -NumWorkers 4 `
  -FrameCacheSize 16

.\tools\monitor_route_b_vatd_motion_action_score.ps1
```

The important output fields are:

```text
motion_action_score
vatd_action_consistency_score
vatd_score
```

The fulltest tracklet file above already carries `image_path` fields from AOT ground truth, so `-FrameRoot` is optional for that path. For paper-comparable claims, the scored tracklets still need to be converted back to frame/AOT prediction outputs and evaluated with the same official evaluator used for TransVisDrone.

VATD-owned weak candidates:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli export-frame-list-from-gt-csv `
  --gt-csv path\to\gt.csv `
  --frame-root path\to\frames `
  --out artifacts\vatd_temporal_saliency\frames.txt `
  --max-frames-per-seq 5

.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli export-temporal-saliency-tracklets `
  --list-files artifacts\vatd_temporal_saliency\frames.txt `
  --gt-csv path\to\gt.csv `
  --out artifacts\vatd_temporal_saliency\tracklets `
  --threshold 24 `
  --min-area 2 `
  --max-area 400 `
  --min-tracklet-rows 2
```

This source is deliberately simple: it supplies detector-free motion candidates for VATD training/ablation. The learned dense proposal head is still the paper target for a fully end-to-end detector.

Detached temporal-saliency export:

```powershell
.\tools\start_vatd_temporal_saliency_tracklets_detached.ps1 `
  -GtCsv path\to\gt.csv `
  -FrameRoot path\to\frames `
  -MaxFramesPerSeq 5 `
  -OutDir artifacts\vatd_temporal_saliency\tracklets `
  -RunId vatd_temporal_saliency_test

.\tools\monitor_vatd_temporal_saliency_tracklets.ps1 `
  -RunId vatd_temporal_saliency_test `
  -OutputRoot artifacts\vatd_temporal_saliency_tracklets_runner

.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli export-tracklet-jsonl-predictions `
  --tracklet-jsonl artifacts\vatd_temporal_saliency\tracklets\proposal_tracklets.jsonl `
  --out-dir artifacts\vatd_temporal_saliency\predictions `
  --dataset-name vatd_independent_saliency `
  --score-field objectness `
  --nms-iou-threshold 0.5 `
  --nms-center-threshold 6

.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli evaluate-flat-tracklet-predictions `
  --gt-csv path\to\gt.csv `
  --prediction-csv artifacts\vatd_temporal_saliency\predictions\flat_xyxy_predictions.csv `
  --out-dir artifacts\vatd_temporal_saliency\eval `
  --iou-threshold 0.5 `
  --fp-limit 0 `
  --fp-limits 0 10 50 100 `
  --max-fppis 0 0.01 0.05 0.10

.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli sweep-flat-tracklet-prediction-nms `
  --tracklet-jsonl artifacts\vatd_temporal_saliency\tracklets\proposal_tracklets.jsonl `
  --gt-csv path\to\gt.csv `
  --out-dir artifacts\vatd_temporal_saliency\nms_sweep `
  --dataset-name vatd_independent_saliency `
  --score-field objectness `
  --nms-iou-thresholds none 0.3 0.5 `
  --nms-center-thresholds none 6 12 18 `
  --fp-limit 0 `
  --fp-limits 0 10 50 100 `
  --max-fppis 0 0.01 0.05 0.10

.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli compare-flat-prediction-eval-summaries `
  --summaries path\to\transvisdrone_eval\flat_prediction_eval_summary.json artifacts\vatd_temporal_saliency\nms_sweep\final_eval\flat_prediction_eval_summary.json `
  --method-names transvisdrone vatd_independent `
  --out-dir artifacts\vatd_temporal_saliency\comparison
```

The monitor reports PID state, `done/total` sequence progress, tracklet JSONL line count, last completed sequence, logs, and GPU signal, matching the repo long-run rules. The flat-prediction evaluator closes the independent branch loop by reporting the threshold sweep plus `best_under_budget`, so we can directly ask how much recall VATD gets at a fixed FP or FPPI budget. It also writes `flat_prediction_fp_budget_curve.csv` and `flat_prediction_fppi_budget_curve.csv` for paper plots/tables. The NMS sweep automates duplicate-suppression tuning before comparing against TransVisDrone at the same false-positive budget, then writes the selected output under `final_predictions/` plus a `final_eval/` summary. The comparison table treats the first method as baseline and adds delta columns. For paper tables, put TransVisDrone first and VATD second; `recall_verdict_vs_baseline=win` or `recall_win_vs_baseline=True` at the same FP/FPPI budget is the direct win signal. `flat_prediction_eval_comparison_summary.json` aggregates all outcomes in `verdict_summary`, aggregates fixed-budget-only outcomes in `fixed_budget_verdict_summary`, and exposes `paper_claim_rows`, which filters to fixed-budget rows (`best_under_budget`, `fp_limit`, `max_fppi`) suitable for paper claims. The same comparison run also writes `flat_prediction_paper_claim_wins.csv` and `.md`; those files are the shortest paper-facing list of fixed-budget wins over the baseline. For automation, `flat_prediction_paper_result_summary.json` contains the same fixed-budget result summary plus the best fixed-budget win row, if one exists, and `flat_prediction_claim_gate.json` exposes the same pass/fail gate in a smaller file for scripts. Its `claim_gate.status` is `pass` only when at least one fixed-budget recall win exists; otherwise it stays `insufficient_evidence`.
