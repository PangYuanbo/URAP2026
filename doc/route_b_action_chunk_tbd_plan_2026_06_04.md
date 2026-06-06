# Route B Plan: Action-Chunk Dynamics for Track-Before-Detect UAV Recognition

Date: 2026-06-04

## Goal

Build a route that can exceed the current NPS / YOLOMG / TransVisDrone baselines by using motion as the recognition signal for tiny UAVs. The first target is Route B: explicit track-before-detect with an action-chunk dynamics scorer. Route A, the later target, is an implicit spatiotemporal detector trained across datasets in an Octo-style unified-data regime.

The core correction is:

- Action chunks should not be the first-stage detector.
- Action chunks should model UAV dynamics and score whether a weak candidate tube is physically and visually consistent with a UAV.

## Route B System

```text
low-threshold detector proposals
  -> candidate tube construction
  -> action-chunk dynamics scoring
  -> evidence integration along the tube
  -> final per-frame UAV detections
```

### Stage 1: Over-detect

Run YOLOMG, TransVisDrone, QSTR YOLO-P2, or other available detectors at low thresholds. The purpose is recall, not precision. Keep weak proposals that would normally be filtered out.

Expected output per frame:

```text
frame_id, bbox_xyxy, objectness, detector_source, optional_roi_feature
```

### Stage 2: Build candidate tubes

Link nearby weak detections into short tubes with generous gates. This is not a final tracker; it is a hypothesis generator.

Tube fields:

```text
seq, tube_id, frame_id, bbox_xyxy, score, source, visibility
```

### Stage 3: Action-chunk dynamics model

For a tube, convert boxes into normalized action chunks:

```text
box_t = [cx/W, cy/H, w/W, h/H]
action_t = [delta_cx, delta_cy, delta_log_w, delta_log_h]
chunk = action_{t+1:t+H}
```

Condition:

```text
past N boxes
past N scores
visibility flags
optional YOLOMG ROI/FPN feature
optional camera/global-motion quality feature
```

Target:

```text
future H action vectors
```

The first MVP can be GRU/Transformer regression. The stronger version can use diffusion policy to estimate a likelihood over future action chunks.

Early sanity result from 2026-06-04:

- Constant velocity is a strong short-horizon physics baseline.
- Direct action prediction underperformed constant velocity, even after row-level coordinate normalization.
- A residual policy that predicts `future_actions - constant_velocity_actions` was much better than direct MLP/diffusion and nearly matched constant velocity on held-out oracle clips.
- The next Route B scorer should keep constant velocity as the explicit baseline and add a learned residual / likelihood term, rather than replacing constant velocity with a black-box policy.

### Stage 4: Score and integrate

A candidate tube is accepted when the following agree:

```text
detector evidence along tube
motion/dynamics likelihood
shape and scale stability
low background/artifact probability
```

The action-chunk model is used as a dynamics prior:

```text
score_dyn = - reconstruction_error_or_diffusion_nll
final_score = detector_score + temporal_evidence + score_dyn - inconsistency_penalty
```

Implemented scorer modes:

```text
learned_consistency = exp(-learned_error / scale)
cv_consistency      = exp(-constant_velocity_error / scale)
improvement         = sigmoid((constant_velocity_error - learned_error) / scale)
hybrid              = learned_consistency * improvement
```

For normalized action chunks, `scale` must be in normalized coordinates, not pixels. The default pixel-oriented scale is intentionally not sufficient for normalized multi-dataset scoring.

## How This Connects to YOLOMG

### B1: External learned tracker/scorer

Do not modify YOLOMG. Use YOLOMG detections to form tubes, then apply the action-chunk scorer.

This is the lowest-risk first implementation and gives clean ablations:

```text
YOLOMG
YOLOMG + IoU/Kalman tube scoring
YOLOMG + learned action-chunk scoring
YOLOMG + diffusion action-chunk scoring
```

### B2: Replace the motion branch with a trajectory-prior branch

YOLOMG currently takes:

```text
RGB image + pixel-level motion map
```

The action model can produce a causal prior heatmap:

```text
past boxes -> action chunk -> predicted current/future boxes -> Gaussian prior heatmap
```

Then YOLOMG becomes:

```text
RGB image + trajectory-prior heatmap -> detection
```

Important constraint: the prior heatmap must be causal. It can use past boxes, but not current-frame ground truth.

## Dataset Unification

The Octo-style lesson is to normalize many datasets into one shared sample format. For Route B, the shared sample is not a robot episode yet; it is a UAV tube episode.

Candidate sources:

- NPS / Dogfight annotations
- ARD100 / YOLOMG data
- AOT / Airborne Object Tracking
- Anti-UAV visible subsets
- Synthetic drone datasets when their motion labels are usable

Unified training row:

```text
dataset, seq, frame_id, width, height,
x1, y1, x2, y2,
class, visible, source
```

Unified action-chunk sample:

```text
dataset, seq, anchor_frame,
past_boxes[N,4], past_scores[N], past_visible[N],
future_actions[H,4], future_boxes[H,4],
positive_label
```

The same format supports:

- deterministic regression policy
- diffusion policy
- tube-likelihood scorer
- prior-heatmap generation for YOLOMG

Normalization rule:

- Multi-dataset action chunks must use a shared coordinate system.
- YOLO-format oracle exports should build action chunks with row-level `image_width/image_height` normalization.
- Pixel-space action chunks make NPS / AOT / ARD100 dynamics incomparable and caused learned policies to underperform simple constant velocity in sanity tests.

## Metrics

Use the same metrics as the existing baselines, plus tube-specific diagnostics.

Primary:

- NPS: AP@0.5, recall, precision
- AOT: HFAR/FAR, FPPI, EDR@300
- QSTR heldout: frame-level recall, false positives, hard-tiny recovery

Route B must show:

- higher recall than YOLOMG/TVD at matched or lower FPPI/FAR
- lower false positives than low-threshold detectors
- better missed-frame recovery than constant velocity / Kalman
- ablation proof that dynamics scoring is responsible

## AOT Part0 Official Baseline Snapshot

TransVisDrone pretrained AOT inference on fulltest part0 was run with `conf=0.2`, `img=1280`, 3 frames, half precision, and the IoU post-tracker enabled. The AOT pkl schema was validated as center `x,y,w,h` before evaluation.

Inference output:

```text
records: 3712 image records
detections: 5144
clips: 687-696
```

Official AOT metrics on part0:

| Variant | Threshold | FPPI | HFAR | AFDR range | AFDR area > 200 | AFDR area <= 200 | Detection encounters @300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TVD + IoU track | 0.200 | 0.28172 | 6699.0 | 0.88348 | 0.68358 | 0.05002 | 9/10 |
| TVD + IoU track | 0.350 | 0.08918 | 2142.0 | 0.82817 | 0.57783 | 0.01938 | 9/10 |
| TVD + IoU track | 0.551 | 0.01310 | 279.0 | 0.74484 | 0.45330 | 0.00844 | 7/10 |
| TVD + IoU track | 0.801 | 0.00008 | 3.0 | 0.37242 | 0.21237 | 0.00375 | 3/10 |
| TVD + temporal length >= 2 | 0.200 | 0.13907 | 1569.0 | 0.67183 | 0.47122 | 0.02938 | 6/10 |
| TVD + temporal length >= 3 | 0.200 | 0.09168 | 717.0 | 0.54794 | 0.37527 | 0.02157 | 6/10 |
| TVD + temporal length >= 4 | 0.200 | 0.07240 | 443.33 | 0.46123 | 0.31917 | 0.02340 | 4/9 |
| TVD + gap2 length >= 2 | 0.201 | 0.12622 | 1464.0 | 0.65560 | 0.46141 | 0.02845 | 6/10 |
| TVD + gap2 length >= 3 + CV score >= 0.75 | 0.201 | 0.06157 | 483.0 | 0.43658 | 0.28742 | 0.01250 | 4/10 |
| TVD + gap2 length >= 3 + CV score >= 0.85 | 0.202 | 0.04244 | 356.25 | 0.37896 | 0.24869 | 0.00557 | 2/8 |
| TVD + gap2 length >= 3 + CV score >= 0.93 | 0.202 | 0.01460 | 145.0 | 0.07616 | 0.05067 | 0.00000 | 1/6 |

Interpretation:

- Confidence thresholding alone shows the precision/recall wall: `0.35` keeps 9/10 encounters but still has very high FAR; `0.80` gives low FAR but collapses encounter detection.
- Simple temporal persistence is a useful Route B baseline because it reduces FAR, but it also drops AFDR sharply.
- Splitting reused `track_id`s by frame gaps is required before action scoring. Some official pkl track IDs reappear after large gaps, and those discontinuities produce invalid action chunks if not segmented.
- Constant-velocity normalized consistency is a useful physics baseline, but it is too brittle by itself: it reduces FAR from `1464` to `483` at CV score `0.75`, but drops encounter detection from `6/10` to `4/10` and AFDR range from `0.65560` to `0.43658`.
- The next model should not merely require longer tracks. It should score whether a persistent tracklet has UAV-like dynamics and visual evidence, so the target is to recover the `0.2-0.35` encounter rate with FAR closer to the `0.55-0.80` operating points.
- The learned action-chunk model should therefore be a residual / likelihood model over the constant-velocity baseline, not a hard constant-velocity gate.

## Ablations

Minimum ablation table:

```text
detector only
detector + constant velocity tube score
detector + GRU action chunk score
detector + diffusion action chunk score
detector + action prior heatmap into YOLOMG
```

Important stress tests:

- tiny box side <= 24 px
- fast host-camera motion
- target acceleration
- short occlusion / detector dropout
- hard negatives: birds, insects, static specks, alignment artifacts

## Route A Handoff

Route A becomes natural after Route B data exists. Replace the explicit tube scorer with a clip model:

```text
video clip tokens + candidate/query tokens -> boxes/classes/action chunks
```

Route A can still use the action-chunk target as an auxiliary loss:

```text
L = L_detection + lambda * L_action_chunk + mu * L_temporal_consistency
```

This gives the later "pretty" goal: one spatiotemporal detector trained on unified multi-dataset UAV episodes, while Route B supplies the data format, baselines, and proof that dynamics is the right signal.

## Immediate Implementation Tasks

1. Define bbox/action-chunk conversion utilities.
2. Export action-chunk samples from existing tracklet JSONL.
3. Train a small deterministic action model as the first baseline.
4. Score candidate tubes with reconstruction error.
5. Compare against constant velocity on frozen heldout sequences.
6. Add diffusion policy only after the deterministic model exposes where multimodality matters.

## Official Run Workflow

Use a run manifest for every real Route B comparison. The manifest freezes the multi-source inputs, the action-policy settings, the preflight result, and the exact detached start/monitor commands.

If the run roots and GT CSVs are not already known, scan likely output/dataset roots first:

```powershell
.\tools\start_route_b_input_scan_detached.ps1 `
  -ScanRoots @('artifacts', 'runs', 'D:\datasets', 'D:\URAP_datasets') `
  -Out 'artifacts\route_b_official\proposal_input_scan.json' `
  -MaxDepth 8 `
  -MaxFiles 20000
```

Monitor it with:

```powershell
.\tools\monitor_route_b_input_scan.ps1
```

Use only candidates with positive `best_gt_sequence_overlap` and nonzero `sampled_bbox_rows`.

If the scan only finds YOLO-format datasets, export oracle tracklets for action-dynamics pretraining only:

```powershell
python -m qstr_dronedet.cli export-yolo-oracle-tracklets `
  --list-files D:\URAP_datasets\YOLOMG_eval\NPS_test\val.txt `
  --out artifacts\route_b_official\oracle_tracklets_nps `
  --dataset-source nps_yolomg_oracle `
  --min-tracklet-rows 4

python -m qstr_dronedet.cli build-action-chunk-dataset `
  --tracklet-jsonl artifacts\route_b_official\oracle_tracklets_nps\oracle_tracklets.jsonl `
  --out artifacts\route_b_official\oracle_tracklets_nps\action_chunk_samples.jsonl `
  --past-len 8 `
  --future-len 8 `
  --positives-only
```

These oracle YOLO-label tracklets are valid for learning UAV motion priors and Octo-style multi-source action chunks. They are not detector proposals and must not be reported as Route B detector results against NPS / YOLOMG / TransVisDrone baselines.

For multi-source oracle action-chunk pretraining, use the detached wrapper so long exports have a PID, logs, and a monitorable summary:

```powershell
$params = @{
  ListFiles = @(
    'D:\URAP_datasets\YOLOMG_eval\NPS_test\val.txt',
    'D:\URAP_datasets\YOLOMG_eval\AOT_part0\val.txt'
  )
  SourceNames = @('nps_yolomg_oracle', 'aot_yolomg_oracle')
  SkipImagesPerSource = @(0, 980)
  OutDir = 'artifacts\route_b_official\oracle_action_chunks'
  OutputRoot = 'artifacts\route_b_oracle_action_chunks'
  MaxImagesPerSource = 200
  PastLen = 8
  FutureLen = 8
  OracleMinTrackletRows = 4
}
.\tools\start_route_b_oracle_action_chunks_detached.ps1 @params
```

For full export, omit `MaxImagesPerSource` instead of using `0`. `SkipImagesPerSource` is optional; it is useful for short sanity runs on datasets such as AOT where the first many frames may contain no labeled target.

Monitor with:

```powershell
.\tools\monitor_route_b_oracle_action_chunks.ps1
```

The monitor reports `RUNNING=true` only when the PID exists and the command line matches the generated worker script. Otherwise it reports `NOT RUNNING` with the last completed source, logs, merged action chunk path, and split path.

First seed the baseline CSV with source-backed values and explicit placeholders:

```powershell
python -m qstr_dronedet.cli write-route-b-official-baseline-seed `
  --out artifacts/route_b_official/baselines.csv
```

Fill any `needs_fill=yes` rows before strict reporting. For draft checks, use:

```powershell
python -m qstr_dronedet.cli validate-route-b-baselines `
  --baseline-csv artifacts/route_b_official/baselines.csv `
  --out artifacts/route_b_official/baseline_validation_draft.json `
  --allow-empty-metric
```

```powershell
python -m qstr_dronedet.cli write-route-b-proposal-run-manifest `
  --out-dir artifacts/route_b_official/nps_aot_tvd_v1 `
  --train-run-roots <NPS_RUN_ROOT> <AOT_RUN_ROOT> <YOLOMG_RUN_ROOT> `
  --train-gt-csvs <NPS_GT_CSV> <AOT_GT_CSV> <YOLOMG_GT_CSV> `
  --eval-run-roots <NPS_HELDOUT_ROOT> <AOT_HELDOUT_ROOT> <TVD_HELDOUT_ROOT> `
  --eval-gt-csvs <NPS_HELDOUT_GT> <AOT_HELDOUT_GT> <TVD_HELDOUT_GT> `
  --train-source-names nps aot yolomg `
  --eval-dataset-names nps aot transvisdrone `
  --baseline-csv artifacts/route_b_official/baselines.csv `
  --past-len 8 `
  --future-len 8 `
  --model-types mlp diffusion `
  --epochs 50 `
  --balance-by dataset_source bucket label
```

The command writes:

```text
route_b_proposal_run_manifest.json
proposal_preflight.json
start_route_b_proposal_benchmark.ps1
monitor_route_b_proposal_benchmark.ps1
preflight_route_b_proposal_benchmark.txt
```

Rules for official reporting:

- Do not start the detached benchmark if `proposal_preflight.json` is invalid.
- Keep the generated manifest with the result artifacts.
- Use `tools/monitor_route_b_proposal_benchmark.ps1` for status so reports include PID, done/total, last output timestamp, logs, and preflight validity.
- Compare against NPS / YOLOMG / TransVisDrone only through a filled and validated baseline CSV, so split and metric definitions are explicit.

## 2026-06-04 YOLOMG low-confidence smoke

Artifact root:

```text
artifacts/route_b_official/yolomg_lowconf_smoke
```

This smoke uses 48 ARD100 validation frames from `phantom109` and runs YOLOMG with `--conf-thres 0.001 --save-txt --save-conf`.
It is not an official benchmark; it is the first real detector-proposal wiring test.

Observed chain:

- YOLOMG prediction export: `48` images, `159` low-confidence prediction rows, `0` missing prediction txt files.
- GT export: `48` images, `48` label rows, `1` sequence.
- Route B proposal preflight: valid, `159` bbox rows, `48` GT frames, all diagnostic sequences matched to GT.
- Proposal tracklets: `18` total, `1` positive, `17` easy-background negatives.
- Residual action-policy scoring with `past_len=8`, `future_len=8`, row-normalized boxes: `2` scored tracklets, `16` unscored short tracklets, `30` action windows.

Important result:

- The oracle-pretrained `residual_mlp` action policy did not beat constant velocity on this real-proposal smoke.
- Constant-velocity consistency alone ranked a smooth background false positive above the true positive (`0.8507` vs `0.6351`).

Interpretation:

Motion smoothness is necessary but not sufficient. Route B should not use a dynamics score as the only judge. The production scorer needs a joint score that combines:

- detector evidence: confidence, frame density, matched track span, score persistence;
- dynamics evidence: learned/CV consistency and improvement over CV;
- anti-background evidence: objectness margin, temporal/background dominance, crop/temporal disagreement when available;
- tracklet geometry: box size, speed, frame gaps, and stability.

This smoke supports keeping CV as a strong baseline and using action-chunk models as one feature family inside the track-before-detect scorer, not as a standalone replacement for detection evidence.

## Joint scorer handoff

After action dynamics are attached, export a classifier dataset from the nested proposal tracklets:

```powershell
python -m qstr_dronedet.cli export-tracklet-jsonl-classifier-dataset `
  --tracklet-jsonl artifacts/route_b_official/yolomg_lowconf_smoke/proposal_tracklets_with_dynamics.jsonl `
  --out artifacts/route_b_official/yolomg_lowconf_smoke/joint_tracklet_classifier_dataset
```

This writes:

```text
tracklets.csv
tracklets.jsonl
summary.json
```

The CSV includes detector persistence, temporal/background evidence, geometry/gap features, and the attached action-dynamics fields:

```text
mean_action_dynamics_score
min_action_dynamics_score
mean_action_error_improvement_vs_cv
mean_action_learned_center_error
```

The YOLOMG smoke produced `18` classifier rows: `1` positive, `17` negatives, and `2` tracklets with action-dynamics windows.
A same-smoke train/eval sanity reached `tp=1`, `fp=0`, `fn=0`, `tn=17`, but this is only a wiring check because train and eval are identical.

Official Route B scoring should therefore be:

```text
low-threshold detector proposals
-> proposal tracklets
-> action-dynamics attachment
-> export-tracklet-jsonl-classifier-dataset
-> train/eval joint tracklet classifier on held-out sequences/datasets
-> compare against NPS / YOLOMG / TransVisDrone baselines
```

For paper claims, the joint classifier must be trained on source-balanced multi-dataset tracklets and evaluated on held-out sequences. Same-clip smoke results are not reportable.

## 2026-06-04 held-out sequence smoke

Artifact root:

```text
artifacts/route_b_official/yolomg_lowconf_heldout_smoke
```

Setup:

- Train smoke sequence: ARD100 `phantom109`, first 48 validation frames.
- Held-out eval smoke sequence: ARD100 `phantom28`, first 48 validation frames.
- YOLOMG low-confidence setting: `--conf-thres 0.001 --save-txt --save-conf`, image size `1280`.
- Action-dynamics scorer: oracle-pretrained residual MLP, row-normalized boxes, `past_len=8`, `future_len=8`, improvement mode.

Input validation:

- Strict proposal preflight passed.
- Train route root: 48 images, 159 prediction rows, 48 GT labels.
- Eval route root: 48 images, 98 prediction rows, 48 GT labels.

Proposal tracklets:

```text
train phantom109: 18 tracklets, 1 positive, 17 negatives
eval  phantom28 : 10 tracklets, 2 positives, 8 negatives
```

Action-dynamics coverage:

```text
train: 2 scored tracklets, 30 action windows
eval : 1 scored tracklet, 33 action windows
```

Joint classifier held-out result:

Train on `phantom109`, evaluate on `phantom28`.

```text
threshold 0.5: tp=1, fp=0, fn=1, tn=8, precision=1.0, recall=0.5, f1=0.667
threshold 0.3: tp=2, fp=7, fn=0, tn=1, precision=0.222, recall=1.0, f1=0.364
```

The threshold sweep selected `0.5` by F1. The missed positive is a short weak `hard_tiny_positive` tracklet with classifier probability about `0.33`, while several weak background tracklets sit around `0.35-0.44`.

Interpretation:

Lowering the threshold is not the fix; it recovers the hard positive but also promotes many weak background false positives. The next Route B training set must include many more short weak positives and matched weak-background negatives across sequences/datasets. This is the exact place where the Octo-style multi-dataset mixture matters: source-balanced NPS/AOT/ARD100/TransVisDrone proposal tracklets should train the joint scorer, with held-out sequences used for threshold calibration.

New tooling added for this:

```powershell
python -m qstr_dronedet.cli eval-tracklet-classifier-thresholds `
  --csv <heldout_tracklets.csv> `
  --weights <joint_tracklet_classifier.pt> `
  --out-dir <threshold_sweep_dir> `
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

This writes:

```text
tracklet_classifier_threshold_sweep.csv
tracklet_classifier_threshold_summary.json
```

## Octo-style mixture hook for Route B

The joint tracklet classifier now preserves dataset metadata and supports inverse-frequency sample balancing.

When exporting classifier CSVs from per-dataset proposal tracklets, always tag the source:

```powershell
python -m qstr_dronedet.cli export-tracklet-jsonl-classifier-dataset `
  --tracklet-jsonl <tracklets_with_action_dynamics.jsonl> `
  --out <classifier_dataset_dir> `
  --dataset-source ard100
```

Use source/bucket/label balancing for multi-dataset training:

```powershell
python -m qstr_dronedet.cli train-tracklet-classifier `
  --csv <mixed_train_tracklets.csv> `
  --out <joint_tracklet_classifier.pt> `
  --epochs 50 `
  --hidden 64 `
  --balance-by dataset_source bucket label
```

This is the Route B analogue of Octo-style mixture training: each source keeps its identity, and rare groups such as short hard positives are not drowned by abundant easy background from a larger dataset.

Held-out smoke check:

- Re-exported `phantom109`/`phantom28` classifier CSVs with `dataset_source=ard100`.
- Trained with `--balance-by dataset_source bucket label`.
- Checkpoint balance groups:

```text
ard100|positive|1: 1
ard100|easy_background|0: 17
```

- Held-out `phantom28` best F1 remained `0.667` at threshold `0.5`.

Interpretation:

This single-source smoke is too small for balancing to improve performance, but it verifies the exact machinery needed for multi-source training. The next official-scale run should merge classifier CSVs from ARD100, NPS, AOT, and TransVisDrone-derived proposals, then train with `--balance-by dataset_source bucket label`.

## Mixed classifier dataset manifest

Per-source classifier CSVs can now be merged into one Octo-style mixture file:

```powershell
python -m qstr_dronedet.cli merge-tracklet-classifier-datasets `
  --inputs `
    <ard100_tracklets.csv> `
    <nps_tracklets.csv> `
    <aot_tracklets.csv> `
    <transvisdrone_tracklets.csv> `
  --out artifacts/route_b_official/mixed_classifier_dataset/tracklets.csv `
  --source-names ard100 nps aot transvisdrone `
  --manifest-out artifacts/route_b_official/mixed_classifier_dataset/manifest.json
```

The manifest records:

```text
rows
positives / negatives
bucket_counts
dataset_source_counts
per-input rows / positives / negatives / buckets / sources
```

Smoke verification on ARD100 train/eval slices:

```text
merged rows: 28
positives: 3
negatives: 25
sources:
  ard100_train_smoke: 18
  ard100_eval_smoke : 10
buckets:
  easy_background: 25
  positive: 2
  hard_tiny_positive: 1
```

Balanced training on that merged smoke wrote a checkpoint with these groups:

```text
ard100_train_smoke|positive|1: 1
ard100_train_smoke|easy_background|0: 17
ard100_eval_smoke|positive|1: 1
ard100_eval_smoke|hard_tiny_positive|1: 1
ard100_eval_smoke|easy_background|0: 8
```

For official runs, this manifest is the mixture audit. A result should not be compared against NPS / YOLOMG / TransVisDrone baselines unless the mixed classifier dataset manifest and checkpoint balance summary are preserved with the run artifacts.

## Joint classifier mixture benchmark runner

The official Route B joint-scorer entrypoint is:

```powershell
python -m qstr_dronedet.cli validate-tracklet-classifier-mixture-inputs `
  --train-csvs `
    <ard100_train_tracklets.csv> `
    <nps_train_tracklets.csv> `
    <aot_train_tracklets.csv> `
    <transvisdrone_train_tracklets.csv> `
  --eval-csvs `
    <ard100_heldout_tracklets.csv> `
    <nps_heldout_tracklets.csv> `
    <aot_heldout_tracklets.csv> `
    <transvisdrone_heldout_tracklets.csv> `
  --out artifacts/route_b_official/joint_classifier_benchmark/tracklet_classifier_mixture_preflight.json `
  --train-source-names ard100 nps aot transvisdrone `
  --eval-dataset-names ard100 nps aot transvisdrone
```

This preflight checks classifier schema, per-source rows, positives/negatives, missing `dataset_source`, non-finite features, duplicate eval names, and train/eval sequence overlap. Official benchmark claims should preserve this JSON; a failed preflight means the run is an engineering artifact, not a paper-comparable result.

The benchmark runner performs the same preflight by default before training:

```powershell
python -m qstr_dronedet.cli run-tracklet-classifier-mixture-benchmark `
  --train-csvs `
    <ard100_train_tracklets.csv> `
    <nps_train_tracklets.csv> `
    <aot_train_tracklets.csv> `
    <transvisdrone_train_tracklets.csv> `
  --eval-csvs `
    <ard100_heldout_tracklets.csv> `
    <nps_heldout_tracklets.csv> `
    <aot_heldout_tracklets.csv> `
    <transvisdrone_heldout_tracklets.csv> `
  --out-dir artifacts/route_b_official/joint_classifier_benchmark `
  --train-source-names ard100 nps aot transvisdrone `
  --eval-dataset-names ard100 nps aot transvisdrone `
  --epochs 50 `
  --hidden 64 `
  --balance-by dataset_source bucket label `
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

For any run expected to take more than a few minutes, use the detached wrapper instead of running the CLI directly:

```powershell
.\tools\start_route_b_tracklet_classifier_mixture_detached.ps1 `
  -TrainCsvs `
    <ard100_train_tracklets.csv> `
    <nps_train_tracklets.csv> `
    <aot_train_tracklets.csv> `
    <transvisdrone_train_tracklets.csv> `
  -EvalCsvs `
    <ard100_heldout_tracklets.csv> `
    <nps_heldout_tracklets.csv> `
    <aot_heldout_tracklets.csv> `
    <transvisdrone_heldout_tracklets.csv> `
  -OutDir artifacts\route_b_official\joint_classifier_benchmark `
  -OutputRoot artifacts\route_b_official\joint_classifier_benchmark_runner `
  -TrainSourceNames ard100 nps aot transvisdrone `
  -EvalDatasetNames ard100 nps aot transvisdrone `
  -Epochs 50 `
  -Hidden 64 `
  -BalanceBy dataset_source bucket label `
  -Thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 `
  -BaselineCsv artifacts\route_b_official\tracklet_level_baselines.csv `
  -BaselineMetric tracklet_best_f1 `
  -RunId official_joint_classifier
```

Status must be checked through the monitor, not through an interactive shell session:

```powershell
.\tools\monitor_route_b_tracklet_classifier_mixture.ps1 `
  -OutputRoot artifacts\route_b_official\joint_classifier_benchmark_runner `
  -RunId official_joint_classifier
```

The monitor reports PID/process command, preflight counts, `done/total`, last output timestamp, last completed unit, stdout/stderr logs, and GPU signal when `nvidia-smi` is available.

The classifier mixture runner also writes:

```text
tracklet_classifier_mixture_route_b_results.csv
baseline_report/route_b_tracklet_classifier_baseline_report.md
baseline_report/comparison/route_b_baseline_comparison_summary.json
```

This comparison is deliberately tracklet-level. Use `tracklet_best_f1` only against a baseline CSV computed from the same proposal-tracklet evaluation protocol. Do not report this as paper-level NPS mAP, AOT HFAR/EDR, or TransVisDrone frame/video detection performance. Paper-comparable claims still require applying the learned tracklet scorer back to frame predictions and evaluating with the dataset's official detection metrics.

The next gate applies the trained joint scorer back to held-out frame-level inference outputs without mutating the source run:

```powershell
python -m qstr_dronedet.cli run-tracklet-classifier-frame-benchmark `
  --run-roots `
    <nps_heldout_infer_root> `
    <aot_heldout_infer_root> `
    <transvisdrone_heldout_infer_root> `
  --gt-csvs `
    <nps_heldout_gt.csv> `
    <aot_heldout_gt.csv> `
    <transvisdrone_heldout_gt.csv> `
  --weights artifacts/route_b_official/joint_classifier_benchmark/train/joint_tracklet_classifier.pt `
  --out-dir artifacts/route_b_official/joint_classifier_frame_benchmark `
  --dataset-names nps aot transvisdrone `
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 `
  --iou-threshold 0.3 `
  --baseline-csv artifacts/route_b_official/frame_level_baselines.csv `
  --baseline-metric frame_best_f1
```

This writes raw-vs-filtered frame-level precision/recall/F1 in `tracklet_classifier_frame_benchmark.csv`, keeps copied `predictions_raw.jsonl` / filtered `predictions.jsonl` under per-threshold output directories, and records the best filtered threshold per dataset in `tracklet_classifier_frame_benchmark_summary.json`. It is the correct engineering bridge from Route B tracklet learning to dataset-level detection evaluation. The final paper claim still needs the official evaluator for each target dataset/protocol.

When a frame-level baseline CSV is provided, the same command also writes:

```text
tracklet_classifier_frame_route_b_results.csv
baseline_report/route_b_tracklet_classifier_frame_baseline_report.md
baseline_report/comparison/route_b_baseline_comparison_summary.json
```

This report is a frame-level proxy using the GT-CSV/IoU protocol in this repo. It is suitable for engineering threshold selection and for deciding whether a run is worth pushing through official NPS/AOT/TransVisDrone evaluation, but the final paper table must use the official evaluator outputs.

For full held-out runs, use the detached wrapper:

```powershell
.\tools\start_route_b_tracklet_classifier_frame_benchmark_detached.ps1 `
  -RunRoots <nps_heldout_infer_root>,<aot_heldout_infer_root>,<transvisdrone_heldout_infer_root> `
  -GtCsvs <nps_heldout_gt.csv>,<aot_heldout_gt.csv>,<transvisdrone_heldout_gt.csv> `
  -Weights artifacts\route_b_official\joint_classifier_benchmark\train\joint_tracklet_classifier.pt `
  -OutDir artifacts\route_b_official\joint_classifier_frame_benchmark `
  -OutputRoot artifacts\route_b_official\joint_classifier_frame_benchmark_runner `
  -DatasetNames nps,aot,transvisdrone `
  -Thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 `
  -BaselineCsv artifacts\route_b_official\frame_level_baselines.csv `
  -BaselineMetric frame_best_f1 `
  -RunId official_frame_benchmark
```

The detached wrapper runs `validate-tracklet-classifier-frame-benchmark-inputs` before launching the long benchmark unless `-SkipPreflight` is set. The preflight checks prediction/diagnostics JSONL pairs, GT CSV parsing, sequence overlap, weights, thresholds, and optional baseline CSV validity. For a manual check:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli validate-tracklet-classifier-frame-benchmark-inputs `
  --run-roots <nps_heldout_infer_root> <aot_heldout_infer_root> <transvisdrone_heldout_infer_root> `
  --gt-csvs <nps_heldout_gt.csv> <aot_heldout_gt.csv> <transvisdrone_heldout_gt.csv> `
  --weights artifacts\route_b_official\joint_classifier_benchmark\train\joint_tracklet_classifier.pt `
  --out artifacts\route_b_official\joint_classifier_frame_benchmark_runner\official_frame_benchmark_preflight.json `
  --dataset-names nps aot transvisdrone `
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 `
  --baseline-csv artifacts\route_b_official\frame_level_baselines.csv `
  --baseline-metric frame_best_f1
```

Monitor it with:

```powershell
.\tools\monitor_route_b_tracklet_classifier_frame_benchmark.ps1 `
  -OutputRoot artifacts\route_b_official\joint_classifier_frame_benchmark_runner `
  -RunId official_frame_benchmark
```

The monitor reports PID/process command, preflight validity/counters/errors, per-dataset best threshold/F1/TP/FP/FN, `done/total`, last output timestamp, log paths, baseline comparison wins, and GPU signal when available.

After the frame benchmark finishes, freeze the best-threshold outputs into an official-evaluator bundle:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli build-tracklet-classifier-official-eval-bundle `
  --frame-summary artifacts\route_b_official\joint_classifier_frame_benchmark\tracklet_classifier_frame_benchmark_summary.json `
  --out-dir artifacts\route_b_official\joint_classifier_official_eval_bundle `
  --preflight-json artifacts\route_b_official\joint_classifier_frame_benchmark_runner\official_frame_benchmark_preflight.json `
  --baseline-comparison-json artifacts\route_b_official\joint_classifier_frame_benchmark\baseline_report\comparison\route_b_baseline_comparison_summary.json `
  --require-baseline-comparison
```

This writes:

```text
official_eval_bundle_manifest.json
official_eval_prediction_index.csv
best_filtered/<dataset>/<seq>/predictions.jsonl
best_filtered/<dataset>/<seq>/diagnostics.jsonl
```

This bundle is the handoff to NPS/AOT/TransVisDrone official evaluators. It records the selected threshold, copied filtered predictions, GT CSV, preflight status, proxy frame metrics, and baseline-comparison metadata. It is not itself a paper-level result; paper claims require running the corresponding dataset official evaluator on `best_filtered/.../predictions.jsonl` and preserving those evaluator outputs.

Convert the bundle into evaluator-facing prediction files:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli export-tracklet-classifier-official-predictions `
  --bundle-manifest artifacts\route_b_official\joint_classifier_official_eval_bundle\official_eval_bundle_manifest.json `
  --out-dir artifacts\route_b_official\joint_classifier_official_eval_bundle\eval_inputs `
  --formats flat_csv yolo_txt `
  --image-width 1280 `
  --image-height 720
```

This writes:

```text
eval_inputs/flat_xyxy_predictions.csv
eval_inputs/image_index.csv
eval_inputs/yolo_txt/<dataset>/labels/<image_stem>.txt
eval_inputs/official_prediction_export_summary.json
```

Use `flat_xyxy_predictions.csv` for custom official adapters such as AOT-style FAR/FPPI/EDR scripts, and use the YOLO txt tree when the evaluator expects YOLO-style `class cx cy w h conf` prediction labels. If prediction rows already carry `image_width/image_height`, those are used; otherwise pass the dataset image size explicitly.

For AOT official metrics, convert the flat CSV into the `aotpredictions/*.pkl` format expected by `papers/TransVisDrone/evaluate_aot.py`:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli export-tracklet-classifier-aot-predictions `
  --flat-csv artifacts\route_b_official\joint_classifier_official_eval_bundle\eval_inputs\flat_xyxy_predictions.csv `
  --out-dir artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_input `
  --image-name-mode aot_clip_frame `
  --part-name predictions_split_0.pkl
```

Important: the AOT pkl schema uses center-format boxes: `x` and `y` are box center coordinates, and `w`/`h` are width/height in pixels. The adapter converts Route B flat `x1,y1,x2,y2` rows into this center-xywh format.

Then run the official AOT evaluator with the matching AOT dataset root. This can take long enough to require the detached runner:

```powershell
.\tools\start_route_b_aot_official_eval_detached.ps1 `
  -ResultsFolder artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_input\aotpredictions `
  -EvaluationFolder artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_official_eval `
  -DatasetPath D:\URAP_datasets\AOT\part1 `
  -DetectionThreshold 0.2 `
  -OutputRoot artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_runner `
  -RunId official_aot_eval
```

The detached runner validates the `aotpredictions/*.pkl` files before launch by default. The preflight checks that prediction parts exist, each pkl root is a list, image names match the AOT evaluator convention `Clip_<clip_id>_<frame>.png`, clip IDs are known in `papers\TransVisDrone\aot_flight_ids\aot_clip_id_to_flight_id.pkl`, and detection fields are finite. To run that gate manually:

```powershell
.\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe -m qstr_dronedet.cli validate-tracklet-classifier-aot-eval-inputs `
  --results-folder artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_input\aotpredictions `
  --out artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_runner\aot_eval_preflight.json `
  --clip-id-to-flight-id-path papers\TransVisDrone\aot_flight_ids\aot_clip_id_to_flight_id.pkl
```

Monitor it with:

```powershell
.\tools\monitor_route_b_aot_official_eval.ps1 `
  -OutputRoot artifacts\route_b_official\joint_classifier_official_eval_bundle\aot_eval_runner `
  -RunId official_aot_eval
```

The adapter assumes the exported `image_stem` matches the AOT prepared frame naming, usually `Clip_<clip_id>_<frame_id>.png`. If Route B inference used a different image stem, fix the export template before launching official eval; otherwise the preflight will fail before any detached evaluator process starts.

It writes:

```text
tracklet_classifier_mixture_preflight.json
train/mixed_tracklets.csv
train/mixed_tracklets.manifest.json
train/joint_tracklet_classifier.pt
eval/<dataset>/tracklet_classifier_threshold_sweep.csv
eval/<dataset>/tracklet_classifier_threshold_summary.json
tracklet_classifier_mixture_benchmark_summary.json
```

The top-level summary includes:

```text
mixed_train
checkpoint_balance
best_by_dataset
per-dataset threshold sweep paths
```

Smoke verification on ARD100 train/eval slices produced the expected structure and held-out `ard100_eval_smoke` best F1 `0.667`. This remains a wiring check; official claims require filled multi-dataset train/eval CSVs and baseline comparison.
