# Per-Video +/-3s Window Accuracy Curves

Last updated: 2026-05-22

## Purpose

Use one common post-processing path to see where each video is strong or weak.
For every center frame, the tool scores detections inside the frame window:

```text
[center_frame - round(fps * 3s), center_frame + round(fps * 3s)]
```

Predictions are matched to ground truth on the same frame with greedy IoU
matching. Counts are then summed over the window.

- `accuracy = TP / (TP + FP + FN)`
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = harmonic mean(precision, recall)`

This is detection-window accuracy, not classification accuracy. It is intended
as a timeline diagnostic: low windows identify video segments that deserve
manual inspection.

## Pulled Paper Repositories

Refresh public repos:

```bash
python3 tools/pull_paper_repos.py
```

The AICrowd winner is a special case: unauthenticated `git clone` prompts for
credentials, but the public GitLab API exposes the `submission-v022` source
tree. `tools/pull_paper_repos.py` downloads that code snapshot and skips large
`.pth` model weights, recording the result in `.urap_snapshot.json`.

Current local checkouts under `papers/`:

| Method / paper line | Local path | Remote | HEAD |
| --- | --- | --- | --- |
| YOLOMG | `papers/YOLOMG` | `https://github.com/Irisky123/YOLOMG.git` | `090a74c` |
| TransVisDrone | `papers/TransVisDrone` | `https://github.com/tusharsangam/TransVisDrone.git` | `8b3c760` |
| ESOD | `papers/ESOD` | `https://github.com/alibaba/esod.git` | `bde3571` |
| EDTC | `papers/EDTC` | `https://github.com/xuefeng-zhu5/EDTC.git` | `61d1932` |
| Li-TETC / NPS baseline | `papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking` | `https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking.git` | `317e85a` |
| Dogfight / NPS annotations | `datasets/Drone-Detection` | `https://github.com/mwaseema/Drone-Detection.git` | `14cf376` |
| AICrowd Winner v022 | `papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022` | `https://gitlab.aicrowd.com/dmytro_poplavskiy/airborne-detection-starter-kit.git` | `1fbc2276` API snapshot |

The AICrowd snapshot is source-code only. The skipped winner model weights and
local AOT/NPS prediction `result.json` outputs are still required before real
AICrowd per-video curves can be generated. To inventory the missing Git LFS
weights and exact SHA256/size requirements:

```bash
python3 tools/inventory_aicrowd_lfs_weights.py
```

The current inventory is written to
`runs/window_accuracy/aicrowd_lfs_weight_inventory.md`.

To download and sha256-verify those LFS weights when a GitLab token is
available:

```bash
AICROWD_GITLAB_TOKEN=<token> python3 tools/download_aicrowd_lfs_weights.py
```

Without a token, the script writes an `auth_required` report to
`runs/window_accuracy/aicrowd_lfs_weight_download_report.json`.

External source inventory for missing real inputs:

```bash
python3 tools/inventory_external_window_accuracy_sources.py
```

This writes `runs/window_accuracy/papers/external_source_inventory.md`. It
checks NPS/Purdue download sizes, local Dogfight annotation counts, AOT
`ImageSets` metadata on public S3, EDTC public Google Drive file names/IDs,
EDTC local validation/model readiness, and the YOLOMG ARD100 BaiduYun source.
It is intentionally an inventory command and does not download NPS videos, AOT
frames, or AntiUAV zip files.

To download the public NPS video zip once and extract only the clips needed for
NPS val:

```bash
python3 tools/download_nps_videos.py \
  --clips 37-40 \
  --workers 8 \
  --zip datasets/NPS/raw/Videos.zip \
  --out-dir datasets/NPS/raw/Videos \
  --json runs/window_accuracy/papers/nps_video_download.json
```

Then prepare local TransVisDrone-compatible NPS val frames and labels using
JPEG frames to keep disk use bounded:

```bash
python3 tools/prepare_transvisdrone_nps.py \
  --videos-dir datasets/NPS/raw/Videos \
  --annos-dir datasets/Drone-Detection/annotations/NPS-Drones-Dataset \
  --out-root datasets/TransVisDrone/NPS \
  --only-split val \
  --only-clips 37-40 \
  --image-ext jpg \
  --jpg-quality 85
```

EDTC validation inputs are now local. The downloaded artifacts are:

```text
datasets/AntiUAV600/raw/validation.zip
datasets/AntiUAV600/validation
papers/EDTC/pretrained/UAVTrackEH.pth.tar
papers/EDTC/yolov5/weights/edtc_yolo_best.pt
data_templates/edtc_antiuav.yaml
```

To validate EDTC wiring locally without running the full 56,301-frame
validation set, run one short CPU smoke sequence:

```bash
.venv/edtc-window/bin/python tools/run_edtc_tracker_window_accuracy.py \
  --python .venv/edtc-window/bin/python \
  --dataset-root datasets/AntiUAV600/validation \
  --tracker-model papers/EDTC/pretrained/UAVTrackEH.pth.tar \
  --yolo-weights papers/EDTC/yolov5/weights/edtc_yolo_best.pt \
  --yolo-data data_templates/edtc_antiuav.yaml \
  --out runs/window_accuracy/papers/edtc_antiuav600_smoke_sequence23 \
  --threads 0 \
  --num-gpus 1 \
  --device cpu \
  --sequence 23
```

The full EDTC validation run should use the detached Windows/GPU wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_edtc_tracker_window_accuracy_detached.ps1 `
  -DatasetRoot datasets\AntiUAV600\validation `
  -TrackerModel papers\EDTC\pretrained\UAVTrackEH.pth.tar `
  -YoloWeights papers\EDTC\yolov5\weights\edtc_yolo_best.pt `
  -YoloData data_templates\edtc_antiuav.yaml `
  -Out runs\window_accuracy\papers\edtc_antiuav600 `
  -Fps 30 `
  -WindowSeconds 3 `
  -RunId edtc_antiuav600
```

TransVisDrone NPS pretrained weights are now local at the official README path:

```text
papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt
```

A small CPU smoke with the real checkpoint and prepared NPS frames writes:

```text
runs/window_accuracy/papers/transvisdrone_nps_val_smoke/plots/index.html
```

The full NPS-val generation command is still the detached runner in
`runs/window_accuracy/papers/gap_report.md`; it should run on the Windows/GPU
machine because local CPU inference is about 11 seconds per 1280px frame.

## Tool Entry Point

Single run:

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt <ground-truth path> \
  --gt-format <csv|jsonl|yolo-dir|aot-json|aot-gt-json|xywh-file|antiuav-json|li-tetc-txt|tvd-pkl-gt> \
  --pred <prediction path> \
  --pred-format <csv|jsonl|yolo-dir|aot-json|aot-gt-json|xywh-file|antiuav-json|li-tetc-txt|tvd-pkl-pred> \
  --fps <video fps> \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --segment-threshold 0.5 \
  --frame-manifest <optional image dir / frame csv / dataset root> \
  --frame-manifest-format <image-dir|csv|yolo-dir|aot-json|aot-gt-json|xywh-file|antiuav-json|li-tetc-txt|tvd-pkl-gt|tvd-pkl-pred> \
  --out runs/window_accuracy/<method>/<split-or-video>
```

YOLO-style paper eval plus curves in one command for YOLOMG, TransVisDrone, or
ESOD:

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method <yolomg|transvisdrone|esod> \
  --repo papers/<paper-repo> \
  --data <dataset.yaml> \
  --weights <weights.pt> \
  --gt <ground-truth label path> \
  --gt-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --out runs/window_accuracy/<method>/<run_name>
```

The wrapper calls the paper repo's `val.py` or `test.py` with
`--save-txt --save-conf`, then scores the saved `labels/` directory. Use
`--skip-eval --pred-labels-dir <labels>` when predictions already exist.
Pass `--frame-manifest <image-or-frame-source>` when the dataset has empty
frames that are not represented by label files; otherwise only the span covered
by GT/prediction frame ids is guaranteed.
For long Windows runs, start the same wrapper through the detached launcher:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_yolo_eval_window_accuracy_detached.ps1 `
  -Method yolomg `
  -Data D:\URAP_datasets\ARD100_YOLOMG\ard100.yaml `
  -Weights papers\YOLOMG\runs\train\<run>\weights\best.pt `
  -Gt D:\URAP_datasets\ARD100_YOLOMG\labels\test `
  -Out runs\window_accuracy\yolomg_ard100 `
  -Fps 30

powershell -ExecutionPolicy Bypass -File tools\monitor_yolo_eval_window_accuracy.ps1 `
  -Out runs\window_accuracy\yolomg_ard100
```

Batch run across paper repos:

```bash
python3 tools/run_paper_window_accuracy_batch.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --skip-missing
```

Copy the example manifest to a run-specific JSON, replace `<run_name>` and
placeholder paths with the actual repo outputs, then run without
`--skip-missing` for a strict all-paper pass.

End-to-end smoke run that creates tiny fixture inputs for every supported paper
line and renders real SVG/HTML curves:

```bash
python3 tools/build_paper_window_accuracy_smoke.py
```

Smoke outputs are written under `runs/window_accuracy/smoke/`, with a top-level
`index.html` linking each per-paper `plots/index.html`.

Readiness audit for real paper runs:

```bash
python3 tools/audit_paper_window_accuracy_readiness.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --json runs/window_accuracy/papers/readiness_audit.json \
  --markdown runs/window_accuracy/papers/readiness_audit.md
```

The audit reports each run as `complete_curves`, `ready_to_run`,
`missing_inputs`, or `auth_required`, and prints the exact batch command for
ready runs.

Goal-level audit for the whole paper-repo curve objective:

```bash
python3 tools/audit_paper_window_accuracy_goal.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --json runs/window_accuracy/papers/goal_audit.json \
  --markdown runs/window_accuracy/papers/goal_audit.md
```

This checks the objective gates directly: required paper methods are present,
formats are supported, complete runs have all curve artifacts, dashboard and
batch summary exist, every incomplete run has generation commands, and whether
all manifest runs are actually complete.

Gap report for missing real inputs:

```bash
python3 tools/write_paper_window_accuracy_gap_report.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --json runs/window_accuracy/papers/gap_report.json \
  --markdown runs/window_accuracy/papers/gap_report.md
```

The gap report keeps the missing full-paper work explicit: expected GT/pred
paths, input format, candidate paths found under `datasets/`, `runs/`, and
`papers/`, next commands, method-specific next steps, and concrete generation
commands for known missing paper runs.

One-command pipeline for the normal workflow:

```bash
python3 tools/run_paper_window_accuracy_pipeline.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --smoke
```

The pipeline refreshes public paper repos, audits readiness, runs any
`ready_to_run` curve jobs, audits again, and writes
`runs/window_accuracy/papers/pipeline_report.{json,md}`.

Auto-discover real outputs first when run names are not known:

```bash
python3 tools/discover_paper_window_accuracy_runs.py
python3 tools/run_paper_window_accuracy_pipeline.py \
  --manifest runs/window_accuracy/discovered_manifest.json \
  --smoke
```

The discovery pass scans common paper output locations such as
`papers/*/runs/**/labels`, AOT `result.json` folders, EDTC tracker results, and
Li-TETC `Experiment_Results/Final/txt`. It also scans TransVisDrone
`predictionsgt_split_*.pkl` files, which embed both labels and predictions from
`--save-json-gt` runs. When the same TransVisDrone experiment directory also
contains sibling `best_predictions.pkl` or `last_predictions.pkl` files, the
discovery step pairs those prediction-only pkls with the matching embedded GT
pkl so lower-res, COCO, speed-test, and best-augment variants get separate
curves.

Outputs:

- `per_frame_window_metrics.csv`
- `worst_windows.csv`
- `low_accuracy_segments.csv`
- `summary.json`
- `plots/<video>_window_metrics.svg`
- `plots/index.html`
- batch mode also writes `batch_summary.json`
- batch mode also writes a top-level `index.html` linking all paper/run curves
- batch mode also writes `dashboard.html`, a cross-run overview with the
  continuous low-accuracy segments, lowest-accuracy +/-3s windows, per-video
  scorecards, missing/skipped runs, and a curve gallery

Supported inputs:

- `csv`: columns such as `video,frame_id,x1,y1,x2,y2,score,label`
- `jsonl`: one JSON object per box, with `video`, `frame_id`, and `bbox`
- `yolo-dir`: YOLO txt directory, one file per frame, lines
  `class cx cy w h [score]`
- `aot-json`: AOT/Winner `result.json` file or a directory containing
  per-flight/per-clip `result.json` files
- `aot-gt-json`: official AOT `groundtruth.json` file, a directory containing
  `groundtruth.json`, or a dataset root containing `ImageSets/groundtruth.json`
- `xywh-file`: tracker result txt files with one `x y w h [score]` row per frame
- `antiuav-json`: AntiUAV dataset root containing `list.txt` and `*/IR_label.json`
- `li-tetc-txt`: Li-TETC/NPS text with `time_layer: <frame> detections: (...)`
- `tvd-pkl-gt` / `tvd-pkl-pred`: TransVisDrone `predictionsgt_split_*.pkl`
  files saved by `--save-json-gt`; use the same pkl path for `--gt` and
  `--pred`, or use the `predictionsgt` pkl as GT with a sibling
  prediction-only `best_predictions.pkl` / `last_predictions.pkl` as pred

For true every-frame curves, pass `frame_manifest` in the manifest or
`--frame-manifest` on the CLI. Without a frame manifest, the scorer fills only
the frame span between the first and last frame that appears in GT or
predictions. With a frame manifest, center frames come from the explicit source,
so empty-object frames at the start/end of a video are still represented in the
curve. `image-dir` scans frame image names; if a filename has no numeric frame
id, such as AOT hash images, frames are assigned by sorted order within each
video/flight folder and that lookup is also used to map AOT `result.json`
`img_name` values. `antiuav-json` uses `IR_label.json` lengths including
`exist=0` frames, `xywh-file` uses tracker-result line numbers, `li-tetc-txt`
uses all `time_layer` rows, and `tvd-pkl-*` uses pickle image ids.

For YOLO txt, boxes stay normalized unless `--img-width` and `--img-height` are
provided. IoU is unchanged when both GT and predictions use the same normalized
coordinate system. Provide image size when mixing normalized YOLO GT with
absolute-pixel predictions such as AOT/Winner JSON.

## Per-Repo Usage

### YOLOMG

Generate predictions and curves together:

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method yolomg \
  --repo papers/YOLOMG \
  --data D:/URAP_datasets/ARD100_YOLOMG/ard100.yaml \
  --weights papers/YOLOMG/runs/train/<run_name>/weights/best.pt \
  --gt D:/URAP_datasets/ARD100_YOLOMG/labels/test \
  --gt-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --out papers/YOLOMG/runs/window_accuracy/<run_name>
```

For the small fixture shipped inside the YOLOMG repo, first build YOLO-format
labels from its mask images, then run official YOLOMG eval through the isolated
paper runtime:

```bash
python3 tools/build_yolomg_test_images_dataset.py
python3 tools/run_yolo_eval_window_accuracy.py \
  --method yolomg \
  --repo papers/YOLOMG \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/yolomg_test_images_dataset/yolomg_test_images.yaml \
  --weights papers/YOLOMG/runs/train/ARD100_mask32-640_uavs/weights/best.pt \
  --gt runs/window_accuracy/yolomg_test_images_dataset/labels \
  --gt-format yolo-dir \
  --out runs/window_accuracy/yolomg_test_images_eval \
  --project runs/window_accuracy/yolomg_test_images_eval/eval \
  --name official_val \
  --task val \
  --img 640 \
  --batch-size 1 \
  --device cpu \
  --fps 30 \
  --window-seconds 3 \
  --score-threshold 0.25
```

Or compare labels already saved by YOLOMG's `val.py`:

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/ARD100_YOLOMG/labels/test \
  --gt-format yolo-dir \
  --pred papers/YOLOMG/runs/val/<run_name>/labels \
  --pred-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/YOLOMG/runs/window_accuracy/<run_name>
```

### TransVisDrone

Run TransVisDrone eval and curves together. For the prepared NPS layout, labels
are indexed from 0 while extracted frames and saved predictions are indexed from
1, so add `--gt-frame-offset 1`.

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method transvisdrone \
  --repo papers/TransVisDrone \
  --data D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/nps.yaml \
  --weights papers/TransVisDrone/runs/train/<run_name>/weights/best.pt \
  --gt D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --fps 30 \
  --window-seconds 3 \
  --out papers/TransVisDrone/runs/window_accuracy/<run_name>
```

Or compare labels already saved by `papers/TransVisDrone/val.py`:

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --pred papers/TransVisDrone/runs/val/NPS_URAP_D/<run_name>/labels \
  --pred-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/TransVisDrone/runs/window_accuracy/<run_name>
```

If `val.py` was run with `--save-json-gt`, TransVisDrone may already have a
pickle such as:

```text
papers/TransVisDrone/runs/val/NPS/<train_run>/<eval_run>/predictionsgt/predictionsgt_split_0.pkl
```

That file embeds both native-pixel labels and detections, so it can be plotted
without a separate GT label directory:

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt papers/TransVisDrone/runs/val/NPS/<train_run>/<eval_run>/predictionsgt/predictionsgt_split_0.pkl \
  --gt-format tvd-pkl-gt \
  --pred papers/TransVisDrone/runs/val/NPS/<train_run>/<eval_run>/predictionsgt/predictionsgt_split_0.pkl \
  --pred-format tvd-pkl-pred \
  --fps 30 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/TransVisDrone/runs/window_accuracy/<eval_run>
```

### ESOD

Run `papers/ESOD/test.py` and curves together. The wrapper uses ESOD's
`--img-size` argument, because ESOD's parser does not accept the same short
`--img` path as the YOLOMG/TransVisDrone forks. The wrapper also blocks the
paper fork's YOLOv5 auto-pip install behavior during eval.

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method esod \
  --repo papers/ESOD \
  --data <VisDrone-or-custom-yaml> \
  --weights papers/ESOD/runs/train/<run_name>/weights/best.pt \
  --gt <VisDrone-or-custom-label-dir> \
  --gt-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --out papers/ESOD/runs/window_accuracy/<run_name>
```

For the small local fixture, download the official ESOD YOLOv5m checkpoint
from the paper's Google Drive to `papers/ESOD/weights/esod_yolov5m.pt`, then
run:

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method esod \
  --repo papers/ESOD \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/esod_test_images_dataset/esod_yolomg_test_images.yaml \
  --weights papers/ESOD/weights/esod_yolov5m.pt \
  --gt runs/window_accuracy/yolomg_test_images_dataset/labels \
  --gt-format yolo-dir \
  --out runs/window_accuracy/esod_test_images_eval \
  --project runs/window_accuracy/esod_test_images_eval/eval \
  --name official_test \
  --task val \
  --img 640 \
  --batch-size 1 \
  --device cpu \
  --fps 30 \
  --window-seconds 3 \
  --score-threshold 0.25
```

For the real local VisDrone val run, prepare the public VisDrone val split and
then run ESOD through the same wrapper:

```bash
python3 tools/prepare_visdrone_yolo.py \
  --download \
  --split val \
  --root datasets/VisDrone \
  --yaml-out runs/window_accuracy/papers/visdrone_esod.yaml \
  --summary-json runs/window_accuracy/papers/visdrone_prepare_summary.json

.venv/paper-cv/bin/python tools/run_yolo_eval_window_accuracy.py \
  --method esod \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/papers/visdrone_esod.yaml \
  --weights papers/ESOD/weights/esod_yolov5m.pt \
  --gt datasets/VisDrone/VisDrone2019-DET-val/labels \
  --gt-format yolo-dir \
  --frame-manifest datasets/VisDrone/VisDrone2019-DET-val/images \
  --frame-manifest-format image-dir \
  --out runs/window_accuracy/papers/esod_visdrone_val \
  --project runs/window_accuracy/papers/esod_visdrone_val/eval \
  --name esod_visdrone_val \
  --img 1280 \
  --batch-size 1 \
  --device cpu \
  --fps 30 \
  --window-seconds 3 \
  --score-threshold 0.25 \
  --match-iou 0.5
```

The verified run produced 548 per-image curves under
`runs/window_accuracy/papers/esod_visdrone_val/plots/`.

ESOD's txt writer also uses normalized YOLO `class cx cy w h conf` lines, so
existing saved labels can use the same `yolo-dir` path:

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt <VisDrone-or-custom-label-dir> \
  --gt-format yolo-dir \
  --pred papers/ESOD/runs/test/<run_name>/labels \
  --pred-format yolo-dir \
  --fps 30 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/ESOD/runs/window_accuracy/<run_name>
```

### AICrowd Winner / AOT JSON

The winner source snapshot is available under
`papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022`.
After running it with the required weights/data, its `result.json` can be
scored directly.
For NPS, the winner's official `seg_test.py` expects one folder per
flight/clip. The helper below converts flat NPS frames such as
`Clip_001_00001.png` into `Clip_001/Clip_001_00001.png` hardlinks or copies:

```bash
python3 tools/prepare_aicrowd_nps_flight_dirs.py \
  --frames-dir D:/URAP_datasets/TransVisDrone/NPS/AllFrames/val \
  --out-dir papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/_prepared_nps_val \
  --json papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/prepare_nps_val.json
```

For a full NPS inference run on Windows, use the detached runner and monitor:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_winner_v022_nps_val_detached.ps1 `
  -DatasetPath D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val

powershell -ExecutionPolicy Bypass -File tools\monitor_winner_v022_nps_val.ps1 `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val
```

For NPS-style GT labels, provide image size because winner JSON uses absolute
pixel `x,y,w,h` boxes.

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --pred papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/nps_val \
  --pred-format aot-json \
  --img-width 1280 \
  --img-height 960 \
  --fps 30 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/AICrowd_AOT_Challenge_Winner/runs/window_accuracy/nps_val
```

For official AOT data, score the challenge `groundtruth.json` directly. The GT
loader reads `samples.*.entities[].blob.frame` and converts `bb=[left, top,
width, height]` to xyxy boxes; winner predictions still use `pred-format
aot-json`. When GT is `aot-gt-json`, aggregate AOT prediction files that only
contain `img_name` are mapped back to `flight_id/frame` through the GT
`img_name` lookup, matching the official AOT metrics join.

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json \
  --gt-format aot-gt-json \
  --pred papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_aot_part1/part1 \
  --pred-format aot-json \
  --fps 10 \
  --window-seconds 3 \
  --iou 0.5 \
  --score-threshold 0.25 \
  --out papers/AICrowd_AOT_Challenge_Winner/runs/window_accuracy/aot_part1
```

### EDTC and Li-TETC

EDTC's YOLO detector branch can be run through the same YOLO eval wrapper. For
the small local fixture, download the official Drive `yolo/best.pt` to
`papers/EDTC/yolov5/weights/edtc_yolo_best.pt`, then run:

```bash
python3 tools/run_yolo_eval_window_accuracy.py \
  --method edtc \
  --repo papers/EDTC/yolov5 \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/yolomg_test_images_dataset/yolomg_test_images.yaml \
  --weights papers/EDTC/yolov5/weights/edtc_yolo_best.pt \
  --gt runs/window_accuracy/yolomg_test_images_dataset/labels \
  --gt-format yolo-dir \
  --out runs/window_accuracy/edtc_yolo_test_images_eval \
  --project runs/window_accuracy/edtc_yolo_test_images_eval/eval \
  --name official_val \
  --task val \
  --img 640 \
  --batch-size 1 \
  --device cpu \
  --fps 30 \
  --window-seconds 3 \
  --score-threshold 0.001
```

The EDTC fixture curve uses that lower post-processing threshold because the
official detector's saved fixture confidences are around 0.001.

For the full EDTC tracker branch, tracker outputs are one `x y w h` row per
frame. Its AntiUAV labels are `IR_label.json` files with `gt_rect` / `exist`.
Use the wrapper below to create EDTC `local.py`, generate a temporary tracker
config with the YOLO detector paths, run `tracking/test.py`, and then render
the same +/-3s curves:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_edtc_tracker_window_accuracy_detached.ps1 `
  -DatasetRoot <AntiUAV-dataset-root> `
  -TrackerModel <UAVTrackEH.pth.tar> `
  -YoloWeights papers\EDTC\yolov5\weights\edtc_yolo_best.pt `
  -YoloData <EDTC-yolo-antiuav.yaml> `
  -Out runs\window_accuracy\papers\edtc_antiuav600 `
  -Fps 30 `
  -WindowSeconds 3 `
  -RunId edtc_antiuav600

powershell -ExecutionPolicy Bypass -File tools\monitor_edtc_tracker_window_accuracy.ps1 `
  -Out runs\window_accuracy\papers\edtc_antiuav600 `
  -RunId edtc_antiuav600
```

When EDTC tracker txt outputs already exist, skip tracker inference and score
the result directory directly:

```bash
python3 tools/run_edtc_tracker_window_accuracy.py \
  --dataset-root <AntiUAV-dataset-root> \
  --skip-track \
  --results-dir papers/EDTC/<tracker-results-dir> \
  --fps 30 \
  --window-seconds 3 \
  --out runs/window_accuracy/papers/edtc_antiuav600
```

Li-TETC writes `time_layer: ... detections: (...)` text under
`Experiment_Results/Final/txt/`; the original GT files use the same structure.
`tools/run_li_tetc_demo_compat.py` is a compatibility launcher for the
published demo code. On this Mac it loads the old Keras/AdaBoost models,
patches the legacy sklearn/joblib pickle path, and completed the bundled
`Clip_14.mov` and `Clip_40.mov` demos to produce
`Experiment_Results/Final/txt/14_dt.txt` and `40_dt.txt`.

```bash
.venv/paper-cv/bin/python tools/run_li_tetc_demo_compat.py --video-id 14
.venv/paper-cv/bin/python tools/run_li_tetc_demo_compat.py --video-id 40
python3 tools/discover_paper_window_accuracy_runs.py
python3 tools/run_paper_window_accuracy_batch.py \
  --manifest runs/window_accuracy/discovered_manifest.json \
  --only li_tetc_video_14 \
  --only li_tetc_video_40
```

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925/Video_14_gt.txt \
  --gt-format li-tetc-txt \
  --pred papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Experiment_Results/Final/txt/14_dt.txt \
  --pred-format li-tetc-txt \
  --fps 29 \
  --window-seconds 3 \
  --out runs/window_accuracy/li_tetc_video14
```

Use `Video_40_gt.txt`, `40_dt.txt`, and `--fps 29.8` for the second bundled
demo clip.

## Reading The Curves

- Low accuracy + high recall usually means too many false positives.
- Low accuracy + low recall usually means misses dominate.
- Precision dips with stable recall point to noisy background segments.
- Recall dips with stable precision point to hard visibility, scale, motion blur,
  occlusion, or camera motion.

Use `low_accuracy_segments.csv` first when deciding which spans to inspect.
It groups adjacent center frames below the segment threshold into start/end
frame ranges and time ranges. Then use the SVG curve troughs and
`worst_windows.csv` to inspect the exact local failures.
