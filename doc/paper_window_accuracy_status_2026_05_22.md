# Paper Window Accuracy Status - 2026-05-22

This note records the current verified state for the paper-repo +/-3s
per-video accuracy curves.

## Verified Complete Curve Outputs

The paper manifest currently has twelve complete curve outputs: one YOLOMG
official-eval label run, one ESOD full VisDrone val run, one ESOD
official-eval fixture run, one EDTC YOLO-branch official-eval label run, six
TransVisDrone validation pickle runs, and two Li-TETC demo prediction txt
files. The TransVisDrone runs include two `predictionsgt_split_0.pkl` files
that embed both GT labels and detections plus four sibling prediction-only
pkls paired to the same embedded GT. All twelve runs were converted to
per-frame +/-3s window metrics and per-video SVG curves.

Run:

```bash
python3 tools/discover_paper_window_accuracy_runs.py
python3 tools/run_paper_window_accuracy_batch.py \
  --manifest runs/window_accuracy/discovered_manifest.json
python3 tools/audit_paper_window_accuracy_readiness.py \
  --manifest runs/window_accuracy/discovered_manifest.json \
  --json runs/window_accuracy/discovered/readiness_audit.json \
  --markdown runs/window_accuracy/discovered/readiness_audit.md
```

Outputs:

| Method | Run | Videos | Frames | GT boxes | Prediction boxes | Mean window accuracy | Curve index |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOMG | `yolomg_official_val` | 1 | 11 | 33 | 11 | 0.073 | `runs/window_accuracy/papers/yolomg_official_val/plots/index.html` |
| ESOD | `esod_visdrone_val` | 548 | 548 | 38759 | 44202 | 0.574 | `runs/window_accuracy/papers/esod_visdrone_val/plots/index.html` |
| ESOD | `esod_official_test` | 1 | 11 | 33 | 93 | 0.000 | `runs/window_accuracy/papers/esod_official_test/plots/index.html` |
| EDTC | `edtc_official_val` | 1 | 11 | 33 | 7 | 0.000 | `runs/window_accuracy/papers/edtc_official_val/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_fl_best_augment_full_save` | 14 | 19466 | 8974 | 13904 | 0.465 | `runs/window_accuracy/papers/transvisdrone_pkl_fl_best_augment_full_save/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_fl_640_test` | 14 | 19466 | 8974 | 2036 | 0.148 | `runs/window_accuracy/papers/transvisdrone_pkl_fl_640_test/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_fl_800_test` | 14 | 19466 | 8974 | 2515 | 0.148 | `runs/window_accuracy/papers/transvisdrone_pkl_fl_800_test/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_nps_best_augment_full_save` | 10 | 12350 | 16362 | 21799 | 0.588 | `runs/window_accuracy/papers/transvisdrone_pkl_nps_best_augment_full_save/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_nps_best_coco` | 10 | 12350 | 16362 | 17883 | 0.802 | `runs/window_accuracy/papers/transvisdrone_pkl_nps_best_coco/plots/index.html` |
| TransVisDrone | `transvisdrone_pkl_nps_speedtest` | 10 | 12350 | 16362 | 30000 | 0.097 | `runs/window_accuracy/papers/transvisdrone_pkl_nps_speedtest/plots/index.html` |
| Li-TETC / NPS baseline | `li_tetc_video_14` | 1 | 1800 | 624 | 738 | 0.216 | `runs/window_accuracy/papers/li_tetc_video_14/plots/index.html` |
| Li-TETC / NPS baseline | `li_tetc_video_40` | 1 | 1533 | 1167 | 1141 | 0.325 | `runs/window_accuracy/papers/li_tetc_video_40/plots/index.html` |

Batch index:

```text
runs/window_accuracy/papers/index.html
```

Cross-run dashboard:

```text
runs/window_accuracy/papers/dashboard.html
```

Each completed run now also has:

```text
low_accuracy_segments.csv
```

This file groups adjacent center frames below the segment threshold into
continuous low-accuracy spans with start/end frames, start/end seconds, worst
frame, and aggregated TP/FP/FN. The dashboard surfaces these spans before the
individual worst-window rows.

The curve scorer now supports explicit frame manifests. When a run provides
`frame_manifest` / `frame_manifest_format`, center frames come from that source
instead of only from frames that contain GT or predictions. This is important
for the stated "every frame" goal because YOLO labels, tracker txt files, and
AOT result files often omit empty frames at the start or end of a video.
Supported manifest formats include image directories, frame CSV files, AOT
JSON, AntiUAV `IR_label.json`, TransVisDrone pkl files, Li-TETC txt files, and
YOLO label directories. For AOT-style image directories whose files are hashes
rather than numeric frame names, `image-dir` assigns frame ids by sorted order
inside each flight folder and uses the same filename lookup for AOT
`result.json` predictions.

Readiness audit:

```text
runs/window_accuracy/discovered/readiness_audit.md
```

Goal audit for the full paper-repo curve objective:

```text
runs/window_accuracy/papers/goal_audit.md
```

This audit is intentionally stricter than the readiness report. It keeps the
objective marked incomplete until every manifest run has complete curve
artifacts, while still verifying that repositories, shared format adapters,
dashboards, and generation commands are in place.

Missing-input gap report for the full paper manifest:

```text
runs/window_accuracy/papers/gap_report.md
```

The current strict manifest audit has 12 complete curve runs, 4 missing-input
runs, and 1 ready/running run (`transvisdrone_nps_val`). The gap report lists
all five non-complete manifest entries, their expected GT and prediction
paths/formats, any compatible candidates found under local `datasets/`, `runs/`,
and `papers/`, and the exact batch command to run once the inputs are present.
It also includes concrete generation commands for known missing runs, including
detached start/monitor commands for long YOLO-style, EDTC, and AICrowd winner
inference jobs.

Repository sync report:

```text
runs/window_accuracy/papers/paper_repo_sync_latest.json
```

The latest safe sync refreshed repos not currently being used by active
detached jobs. YOLOMG, ESOD, Li-TETC, and Dogfight/Drone-Detection were already
up to date; AICrowd winner `submission-v022` was refreshed as a GitLab API
source snapshot at commit `1fbc2276` with 100 source files downloaded and 13
binary/model files skipped. TransVisDrone and EDTC were intentionally not
fast-forwarded while their detached inference jobs were using those worktrees.

External source inventory:

```text
runs/window_accuracy/papers/external_source_inventory.md
```

The metadata-only inventory confirms:

- NPS `Videos.zip` is about 1.9 GiB and `Video_Annotation.zip` is about
  741.5 KiB; the local Dogfight annotation checkout has 50 NPS annotation
  files, 45,097 rows, and 59,201 boxes. The local NPS `Videos.zip` is now
  downloaded at `datasets/NPS/raw/Videos.zip`, clips 37-40 are extracted under
  `datasets/NPS/raw/Videos`, and NPS val has been prepared as 5,944 JPEG
  frames plus 3,753 YOLO label files under `datasets/TransVisDrone/NPS`.
  The official TransVisDrone NPS `best.pt` is now downloaded from the paper
  Drive to `papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt`
  (918 MiB). A 5-frame local CPU smoke using this checkpoint completed at
  `runs/window_accuracy/papers/transvisdrone_nps_val_smoke/plots/index.html`;
  full `transvisdrone_nps_val` still needs the complete 5,944-frame prediction
  label directory.
  AICrowd-style hardlinked NPS clip folders were prepared under
  `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/_prepared_nps_val`.
- AOT part1 exposes `ImageSets/groundtruth.json` at about 434 MiB and
  `valid_encounters_maxRange700_maxGap3_minEncLen30.json` at about 9.7 MiB,
  while `Images/` contains multi-MB frame files and should not be recursively
  synced on this Mac with only about 16 GiB free.
- EDTC's public Drive model/raw-results folder lists `yolo/`,
  `CvT-21-384x384-IN-22k.pth`, and `UAVTrackEH.pth.tar`; the AntiUAV600 Drive
  folder lists `test.zip`, `train.zip`, and `validation.zip`. The local EDTC
  inputs are now prepared for validation: `datasets/AntiUAV600/raw/validation.zip`
  is downloaded and extracted to `datasets/AntiUAV600/validation` with 50
  sequences, 50 `IR_label.json` files, and 56,301 JPEG frames. `UAVTrackEH.pth.tar`
  is present under `papers/EDTC/pretrained/`, the EDTC YOLO weight is present
  under `papers/EDTC/yolov5/weights/`, and `data_templates/edtc_antiuav.yaml`
  points YOLOv5 at the extracted validation root.

Current discovered-manifest audit:

```bash
python3 tools/audit_paper_window_accuracy_readiness.py \
  --manifest runs/window_accuracy/discovered_manifest.json
# counts={"complete_curves": 11}
```

## Current Missing Real Outputs

The public repositories were refreshed on 2026-05-22 and are up to date under
`papers/`, but not every method has scorable prediction outputs on this Mac
yet.

| Method | Current local state | Needed to generate real curves |
| --- | --- | --- |
| YOLOMG | A small official-code run is complete from `papers/YOLOMG/data/Test_images` using mask-derived GT labels and `ARD100_mask32-640_uavs/weights/best.pt`. Full ARD100/NPS runs still need real dataset YAML/labels on this Mac. The official README points ARD100 to a BaiduYun share. | For full-paper curves, run YOLOMG eval with `--save-txt --save-conf`, or use `tools/run_yolo_eval_window_accuracy.py` with full dataset YAML and GT labels. |
| TransVisDrone | Six validation pkl runs are complete and plotted: FL `best_augment_full_save`, `640_test`, `800_test`; NPS `best_augment_full_save`, `best_coco`, `speedtest`. Prediction-only pkls are paired to the matching `best_augment_full_save/predictionsgt` GT pkl. The Dogfight annotation repo is now cloned at `datasets/Drone-Detection`; NPS val frames/labels are prepared locally in `datasets/TransVisDrone/NPS`; the official NPS `best.pt` checkpoint is downloaded under `papers/TransVisDrone/pretrained/TransVisDrone_weights`. A 5-frame CPU smoke with the real checkpoint completed, after patching current-runtime compatibility issues (`mmcv` fallback import, `np.int`, `.jpg` temporal sampling, old checkpoint GELU/DropPath attributes, Pillow text sizing). The formal `transvisdrone_nps_val` run is launched through the macOS detached wrapper and is producing prediction labels under `runs/window_accuracy/papers/transvisdrone_nps_val/eval/transvisdrone_nps_val/labels`; it must finish all 5,944 frames before the run can be scored as complete. | Monitor with `bash tools/monitor_yolo_eval_window_accuracy.sh --out runs/window_accuracy/papers/transvisdrone_nps_val --run-id transvisdrone_nps_val`. The monitor reports eval-log image progress separately from saved prediction-label count, because a label file count alone is not a reliable every-frame progress source. After it finishes successfully, rerun the manifest batch/dashboard/audit. |
| ESOD | Code is pulled, `esod_yolov5m.pt` is present, `tools/prepare_visdrone_yolo.py` downloaded/prepared VisDrone val locally, and `test.py --save-txt --save-conf` completed at image size 1280 in `.venv/paper-cv`. The real `esod_visdrone_val` curve is complete with 548 per-image curves, 38,759 GT boxes, and 44,202 predictions. The smaller `esod_official_test` fixture curve is also complete. | Optional: add UAVDT/TinyPerson or other ESOD splits by preparing labels and running `tools/run_yolo_eval_window_accuracy.py --method esod`. |
| AICrowd Winner | The AIcrowd winners page links to `dmytro_poplavskiy/airborne-detection-starter-kit`. Unauthenticated `git clone` still prompts for credentials and archive downloads time out, so `tools/pull_paper_repos.py` now downloads the public `submission-v022` source tree via GitLab API into `papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022`. The snapshot marker records commit `1fbc227686e5721535eefc9bd76e4f523c697c7f`. `tools/inventory_aicrowd_lfs_weights.py` inventories 10 missing Git LFS model weights totaling about 1.01 GiB in `runs/window_accuracy/aicrowd_lfs_weight_inventory.md`. `tools/download_aicrowd_lfs_weights.py --dry-run` currently records `auth_required` in `runs/window_accuracy/aicrowd_lfs_weight_download_report.json`, because GitLab LFS batch requires a token. No local AOT/NPS `result.json` exists yet. The curve loader now supports official AOT `ImageSets/groundtruth.json` via `gt_format=aot-gt-json` and maps winner predictions back to `flight_id/frame` by `img_name`, matching the official metrics join pattern. The NPS runner path was fixed: `tools/prepare_aicrowd_nps_flight_dirs.py` groups flat `Clip_001_00001.png` frames into AICrowd-style clip folders, and `tools/run_winner_v022_nps_val.ps1` now runs the actual snapshot entrypoint `seg_test.py` instead of the missing `seg_test_nps.py`. Local NPS clip folders are now prepared, so `aicrowd_winner_nps_val` is missing only winner `result.json`/weights. | Set `AICROWD_GITLAB_TOKEN` or `GITLAB_TOKEN`, run `tools/download_aicrowd_lfs_weights.py`, then run the winner inference to produce AOT-style `result.json` and score it with `pred_format=aot-json`. For NPS, use `tools/start_winner_v022_nps_val_detached.ps1` and `tools/monitor_winner_v022_nps_val.ps1`. For official AOT data, use `gt_format=aot-gt-json`, `fps=10`, and `D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json`. |
| EDTC | The official Drive `yolo/best.pt` was downloaded to `papers/EDTC/yolov5/weights/edtc_yolo_best.pt`, and EDTC's `yolov5/val.py --save-txt --save-conf` completed on the local YOLOMG Test_images fixture. The `edtc_official_val` detector curve is complete. The full AntiUAV600 validation inputs are now local: `datasets/AntiUAV600/validation`, `papers/EDTC/pretrained/UAVTrackEH.pth.tar`, and `data_templates/edtc_antiuav.yaml`. `tools/run_edtc_tracker_window_accuracy.py` wires EDTC's `tracking/test.py` to the curve generator by creating the required `local.py`, generating a temporary tracker YAML with YOLO paths, running AntiUAV tracking, and scoring the resulting `x y w h` txt files. A CPU smoke on sequence index 23 completed with one 28-frame curve at `runs/window_accuracy/papers/edtc_antiuav600_smoke_sequence23/plots/index.html`; that smoke measured about 0.65 FPS, so large CPU sequences can take tens of minutes before a first result file appears. A live `sample` of the formal tracker child process showed active PyTorch CPU inference inside `conv2d` / `slow_conv2d_forward_cpu`, which supports slow CPU computation rather than an input wait or deadlock. The formal `edtc_antiuav600` detached run has been launched on the local CPU, but the manifest run remains incomplete until the full validation tracker result directory `runs/window_accuracy/papers/edtc_antiuav600/tracking_results/uavtrack_eh/urap_window_accuracy` is populated and scored. The AntiUAV validation audit found 50 unique listed sequences, 56,301 direct JPEG frames, and one nested duplicate directory under `20190925_200320_1_5`; EDTC's loader uses only direct `.jpg` files from each listed sequence, so the nested duplicate does not affect scoring. | Monitor with `bash tools/monitor_edtc_tracker_window_accuracy.sh --out runs/window_accuracy/papers/edtc_antiuav600 --run-id edtc_antiuav600`. The monitor reports child CPU/memory, stale output age, and the next expected sequence/frame count. The local CPU path is useful for compatibility but slow for the full 56,301-frame validation set; use `--skip-track --results-dir` if full validation tracker txt outputs already exist elsewhere. |
| Li-TETC / NPS baseline | `tools/run_li_tetc_demo_compat.py` now loads the old Keras/AdaBoost models and completed both bundled demos, producing `Experiment_Results/Final/txt/14_dt.txt` and `40_dt.txt` plus the `li_tetc_video_14` and `li_tetc_video_40` curves. | Run additional full NPS videos through the same compat runner or original environment to add more `*_dt.txt` files. |

## Local Runtime Notes

The system Anaconda Python still cannot directly run some official paper
inference scripts:

- The system Python had NumPy 2.4 / compiled extension ABI failures. An
  isolated `.venv/paper-cv` runtime with `numpy<2`, `thop`, and
  `torchvision==0.24.1` now runs YOLOMG and ESOD entrypoints. ESOD VisDrone val
  was executed in this environment.
- `tools/run_yolo_eval_window_accuracy.py` preserves the venv Python symlink,
  passes absolute eval input paths to paper repos, and sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted local YOLO checkpoints. It
  also disables paper-fork YOLOv5 auto-pip installs during eval so the isolated
  runtime is not mutated by requirements checks.
- Li-TETC now has `tools/run_li_tetc_demo_compat.py` shims for old
  `sklearn.externals.joblib` joblib pickles, legacy sklearn tree objects,
  NumPy aliases, and stricter OpenCV 4 drawing coordinates. The bundled
  `Clip_14.mov` and `Clip_40.mov` demos completed in `.venv/paper-cv`.
- TransVisDrone now loads the official NPS checkpoint in `.venv/paper-cv`.
  The local checkout includes runtime compatibility patches for missing
  `mmcv` imports, NumPy alias removals, JPEG temporal frame sampling, older
  pickled GELU/DropPath modules, and Pillow 10+ font sizing.
- EDTC now has a separate `.venv/edtc-window` runtime with `numpy<2`. The
  official EDTC code was patched for current PyTorch CPU compatibility in the
  local checkout: removed `torch._six` dependencies, made `jpeg4py` optional,
  moved tensors/preprocessors to the selected device, added a pure PyTorch NMS
  fallback for broken `torchvision.ops.nms`, and added a `grid_sample` fallback
  for CUDA-only PreciseRoIPooling during CPU smoke runs.

This does not affect the post-processing curve tools, which are covered by
tests and have already processed the existing TransVisDrone pickle outputs.

## Verification

```bash
python3 -m pytest tests/test_window_accuracy.py tests/test_paper_repos.py -q
# 40 passed

python3 -m py_compile \
  tools/pull_paper_repos.py \
  tools/download_nps_videos.py \
  tools/inventory_aicrowd_lfs_weights.py \
  tools/inventory_external_window_accuracy_sources.py \
  tools/download_google_drive_file.py \
  tools/download_aicrowd_lfs_weights.py \
  tools/audit_paper_window_accuracy_readiness.py \
  tools/audit_paper_window_accuracy_goal.py \
  tools/write_paper_window_accuracy_gap_report.py \
  tools/build_window_accuracy_dashboard.py \
  tools/prepare_visdrone_yolo.py \
  tools/prepare_aicrowd_nps_flight_dirs.py \
  tools/run_edtc_tracker_window_accuracy.py \
  tools/run_li_tetc_demo_compat.py \
  tools/run_yolo_eval_window_accuracy.py \
  tools/discover_paper_window_accuracy_runs.py \
  qstr_dronedet/evaluation/window_accuracy.py
# pass

git diff --check
# pass

python3 tools/build_paper_window_accuracy_smoke.py
# complete=6

python3 tools/run_paper_window_accuracy_batch.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --skip-missing
# complete=12, skipped_missing=5; complete runs were regenerated with frame_manifest where configured

python3 tools/pull_paper_repos.py \
  --json runs/window_accuracy/papers/paper_repo_pull_report.json
# public git/data repos already up to date; AICrowd source snapshot refreshed at 1fbc2276

python3 tools/inventory_external_window_accuracy_sources.py \
  --json runs/window_accuracy/papers/external_source_inventory.json \
  --markdown runs/window_accuracy/papers/external_source_inventory.md
# metadata-only external source report refreshed

python3 tools/write_paper_window_accuracy_gap_report.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --json runs/window_accuracy/papers/gap_report.json \
  --markdown runs/window_accuracy/papers/gap_report.md
# gaps=5

python3 tools/audit_paper_window_accuracy_goal.py \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --json runs/window_accuracy/papers/goal_audit.json \
  --markdown runs/window_accuracy/papers/goal_audit.md
# status=incomplete; frame_manifests_configured=true; all_manifest_runs_complete=false

python3 tools/run_paper_window_accuracy_pipeline.py \
  --skip-pull \
  --manifest data_templates/paper_window_accuracy_runs.example.json \
  --smoke \
  --json runs/window_accuracy/papers/pipeline_report.json \
  --markdown runs/window_accuracy/papers/pipeline_report.md
# final_counts={"complete_curves": 12, "missing_inputs": 5}
```

The readiness audit now validates inputs by format, not just by path existence:
YOLO label directories must contain `.txt` files, AOT outputs must contain
`result.json`, AOT ground truth must contain `groundtruth.json`, AntiUAV roots
must contain label JSON, and TransVisDrone pkl inputs must be real `.pkl`
files. Optional frame manifests are also validated by format, so configured
image dirs must contain image files and AntiUAV frame manifests must contain
`list.txt` or `IR_label.json`. Completed curve outputs must include `per_frame_window_metrics.csv`,
`worst_windows.csv`, `low_accuracy_segments.csv`, `summary.json`, plot HTML,
and at least one SVG curve.
