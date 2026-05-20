# Official Dataset Download + Protocol Metrics Recompute

This document tracks what has been downloaded (from official sources when available) and how metrics are recomputed following each paper/repo's protocol.

Last updated: 2026-02-09

## TransVisDrone (ICRA 2023) - `papers/TransVisDrone`

Paper: "TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos" (Sangam et al.)

### Datasets Used By The Repo

- NPS (Purdue UAV dataset videos) + Dogfight annotations (public)
- FL-Drones (NOT public; requires author permission)
- AOT (Airborne Object Tracking; public on AWS Open Data Registry)

### NPS (Public) - Status

Official videos (Purdue): already present under:
- `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos/Clip_*.mov` (50 clips)

Annotations (Dogfight GitHub): already present under:
- `datasets/Drone-Detection/annotations/NPS-Drones-Dataset/Clip_*.txt` (50 annotation files)

We follow the repo README's indexing convention for NPS:
- Extract frames with filenames starting at 1: `Clip_{id}_{frame:05}.png`
- Keep labels indexed from 0 (as in Dogfight annotations): `Clip_{id}_{frame:05}.txt`
- This matches `papers/TransVisDrone/utils/datasets.py` behavior for NPS (image index - 1 when locating label index).

Prepared dataset output root (on `D:` for capacity):
- `D:/URAP_datasets/TransVisDrone/NPS`

Converter used:
- `tools/prepare_transvisdrone_nps.py`

Output layout:
- Frames: `D:/URAP_datasets/TransVisDrone/NPS/AllFrames/{train,val,test}/Clip_XXX_00001.png ...`
- YOLO labels: `D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/{train,val,test}/labels/Clip_XXX_00000.txt ...`
- Video lengths: `D:/URAP_datasets/TransVisDrone/NPS/Videos/{train,val,test}/video_length_dict.pkl`

Dataset yaml (URAP-local):
- `papers/TransVisDrone/data/NPS_URAP_D.yaml`

#### Metrics Recompute (Protocol)

Repo-provided evaluation command (from `papers/TransVisDrone/submit-test.slurm`):
- `python val.py --task test --img 1280 --num-frames 5 --batch-size 1 --augment --save-json --save-json-gt`

Weights used (official pretrained provided by repo author):
- `papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt`

#### NPS Metrics (Pretrained Best Weights)

Command used (same as repo protocol, but with `--batch-size 8 --half` for speed):

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\TransVisDrone
$w='pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt'
.\.venv\Scripts\python val.py --data .\data\NPS_URAP_D.yaml --weights $w --task test --img 1280 --num-frames 5 --augment --save-json --save-json-gt --device 0 --batch-size 8 --half --project .\runs\val\NPS_URAP_D --name nps_test_best_aug_bs8_half --exist-ok
```

Test split results (from console metrics):
- P: 0.916
- R: 0.901
- mAP@0.5: 0.938
- mAP@0.5:0.95: 0.468

Artifacts:
- Log: `artifacts/logs/transvisdrone_nps_test_bs8_half_aug.log`
- Output dir: `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half`

Notes:
- `val.py` also tries to run COCO-api (`pycocotools`) evaluation when `--save-json` is enabled. NPS does not ship a COCO-style `annotations/instances_val2017.json`, so that optional COCO-api step logs a missing-file warning but does not affect the printed YOLO metrics above.
- NPS `--task val` run: DONE (log: `artifacts/logs/transvisdrone_nps_val_bs8_half_aug.log`, output dir: `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_val_best_aug_bs8_half`).

Val split results (from console metrics):
- P: 0.901
- R: 0.881
- mAP@0.5: 0.948
- mAP@0.5:0.95: 0.464

### FL-Drones (Not Public) - Status

The repo README states FL-Drones is not publicly available and requires author permission. We cannot "fully download" it from official sources without access approval.

### AOT (Public) - Status

Repo README points to AWS Open Data Registry:
- `https://registry.opendata.aws/airborne-object-tracking/`

Planned:
- Download AOT part1 to `D:/URAP_datasets/AOT/part1`
- Run conversion script to VisDrone-style layout per repo instructions
- Run `val.py` to produce AOT-style predictions (`--save-aot-predictions`) and then recompute official AOT metrics via `papers/TransVisDrone/evaluate_aot.py`

Completed:
- Patched `papers/TransVisDrone/evaluate_aot.py` to accept `--dataset-path` and to avoid double-`argparse` parsing (Windows-friendly).
- Downloaded AOT part1 ground truth:
  - `D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json`
- Implemented a Windows-friendly partial downloader/converter for AOT test parts:
  - `tools/prepare_transvisdrone_aot_part1.py`
  - Generates YAMLs under `papers/TransVisDrone/data/AOTTestSplits_URAP/`
  - Output dataset root example: `D:/URAP_datasets/TransVisDrone/AOT_part1_yolo/`
- Fixed Windows + modern dependency incompatibilities required to download/evaluate AOT:
  - S3 key joining on Windows (backslashes -> 404): `papers/TransVisDrone/aotcore/file_handler.py`
  - Optional imgaug import (NumPy 2): `papers/TransVisDrone/aotcore/frame.py`
  - pandas 2.x (DataFrame.append removed) + groupby key tuple: `papers/TransVisDrone/aotcore/metrics/airborne_metrics/calculate_airborne_metrics.py`

Validated end-to-end on a single test flight (part0, 1 flight):
- Build part0 data:
  - `C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python tools\prepare_transvisdrone_aot_part1.py --aot-root D:\URAP_datasets\AOT\part1 --out-root D:\URAP_datasets\TransVisDrone\AOT_part1_yolo --split test --part-size 1 --max-flights 1`
- Run detection + dump AOT predictions:
  - Log: `artifacts/logs/transvisdrone_aot_test_part0_bs2_half.log`
  - Output: `papers/TransVisDrone/runs/val/AOT_URAP/part0/aotpredictions/predictions_split_0.pkl`
- Run official AOT airborne metrics (on this partial set):
  - Log: `artifacts/logs/transvisdrone_aot_eval_part0.log`

#### Full AOT Test (172 flights) - In Progress

Started: 2026-02-09 09:46 (local)

Runner:
- `tools/run_aot_full_test.ps1`

Dataset output root:
- `D:/URAP_datasets/TransVisDrone/AOT_part1_yolo_fulltest`

Expected split count:
- 172 flights -> 18 parts (`part_size=10`)
- YAMLs: `papers/TransVisDrone/data/AOTTestSplits_URAP/AOTTest_0.yaml` ... `AOTTest_17.yaml`

Inference outputs (after completion):
- `papers/TransVisDrone/runs/val/AOT_URAP/fulltest_conf0p2/aotpredictions/predictions_split_*.pkl`

Logs:
- Driver stdout/stderr: `artifacts/logs/aot_fulltest/driver_stdout.log`, `artifacts/logs/aot_fulltest/driver_stderr.log`
- Prepare: `artifacts/logs/aot_fulltest/prepare.log`
- Inference: `artifacts/logs/aot_fulltest/infer_split_*.log`
- Evaluation: `artifacts/logs/aot_fulltest/eval.log`

Notes:
- `val.py` is run with `--conf-thres 0.2` to avoid writing extremely large AOT prediction pickles containing detections that would be discarded by the official threshold anyway.
- Official evaluation is run with `--detection_threshold 0.2` as recommended in the repo README.

Completed: 2026-02-09 20:02 (local)

Official AOT airborne metrics recompute outputs:
- Evaluation root: `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2`
- Summary JSON: `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/result_metrics_min_track_len_0_summary_far_89_47674_min_intruder_fl_dr_0p5_in_win_30.json`

Key numbers (min score ~0.2):
- FPPI: 0.262318
- HFAR: 89.47674
- AFDR (range<=700): 0.868472 (29251/33681)
- AFDR (area>200): 0.589017 (39397/66886)
- AFDR (area<=200): 0.089446 (5695/63670)
- Encounter DR @ max range 300 (All): 0.925714 (162/175)

Data footprint snapshot (logical sizes, 2026-02-09 12:16 local; AOT download still in progress so this will grow):
- AOT raw downloaded so far: ~295.45 GiB (`D:/URAP_datasets/AOT/part1`)
- NPS prepared frames+labels: ~17.32 GiB (`D:/URAP_datasets/TransVisDrone/NPS`)
- TransVisDrone pretrained weights: ~5.41 GiB (`papers/TransVisDrone/pretrained/TransVisDrone_weights`)
- Papers PDFs: ~67.35 MiB (`doc/`)
- Note: `D:/URAP_datasets/TransVisDrone/AOT_part1_yolo_fulltest` is mostly hardlinks to `D:/URAP_datasets/AOT/part1`, so it does *not* double physical disk usage.

## EDTC / AntiUAV600 (arXiv 2023) - `papers/EDTC`

Paper: "Evidential Detection and Tracking Collaboration: New Problem, Benchmark and Algorithm for Robust Anti-UAV System"

Repo README states:
- "The dataset will be released soon."

Status:
- Code + pretrained models are available (downloaded in `papers/EDTC/pretrained/`).
- The official AntiUAV600 dataset is not included in the repo. We attempted to locate an official public release (ModelScope `ly261666/3rd_Anti-UAV` download started into `D:/modelscope_cache`), but full metric recomputation is pending until we confirm the dataset matches AntiUAV600 and map it into the repo's expected structure.
