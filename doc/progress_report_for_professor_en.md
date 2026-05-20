# URAP Project Progress Report (From Zero to Current State)

## Reference

| Name | Meaning in this report | Related paper/dataset/project | Local entry points (for verification) |
|---|---|---|---|
| TransVisDrone / TVD | Our reproduced video-based detection baseline | ICRA 2023: *TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos* | Code: `papers/TransVisDrone`; paper: `doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf` |
| AOT | Airborne Object Tracking dataset (airborne object detection/tracking challenge data) | AOT from AWS Open Data Registry (this project uses part1) | Raw data: `D:\URAP_datasets\AOT\part1`; TVD read format: `D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest` |
| NPS (TVD) | NPS Drones dataset under the TVD training/evaluation protocol | Videos from Purdue/Bouman UAV Dataset; annotations in Dogfight/Drone-Detection format | Raw videos: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos/Clip_*.mov`; annotations: `datasets/Drone-Detection/annotations/NPS-Drones-Dataset/Clip_*.txt`; converted: `D:\URAP_datasets\TransVisDrone\NPS` |
| NPS (Li-TETC) | Original NPS protocol used in Li et al. TETC 2021 (time_layer GT) and our runnable baseline | IEEE TETC 2021: *Fast and Robust UAV to UAV Detection and Tracking from Video* (DOI: `10.1109/TETC.2021.3104555`) | Original GT: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925/Video_*_gt.txt` + `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Video_Annotation/Clip_*_gt.txt`; PyTorch baseline: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline`; result: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| Winner / submission-v022 | AOT challenge winning solution code | AIcrowd Airborne Object Tracking Challenge winner `airborne-detection-starter-kit`, tag `submission-v022` | Code: `papers/AICrowd_AOT_Challenge_Winner/submission-v022`; notes: `doc/aicrowd_aot_winner_code.md` |

## 0) Executive Summary

| Topic | Conclusion/Status | Evidence (reproducible paths) |
|---|---|---|
| Main reproduction line (TransVisDrone) | Running end-to-end on Windows + modern PyTorch/CUDA, with official-protocol metric recomputation completed on NPS (val/test) and AOT (fulltest) | `doc/official_datasets_and_metrics.md`; `papers/TransVisDrone/runs/val/NPS_URAP_D/*/results.txt`; `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/*.json` |
| Original NPS paper baseline (Li-TETC 2021) | Original repo (Keras) reviewed; `pt_pipeline` (PyTorch/uv) established as a runnable sanity baseline; annotation-source/protocol differences between the two NPS variants were clarified to avoid mixing | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/README.md`; `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md`; `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| Competition Winner (AOT #1 submission-v022) | AOT fulltest **172/172 flights inference + official airborne metric recomputation completed** | `tools/monitor_winner_v022_fulltest.ps1`; `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json` |
| Winner method cross-domain generalization (NPS) | Directly applying Winner to NPS performs poorly (most candidates filtered out by thresholding/confirmation chain; still far below TVD after relaxed settings) | `doc/winner_v022_on_nps_val.md`; `doc/winner_vs_tvd_aot_nps_analysis.md` |
| Key incremental experiments (Winner -> TVD ablation transfer) | `baseline / border10 / tracker` variants completed on NPS+AOT; `confirm` is running on NPS | `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv`; `tools/monitor_tvd_winner_ablation.ps1` |
| ESOD reproduction (high-resolution small-object detection) | Full VisDrone download + preprocessing done; pretrained-weight evaluation done; one `50 epochs` run completed (another run interrupted at epoch 19) | `doc/repro_esod.md`; `papers/ESOD/VisDrone/split/*.txt`; `papers/ESOD/runs/train/visdrone_esod_yolov5m_e50_b8_img1536_20260210_1700362/weights/best.pt` |

## 1) Objective and Constraints (Project Positioning)

| Item | Content |
|---|---|
| Final objective | Detect and stably track ultra-small targets (small UAVs/small obstacles) in complex backgrounds (urban/buildings/wires), and support path planning (collision avoidance) |
| Training constraint | Heavy training is allowed on high-end GPU workstations (current GPU: RTX 5090) |
| Inference constraint | Inference must run onboard UAV hardware (so we prioritize "heavy training, lightweight inference" strategies: two-stage/ROI/triggered compute, etc.) |
| Evaluation focus | AOT protocol emphasizes **low false alarms** (FAR/HFAR, FPPI) and Encounter DR (EDR@300); NPS is used as a complementary mAP/Recall benchmark |

## 2) Mapping Between Repos and Papers/Code

| Track | Paper/System | Year | Local PDF | Code location | commit/tag | Current status |
|---|---|---:|---|---|---|---|
| Baseline (video detection) | TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos | 2023 | `doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf` | `papers/TransVisDrone` | `8b3c760` | Reproduced and fully evaluated on NPS/AOT |
| Original baseline (NPS) | Fast and Robust UAV to UAV Detection and Tracking from Video (Li et al., TETC) | 2021 | N/A (DOI: `10.1109/TETC.2021.3104555`) | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking`; `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline` | `317e85a` | `pt_pipeline` runs and produces `eval_v1.json`; full Keras training reproduction not aligned end-to-end yet (used as sanity baseline) |
| Comparison (competition winner) | AIcrowd AOT Challenge Winner (`airborne-detection-starter-kit`) submission-v022 | 2022 | N/A (challenge solution) | `papers/AICrowd_AOT_Challenge_Winner/submission-v022/...` | tag `submission-v022` (see `doc/aicrowd_aot_winner_code.md`) | fulltest inference `172/172` done; official airborne metric recomputation done |
| High-resolution small-object detection | ESOD: Efficient Small Object Detection on High-Resolution Images | 2025 | `doc/ESOD Efficient Small Object Detection on High-Resolution Images.pdf` | `papers/ESOD` | `bde3571` | VisDrone preprocessing completed; pretrained evaluation completed; one 50e run completed |
| Detection-tracking collaboration | EDTC / AntiUAV600 | 2023 | `doc/Evidential Detection and Tracking Collaboration New Problem, Benchmark and Algorithm for Robust Anti-UAV System.pdf` | `papers/EDTC` | `d113d51` | Windows + modern PyTorch compatibility finished; smoke test only (dataset not public) |

## 3) Environment and Toolchain (PyTorch + uv Route)

| Environment | Location | Python | torch / torchvision | CUDA (driver) | Build method |
|---|---|---:|---|---|---|
| TransVisDrone venv | `papers/TransVisDrone/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | CUDA `13.1` (Driver `591.86`) | venv (ready) |
| ESOD venv | `papers/ESOD/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | same | `uv venv` + `uv pip install` (see `doc/repro_esod.md`) |
| EDTC venv | `papers/EDTC/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | same | `uv venv` + `uv pip install` (see `doc/repro_edtc.md`) |
| NPS pt_pipeline venv | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/.venv` | 3.11.x | `2.10.0+cu130` / `0.25.0+cu130` | same | `uv sync` (see `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md`) |

## 4) Original NPS Paper (Li-TETC 2021): Reproduction Issues and Handling

### 4.1 Two NPS Annotation Sources (TVD vs Li-TETC)

| Usage | Annotation source | Format highlights | Local path | Risk |
|---|---|---|---|---|
| TVD train/eval | Dogfight/Drone-Detection | `frame_no,num_obj,x1,y1,x2,y2,...` (one line per frame, multi-object allowed) | `datasets/Drone-Detection/annotations/NPS-Drones-Dataset/Clip_*.txt` | Not compatible with Li-TETC `time_layer`; mixing causes train/eval misalignment |
| Li-TETC original GT | Li repo (time_layer) | `frame_id y1 x1 y2 x2` (frame_id is 1-based; coordinate order is yxxy) | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925/Video_*_gt.txt`; `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Video_Annotation/Clip_*_gt.txt` | same |

### 4.2 Key Issues Found During Reproduction (and Our Handling)

| Issue | Impact | Our handling | Evidence/entry |
|---|---|---|---|
| Original repo depends on legacy Keras/TensorFlow + old CUDA | Hard to run directly on RTX 5090 + modern CUDA/PyTorch | Created `pt_pipeline` (uv + PyTorch) as runnable baseline decoupled from legacy stack | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md` |
| Two NPS annotation formats (Dogfight vs time_layer), with different frame indexing/coordinate order | Direct reuse can produce near-zero eval or shifted boxes | Explicit split: TVD uses Dogfight, `pt_pipeline` eval uses time_layer; parser rules fixed in `uav_annotations.py` | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/uav_annotations.py` |
| Highly sparse data (many empty frames) | Training can be dominated by negatives; thresholds and false-positive interpretation become misleading | Used stride sampling (v1: stride-3) + max-frame caps; report `fp_per_frame` in eval | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/run_v1_train.ps1`; `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| Tiny target scale | Recall is hard without high resolution/local focus | `pt_pipeline` includes optional motion-guided crops / track filtering (to combine with ESOD/ROI ideas later) | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md` |

### 4.3 PyTorch Sanity Baseline (Recomputable)

| split | videos | frames | IoU | score thr | P | R | FP/frame | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| holdout eval | 41-50 | 3177 | 0.5 | 0.3 | 0.149 | 0.227 | 2.377 | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| holdout eval | 41-50 | 3177 | 0.5 | 0.5 | 0.159 | 0.180 | 1.732 | same |

## 5) Key Timeline (From Zero to Current)

| Time (local) | Task | Output/Conclusion | Evidence path |
|---|---|---|---|
| 2026-02-06 ~ 02-08 | Pulled major paper repos and aligned reproduction workflows | `papers/TransVisDrone`, `papers/ESOD`, `papers/EDTC`, Winner code all available locally | `papers/*` |
| 2026-02-07 | Reviewed original NPS paper (Li-TETC 2021) code + built PyTorch `pt_pipeline` baseline | Runnable train/eval scripts and recomputable result (`eval_v1.json`) | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline` |
| 2026-02-08 ~ 02-10 | Downloaded/sorted paper PDFs + extracted text | PDFs + `doc_texts/*.txt` | `doc/*.pdf`; `doc_texts/*` |
| 2026-02-09 | AOT download + Windows compatibility fixes + full split evaluation | TVD AOT fulltest (172 flights) metrics recomputed | `doc/official_datasets_and_metrics.md` |
| 2026-02-10 | ESOD: full VisDrone download, preprocessing, pretrained evaluation | Preprocessing completed (masks + splits complete); pretrained metrics recorded | `doc/repro_esod.md`; `papers/ESOD/VisDrone/split/*.txt` |
| 2026-02-10 | Winner v022: AOT fulltest inference launched (detached) | fulltest inference `172/172` complete | `doc/aot_fulltest_run_status.md`; `tools/monitor_winner_v022_fulltest.ps1` |
| 2026-02-12 ~ present | Winner -> TVD ablation transfer (border / tracker / confirm) with dual-domain tests (AOT + NPS) | baseline/border10/tracker completed; confirm running on NPS | `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv` |
| 2026-02-14 | Winner v022: official airborne metric recomputation on AOT fulltest | final fulltest summary generated (HFAR/EDR/FPPI/AFDR) | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json` |

## 6) Why We Include TVD and Winner (and Cross-Domain Behavior)

| baseline | Why included | NPS (recomputed) | Paper/challenge alignment | AOT (official airborne recompute) | Key conclusion/insight | Evidence (verifiable paths) |
|---|---|---|---|---|---|---|
| TransVisDrone (TVD) | Most aligned public baseline for our task: explicit temporal modeling (VideoSwin), edge-throughput oriented, and covers both NPS and AOT | `mAP@0.5 ~ 0.95` on NPS (val=0.948, test=0.938) | Paper reports `AP@0.5IoU (NPS)=0.95`; close match indicates a credible reproduction chain | Can run and evaluate stably on AOT fulltest (172 flights) with official airborne metrics | TVD is a strong "reproducible + extensible" research baseline; however, AOT needs stronger low-false-positive engineering | NPS: `papers/TransVisDrone/runs/val/NPS_URAP_D/*/results.txt`; AOT: `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/*.json`; paper text: `doc/_texts/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.txt` |
| Winner (submission-v022) | AOT challenge winner from GitLab/AIcrowd; much stronger than academic baselines in the AOT metric regime (very strict false-alarm penalties) | On NPS-val, default setting outputs almost nothing (5944 frames, only 3 boxes, `AP@0.5 ~ 0.00029`); still weak after relaxation (`AP@0.5 ~ 0.0446`) | In subset10/challenge-style evaluation, can achieve extremely low false alarms with high EDR, showing strong AOT specialization | AOT fulltest recompute: `HFAR ~ 0.523`, `FPPI ~ 1.46e-05`, `EDR@300(All) ~ 0.989` | Winner's strong priors/threshold chain is highly effective for AOT but generalizes poorly to NPS (visible-spectrum differences in background/scale/statistics); best path is to transfer its low-FP post-processing ideas into TVD instead of replacing backbone directly | NPS: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val*/winner_v022/summaries/*.json`; AOT: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json`; subset10 comparison: `doc/compare_winner_v022_vs_transvisdrone_subset10.md` |

> Note: TVD paper AP is `AP@0.5IoU` (11-point PR operating points in the paper). Our NPS recomputation uses repo detection outputs (`mAP@0.5` / `mAP@0.5:0.95`). Values are close but not guaranteed to be strictly identical per metric definition.

## 7) Reproduction Results (Core Benchmark Tables)

### 7.1 TransVisDrone: NPS (Repo Protocol)

| split | P | R | mAP@0.5 | mAP@0.5:0.95 | Output dir/log |
|---|---:|---:|---:|---:|---|
| NPS val (best weights) | 0.901 | 0.881 | 0.948 | 0.464 | `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_val_best_aug_bs8_half`; `artifacts/logs/transvisdrone_nps_val_bs8_half_aug.log` |
| NPS test (best weights) | 0.916 | 0.901 | 0.938 | 0.468 | `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half`; `artifacts/logs/transvisdrone_nps_test_bs8_half_aug.log` |

### 7.1.1 Winner v022: NPS-val (Generalization test, recomputed with AP@0.5IoU)

| setting | num_images | num_gt_boxes | num_predictions | AP@0.5 | precision | recall | f1 | summary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| default | 5944 | 4656 | 3 | 0.000286 | 0.666667 | 0.000430 | 0.000859 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json` |
| relaxed | 5944 | 4656 | 923 | 0.044622 | 0.429036 | 0.085052 | 0.141961 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val_relaxed/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json` |

### 7.2 TransVisDrone: AOT fulltest (Official airborne metrics recomputation)

| variant | min_det_score | FPPI | FAR(HFAR) | AFDR(range<=700) | AFDR(area>200) | AFDR(area<=200) | EDR@300(Detection, All) | EDR@300(Tracking, All) | summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (conf=0.2) | 0.200195 | 0.262318 | 89.476744 | 0.868472 | 0.589017 | 0.089446 | 0.925714 | 0.925714 | `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/result_metrics_min_track_len_0_summary_far_89_47674_min_intruder_fl_dr_0p5_in_win_30.json` |
| wport_border10 (conf=0.2) | 0.200195 | 0.246720 | 84.593023 | 0.857961 | 0.582618 | 0.088723 | 0.925714 | 0.925714 | `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2_wport_border10/summaries/result_metrics_min_track_len_0_summary_far_84_59302_min_intruder_fl_dr_0p5_in_win_30.json` |
| Winner v022 (submission-v022) | 0.600198 | 1.46e-05 | 0.523256 | 0.955672 | 0.529184 | 0.021329 | 0.988571 | 0.988571 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_52326_min_intruder_fl_dr_0p5_in_win_30.json` |

### 7.3 Winner v022: Core AOT/NPS Observations

| Item | Conclusion | Evidence |
|---|---|---|
| AOT subset10 (official recompute) | Winner reaches `HFAR=0` and `EDR@300(All)=1.0` at `min_det_score ~ 0.602` | `doc/compare_winner_v022_vs_transvisdrone_subset10.md` |
| AOT fulltest (official recompute) | `HFAR ~ 0.523` and `EDR@300(All) ~ 0.989` (a major gain vs TVD baseline `HFAR ~ 89.48`) | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_52326_min_intruder_fl_dr_0p5_in_win_30.json` |
| NPS-val (direct transfer) | Default only 3 predicted boxes: `AP@0.5 ~ 0.00029`; after relaxation `AP@0.5 ~ 0.0446` (still far below TVD) | `doc/winner_v022_on_nps_val.md`; `doc/winner_vs_tvd_aot_nps_analysis.md`; `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val*/winner_v022/summaries/*.json` |

### 7.4 Winner -> TVD Ablation Transfer (same weights, same conf=0.2, dual eval on AOT+NPS)

> This ablation is driven by `tools/run_tvd_winner_ablation.ps1`; results are auto-collected into `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv`.

| variant | NPS P | NPS R | NPS mAP@0.5 | NPS mAP@0.5:0.95 | AOT FAR(HFAR) | AOT FPPI | EDR@300(Det,All) | EDR@300(Trk,All) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.890 | 0.890 | 0.945 | 0.475 | 89.4767 | 0.2623 | 0.9257 | 0.9257 | control |
| border10 | 0.897 | 0.863 | 0.921 | 0.465 | 84.5930 | 0.2467 | 0.9257 | 0.9257 | AOT FAR/FPPI drops, but NPS recall also drops |
| tracker (IoU tracker) | 0.890 | 0.890 | 0.945 | 0.475 | 5447.0930 | 0.2623 | 0.9257 | 0.0286 | tracking metrics collapse (needs further debugging/fix) |
| confirm | running | running | running | running | pending | pending | pending | pending | currently running (monitor: `tools/monitor_tvd_winner_ablation.ps1`) |

## 8) ESOD: Data Preparation and Training Status

### 8.1 VisDrone Preprocessing Integrity Check

| subset | images | labels(txt) | masks(npy) | split lines | Evidence |
|---|---:|---:|---:|---:|---|
| train | 6471 | 6471 | 6471 | 6471 | `tools/monitor_esod_visdrone_prepare.ps1` |
| val | 548 | 548 | 548 | 548 | same |
| test-dev | 1610 | 1610 | 1610 | 1610 | same |

### 8.2 Training Runs (50 epochs)

| run_name | Training status | results.txt line count | last epoch line | best/last weights | Evidence |
|---|---|---:|---|---|---|
| `...170036` | interrupted/not completed | 19 | `18/49 ...` | `papers/ESOD/runs/train/...170036/weights/{best,last}.pt` | `tools/monitor_esod_train_visdrone_yolov5m.ps1` |
| `...1700362` | completed | 50 | `49/49 ...` | `papers/ESOD/runs/train/...1700362/weights/{best,last}.pt` | `papers/ESOD/runs/train/...1700362/results.txt` |
