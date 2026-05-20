# URAP 项目阶段性汇报（从 0 到当前）

## Reference

| 名称 | 在本文中指 | 对应论文/数据集/项目 | 本地入口（便于复核） |
|---|---|---|---|
| TransVisDrone / TVD | 我们复现的视频检测 baseline | ICRA 2023: *TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos* | 代码：`papers/TransVisDrone`；论文：`doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf` |
| AOT | Airborne Object Tracking 数据集（空中目标检测/跟踪挑战数据） | AWS Open Data Registry 的 AOT（本项目使用 part1） | 原始数据：`D:\URAP_datasets\AOT\part1`；TVD 读取格式：`D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest` |
| NPS (TVD) | NPS Drones 视频数据集（TVD 训练/评测协议） | 视频来自 Purdue/Bouman UAV Dataset；标注来自 Dogfight/Drone-Detection 格式 | 原始视频：`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos/Clip_*.mov`；标注：`datasets/Drone-Detection/annotations/NPS-Drones-Dataset/Clip_*.txt`；转换后：`D:\URAP_datasets\TransVisDrone\NPS` |
| NPS (Li-TETC) | NPS 原始论文（Li et al., TETC 2021）使用的 GT 协议（time_layer）与我们的可跑 baseline | IEEE TETC 2021: *Fast and Robust UAV to UAV Detection and Tracking from Video*（DOI: `10.1109/TETC.2021.3104555`） | 原始 GT：`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925/Video_*_gt.txt` + `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Video_Annotation/Clip_*_gt.txt`；PyTorch baseline：`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline`；结果：`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| Winner / submission-v022 | AOT Challenge 竞赛冠军方案代码 | AIcrowd Airborne Object Tracking Challenge winner 的 `airborne-detection-starter-kit`，tag `submission-v022` | 代码：`papers/AICrowd_AOT_Challenge_Winner/submission-v022`；说明：`doc/aicrowd_aot_winner_code.md` |



## 0) 结论速览

| 维度 | 结论/状态 | 证据（可复算路径） |
|---|---|---|
| 复现主线（TransVisDrone） | 已在 Windows + 新 PyTorch/CUDA 跑通，并按官方协议完成 NPS(val/test) 与 AOT(fulltest) 的指标复算 | `doc/official_datasets_and_metrics.md`；`papers/TransVisDrone/runs/val/NPS_URAP_D/*/results.txt`；`papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/*.json` |
| NPS 原始论文 baseline（Li-TETC 2021） | 已核对原 repo（Keras）并建立 `pt_pipeline`（PyTorch/uv）作为可跑 sanity baseline；同时厘清 NPS 两套标注来源/协议差异，避免“同名数据集混用” | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/README.md`；`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md`；`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| 竞赛 Winner（AOT 第一名 submission-v022） | AOT fulltest **172/172 flights 推理 + 官方 airborne 指标复算完成** | `tools/monitor_winner_v022_fulltest.ps1`；`papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json` |
| Winner 方法的“跨域泛化”（NPS） | 直接套用 Winner 在 NPS 上表现很差（大量候选被其门限链路压没；放宽门限后也仍远弱于 TVD） | `doc/winner_v022_on_nps_val.md`；`doc/winner_vs_tvd_aot_nps_analysis.md` |
| 关键增量实验（Winner→TVD 的消融移植） | 已完成 `baseline / border10 / tracker` 三个 variant 的 NPS+AOT 测试；`confirm` 正在跑 NPS | `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv`；`tools/monitor_tvd_winner_ablation.ps1` |
| ESOD 复现（高分辨率小目标） | VisDrone 全量下载+预处理完成；预训练权重评测完成；`50 epochs` 训练已完成一条 run（另一个 run 中断在 epoch 19） | `doc/repro_esod.md`；`papers/ESOD/VisDrone/split/*.txt`；`papers/ESOD/runs/train/visdrone_esod_yolov5m_e50_b8_img1536_20260210_1700362/weights/best.pt` |

## 1) 目标与约束（项目定位）

| 项 | 内容 |
|---|---|
| 最终目标 | 无人机在复杂背景（城市/建筑/线缆等）中对超小目标（小型无人机/小障碍物）进行检测与稳定跟踪，并服务于路径规划（避免碰撞） |
| 训练约束 | 可在高 GPU 工作站训练（当前 GPU：RTX 5090） |
| 推理约束 | 必须在机载设备推理（因此优先考虑“训练可重、推理要轻”的方法；两阶段/ROI/触发式计算等） |
| 评价关注 | AOT 协议更强调 **低误报（FAR/HFAR、FPPI）与 Encounter DR(EDR@300)**；NPS 侧以 mAP/Recall 作为补充 |

## 2) 仓库与论文/代码对应关系

| 方向 | 论文/系统 | 年份 | 本地 PDF | 代码位置 | commit/tag | 当前状态 |
|---|---|---:|---|---|---|---|
| Baseline（视频检测） | TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos | 2023 | `doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf` | `papers/TransVisDrone` | `8b3c760` | 已跑通 + NPS/AOT 指标复算完成 |
| 原始 baseline（NPS） | Fast and Robust UAV to UAV Detection and Tracking from Video（Li et al., TETC） | 2021 | N/A（DOI: `10.1109/TETC.2021.3104555`） | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking`；`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline` | `317e85a` | `pt_pipeline` 可跑并产出 `eval_v1.json`；Keras 全流程未做精确对齐复刻（当前作为 sanity baseline） |
| 对比（竞赛冠军） | AIcrowd AOT Challenge Winner（airborne-detection-starter-kit）submission-v022 | 2022 | N/A（竞赛方案） | `papers/AICrowd_AOT_Challenge_Winner/submission-v022/...` | tag `submission-v022`（说明见 `doc/aicrowd_aot_winner_code.md`） | fulltest 推理 172/172 完成；官方 airborne 指标复算完成 |
| 小目标高分辨率检测 | ESOD: Efficient Small Object Detection on High-Resolution Images | 2025 | `doc/ESOD Efficient Small Object Detection on High-Resolution Images.pdf` | `papers/ESOD` | `bde3571` | VisDrone 预处理完成；预训练评测完成；1 条 50e 训练完成 |
| 检测-跟踪协作 | EDTC / AntiUAV600 | 2023 | `doc/Evidential Detection and Tracking Collaboration New Problem, Benchmark and Algorithm for Robust Anti-UAV System.pdf` | `papers/EDTC` | `d113d51` | 已做 Windows+新 PyTorch 兼容；仅 smoke test（数据集未公开） |

## 3) 环境与工具链（PyTorch 路线 + uv）

| 环境 | 位置 | Python | torch / torchvision | CUDA(驱动) | 创建方式 |
|---|---|---:|---|---|---|
| TransVisDrone venv | `papers/TransVisDrone/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | CUDA `13.1`（Driver `591.86`） | venv（已可用） |
| ESOD venv | `papers/ESOD/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | 同上 | `uv venv` + `uv pip install`（见 `doc/repro_esod.md`） |
| EDTC venv | `papers/EDTC/.venv` | 3.10.19 | `2.10.0+cu130` / `0.25.0+cu130` | 同上 | `uv venv` + `uv pip install`（见 `doc/repro_edtc.md`） |
| NPS pt_pipeline venv | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/.venv` | 3.11.x | `2.10.0+cu130` / `0.25.0+cu130` | 同上 | `uv sync`（见 `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md`） |



## 4) NPS 原始论文（Li-TETC 2021）复现：问题与处理（避免同名数据集混淆）

### 4.1 NPS 两套标注来源对照（TVD vs Li-TETC）

| 用途 | 标注来源 | 格式要点 | 本地路径 | 风险 |
|---|---|---|---|---|
| TVD 训练/评测 | Dogfight/Drone-Detection | `frame_no,num_obj,x1,y1,x2,y2,...`（每帧一行，可多目标） | `datasets/Drone-Detection/annotations/NPS-Drones-Dataset/Clip_*.txt` | 与 Li-TETC 的 `time_layer` 协议不兼容；混用会导致训练/评测错位 |
| Li-TETC 原始论文 GT | Li repo (time_layer) | `frame_id y1 x1 y2 x2`（frame_id 1-based；坐标顺序为 yxxy） | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925/Video_*_gt.txt`；`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Video_Annotation/Clip_*_gt.txt` | 同上 |

### 4.2 复现中发现的关键问题（以及我们怎么处理）

| 问题 | 影响 | 我们的处理 | 证据/入口 |
|---|---|---|---|
| 原 repo 依赖为 legacy Keras/TensorFlow + 旧 CUDA | 在 RTX 5090 + 新 CUDA/PyTorch 环境下难以直接复跑 | 新建 `pt_pipeline`（uv + PyTorch）做可跑 baseline（不与 legacy 环境耦合） | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md` |
| NPS 标注存在两种格式（Dogfight vs time_layer），且 frame index/坐标顺序不同 | 若直接替换/复用标注，会出现“模型输出正常但评测近似为 0”或框位置错位 | 明确区分：TVD 使用 Dogfight；`pt_pipeline` 评测使用 time_layer；并在 `uav_annotations.py` 固化解析规则 | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/uav_annotations.py` |
| 数据极度稀疏（大量 empty frames） | 训练易被负样本主导，且阈值/误报解释容易误导 | 训练采用 stride sampling（v1：stride-3）+ 限制 max frames；评测输出 `fp_per_frame` | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/run_v1_train.ps1`；`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| 目标尺度极小 | 需要高分辨率/局部聚焦，否则 recall 难提升 | `pt_pipeline` 提供 motion-guided crops / track filtering 作为可选改进（后续可与 ESOD/ROI 思路结合） | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/README.md` |

### 4.3 PyTorch sanity baseline（可复算）

| split | videos | frames | IoU | score thr | P | R | FP/frame | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| holdout eval | 41-50 | 3177 | 0.5 | 0.3 | 0.149 | 0.227 | 2.377 | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json` |
| holdout eval | 41-50 | 3177 | 0.5 | 0.5 | 0.159 | 0.180 | 1.732 | 同上 |

## 5) 从 0 到当前：关键工作时间线（表格化）

| 时间（本机） | 任务 | 产出/结论 | 证据路径 |
|---|---|---|---|
| 2026-02-06 ~ 02-08 | 拉起各论文代码仓并梳理复现链路 | `papers/TransVisDrone`、`papers/ESOD`、`papers/EDTC`、Winner 代码落盘 | `papers/*` |
| 2026-02-07 | NPS 原始论文（Li-TETC 2021）代码复核 + PyTorch `pt_pipeline` baseline | 可跑训练/评测脚本与可复算结果（`eval_v1.json`） | `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline` |
| 2026-02-08 ~ 02-10 | 下载/整理论文 PDF + 文本抽取 | 论文 PDF + `doc_texts/*.txt` | `doc/*.pdf`；`doc_texts/*` |
| 2026-02-09 | AOT 下载 + Windows 兼容修复 + 分片全量评测 | TVD AOT fulltest（172 flights）指标复算完成 | `doc/official_datasets_and_metrics.md` |
| 2026-02-10 | ESOD：VisDrone 全量下载、预处理与预训练评测 | 预处理完成（masks 与 split 完整）；预训练权重指标已记录 | `doc/repro_esod.md`；`papers/ESOD/VisDrone/split/*.txt` |
| 2026-02-10 | Winner v022：AOT fulltest 推理启动（detached） | fulltest 推理 172/172 完成 | `doc/aot_fulltest_run_status.md`；`tools/monitor_winner_v022_fulltest.ps1` |
| 2026-02-12 ~ 至今 | Winner→TVD 消融移植（border / tracker / confirm）并双域测试（AOT+NPS） | baseline/border10/tracker 完成；confirm 正在跑 NPS | `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv` |
| 2026-02-14 | Winner v022：AOT fulltest 官方 airborne 指标复算 | 产出 fulltest 指标 summary（HFAR/EDR/FPPI/AFDR） | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json` |

## 6) 为什么纳入 TVD 与 Winner（以及跨域现象）

| baseline | 为什么纳入 | NPS（复算） | 对照（论文/竞赛） | AOT（官方 airborne 复算） | 关键结论/启示 | 证据（可复核路径） |
|---|---|---|---|---|---|---|
| TransVisDrone (TVD) | 与我们任务最对齐的公开视频检测 baseline：显式利用时序（VideoSwin）并强调 edge throughput；且同时覆盖 NPS/AOT | NPS 上 `mAP@0.5≈0.95`（val=0.948，test=0.938） | 论文报告 `AP@0.5IoU (NPS)=0.95`；数值接近，说明复现链路可信 | 能在 AOT fulltest（172 flights）稳定跑完并复算 airborne 指标（HFAR/EDR） | TVD 是“可复现 + 可扩展”的研究 baseline；但 AOT 上需要进一步引入“低误报工程” | NPS：`papers/TransVisDrone/runs/val/NPS_URAP_D/*/results.txt`；AOT：`papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/*.json`；论文文本：`doc/_texts/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.txt` |
| Winner (submission-v022) | AOT 数据集对应 GitLab/AIcrowd 竞赛第一名方案：在 AOT 指标体系（极端惩罚误报）下明显强于学术 baseline | NPS-val 默认几乎不出框（5944 帧仅 3 框，`AP@0.5≈0.00029`）；放宽门限后仍弱（`AP@0.5≈0.0446`） | subset10/竞赛体系下可做到极低误报 + 高 EDR，体现其 AOT 定制强度 | AOT fulltest 复算：`HFAR≈0.523`、`FPPI≈1.46e-05`、`EDR@300(All)≈0.989` | Winner 的强先验/门限链路对 AOT 极有效，但跨域到 NPS（可见光、背景/尺度/统计不同）泛化性差；后续更适合“吸收其低误报后处理思想”而非直接替换 backbone | NPS：`papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val*/winner_v022/summaries/*.json`；AOT：`papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/*.json`；subset10 对照：`doc/compare_winner_v022_vs_transvisdrone_subset10.md` |

> 注：TVD 论文中 AP 为 `AP@0.5IoU`（文中说明为 11-point PR operating points）；本项目 NPS 侧复算使用 repo 的检测评测输出（`mAP@0.5` / `mAP@0.5:0.95`）。两者数值接近但不保证逐项严格一致。

## 7) 复现结果（核心指标表）

### 7.1 TransVisDrone：NPS（按 repo protocol）

| split | P | R | mAP@0.5 | mAP@0.5:0.95 | 输出目录/日志 |
|---|---:|---:|---:|---:|---|
| NPS val（best weights） | 0.901 | 0.881 | 0.948 | 0.464 | `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_val_best_aug_bs8_half`；`artifacts/logs/transvisdrone_nps_val_bs8_half_aug.log` |
| NPS test（best weights） | 0.916 | 0.901 | 0.938 | 0.468 | `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half`；`artifacts/logs/transvisdrone_nps_test_bs8_half_aug.log` |

### 7.1.1 Winner v022：NPS-val（泛化测试，按 AP@0.5IoU 复算）

| setting | num_images | num_gt_boxes | num_predictions | AP@0.5 | precision | recall | f1 | summary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| default | 5944 | 4656 | 3 | 0.000286 | 0.666667 | 0.000430 | 0.000859 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json` |
| relaxed | 5944 | 4656 | 923 | 0.044622 | 0.429036 | 0.085052 | 0.141961 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val_relaxed/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json` |

### 7.2 TransVisDrone：AOT fulltest（官方 airborne metrics 复算）

| variant | min_det_score | FPPI | FAR(HFAR) | AFDR(range<=700) | AFDR(area>200) | AFDR(area<=200) | EDR@300(Detection, All) | EDR@300(Tracking, All) | summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline（conf=0.2） | 0.200195 | 0.262318 | 89.476744 | 0.868472 | 0.589017 | 0.089446 | 0.925714 | 0.925714 | `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/summaries/result_metrics_min_track_len_0_summary_far_89_47674_min_intruder_fl_dr_0p5_in_win_30.json` |
| wport_border10（conf=0.2） | 0.200195 | 0.246720 | 84.593023 | 0.857961 | 0.582618 | 0.088723 | 0.925714 | 0.925714 | `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2_wport_border10/summaries/result_metrics_min_track_len_0_summary_far_84_59302_min_intruder_fl_dr_0p5_in_win_30.json` |
| Winner v022（submission-v022） | 0.600198 | 1.46e-05 | 0.523256 | 0.955672 | 0.529184 | 0.021329 | 0.988571 | 0.988571 | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_52326_min_intruder_fl_dr_0p5_in_win_30.json` |

### 7.3 Winner v022：AOT/NPS 核心观察（对照）

| 项 | 结论 | 证据 |
|---|---|---|
| AOT subset10（官方复算） | Winner 在 `min_det_score≈0.602` 时做到 `HFAR=0` 且 `EDR@300(All)=1.0` | `doc/compare_winner_v022_vs_transvisdrone_subset10.md` |
| AOT fulltest（官方复算） | `HFAR≈0.523` 且 `EDR@300(All)≈0.989`（相比 TVD baseline 的 `HFAR≈89.48` 提升极大） | `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_52326_min_intruder_fl_dr_0p5_in_win_30.json` |
| NPS-val（直接套用） | 默认仅 3 个预测框：`AP@0.5≈0.00029`；放宽门限后 `AP@0.5≈0.0446`（仍显著落后于 TVD） | `doc/winner_v022_on_nps_val.md`；`doc/winner_vs_tvd_aot_nps_analysis.md`；`papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val*/winner_v022/summaries/*.json` |

### 7.4 Winner→TVD 消融移植（同权重，同 conf=0.2，AOT+NPS 双评测）

> 该消融由 `tools/run_tvd_winner_ablation.ps1` 驱动，结果自动汇总到 `papers/TransVisDrone/runs/ablation/winner_port_v1/results.csv`。

| variant | NPS P | NPS R | NPS mAP@0.5 | NPS mAP@0.5:0.95 | AOT FAR(HFAR) | AOT FPPI | EDR@300(Det,All) | EDR@300(Trk,All) | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.890 | 0.890 | 0.945 | 0.475 | 89.4767 | 0.2623 | 0.9257 | 0.9257 | 对照组 |
| border10 | 0.897 | 0.863 | 0.921 | 0.465 | 84.5930 | 0.2467 | 0.9257 | 0.9257 | AOT FAR/FPPI 降低，但 NPS recall 下降 |
| tracker（IoU tracker） | 0.890 | 0.890 | 0.945 | 0.475 | 5447.0930 | 0.2623 | 0.9257 | 0.0286 | tracking 指标严重崩坏（需进一步排查/修正） |
| confirm | running | running | running | running | pending | pending | pending | pending | 当前在跑（监控：`tools/monitor_tvd_winner_ablation.ps1`） |

## 8) ESOD：数据准备与训练状态

### 8.1 VisDrone 预处理完整性检查

| subset | images | labels(txt) | masks(npy) | split 行数 | 证据 |
|---|---:|---:|---:|---:|---|
| train | 6471 | 6471 | 6471 | 6471 | `tools/monitor_esod_visdrone_prepare.ps1` |
| val | 548 | 548 | 548 | 548 | 同上 |
| test-dev | 1610 | 1610 | 1610 | 1610 | 同上 |

### 8.2 训练 runs（50 epochs）

| run_name | 训练状态 | results.txt 行数 | last epoch 行 | best/last 权重 | 证据 |
|---|---|---:|---|---|---|
| `...170036` | 中断/未完成 | 19 | `18/49 ...` | `papers/ESOD/runs/train/...170036/weights/{best,last}.pt` | `tools/monitor_esod_train_visdrone_yolov5m.ps1` |
| `...1700362` | 已完成 | 50 | `49/49 ...` | `papers/ESOD/runs/train/...1700362/weights/{best,last}.pt` | `papers/ESOD/runs/train/...1700362/results.txt` |
