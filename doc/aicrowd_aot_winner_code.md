# AIcrowd Airborne Object Tracking Challenge 第一名代码定位（Detection+Tracking Benchmark）

> 目的：把“第一名的可复现代码”在本机落盘，并标出入口/关键模块，方便你直接阅读和二次改造。

## Leaderboard 对应仓库

- 选手（Detection+Tracking Benchmark 排名第 1）：`dmytro_poplavskiy`
- 提交信息里给出的仓库（GitLab, AIcrowd）：`airborne-detection-starter-kit`

AIcrowd winners 页面公开链接到该 GitLab 项目。当前机器上，
unauthenticated `git clone` 会要求用户名，GitLab archive 下载会超时；
但 GitLab API 可以读取公开 tag 和 source tree。因此本机现在保存的是
`submission-v022` 的**源码快照**，不是完整 git checkout，也不包含大模型
权重。

## 当前本机快照

1. `submission-v022`（commit `1fbc227686e5721535eefc9bd76e4f523c697c7f`，tag 标题：`Threshold 0.6 submission`）
   - 本地路径：`papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022`
   - 快照记录：`.urap_snapshot.json`
   - 已下载：100 个源码/文档/配置文件
   - 已跳过：13 个 `.pth`/`.pyc` 二进制文件，主要是 winner 模型权重
   - 权重清单：`runs/window_accuracy/aicrowd_lfs_weight_inventory.md`
   - 缺失权重：10 个 Git LFS 模型文件，总计约 1.01 GiB
   - 下载工具：`tools/download_aicrowd_lfs_weights.py`
   - 当前下载状态：无 token 时 GitLab LFS batch 返回 `auth_required`，报告在 `runs/window_accuracy/aicrowd_lfs_weight_download_report.json`

2. `submission-v021`（commit `f68d48e562a76d1ec15c6b14a73d05ef270d0fcc`，tag 标题：`Conservative submission with 0.75 threshold`）
   - 当前未下载；只通过 GitLab API 确认了 tag 元数据

> 哪个 tag 对应“第一名那次提交”：从 tag 描述看，`v022`/`v021` 是同一套方法的两次阈值版本；`v022` 更激进（更高召回），`v021` 更保守（更低误报）。如果你要对照排行榜分数，通常更可能是 `v022`（但需要在相同评测集上验证）。

## 代码入口在哪里

这套提交不是跑 `test.py`，而是：

- 入口脚本：`run.sh`
- 实际执行：`python seg_test.py`
- 主要实现：
  - `seg_test.py`：每个 flight 逐帧读灰度图，调用 tracker 输出 `(track_id, conf, cx, cy, w, h, offset)` 再写入提交格式。
  - `seg_tracker/seg_tracker.py`：Detector + Tracker 的核心逻辑（包含 ensemble + offset tracking 的阈值）。
  - `seg_tracker/predict_ensemble.py`：多模型组合推理（看起来是 full-res + crop models 的融合）。

## `v021` vs `v022` 的关键差异（非常核心）

差异主要在 `seg_tracker/seg_tracker.py`（阈值策略）：

- `full_res_threshold`
  - `v021`: `0.5`
  - `v022`: `0.35`
- `threshold_to_find` / `threshold_to_continue`（用于跟踪器建立/延续轨迹）
  - `v021`: `0.75`
  - `v022`: `0.6`

直觉解释：`v022` 更容易“启动和维持”track，因此更偏召回；`v021` 更偏低误报（HFAR/FPPI）。

## 本机跑起来需要的数据路径（和我们已下载的数据一致）

我们之前全量跑 TransVisDrone 的 AOT part1 用的是：

- AOT part1 数据根目录：`D:/URAP_datasets/AOT/part1`
  - 原始帧：`D:/URAP_datasets/AOT/part1/Images/<flight_id>/*.png`
  - 官方 GT：`D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json`

这份 winner 提交脚本默认从环境变量读取帧目录：

- `TEST_DATASET_PATH` 应该指向：`D:/URAP_datasets/AOT/part1/Images`

NPS 是平铺帧目录（例如 `Clip_001_00001.png`），而 winner 的
`seg_test.py` 期望 `TEST_DATASET_PATH/<flight_id>/*.png`。本仓库已补
`tools/prepare_aicrowd_nps_flight_dirs.py`，会把平铺 NPS 帧按 `Clip_001`
这类前缀分组成 flight/clip 目录。`tools/run_winner_v022_nps_val.ps1`
现在调用该准备步骤后再运行实际存在的 `seg_test.py`，不再依赖缺失的
`seg_test_nps.py`。

## 下一步（我可以继续做）

1. 用 `submission-v022` 在 `AOT part1` 上跑一遍生成 `result.json`，并用官方 airborne metrics 复算（和你给的排行榜指标口径一致），得到可对比的 `EDR/HFAR/FPPI/AFDR`。
2. 如果需要，也可以把 `v021` 同样复算，直接画出“阈值-EDR/HFAR”曲线，方便你选推理侧的 operating point。

## 接入 +/-3s 曲线

NPS winner 推理建议用 detached runner：

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_winner_v022_nps_val_detached.ps1 `
  -DatasetPath D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val

powershell -ExecutionPolicy Bypass -File tools\monitor_winner_v022_nps_val.ps1 `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val
```

然后把 NPS winner `result.json` 接入曲线：

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
  --score-threshold 0.25 \
  --out runs/window_accuracy/papers/aicrowd_winner_nps_val
```

官方 AOT GT 可以直接进入当前曲线工具：

```bash
python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json \
  --gt-format aot-gt-json \
  --pred papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_aot_part1/part1 \
  --pred-format aot-json \
  --fps 10 \
  --window-seconds 3 \
  --score-threshold 0.25 \
  --out papers/AICrowd_AOT_Challenge_Winner/runs/window_accuracy/aot_part1
```

这里 `aot-gt-json` 会读 `samples.*.entities[].blob.frame` 和 `bb=[left,
top, width, height]`。当 winner `result.json` 只有 `img_name`、没有显式
`frame` 时，曲线工具会用 GT 里的 `img_name -> flight_id/frame` 映射补齐，
避免把预测按 JSON 列表下标错位。
