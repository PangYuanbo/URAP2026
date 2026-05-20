# AIcrowd Airborne Object Tracking Challenge 第一名代码定位（Detection+Tracking Benchmark）

> 目的：把“第一名的可复现代码”在本机落盘，并标出入口/关键模块，方便你直接阅读和二次改造。

## Leaderboard 对应仓库

- 选手（Detection+Tracking Benchmark 排名第 1）：`dmytro_poplavskiy`
- 提交信息里给出的仓库（GitLab, AIcrowd）：`airborne-detection-starter-kit`

该仓库的 Tags 页面公开了多次提交快照，我已把最新的两个 `submission` tag 都下载并解压到本机（包含模型权重，所以每个压缩包约 974MB）。

## 已下载的两个候选“第一名提交快照”

1. `submission-v022`（commit `1fbc227686e5721535eefc9bd76e4f523c697c7f`，tag 标题：`Threshold 0.6 submission`）
   - 本地路径：`papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022`
2. `submission-v021`（commit `f68d48e562a76d1ec15c6b14a73d05ef270d0fcc`，tag 标题：`Conservative submission with 0.75 threshold`）
   - 本地路径：`papers/AICrowd_AOT_Challenge_Winner/submission-v021/airborne-detection-starter-kit-submission-v021`

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

## 下一步（我可以继续做）

1. 用 `submission-v022` 在 `AOT part1` 上跑一遍生成 `result.json`，并用官方 airborne metrics 复算（和你给的排行榜指标口径一致），得到可对比的 `EDR/HFAR/FPPI/AFDR`。
2. 如果需要，也可以把 `v021` 同样复算，直接画出“阈值-EDR/HFAR”曲线，方便你选推理侧的 operating point。

