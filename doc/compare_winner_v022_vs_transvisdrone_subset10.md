# AOT Subset10: Winner (submission-v022) vs TransVisDrone Baseline 对比

本对比用于快速判断：AICrowd AOT Challenge 第一名方案（`submission-v022`）在 AOT 子集上的表现，和我们复现的 TransVisDrone baseline 的差异。

## 子集定义
- 子集 flights: `papers/TransVisDrone/aot_flight_ids/testflightidsfull1.json` 的前 10 个 flight id
- 10 个 flight id（按文件顺序）:
  - `40a522741b434597869ea7d751ad6c7d`
  - `6fe6d5fe0309403783bb9d18782c9288`
  - `65e39dcc372d4220b6ba33721f69e3f0`
  - `0cc69f4b81f242c6839889016ff3a942`
  - `0a98e01c4aa84307a82dcba3e5af86a3`
  - `4761fc2851a34885b48d5e9bf3fbae86`
  - `01a93940244447229030f54ae0ab69af`
  - `49f4d9ca19004b81b905db6acab3d7d8`
  - `6896849b32f844b4bbea5361c71f8783`
  - `45ea01283a8b4e4b80210a164fbc9baf`

## Ground Truth
- 原始 AOT part1 images: `D:/URAP_datasets/AOT/part1/Images`
- 子集 GT（从官方 `groundtruth.json` 过滤得到）:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/gt_subset10_first10/groundtruth.csv`
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/gt_subset10_first10/groundtruth_with_encounters_maxRange700_maxGap3_minEncLen30.csv`
  - unique frames: `11987`

## Winner 方案 (submission-v022)

### 结果生成
- per-flight 输出: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_subset10/subset10/<flight_id>/result.json`
- 合并为一个 `result.json`（供 airborne metrics 读取）:
  - 脚本: `tools/merge_airborne_results.py`
  - 输出: `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_subset10/subset10/result.json`

### 官方 airborne metrics（子集）复算
- Summary JSON:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_0_min_intruder_fl_dr_0p5_in_win_30.json`
- 关键值（该 summary 的 score 阈值等于该算法输出的最小分数 `min_det_score≈0.602`）:
  - `min_det_score`: `0.601990`
  - `HFAR`: `0.0`
  - `FPPI`: `0.0`
  - `AFDR(range<=700)`: `0.957227` (`1298/1356`)
  - `EDR@300(All, Tracking)`: `1.0` (`10/10`)

## TransVisDrone baseline（子集）

### A) Baseline @ min_det_score≈0.2（原先子集对齐点）
- Summary JSON:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/baseline_transvisdrone/summaries/result_metrics_min_track_len_0_summary_far_102_0_min_intruder_fl_dr_0p5_in_win_30.json`
- 关键值:
  - `min_det_score`: `0.200195`
  - `HFAR`: `102.0`
  - `FPPI`: `0.281722`
  - `AFDR(range<=700)`: `0.883481` (`1198/1356`)
  - `EDR@300(All, Tracking)`: `0.9` (`9/10`)

### B) Baseline @ score=0.602（和 winner 同阈值做更公平对比）
- Summary JSON:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/baseline_transvisdrone/score_0p602/summary_far_12_0_min_intruder_fl_dr_0p5_in_win_30.json`
- 关键值:
  - `min_det_score`: `0.602`
  - `HFAR`: `12.0`
  - `FPPI`: `0.006757`
  - `AFDR(range<=700)`: `0.710177` (`963/1356`)
  - `EDR@300(All, Tracking)`: `0.7` (`7/10`)

## 结论（subset10）
- Winner (`submission-v022`) 在 subset10 上明显更强：在 `min_det_score≈0.602` 的运行点达到 `HFAR=0` 且 `EDR@300(All)=1.0`。
- TransVisDrone baseline 在低阈值下召回较高但误报高；把阈值抬到 0.602 后，误报下降但 `AFDR/EDR` 明显掉。

备注:
- 这里是 **subset10 快速对比**（10 flights）。如果要做最终结论，还需要在更大子集甚至完整 test set 上复算。

