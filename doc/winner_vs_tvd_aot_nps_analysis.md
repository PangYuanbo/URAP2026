# Winner v022 vs TransVisDrone: AOT Strong, NPS Weak 的根因分析

## 1) 先看证据（不是拍脑袋）

### AOT（subset10，官方 airborne metrics 口径）

数据来源：

- `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_0_min_intruder_fl_dr_0p5_in_win_30.json`
- `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/baseline_transvisdrone/summaries/result_metrics_min_track_len_0_summary_far_102_0_min_intruder_fl_dr_0p5_in_win_30.json`
- `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_subset10/baseline_transvisdrone/score_0p602/summary_far_12_0_min_intruder_fl_dr_0p5_in_win_30.json`

关键结论：

- Winner v022 (`min_det_score≈0.602`)：
  - `HFAR=0`
  - `FPPI=0`
  - `EDR@300(All)=1.0`
  - `AFDR(range<=700)=0.9572`
- TVD baseline (`min_det_score≈0.200`)：
  - `HFAR=102`
  - `EDR@300(All)=0.9`
  - `AFDR=0.8835`
- TVD baseline (`score=0.602`, 对齐 winner 阈值)：
  - `HFAR=12`
  - `EDR@300(All)=0.7`
  - `AFDR=0.7102`

解释：在 AOT 的“低误报优先”评价体系下，Winner 的 operating point 设计明显更贴题。

### NPS（val，IoU=0.5）

默认 Winner v022：

- `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json`
- `AP@0.5=0.000286`
- `Recall=0.000430`
- `num_predictions=3/5944帧`

放宽门限（relaxed）后：

- `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_nps_val_relaxed/winner_v022/summaries/winner_v022_nps_val_ap_iou0.5_minScore0.json`
- `AP@0.5=0.044622`
- `Recall=0.085052`
- `Precision=0.4290`
- `num_predictions=923`

解释：不是“完全不能检”，而是默认门限链路把候选大量压没；即使放宽后仍远低于 NPS 原生方法，说明模型表征本体跨域不足。

## 2) 为什么 Winner 在 AOT 显著强于 TVD

### 2.1 目标函数匹配（最关键）

AOT 排名高度惩罚误报（HFAR/FPPI），Winner v022 是围绕这个目标做系统设计：

- 检测不是只看单帧框，而是“检测 + 轨迹确认”后再报。
- 这在 AOT 的评分函数里收益巨大（你的排行榜结果也印证）。

TVD baseline 更偏检测召回范式，阈值一压就掉 EDR，阈值一放就爆 HFAR。

### 2.2 Winner 的工程先验高度 AOT 化

代码证据：

- 灰度输入：`seg_test_nps.py:176`（`cv2.IMREAD_GRAYSCALE`）
- 固定几何：`seg_test_nps.py:38-39`（`2048x2448`）
- 全局运动补偿：`seg_tracker/seg_tracker.py:180`（`estimate_transformation_full`）
- 检测后多级门控：`seg_tracker/tracking.py:110,113,151`

这些先验在 AOT（远距离小目标、背景分布相对单一）非常有效。

### 2.3 Winner 的“低误报工具链”比 TVD 更完整

- 掩码边缘抑制：`predict_ensemble.py:46-54`
- 仅保留少量 top 候选并做 crop 精修：`predict_ensemble.py:74-90`
- 跟踪滞回与确认：`tracking.py:55-66,144-152`

本质上 Winner 是“把误报杀到很低”的系统工程，而不是只拼 backbone。

## 3) 为什么到了 NPS 几乎失效

### 3.1 跨域错配：数据分布和任务先验都变了

- AOT 偏“开阔背景 + 远距离目标 + 特定相机分布”
- NPS 偏“复杂城市纹理 + 更多干扰物 + 更强外观变化”

Winner 的 hard prior 在 NPS 上不再成立。

### 3.2 默认门限链路是直接瓶颈（已被消融验证）

消融结论（Clip_37）：

- 默认：`pred=0`
- 仅降检测阈值：`pred=0`
- 仅放宽跟踪门限：`pred=65`
- 全放宽：`pred=149`

对应问题代码：

- `tracking.py:110` 低于 `threshold_to_continue` 直接丢
- `tracking.py:113` 距离大于 `min_distance` 直接丢
- `tracking.py:151` 不满足 `min_track_size` 不上报

### 3.3 还有真实的模型表征跨域问题（不是纯工程问题）

即使 relaxed 后，NPS 也只有 `AP@0.5=0.0446`。  
这说明“门限修正”只解决了输出被压没的问题，没解决表征迁移。

### 3.4 NPS 适配里的额外损失（次要但存在）

- 拉伸到 AOT 尺寸：`seg_test_nps.py:187-188`
- 再缩回原图：`seg_test_nps.py:210-213`
- RGB->灰度丢失颜色判别信息：`seg_test_nps.py:176`

这些会进一步放大域差。

## 4) 结论：到底是 Winner 设计不行，还是我们工程接错了？

结论是“两者都有”，但主次明确：

- **第一主因**：Winner 是“为 AOT 指标强定制”的方案，不是跨域泛化方案。
- **第二主因**：默认跟踪门限链路在 NPS 上过硬，导致近零输出。
- **第三层**：NPS 适配（尺寸/灰度）有损，但不是根本原因（因门限放宽后已能显著恢复输出）。

## 5) 从 AOT 角度，TVD 最值得借鉴 Winner 的点

### 5.1 可直接借鉴（优先级高）

1. 运动补偿前置（CMC/GMC）  
目标：把“相机运动”从检测里剥离，减少假运动响应。

2. 跟踪确认式输出（hysteresis + min-track）  
目标：把单帧偶发 FP 挡在提交层，直接压 HFAR。

3. 双阶段检测（全图粗检 + ROI 精修）  
目标：在 tiny target 下提高定位稳定性。

4. 评分 operating point 按 AOT 指标标定  
目标：不是追 mAP，而是在 EDR/HFAR 曲线上选最佳点。

### 5.2 可选借鉴（中优先级）

1. 辅助头（如运动向量/距离或地平线先验）帮助关联  
2. 结构化 hard negative 策略（AOT 非计划目标/背景干扰）

## 6) 推荐的混合路线（TVD on AOT）

建议做成 TVD-Hybrid-AOT：

1. 保留 TVD 主干（时空表征能力）
2. 加 Winner 风格的 CMC + track-confirmation 头部后处理
3. 用 AOT 指标做阈值/门控网格搜索（而不是只看 mAP）
4. 在 subset10 先打出 “EDR>=0.95, HFAR<=5” 的可行点，再扩全量

这条路线的优点是：  
既保留 TVD 的可扩展性，又吸收 Winner 在 AOT 指标上的“工程致胜点”。
