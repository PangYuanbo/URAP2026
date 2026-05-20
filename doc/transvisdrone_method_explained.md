# TransVisDrone (ICRA 2023) 论文思路速读笔记

论文: **TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos** (`doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf`)

目标: 面向“机载相机视频”的 **drone-to-drone detection**。强调:
- 小目标: 目标无人机通常只占画面极小比例
- 运动与模糊: 目标机/自机都在运动，导致抖动、拖影、快速尺度变化
- 背景复杂/光照变化: below-the-horizon 的杂波、植被/车辆运动、强光等
- 部署约束: 需要高吞吐，能在边缘设备上实时推理

## 1. 总体思路 (一句话)
用 **YOLOv5 风格的多尺度 CNN** 抽取每帧空间特征，再用 **Video Swin 的 3D 窗口注意力** 在短时间窗口内建模运动/时序一致性，端到端直接输出每帧检测框序列，避免“多阶段裁剪 + CPU 跟踪/连通域”等非端到端 pipeline。

## 2. 框架分解 (对应论文 Fig.2 / Sec.III)

### 2.1 输入: 短 clip + 推理时滑窗
- 从任意 flight video 里取一个 temporal window `τ` 的 clip `S(x,y,t)`
- 训练: 从随机时间点采样
- 推理: 用滑窗覆盖整段视频 (保持短期时序上下文)

仓库对应:
- `papers/TransVisDrone/utils/datasets.py` 的 `sample_temporal_frames(...)` 体现了 “取 num_frames 帧” 的 clip 采样逻辑
- `papers/TransVisDrone/utils/datasets_inference.py` 有对应的推理侧采样

### 2.2 Temporally Consistent Augmentation (TCA)
论文强调: 对一个 clip 的每一帧必须施加 **相同的随机增强**，否则会破坏时序关系，学不到“真实运动”。

仓库对应:
- `papers/TransVisDrone/utils/augmentations.py` 里的一组 `*_temporal` 增强与 `AlbumentationsTemporal`

### 2.3 空间特征: CSPDarkNet-53 + 多尺度特征 (P3/P4/P5)
小目标检测要保留细节，论文选择 YOLO 系列常用的 CSPDarkNet-53，并通过 SPP 等结构得到多尺度特征。

仓库对应:
- YOLOv5 backbone/head 配置 YAML: `papers/TransVisDrone/models/yolov5l-xs-tph-temporal.yaml`
- 多尺度 head (实现里还加了更高分辨率的 P2/4 分支来照顾更小目标)

### 2.4 时序模块: Video Swin 的 3D Window Attention
关键点不是“对整幅图做全局时序注意力”，而是:
- 在特征图上做 **局部 3D 窗口** (空间 8x8 + 时间 τ) 的注意力
- 再用 **shifted window** 跨窗口建立联系

直觉:
- 连续帧的位移通常不大，局部时空注意力更划算
- 能在 blur/短时遮挡/形变下，让目标在时序上“被记住”，减少漏检与抖动

仓库对应:
- `papers/TransVisDrone/models/common.py` 中的 `C3STTR` / `C3Temporal`：在 YOLO 的 C3 模块后接 `SwinTransformerBlock3D(...)`
- `papers/TransVisDrone/models/yolo.py`：把这些模块串到检测 head 里

### 2.5 输出与损失
输出: **逐帧的目标检测框序列** (论文用于 AP@0.5IoU 等检测指标；在 AOT 上也做 FPPI/EDR 等安全相关指标)。

损失: 论文直接用 YOLO 的标准损失:
- objectness
- classification
- localization

## 3. 论文里最重要的“可落地结论”
1) **短期时序就够**: `τ` 从 1 到 3 有明显收益，继续加到 5 收益变小但仍有提升 (Sec.IV-E.1)  
2) **分辨率 vs 吞吐可控**: 640 分辨率 AP 小幅下降但 FPS 大幅提升，利于机载实时 (Sec.IV-E.2)  
3) **TCA 有用**: 时序一致增强比逐帧不同增强更稳 (Sec.IV-E.4)  
4) **边缘部署可行**: 论文报告 Jetson Xavier NX 上 640 分辨率可实时 (不依赖复杂 TensorRT 工程) (Sec.V-A)

## 4. 和我们当前复现/对比相关的定位点
- 论文方法本质是 **Video-based detector (clip 输入)**，不是把视频当成独立单帧图像来做。
- 如果你关心“复杂城市背景 + 极小障碍物”，TransVisDrone 给我们的启发主要在:
  - 用低成本时序模块提升“稳定性与召回”
  - 通过多尺度 head (尤其更高分辨率的浅层特征) 兜住 tiny objects

