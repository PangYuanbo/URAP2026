# TransVisDrone 各模块入门资料 (新手友好)

这份清单按 TransVisDrone 的模块拆开，每个模块给 1 个“最好理解”的主推荐，并给 1 个备选（论文/视频二选一）。

## 0) 先补齐的基础 (建议先看)
- 主推荐(视频): **Roboflow** 的 IoU / Precision-Recall / mAP 系列讲解（YouTube；B 站也有人搬运）
- 备选(书/课程): **Dive into Deep Learning (D2L)** 的目标检测章节（含 IoU、anchor、NMS、mAP）

## 1) 输入 Clip + 滑窗推理 (为什么不是“逐帧单图”)
- 主推荐(视频/教程关键词): 搜索 `video clip sliding window inference deep learning` 或 `PyTorchVideo tutorial`（先把 “T x H x W x C” 的输入范式搞清楚）
- 备选(论文): **SlowFast Networks for Video Recognition**（不做检测，但对“视频模型如何吃 clip、为何要短时序窗口”很直观）

## 2) Temporally Consistent Augmentation (TCA, 时序一致增强)
- 主推荐(视频/实践): **Albumentations** 入门教程（看懂 random crop / color jitter / perspective；再把“同一随机参数复制到每一帧”实现出来）
- 备选(论文): **RandAugment: Practical Automated Data Augmentation with a Reduced Search Space**（把“增强为什么有效、怎么系统化”理解清楚）

## 3) 空间特征提取 (YOLOv5 系列 + CSPDarkNet / CSPNet)
- 主推荐(视频): `Ultralytics YOLOv5 tutorial`（训练、推理、数据格式、anchors 的基本概念讲得最贴工程）
- 备选(论文): **YOLOv3: An Incremental Improvement**（一阶段检测、anchors、多尺度 head 的核心思想；比 YOLOv1 更贴近现代实现）

## 4) 多尺度特征 (FPN/金字塔, 解决 tiny object)
- 主推荐(论文): **Feature Pyramid Networks for Object Detection (FPN)**（经典且相对好读，理解 P3/P4/P5 这类多尺度特征从哪来）
- 备选(视频/关键词): 搜索 `Feature Pyramid Network explained`（很多 10-20 分钟的图解视频）

## 5) Transformer 基础 (为理解 Swin/VideoSwin 打底)
- 主推荐(图文): **The Illustrated Transformer (Jay Alammar)**（强烈建议先看这个再读任何 Transformer 论文）
- 备选(视频/课程): `李宏毅 Transformer`（B 站/YouTube 都很好找，适合中文入门）

## 6) Swin Transformer (shifted window attention, 为什么比全局注意力省)
- 主推荐(论文): **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
- 备选(视频/关键词): 搜索 `Swin Transformer explained shifted window attention`

## 7) Video Swin / 3D Window Attention (TransVisDrone 的核心时序模块)
- 主推荐(论文): **Video Swin Transformer**
- 备选(视频/关键词): 搜索 `Video Swin Transformer 3D shifted window attention explained`

## 8) 检测头 + NMS (输出框怎么得到、为什么要 NMS)
- 主推荐(视频/关键词): 搜索 `Non Maximum Suppression explained`（先彻底搞懂 NMS 在干嘛）
- 备选(论文): **Soft-NMS: Improving Object Detection with One Line of Code**（很短、很实用，也能理解 NMS 的局限）

---

如果你愿意，我可以把上面提到的“论文 PDF”也按标题下载到 `doc/`（像你之前那批一样），并在这里加上对应文件名，方便你对照阅读。

