# V1 Plan: Edge Tiny-UAV Detection in Urban Clutter (Video + AutoFocus-Style ROI)

Goal (ours):
- Train on a high-GPU workstation.
- Run inference on-board a UAV (edge compute), so we need a design that is both accurate on tiny targets and latency/energy aware.
- Target setting: tiny UAV / tiny obstacles in complex "below-the-horizon" urban backgrounds (buildings, edges, windows, wires), not only open sky.

## 2026-06-23 Direction Update: SAMURAI-Style Motion-Aware Memory

The current main R&D direction is now the SAMURAI-style motion-aware memory route:

```text
detector candidates
  -> motion prediction / Kalman-style consistency
  -> selective clean-memory write gate
  -> memory-guided rescoring and zoom re-detection
  -> hard-reset stale-memory correction
```

This does not mean copying SAM 2 wholesale. The transferable idea is motion-aware
memory selection: only high-quality observations should condition future frames.
For the detailed plan, see:

`doc/samurai_motion_memory_research_direction_2026_06_23.md`

## What The Current Baselines Actually Do

### TransVisDrone (ICRA 2023)
- Task: video-based drone-to-drone detection (per-frame boxes), using a temporal window (`tau`).
- Key idea: insert a lightweight spatio-temporal transformer into a YOLO-style detector to aggregate motion cues.
- Edge evidence: paper reports deployment on Jetson Xavier NX (33 FPS at 640) without TensorRT.
- Why it matters to us: it already targets egomotion + small object video detection and uses FPPI framing (false positives per image) that matches "collision avoidance cost".
- Local PDF: `doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf`

### EDTC / AntiUAV600 (arXiv 2023)
- Task framing: real-time anti-UAV system on streaming video without "first frame template".
- Key idea: detector + tracker collaboration with an uncertainty/evidence head that decides when to trust tracking vs re-run global detection.
- Edge evidence: paper states ~35 FPS inference.
- Why it matters to us: this is exactly a compute-allocation strategy (spend expensive compute only when needed).
- Local PDF: `doc/Evidential Detection and Tracking Collaboration New Problem, Benchmark and Algorithm for Robust Anti-UAV System.pdf`

## Core V1 Design (Not A Totally Different Method)

V1 is a "tight" upgrade of the above two baselines, with one additional proven ingredient: AutoFocus-style ROI inference.

### V1-A: Global Low-Res Video Detector (Always-On)
- Run a lightweight video detector at 640 (or 800) continuously.
- Architecture: keep TransVisDrone-style temporal aggregation, but use an edge-friendly backbone.
  - Option 1: keep their YOLOv5-style backbone and shrink channels (YOLO5n/s class).
  - Option 2: replace backbone with an efficient CNN/ViT hybrid, but keep the same head and temporal block concept.
- Output: coarse candidate boxes + confidence + a few intermediate feature maps.

Why this works:
- Low-res pass maintains global situational awareness and prevents missing objects that appear far from prior tracks.

Reference anchors:
- TransVisDrone (temporal window + video transformer)

### V1-B: AutoFocus-Style High-Res ROI Detector (Triggered)
Problem:
- Tiny UAVs in urban clutter are often a few pixels. Global 640 inference can miss them.

Solution:
- Use ROI proposals from the low-res pass (and/or motion residual map) to trigger a second-stage zoomed detector on only a few crops.
- This is the exact "zoom small regions" approach you described, but in a principled, literature-backed way.

Reference:
- "AutoFocus: Efficient Multi-Scale Inference" (ICCV 2019) - performs sparse high-res processing only where needed.
  - Local PDF: `doc/AutoFocus Efficient Multi-Scale Inference.pdf`
- "SNIPER: Efficient Multi-Scale Training" (NeurIPS 2018) - multi-scale training trick that improves tiny object detection.
  - Local PDF: `doc/SNIPER Efficient Multi-Scale Training.pdf`

Implementation sketch:
- Maintain a small queue of ROIs per frame: top-K boxes by confidence + tracked boxes.
- Expand each ROI by a margin (context) and clamp to image bounds.
- Run a tiny detector on each crop at higher effective resolution (e.g., crop -> resize to 640).
- Merge crop detections back to full image coordinates with NMS + track association.

Why it is edge-friendly:
- K is small (e.g., 2-8). Worst-case compute is bounded.
- This is the same compute allocation philosophy as EDTC.

### V1-C: Detection-Tracking Collaboration (Uncertainty Switch)
- Use EDTC's core idea but keep it lightweight:
  - When track confidence is high: track forward cheaply.
  - When confidence drops or background distractors spike: re-run global detection and/or trigger ROI detector.
- We can keep the evidential/uncertainty head concept, but if it is too heavy, we can approximate:
  - combine detector score stability + track motion consistency + ROI re-detection agreement.

Reference anchors:
- EDTC switching logic and uncertainty modeling.

### V1-D: Motion/Residual Guidance For Urban Clutter (Training + ROI Selection)
Urban background problem:
- Without compensation, frame differencing "lights up everything" due to egomotion.

Practical compromise:
- Compute a cheap global motion model (e.g., homography from sparse features, or IMU/gyro if available), stabilize, then take a residual map.
- Use residual map as:
  - an ROI proposal hint (cheap)
  - an auxiliary supervision / feature guidance idea (learned), inspired by SR-TOD

Reference:
- "Visible and Clear: Finding Tiny Objects in Difference Map (SR-TOD)" (ECCV 2024) - learns to leverage difference maps to enhance tiny objects.
  - Local PDF: `doc/Visible and Clear Finding Tiny Objects in Difference Map (SR-TOD).pdf`

## Training/Data Plan (To Make "Sky -> City" Actually Work)

We should not expect the paper baselines (trained mostly on open-sky) to generalize to urban clutter without hard negatives and domain shift coverage.

### Datasets to leverage
- DrIFT (domain shift emphasis): `doc/DrIFT Autonomous Drone Dataset with Integrated Real and Synthetic Data.pdf`
- SynDroneVision / SimD3 (synthetic scaling + bird distractors): in `doc/`
- Our local Purdue NPS clips + additional urban captures (if we can record).

### Training recipe
- Train global detector on mixed data with strong background randomization (urban textures).
- Train ROI detector with "tiny-object biased sampling": always include crops around small GT boxes + hard-negative crops.
- Optional (high-GPU only): distill from a stronger teacher (ViT/DETR class) into the edge student.

## What V1 Outputs (For Path Planning)

For each frame:
- A set of obstacle tracks (ID, bbox, confidence).
- Short-horizon track velocity in image space (for collision risk heuristics).

This is enough to feed into:
- collision avoidance rules (if any track enters a risk cone),
- or a planner that uses uncertainty-aware constraints.

## Immediate Engineering Steps In This Repo (After We Finish Baseline Recompute)

1) Finish TransVisDrone NPS val metrics recompute and write numbers to `doc/official_datasets_and_metrics.md`.
2) Confirm EDTC dataset availability (ModelScope download) and map it into EDTC expected format if possible.
3) Implement a minimal "ROI inference wrapper" around TransVisDrone inference:
   - global pass at 640
   - top-K ROI crops at higher effective resolution
   - merge + track association
4) Add an EDTC-like switch policy (confidence + uncertainty proxy) to decide when to run ROI detector.
