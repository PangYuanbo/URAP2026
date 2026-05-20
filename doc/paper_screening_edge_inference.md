# Paper Screening (Edge Inference Constraint)

This note is a companion to the PDFs in `doc/` and is written for our goal:

- Training can use a high-GPU workstation.
- Inference must run on-board a UAV (edge compute), so we prioritize methods that are edge-deployable or whose ideas can be adapted into a lightweight model.

## How I Screened

For each paper in `doc/`, I looked for:

- Explicit edge deployment (Jetson Xavier/Orin, TensorRT, latency/FPS).
- Whether the method is detector-only vs video detection / detect-track switching.
- Whether the core method is tied to heavy backbones (large transformers / multi-stage cascades) with no edge story.
- Whether it's mainly a dataset / benchmark contribution (useful for training/evaluation, not necessarily for on-board compute).

## Category A: Edge-Ready / Directly Referenceable for UAV On-Board Inference

### 1) TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos

- Why it is top-tier for us:
- It is explicitly about drone-to-drone, tiny targets, egomotion, and even "below-the-horizon" clutter (i.e., not only open sky).
- It has an explicit edge deployment section: reports real-time FPS on NVIDIA Jetson Xavier NX at 640 resolution without complex TensorRT optimizations.
- In the paper text: "Our 640 resolution model obtains the real-time fps of 33 without any complex TensorRT optimizations" (Jetson Xavier NX).
- What to extract for our V2:
- "Temporal-consistent augmentation" and video-aware detection design (but keep the model lightweight).
- Their FPPI framing is aligned with our "avoid collisions = reduce catastrophic false positives" objective.
- Local PDF:
  `doc/TransVisDrone Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos.pdf`

### 2) Evidential Detection and Tracking Collaboration: New Problem, Benchmark and Algorithm for Robust Anti-UAV System (EDTC / AntiUAV600)

- Why it is top-tier for us:
- Defines a very practical system: global detection when needed, local tracking when confident, and an adaptive switcher (uncertainty/evidential head).
- This structure is "Sony-like" in the sense of allocating compute where needed (coarse-to-fine; persistence; recover on failure).
- Uses a lightweight detector in the global branch (YOLOv5s) and reports real-time-ish system speed (paper mentions ~35 fps).
- The paper explicitly states: "The inference speed of the proposed EDTC is about 35 fps."
- Caveat:
- Their tracker backbone is transformer-based; we may keep the switching logic but swap in a more efficient tracker if needed.
- Local PDF:
  `doc/Evidential Detection and Tracking Collaboration New Problem, Benchmark and Algorithm for Robust Anti-UAV System.pdf`

## Category B: Highly Relevant Ideas, But Not Directly Deployable as Written (Need "Lightweight Re-Implementation")

### Visible and Clear: Finding Tiny Objects in Difference Map (SR-TOD)

- Why relevant:
- Attacks the hardest part of tiny object detection in textured backgrounds: spurious textures / artifacts drowning tiny targets.
- Uses difference-map guidance to enhance tiny-object features (conceptually close to our motion/residual guidance, but learned).
- Why not directly edge-ready:
- Experiments are in an MMDetection style with heavier baselines (e.g., two-stage / cascade). No Jetson/latency story in the paper.
- How to use it:
- Take the idea: "difference-map guided feature enhancement" and adapt it into a lightweight detector (YOLO/RT-DETR-R18/etc.) or as an auxiliary head to guide our crop selection.
- Local PDF:
  `doc/Visible and Clear Finding Tiny Objects in Difference Map (SR-TOD).pdf`

## Category C: Dataset / Domain Shift Papers (Key for "Sky -> City" Generalization)

### DrIFT: Autonomous Drone Dataset with Integrated Real and Synthetic Data, Flexible Views, and Transformed Domains

- Why relevant:
- This is almost exactly our problem statement: domain shifts by environment, viewpoint, and especially background shift.
- Useful for designing evaluation and for training with domain-aware sampling / hard negatives.
- Why not a "deployment method":
- It's mainly a dataset + evaluation/uncertainty analysis. Use it to structure our training/eval; do not copy a heavy model.
- Local PDF:
  `doc/DrIFT Autonomous Drone Dataset with Integrated Real and Synthetic Data.pdf`

### SynDroneVision / SimD3

- Why relevant:
- Synthetic data for data scaling, especially for bird distractors and complex backgrounds.
- Training-time leverage: improve robustness without changing on-board runtime much.
- Deployment impact:
- Use these datasets to train a lightweight model; do not assume their training pipeline is what we deploy.
- Local PDFs:
  `doc/SynDroneVision A Synthetic Dataset for Image-Based Drone Detection.pdf`
  `doc/SimD3 A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection.pdf`

## Category D: Off-Track for Our Current Hardware / Sensor Assumptions

### Event-based Tiny Object Detection: A Benchmark Dataset and Baseline (EV-UAV / EVSOD)

- Why relevant academically:
- Extremely tiny targets and urban clutter; event cameras are strong for high dynamic range and temporal precision.
- Why off-track right now:
- Requires event camera input and a different inference stack. Only pursue if we commit to event sensors.
- Local PDF:
  `doc/Event-based Tiny Object Detection A Benchmark Dataset and Baseline.pdf`

### Multi-Modal UAV Detection, Classification and Tracking (UG2+ technical report)

- Why relevant:
- Has practical tricks like ROI cropping and keyframe selection for efficiency.
- Why off-track:
- Multi-modal sensors and a more complex pipeline than our current monocular-video focus.
- Local PDF:
  `doc/Multi-Modal UAV Detection, Classification and Tracking Algorithm -- Technical Report for CVPR 2024 UG2 Challenge.pdf`

## What We Will Reproduce Next (Per Your Request)

We will pull and try to reproduce the top two "edge-ready" references:

1. TransVisDrone
2. EDTC (AntiUAV600)

If any dataset is not publicly downloadable, we will still:

- Make the code run end-to-end on sample inputs.
- Document exactly what is missing for full benchmark reproduction.
- Provide a minimal adapter to run on our current video dataset (Purdue UAV-to-UAV) for sanity, while clearly separating that from the paper's official results.
