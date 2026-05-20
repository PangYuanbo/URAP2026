# Repro Notes: TransVisDrone (ICRA 2023)

Repo: `papers/TransVisDrone` (commit `8b3c76037edae99a94e1461678ffa685b1333fe2`)

Goal of this repro pass:
- Make the code run end-to-end on Windows + modern PyTorch/CUDA.
- Run a smoke-test inference on our local URAP sample video frames.
- This is NOT the full paper benchmark reproduction (official datasets/splits not downloaded here).

## Environment

Machine:
- Windows
- GPU: NVIDIA GeForce RTX 5090

Python:
- CPython 3.10.19 (venv at `papers/TransVisDrone/.venv`)

Key packages:
- torch `2.10.0+cu130` (CUDA 13.0)
- torchvision `0.25.0+cu130`
- timm `0.6.13` (needed to unpickle old checkpoints)
- einops `0.8.2`
- numpy is v2.x (required minor compatibility patches)

## Pretrained Weights

Downloaded the official pretrained weights folder into:
`papers/TransVisDrone/pretrained/TransVisDrone_weights/...`

Used for smoke test:
`papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt`

## Code Compatibility Fixes Applied (Windows + New PyTorch/Numpy)

These were required to run the repo in our environment:

1) Windows: remove Linux-only `curses` import
- `papers/TransVisDrone/utils/augmentations.py`
- `papers/TransVisDrone/conversion_scripts/fl_drones_to_visdrone.py`

2) Remove MMCV dependency (not needed for inference, painful on Windows)
- `papers/TransVisDrone/models/video_swin_transformer.py`

3) Replace `pkg_resources` usage with `packaging` + `importlib.metadata` fallback
- `papers/TransVisDrone/utils/general.py`

4) PyTorch torch.load security change (weights_only default)
- `papers/TransVisDrone/models/experimental.py`:
  - call `torch.load(..., weights_only=False)` when supported
  - patch missing module attributes for old pickled modules:
    - nn.GELU missing `.approximate`
    - DropPath missing `.scale_by_keep`

5) Numpy 2: replace deprecated `np.int` aliases
- `papers/TransVisDrone/utils/datasets_inference.py`
- `papers/TransVisDrone/utils/datasets.py`
- `papers/TransVisDrone/utils/general.py`

## Smoke-Test Dataset (URAP)

Extracted 60 frames from:
`Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Videos/Clip_14.mov`

Into:
`artifacts/transvisdrone_smoke/val/frames/Clip_14_00000.png` ... `Clip_14_00059.png`

And wrote:
`artifacts/transvisdrone_smoke/val/videos/video_length_dict.pkl`

Dataset YAML:
`papers/TransVisDrone/data/URAP_smoke.yaml`

## Smoke-Test Command

Run inference:
```powershell
cd C:\Users\aaron\Desktop\URAP\papers\TransVisDrone
$w='pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt'
.\.venv\Scripts\python inference.py --data data\URAP_smoke.yaml --weights $w --task inference --imgsz 640 --batch-size 1 --device 0 --half --project runs\inference\URAP_smoke --name nps_best --exist-ok --save-txt
```

Outputs:
- Predictions text files: `papers/TransVisDrone/runs/inference/URAP_smoke/nps_best/labels/*.txt`
- Runtime printed per-image (smoke run, no GT labels so mAP is 0 by design).

Notes:
- This verifies code executes and produces detections on our local frames.
- To reproduce paper numbers, we still need the official datasets (NPS/FL/AOT) and their evaluation protocol.

