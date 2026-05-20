# Repro Notes: EDTC (AntiUAV600) (arXiv 2023)

Repo: `papers/EDTC` (commit `d113d516853a5913c3dbbe3c69ace6833361f5a7`)

Goal of this repro pass:
- Make the code run end-to-end on Windows + modern PyTorch/CUDA.
- Since AntiUAV600 dataset is not bundled in the repo, do a smoke run on our local URAP frames:
  - Verify YOLO detector loads and runs.
  - Verify UAVTrackEH tracking branch loads and runs (detector init -> track loop).
- This is NOT the full benchmark reproduction on AntiUAV600 (no official dataset/split here).

## Environment

Python:
- CPython 3.10.19 (venv at `papers/EDTC/.venv`, created with `uv`)

Key packages:
- torch `2.10.0+cu130` (CUDA 13.0)
- torchvision `0.25.0+cu130`
- timm `0.6.13`
- einops `0.8.2`
- lmdb `1.7.5`
- easydict `1.13`

## Pretrained Weights Downloaded

Downloaded from the official Google Drive into `papers/EDTC/pretrained/`:
- `papers/EDTC/pretrained/yolo/best.pt`
- `papers/EDTC/pretrained/UAVTrackEH.pth.tar`
- `papers/EDTC/pretrained/CvT-21-384x384-IN-22k.pth` (backbone pretrain)

## Code Compatibility Fixes Applied (Windows + New PyTorch)

1) PyTorch torch.load security change (weights_only default)
- `papers/EDTC/yolov5/models/experimental.py`: use `torch.load(..., weights_only=False)` when supported
- `papers/EDTC/lib/test/tracker/uavtrack_eh.py`: same for `UAVTrackEH.pth.tar`

2) `pkg_resources` missing in newer setuptools
- `papers/EDTC/yolov5/utils/general.py`: fallback to `packaging` + `importlib.metadata`

3) PyTorch >= 2.0 removed `torch._six`
- `papers/EDTC/lib/train/data/loader.py`: fallback `string_classes`, and use `collections.abc` for Mapping/Sequence

4) Optional dependency: jpeg4py
- `papers/EDTC/lib/train/data/image_loader.py`: jpeg4py is optional; fallback to OpenCV loader

5) Ensure YOLOv5 root is on PYTHONPATH during tracking runs
- `papers/EDTC/tracking/test.py`: append `.../EDTC/yolov5` to `sys.path`

6) Fix broken local env file generation + provide a valid local env file
- `papers/EDTC/lib/test/evaluation/environment.py`: correct import path + safe repr() for Windows paths
- `papers/EDTC/lib/test/evaluation/local.py`: minimal working config (sets `prj_dir` and `save_dir`)

7) Add a tiny smoke dataset entry (1 sequence from our URAP frames)
- `papers/EDTC/lib/test/evaluation/datasets.py`: add `urap_smoke`
- `papers/EDTC/lib/test/evaluation/urap_smokedataset.py`: sequence loader

8) Avoid compiling the CUDA PreciseRoIPooling extension (Windows build tooling pain)
- `papers/EDTC/external/PreciseRoIPooling/pytorch/prroi_pool/functional.py`:
  - fall back to `torchvision.ops.roi_align` if extension compilation fails

9) Point the tracker config to local YOLO weights + dataset yaml
- `papers/EDTC/experiments/uavtrack_eh/baseline.yaml`
- `papers/EDTC/data/urap_smoke_yolo.yaml`

## Smoke-Test Commands

1) YOLO detector (single frame):
```powershell
cd C:\Users\aaron\Desktop\URAP\papers\EDTC\yolov5
..\..\.venv\Scripts\python detect.py --weights ..\pretrained\yolo\best.pt --source C:\Users\aaron\Desktop\URAP\artifacts\transvisdrone_smoke\val\frames\Clip_14_00000.png --imgsz 640 --conf-thres 0.25 --device 0 --project runs\detect --name smoke --exist-ok
```

2) Detector-init + tracking (1 sequence):
```powershell
cd C:\Users\aaron\Desktop\URAP\papers\EDTC
.\.venv\Scripts\python tracking\test.py --tracker_name uavtrack_eh --tracker_param baseline --dataset_name urap_smoke --num_gpus 1 --threads 0 --params__model C:\Users\aaron\Desktop\URAP\papers\EDTC\pretrained\UAVTrackEH.pth.tar --params__search_area_scale 4.55
```

Outputs:
- Tracking results:
  - `papers/EDTC/lib/test/tracking_results/uavtrack_eh/baseline/Clip_14.txt`
  - `papers/EDTC/lib/test/tracking_results/uavtrack_eh/baseline/Clip_14_time.txt`

## Limitations / Next Step To Fully Reproduce Paper Numbers

- The official AntiUAV600 dataset is not included in this repo. To reproduce paper metrics:
  - obtain AntiUAV600 data in the expected folder structure
  - set dataset root paths in `papers/EDTC/lib/test/evaluation/local.py`
  - run `tracking/test.py ... --dataset_name antiuav` with the official sequences

