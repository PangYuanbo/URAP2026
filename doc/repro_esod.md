# Repro Notes: ESOD (TIP 2025) on Windows + uv

Repo: `papers/ESOD`

Paper PDF:
- `doc/ESOD Efficient Small Object Detection on High-Resolution Images.pdf`

Goal of this repro pass:
- Make the **official ESOD code** run end-to-end on Windows with a modern CUDA-enabled PyTorch.
- Download the **official dataset (VisDrone)** and run ESOD preprocessing + evaluation.

## Environment (uv)

Virtualenv:
- `papers/ESOD/.venv` (created via `uv`)
- Python: 3.10.19

Key packages:
- torch `2.10.0+cu130` (CUDA 13.0)
- torchvision `0.25.0+cu130`

How it was created:

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
C:\Users\aaron\.local\bin\uv.exe venv --python 3.10.19 .venv
C:\Users\aaron\.local\bin\uv.exe pip install -r requirements.txt
C:\Users\aaron\.local\bin\uv.exe pip install torch torchvision --torch-backend cu130
```

## Compatibility Fixes Applied

Two small patches were needed for current toolchains:

1) `pkg_resources` missing in newer environments
- File: `papers/ESOD/utils/general.py`
- Fix: guard `import pkg_resources` and fall back to `packaging` + `importlib.metadata`.

2) PyTorch `torch.load` security default changed (`weights_only=True`)
- File: `papers/ESOD/models/experimental.py`
- Fix: call `torch.load(..., weights_only=False)` when supported (only if the checkpoint is trusted).

3) PyTorch `torch.load` default (`weights_only=True`) also breaks YOLO-style dataset caches (`*.cache`)
- File: `papers/ESOD/utils/datasets.py`
- Fix: load cache with `torch.load(..., weights_only=False)` when supported.

4) Training checkpoint/optimizer stripping must also force `weights_only=False` on modern PyTorch
- Files:
  - `papers/ESOD/train.py`
  - `papers/ESOD/utils/general.py` (`strip_optimizer`)
- Fix: guarded `torch.load(..., weights_only=False)` to keep training end-to-end.

5) Dataloader workers flag was ignored (hard-coded to 8)
- File: `papers/ESOD/utils/datasets.py`
- Fix: respect `workers` argument so `--workers 0` actually disables multiprocessing on Windows.

## Pretrained ESOD Weights

Downloaded from the repo's Google Drive link into:
- `papers/ESOD/weights/esod_pretrained/esod_yolov5m.pt`
- `papers/ESOD/weights/esod_pretrained/esod_yolov8m.pt`

Download command:

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
C:\Users\aaron\.local\bin\uv.exe pip install gdown
.\.venv\Scripts\python.exe -m gdown --folder "https://drive.google.com/drive/folders/1jrrtG34q6gqdGMx7SflNK1uM6iiVKpin?usp=drive_link" -O weights\esod_pretrained
```

Smoke-test inference (works with ESOD weights):

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
.\.venv\Scripts\python.exe detect.py `
  --weights weights\esod_pretrained\esod_yolov5m.pt `
  --source C:\Users\aaron\Desktop\URAP\artifacts\transvisdrone_smoke\val\frames\Clip_14_00000.png `
  --img-size 640 --device 0 `
  --project runs\detect --name smoke_esod_yolov5m --exist-ok
```

Note:
- A *vanilla* YOLOv5 checkpoint (e.g. `yolov5m.pt`) does **not** work with `detect.py` here, because ESOD's model forward returns extra mask/heatmap outputs. Use ESOD-trained weights for inference, or train ESOD first.

## Dataset: VisDrone (Full Download)

Downloaded and extracted all 4 official subsets:
- `VisDrone2019-DET-train`
- `VisDrone2019-DET-val`
- `VisDrone2019-DET-test-dev`
- `VisDrone2019-DET-test-challenge`

Local dataset root:
- `C:\URAP_datasets\VisDrone`

Repo-local junction created (Windows replacement for `ln -sf`):
- `papers/ESOD/VisDrone` -> `C:\URAP_datasets\VisDrone`

One-shot download + extract:

```powershell
cd C:\Users\aaron\Desktop\URAP
tools\download_visdrone_esod.ps1
```

## VisDrone Preprocessing (ESOD Format)

This converts VisDrone annotations into YOLO/Darknet-style `labels/` files and generates ESOD masks + splits:

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
.\.venv\Scripts\python.exe scripts\data_prepare.py --dataset VisDrone
```

Detached (recommended for long runs on Windows):

```powershell
cd C:\Users\aaron\Desktop\URAP
tools\start_esod_visdrone_prepare_detached.ps1
tools\monitor_esod_visdrone_prepare.ps1
```

Outputs (created under `papers/ESOD/VisDrone/...`):
- `VisDrone2019-DET-*/labels/*.txt`
- `VisDrone2019-DET-*/masks/*.npy`
- `split/train.txt`, `split/val.txt`, `split/test-dev.txt`, ...

## Evaluation (After Preprocess Finishes)

Observed results (ESOD pretrained `esod_yolov5m.pt`, `--img-size 1536`, `--batch-size 4`):
- `val` (548 images): `P=0.635`, `R=0.541`, `mAP@0.5=0.564`, `mAP@0.5:0.95=0.331` (saved under `papers/ESOD/runs/test/exp`).
- `test-dev` (1610 images): `P=0.55`, `R=0.45`, `mAP@0.5=0.444`, `mAP@0.5:0.95=0.252` (saved under `papers/ESOD/runs/test/exp2`).

Observed results (ESOD pretrained `esod_yolov8m.pt`, `--img-size 1536`, `--batch-size 4`):
- `val` (548 images): `P=0.65`, `R=0.559`, `mAP@0.5=0.586`, `mAP@0.5:0.95=0.346` (saved under `papers/ESOD/runs/test/exp4`).
- `test-dev` (1610 images): `P=0.558`, `R=0.483`, `mAP@0.5=0.468`, `mAP@0.5:0.95=0.266` (saved under `papers/ESOD/runs/test/exp5`).

Vanilla evaluation (integrated metrics):

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
.\.venv\Scripts\python.exe test.py `
  --data data\visdrone.yaml `
  --weights weights\esod_pretrained\esod_yolov5m.pt `
  --batch-size 8 --img-size 1536 --device 0 --workers 0
```

Windows note:
- If you hit `OSError: [WinError 1455] The paging file is too small ... cublas64_13.dll` when the dataloader spawns CUDA-enabled workers, run with `--workers 0` (or increase the Windows paging file size).

Compute FPS / GFLOPs:

```powershell
.\.venv\Scripts\python.exe test.py `
  --data data\visdrone.yaml `
  --weights weights\esod_pretrained\esod_yolov5m.pt `
  --batch-size 1 --img-size 1536 --device 0 --task measure
```

## Training (50 Epochs Repro)

Single-GPU training command (PowerShell):

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
.\.venv\Scripts\python.exe train.py `
  --data data\visdrone.yaml `
  --cfg models\cfg\esod\visdrone_yolov5m.yaml `
  --weights weights\pretrained\yolov5m.pt `
  --hyp data\hyps\hyp.visdrone.yaml `
  --batch-size 8 --img-size 1536 --epochs 50 --device 0 --workers 0 `
  --project runs\train --name visdrone_esod_yolov5m_e50_b8_img1536
```

Detached runner (recommended for long runs on Windows):

```powershell
cd C:\Users\aaron\Desktop\URAP
tools\start_esod_train_visdrone_yolov5m_detached.ps1
tools\monitor_esod_train_visdrone_yolov5m.ps1
```

After training completes, evaluate the trained weights:

```powershell
cd C:\Users\aaron\Desktop\URAP\papers\ESOD
.\.venv\Scripts\python.exe test.py `
  --data data\visdrone.yaml `
  --weights runs\train\<RUN_NAME>\weights\best.pt `
  --batch-size 4 --img-size 1536 --device 0 --task val --workers 0
```
