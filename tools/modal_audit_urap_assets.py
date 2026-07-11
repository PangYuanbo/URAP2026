from __future__ import annotations

import json
import re
from pathlib import Path

import modal


app = modal.App("urap-asset-audit-v1")
image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "numpy==2.1.3"))
nps = modal.Volume.from_name("urap-nps-formatted-v1")
yolomg = modal.Volume.from_name("urap-nps-yolomg-v1")
weights = modal.Volume.from_name("urap-model-weights-v1")
code = modal.Volume.from_name("urap-code-artifacts-v1")
FRAME_RE = re.compile(r"^(Clip_\d+)_(\d+)$")


def count_files(path: Path, pattern: str) -> tuple[int, int]:
    count = 0
    empty = 0
    for item in path.glob(pattern):
        if item.is_file():
            count += 1
            empty += int(item.stat().st_size == 0)
    return count, empty


@app.function(
    image=image,
    volumes={"/nps": nps, "/yolomg": yolomg, "/weights": weights, "/code": code},
    cpu=4,
    memory=16384,
    timeout=7200,
)
def audit() -> dict:
    import cv2
    import numpy as np

    result = {"nps": {}, "yolomg": {}, "weights": {}, "code": {}}
    for split in ("train", "val", "test"):
        frame_dir = Path("/nps/NPS/AllFrames") / split
        label_dir = Path("/nps/NPS/NPSvisdroneStyle") / split / "labels"
        frame_count, empty_frames = count_files(frame_dir, "*.png")
        label_count, empty_labels = count_files(label_dir, "*.txt")
        result["nps"][split] = {
            "frames": frame_count, "empty_frames": empty_frames,
            "labels": label_count, "empty_labels": empty_labels,
            "complete_marker": (Path("/nps/NPS") / f"build_complete_{split}.json").exists(),
        }

        images = Path("/yolomg/NPS_YOLOMG/images") / split
        images2 = Path("/yolomg/NPS_YOLOMG/images2") / split
        labels = Path("/yolomg/NPS_YOLOMG/labels") / split
        image_count, empty_images = count_files(images, "*.png")
        image2_count, empty_images2 = count_files(images2, "*.png")
        yolo_label_count, yolo_empty_labels = count_files(labels, "*.txt")
        first_masks = []
        for first_frame in sorted(images.glob("Clip_*_00001.png")):
            mask = cv2.imread(str(images2 / first_frame.name), cv2.IMREAD_UNCHANGED)
            first_masks.append({
                "frame": first_frame.name,
                "exists": mask is not None,
                "nonzero": int(np.count_nonzero(mask)) if mask is not None else None,
            })
        result["yolomg"][split] = {
            "images": image_count, "empty_images": empty_images,
            "images2": image2_count, "empty_images2": empty_images2,
            "labels": yolo_label_count, "empty_labels": yolo_empty_labels,
            "first_masks": first_masks,
            "complete_marker": (Path("/yolomg/NPS_YOLOMG") / f"build_complete_{split}.json").exists(),
        }

    required_weights = [
        "TransVisDrone/AOT/best.pt", "TransVisDrone/NPS/best.pt",
        "TransVisDrone/FL/best.pt", "YOLOMG/pretrained/yolov5s.pt",
        "YOLOMG/ARD100_mask32-1280/best.pt", "SAMURAI/sam2.1_hiera_base_plus.pt",
    ]
    for relative in required_weights:
        path = Path("/weights") / relative
        result["weights"][relative] = {"exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
    for relative in ("repo/tools", "repo/qstr_dronedet", "models/YOLOMG", "models/TransVisDrone"):
        path = Path("/code") / relative
        result["code"][relative] = {"exists": path.exists()}
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(audit.remote(), indent=2), flush=True)
