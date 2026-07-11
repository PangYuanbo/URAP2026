from __future__ import annotations

import json
import pickle
from pathlib import Path

import modal


app = modal.App("urap-final-training-readiness-v1")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pillow==11.0.0", "pyyaml==6.0.2"
)
volumes = {
    "/code": modal.Volume.from_name("urap-code-artifacts-v1"),
    "/weights": modal.Volume.from_name("urap-model-weights-v1"),
    "/nps": modal.Volume.from_name("urap-nps-formatted-v1"),
    "/nps_yolomg": modal.Volume.from_name("urap-nps-yolomg-v1"),
    "/motion_original": modal.Volume.from_name("urap-nps-motion-original-v1"),
    "/motion_variants": modal.Volume.from_name("urap-nps-motion-variants-v1"),
    "/ard_raw": modal.Volume.from_name("urap-ard100-raw-v1"),
    "/data_train": modal.Volume.from_name("urap-ard100-yolomg-train-v1"),
    "/data_eval": modal.Volume.from_name("urap-ard100-yolomg-eval-v1"),
    "/ard_tvd": modal.Volume.from_name("urap-ard100-transvisdrone-links-v1"),
    "/aot": modal.Volume.from_name("urap-aot-part1-raw-v1"),
}


@app.function(image=image, volumes=volumes, cpu=4, memory=8192, timeout=1800)
def check() -> dict:
    from PIL import Image
    import yaml

    for volume in volumes.values():
        volume.reload()
    failures = []
    samples = {}

    image_paths = {
        "nps_tvd": next(Path("/nps/NPS/AllFrames/train").glob("*")),
        "nps_yolomg_rgb": next(Path("/nps_yolomg/NPS_YOLOMG/images/train").glob("*")),
        "nps_yolomg_motion": next(Path("/nps_yolomg/NPS_YOLOMG/images2/train").glob("*")),
        "ard_yolomg_train": next(Path("/data_train/ARD100_YOLOMG/images/train").glob("*")),
        "ard_yolomg_eval": next(Path("/data_eval/ARD100_YOLOMG/images/test").glob("*")),
        "ard_tvd_link": next(Path("/ard_tvd/ARD100_TVD/AllFrames/test").iterdir()),
        "aot_raw": next(Path("/aot/AOT_part1/Images").glob("*/*.png")),
    }
    for name, path in image_paths.items():
        try:
            with Image.open(path) as image_handle:
                image_handle.verify()
            samples[name] = {"path": str(path), "bytes": path.stat().st_size, "readable": True}
        except Exception as error:
            failures.append(f"{name}: {error}")

    required_weights = {
        "tvd_nps": Path("/weights/TransVisDrone/NPS/best.pt"),
        "tvd_aot": Path("/weights/TransVisDrone/AOT/best.pt"),
        "yolomg_ard100": Path("/weights/YOLOMG/ARD100_mask32-1280/best.pt"),
        "yolov5s": Path("/weights/YOLOMG/pretrained/yolov5s.pt"),
        "samurai": Path("/weights/SAMURAI/sam2.1_hiera_base_plus.pt"),
    }
    weight_status = {}
    for name, path in required_weights.items():
        exists = path.is_file() and path.stat().st_size > 0
        weight_status[name] = {"path": str(path), "exists": exists, "bytes": path.stat().st_size if exists else 0}
        if not exists:
            failures.append(f"missing weight: {name}")

    required_code = [
        Path("/code/models/YOLOMG/train.py"),
        Path("/code/models/TransVisDrone/train.py"),
        Path("/code/repo/tools/modal_build_ard100_yolomg.py"),
        Path("/code/repo/qstr_dronedet/nps_motion_interventions.py"),
    ]
    code_status = {str(path): path.is_file() for path in required_code}
    failures.extend(f"missing code: {path}" for path, exists in code_status.items() if not exists)

    config_paths = [
        Path("/nps_yolomg/NPS_YOLOMG/NPS_yolomg.yaml"),
        Path("/data_train/ARD100_YOLOMG/ARD100_mask32_local.yaml"),
        Path("/ard_tvd/ARD100_TVD/ARD100_TVD.yaml"),
    ]
    configs = {}
    for path in config_paths:
        try:
            configs[str(path)] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append(f"config {path}: {error}")

    video_lengths = {}
    for split in ("train", "val", "test"):
        path = Path(f"/ard_tvd/ARD100_TVD/Videos/{split}/video_length_dict.pkl")
        with path.open("rb") as handle:
            video_lengths[split] = len(pickle.load(handle))

    motion_markers = {
        "original": Path("/motion_original/motion_v1/complete_original.json").is_file(),
        "slow_0p5": Path("/motion_variants/motion_v1/complete_slow_0p5.json").is_file(),
        "fast_2x": Path("/motion_variants/motion_v1/complete_fast_2x.json").is_file(),
        "accelerate_g2": Path("/motion_variants/motion_v1/complete_accelerate_g2.json").is_file(),
        "decelerate_g2": Path("/motion_variants/motion_v1/complete_decelerate_g2.json").is_file(),
    }
    failures.extend(f"missing motion marker: {name}" for name, exists in motion_markers.items() if not exists)

    result = {
        "complete": not failures,
        "samples": samples,
        "weights": weight_status,
        "code": code_status,
        "configs": configs,
        "ard_tvd_video_counts": video_lengths,
        "motion_markers": motion_markers,
        "failures": failures,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(check.remote(), indent=2))
