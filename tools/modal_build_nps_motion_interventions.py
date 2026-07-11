from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import modal


app = modal.App("urap-nps-motion-build-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy==2.1.3",
        "scipy==1.14.1",
    )
    .add_local_file(
        "qstr_dronedet/nps_motion_interventions.py",
        remote_path="/opt/urap/nps_motion_interventions.py",
        copy=True,
    )
)

source_volume = modal.Volume.from_name("urap-nps-formatted-v1")
original_volume = modal.Volume.from_name("urap-nps-motion-original-v1")
variants_volume = modal.Volume.from_name("urap-nps-motion-variants-v1")

INTERVENTIONS = ("original", "slow_0p5", "fast_2x", "accelerate_g2", "decelerate_g2")


def load_module():
    path = "/opt/urap/nps_motion_interventions.py"
    spec = importlib.util.spec_from_file_location("nps_motion_interventions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@app.function(
    image=image,
    volumes={
        "/source": source_volume,
        "/original": original_volume,
        "/variants": variants_volume,
    },
    cpu=16,
    memory=65536,
    timeout=86400,
)
def build_intervention(intervention: str) -> dict:
    if intervention not in INTERVENTIONS:
        raise ValueError(intervention)
    module = load_module()
    source_root = Path("/source/NPS")
    completion = source_root / "build_complete_test.json"
    for _ in range(720):
        source_volume.reload()
        if completion.exists():
            break
        time.sleep(10)
    if not completion.exists():
        raise TimeoutError("Formatted NPS test split did not become ready")

    output_mount = Path("/original") if intervention == "original" else Path("/variants")
    output_root = output_mount / "motion_v1"
    intervention_root = output_root / intervention
    frames_dir = source_root / "AllFrames" / "test"
    labels_dir = source_root / "NPSvisdroneStyle" / "test" / "labels"
    clips = module.discover_clips(frames_dir)
    interpolator = module.DISFrameInterpolator(0.5, 3.0, 0.25)
    summaries = []
    progress_path = output_root / f"progress_{intervention}.json"
    target_volume = original_volume if intervention == "original" else variants_volume
    for index, clip_name in enumerate(clips, start=1):
        summary = module.build_clip(
            frames_dir,
            labels_dir,
            intervention_root,
            "test",
            clip_name,
            intervention,
            interpolator,
            motion_threshold=16,
            max_frames=None,
            seed=59,
        )
        summaries.append(summary)
        progress = {
            "intervention": intervention,
            "done": index,
            "total": len(clips),
            "last_clip": clip_name,
            "last_output": summary["last_output"],
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        target_volume.commit()
        print(json.dumps(progress), flush=True)

    split_lengths = {
        "test": {
            int(summary["clip"].split("_")[-1]): int(summary["output_frames"])
            for summary in summaries
        }
    }
    module.write_dataset_metadata(intervention_root, intervention, split_lengths)
    integrity = module.validate_intervention(intervention_root, intervention, ("test",))
    result = {"intervention": intervention, "integrity": integrity}
    (output_root / f"complete_{intervention}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    target_volume.commit()
    return result


@app.function(
    image=image,
    volumes={
        "/source": source_volume,
        "/original": original_volume,
        "/variants": variants_volume,
    },
    cpu=4,
    memory=16384,
    timeout=86400,
)
def validate_existing(intervention: str) -> dict:
    if intervention not in INTERVENTIONS:
        raise ValueError(intervention)
    module = load_module()
    output_mount = Path("/original") if intervention == "original" else Path("/variants")
    output_root = output_mount / "motion_v1"
    intervention_root = output_root / intervention
    result = {"intervention": intervention, "integrity": module.validate_intervention(intervention_root, intervention, ("test",))}
    (output_root / f"complete_{intervention}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (original_volume if intervention == "original" else variants_volume).commit()
    return result


@app.local_entrypoint()
def main(interventions: str = ",".join(INTERVENTIONS), validate_only: bool = False) -> None:
    names = [value.strip() for value in interventions.split(",") if value.strip()]
    if validate_only:
        results = []
        for intervention in names:
            result = validate_existing.remote(intervention)
            results.append({"intervention": intervention, "result": result})
        print(json.dumps({"complete": True, "results": results}, indent=2), flush=True)
        return
    calls = []
    for intervention in names:
        call = build_intervention.spawn(intervention)
        calls.append((intervention, call))
    print(
        json.dumps(
            {"calls": [{"intervention": name, "call_id": call.object_id} for name, call in calls]},
            indent=2,
        ),
        flush=True,
    )
    results = [{"intervention": name, "result": call.get()} for name, call in calls]
    print(json.dumps({"complete": True, "results": results}, indent=2), flush=True)
