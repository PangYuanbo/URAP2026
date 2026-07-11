from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


INTERVENTIONS = ("original", "slow_0p5", "fast_2x", "accelerate_g2", "decelerate_g2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TransVisDrone and YOLOMG over NPS motion interventions.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1"))
    parser.add_argument("--out-root", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\model_evals"))
    parser.add_argument("--interventions", nargs="+", choices=INTERVENTIONS, default=list(INTERVENTIONS))
    parser.add_argument("--models", nargs="+", choices=("transvisdrone", "yolomg_native", "yolomg_ard100"), default=["transvisdrone", "yolomg_native", "yolomg_ard100"])
    parser.add_argument("--tvd-repo", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\papers\TransVisDrone"))
    parser.add_argument("--tvd-python", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe"))
    parser.add_argument("--tvd-weights", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt"))
    parser.add_argument("--yolomg-repo", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG"))
    parser.add_argument("--yolomg-python", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"))
    parser.add_argument("--yolomg-native-weights", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\yolomg_nps_train50\weights\best.pt"))
    parser.add_argument("--yolomg-ard-weights", type=Path, default=Path(r"C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt"))
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def safe_reset(run_dir: Path, out_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = out_root.resolve()
    if resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to reset path outside output root: {resolved_run}")
    if run_dir.exists():
        shutil.rmtree(run_dir)


def command_for(args: argparse.Namespace, model: str, intervention: str, run_dir: Path) -> tuple[list[str], Path, Path]:
    if model == "transvisdrone":
        data = args.dataset_root / intervention / f"{intervention}_tvd.yaml"
        command = [
            str(args.tvd_python), "val.py", "--task", "test", "--data", str(data), "--weights", str(args.tvd_weights),
            "--img", "1280", "--batch-size", "2", "--half", "--num-frames", "5", "--conf-thres", "0.001",
            "--iou-thres", "0.6", "--device", args.device, "--save-txt", "--save-conf", "--project", str(run_dir.parent),
            "--name", run_dir.name, "--exist-ok",
        ]
        return command, args.tvd_repo, args.tvd_weights
    weights = args.yolomg_native_weights if model == "yolomg_native" else args.yolomg_ard_weights
    data = args.dataset_root / intervention / f"{intervention}_yolomg.yaml"
    command = [
        str(args.yolomg_python), "val.py", "--task", "test", "--data", str(data), "--weights", str(weights),
        "--img", "1280", "--batch-size", "4", "--workers", "4", "--conf-thres", "0.001", "--iou-thres", "0.4",
        "--device", args.device, "--save-txt", "--save-conf", "--project", str(run_dir.parent), "--name", run_dir.name, "--exist-ok",
    ]
    return command, args.yolomg_repo, weights


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.out_root = args.out_root.resolve()
    args.tvd_repo = args.tvd_repo.resolve()
    args.yolomg_repo = args.yolomg_repo.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    units = [(model, intervention) for model in args.models for intervention in args.interventions]
    progress_path = args.out_root / "progress.json"
    completed = 0
    skipped = []
    progress_path.write_text(json.dumps({"done": 0, "total": len(units), "last_completed_unit": None, "status": "starting", "skipped": []}, indent=2), encoding="utf-8")
    child_environment = os.environ.copy()
    for name in ("PYTHONHOME", "UV_INTERNAL__PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        child_environment.pop(name, None)
    for model, intervention in units:
        run_dir = args.out_root / model / intervention
        complete_path = run_dir / "complete.json"
        if complete_path.exists():
            completed += 1
            status = "already_complete"
        else:
            command, working_directory, weights = command_for(args, model, intervention, run_dir)
            data_path = Path(command[command.index("--data") + 1])
            missing = [str(path) for path in (Path(command[0]), working_directory, weights, data_path) if not path.exists()]
            if missing:
                skipped.append({"model": model, "intervention": intervention, "missing": missing})
                status = "skipped_missing_dependency"
            else:
                safe_reset(run_dir, args.out_root)
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                started = datetime.now().isoformat(timespec="seconds")
                print(f"START {model}/{intervention}: {' '.join(command)}", flush=True)
                result = subprocess.run(command, cwd=working_directory, env=child_environment, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Evaluation failed for {model}/{intervention}: returncode={result.returncode}")
                complete_path.write_text(
                    json.dumps({"model": model, "intervention": intervention, "started": started, "finished": datetime.now().isoformat(timespec="seconds"), "command": command}, indent=2),
                    encoding="utf-8",
                )
                completed += 1
                status = "completed"
        progress = {
            "done": completed,
            "total": len(units),
            "last_completed_unit": f"{model}/{intervention}",
            "status": status,
            "skipped": skipped,
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"[{completed}/{len(units)}] {model}/{intervention} {status}", flush=True)
    (args.out_root / "summary.json").write_text(json.dumps({"done": completed, "total": len(units), "skipped": skipped}, indent=2), encoding="utf-8")
    if skipped:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
