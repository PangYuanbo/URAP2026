from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def write_state(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kwargs, indent=2), encoding="utf-8")


def run(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, state_path: Path, stage: str) -> None:
    write_state(
        state_path,
        status="running",
        stage=stage,
        current_stdout=str(stdout_path),
        current_stderr=str(stderr_path),
        command=" ".join(cmd),
    )
    with stdout_path.open("w", encoding="utf-8", errors="ignore") as out, stderr_path.open("w", encoding="utf-8", errors="ignore") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        write_state(
            state_path,
            status="failed",
            stage=stage,
            current_stdout=str(stdout_path),
            current_stderr=str(stderr_path),
            returncode=proc.returncode,
            command=" ".join(cmd),
        )
        raise SystemExit(proc.returncode)


def collect_by_video(root: Path, dataset: str, videos: list[int]) -> list[Path]:
    wanted = {f"Clip_{v:03d}" for v in videos}
    out: list[Path] = []
    for p in sorted((root / "images" / dataset).glob("**/*.jpg")):
        if p.parent.name in wanted:
            out.append(p.resolve())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--train-videos", default="1-40")
    p.add_argument("--val-videos", default="41-50")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-boxes", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    repo = args.repo.resolve()
    out = args.out.resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = out / "pipeline_state.json"
    py = repo / "papers" / "ESOD" / ".venv" / "Scripts" / "python.exe"
    build_script = repo / "tools" / "build_motion_esod_rois.py"
    esod_dir = repo / "papers" / "ESOD"

    def expand_ranges(spec: str) -> list[int]:
        vals: list[int] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-", 1)
                vals.extend(range(int(a), int(b) + 1))
            else:
                vals.append(int(chunk))
        return vals

    train_videos = expand_ranges(args.train_videos)
    val_videos = expand_ranges(args.val_videos)
    all_videos = sorted(set(train_videos + val_videos))
    videos_arg = ",".join(str(v) for v in all_videos)

    build_cmd = [
        str(py),
        str(build_script),
        "--dataset",
        "nps",
        "--videos",
        videos_arg,
        "--out",
        str(out),
        "--stride",
        str(args.stride),
        "--max-boxes",
        str(args.max_boxes),
    ]
    if args.max_frames > 0:
        build_cmd += ["--max-frames", str(args.max_frames)]
    run(build_cmd, repo, logs / "01_build_rois.out.log", logs / "01_build_rois.err.log", state, "build_rois")

    train_images = collect_by_video(out, "nps", train_videos)
    val_images = collect_by_video(out, "nps", val_videos)
    train_txt = out / "nps_train_1_40.txt"
    val_txt = out / "nps_val_41_50.txt"
    data_yaml = esod_dir / "data" / "nps_motion_esod.yaml"
    train_txt.write_text("\n".join(str(p) for p in train_images) + ("\n" if train_images else ""), encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in val_images) + ("\n" if val_images else ""), encoding="utf-8")
    data_yaml.write_text(
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        "nc: 1\n"
        "names: ['UAV']\n",
        encoding="utf-8",
    )
    write_state(
        state,
        status="running",
        stage="train_esod",
        train_images=len(train_images),
        val_images=len(val_images),
        data_yaml=str(data_yaml),
    )

    train_cmd = [
        str(py),
        "train.py",
        "--data",
        str(data_yaml),
        "--cfg",
        "models\\cfg\\esod\\visdrone_yolov5m.yaml",
        "--weights",
        "weights\\esod_pretrained\\esod_yolov5m.pt",
        "--hyp",
        "data\\hyps\\hyp.visdrone.yaml",
        "--batch-size",
        str(args.batch_size),
        "--img-size",
        str(args.img_size),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--workers",
        "0",
        "--single-cls",
        "--project",
        "runs\\train",
        "--name",
        f"nps_motion_esod_e{args.epochs}_img{args.img_size}",
    ]
    run(train_cmd, esod_dir, logs / "02_train_esod.out.log", logs / "02_train_esod.err.log", state, "train_esod")
    write_state(
        state,
        status="complete",
        stage="complete",
        train_images=len(train_images),
        val_images=len(val_images),
        data_yaml=str(data_yaml),
        train_log=str(logs / "02_train_esod.err.log"),
    )


if __name__ == "__main__":
    main()
