from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.evaluation.window_accuracy import run_window_accuracy


METHODS = {
    "yolomg": {
        "label": "YOLOMG",
        "repo": "papers/YOLOMG",
        "script": "val.py",
        "img_arg": "--img",
        "supports_num_frames": False,
    },
    "transvisdrone": {
        "label": "TransVisDrone",
        "repo": "papers/TransVisDrone",
        "script": "val.py",
        "img_arg": "--img",
        "supports_num_frames": True,
    },
    "esod": {
        "label": "ESOD",
        "repo": "papers/ESOD",
        "script": "test.py",
        "img_arg": "--img-size",
        "supports_num_frames": False,
    },
    "edtc": {
        "label": "EDTC",
        "repo": "papers/EDTC/yolov5",
        "script": "val.py",
        "img_arg": "--img",
        "supports_num_frames": False,
    },
}


def _default_python(repo: Path) -> Path:
    win_venv = repo / ".venv" / "Scripts" / "python.exe"
    unix_venv = repo / ".venv" / "bin" / "python"
    if win_venv.exists():
        return win_venv
    if unix_venv.exists():
        return unix_venv
    return Path(sys.executable)


def _absolute_without_resolving_symlink(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).absolute()


def _split_labels(value: str) -> list[str] | None:
    labels = [part.strip() for part in value.split(",") if part.strip()]
    return labels or None


def build_eval_command(args: argparse.Namespace, repo: Path, python_exe: Path, project: Path) -> list[str]:
    spec = METHODS[args.method]
    cmd = [
        str(python_exe),
        str(repo / spec["script"]),
        "--task",
        args.task,
        "--data",
        str(args.data),
        "--weights",
        str(args.weights),
        str(spec["img_arg"]),
        str(args.img),
        "--batch-size",
        str(args.batch_size),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.nms_iou),
        "--device",
        args.device,
        "--save-txt",
        "--save-conf",
        "--project",
        str(project),
        "--name",
        args.name,
        "--exist-ok",
    ]
    if args.half:
        cmd.append("--half")
    if args.augment:
        cmd.append("--augment")
    if spec["supports_num_frames"]:
        cmd.extend(["--num-frames", str(args.num_frames)])
    cmd.extend(args.extra_eval_arg or [])
    return cmd


def run_eval(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    # These paper forks call YOLOv5 auto-requirement installers at startup.
    # Keep eval reproducible and prevent them from mutating the selected venv.
    env.setdefault("PIP_NO_INDEX", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    python_bin = Path(cmd[0]).parent
    if python_bin.is_dir():
        env["PATH"] = str(python_bin) + os.pathsep + env.get("PATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(f"Eval command failed with exit code {returncode}. Log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOMG/TransVisDrone/ESOD eval with saved labels, then render +/-3s per-video accuracy curves."
    )
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML for the paper eval command.")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--gt-format", default="yolo-dir", choices=["csv", "jsonl", "yolo-dir", "aot-json", "aot-gt-json", "xywh-file", "antiuav-json", "li-tetc-txt"])
    parser.add_argument("--out", type=Path, required=True, help="Curve output directory.")
    parser.add_argument("--project", type=Path, default=None, help="Paper eval project directory. Default: <out>/eval")
    parser.add_argument("--name", default="window_accuracy_eval")
    parser.add_argument("--task", default="val")
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--segment-threshold", type=float, default=0.5)
    parser.add_argument("--gt-frame-offset", type=int, default=0)
    parser.add_argument("--pred-frame-offset", type=int, default=0)
    parser.add_argument("--frame-manifest", type=Path, default=None)
    parser.add_argument("--frame-manifest-format", default=None, choices=["csv", "jsonl", "yolo-dir", "aot-json", "aot-gt-json", "xywh-file", "antiuav-json", "li-tetc-txt", "tvd-pkl-gt", "tvd-pkl-pred", "image-dir"])
    parser.add_argument("--frame-manifest-offset", type=int, default=0)
    parser.add_argument("--img-width", type=float, default=None)
    parser.add_argument("--img-height", type=float, default=None)
    parser.add_argument("--gt-labels", default="")
    parser.add_argument("--pred-labels", default="")
    parser.add_argument("--skip-eval", action="store_true", help="Use an existing --pred-labels-dir instead of running paper eval.")
    parser.add_argument("--pred-labels-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the eval command and exit before running anything.")
    parser.add_argument(
        "--extra-eval-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the paper eval command. Use --extra-eval-arg=--flag for values that start with '-'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = METHODS[args.method]
    repo = (args.repo or (ROOT / spec["repo"])).resolve()
    project = (args.project or (args.out / "eval")).resolve()
    python_exe = _absolute_without_resolving_symlink(args.python or _default_python(repo))
    if args.data is not None:
        args.data = _absolute_without_resolving_symlink(args.data)
    if args.weights is not None:
        args.weights = _absolute_without_resolving_symlink(args.weights)
    labels_dir = args.pred_labels_dir.resolve() if args.pred_labels_dir else (project / args.name / "labels")

    if (args.img_width is None) != (args.img_height is None):
        raise SystemExit("--img-width and --img-height must be provided together.")
    img_size = (args.img_width, args.img_height) if args.img_width is not None else None

    command: list[str] | None = None
    if not args.skip_eval:
        if args.data is None or args.weights is None:
            raise SystemExit("--data and --weights are required unless --skip-eval is set.")
        command = build_eval_command(args, repo=repo, python_exe=python_exe, project=project)
        if args.dry_run:
            print(json.dumps({"method": spec["label"], "command": command, "labels_dir": str(labels_dir)}, indent=2))
            return 0
        run_eval(command, cwd=repo, log_path=args.out / "eval.log")
    elif args.dry_run:
        print(json.dumps({"method": spec["label"], "skip_eval": True, "labels_dir": str(labels_dir)}, indent=2))
        return 0

    if not labels_dir.is_dir():
        raise SystemExit(f"Prediction labels directory not found: {labels_dir}")

    summary = run_window_accuracy(
        gt=args.gt,
        pred=labels_dir,
        out_dir=args.out,
        fps=args.fps,
        gt_format=args.gt_format,
        pred_format="yolo-dir",
        window_seconds=args.window_seconds,
        iou_threshold=args.match_iou,
        score_threshold=args.score_threshold,
        segment_threshold=args.segment_threshold,
        gt_labels=_split_labels(args.gt_labels),
        pred_labels=_split_labels(args.pred_labels),
        gt_frame_offset=args.gt_frame_offset,
        pred_frame_offset=args.pred_frame_offset,
        frame_manifest=args.frame_manifest,
        frame_manifest_format=args.frame_manifest_format,
        frame_manifest_offset=args.frame_manifest_offset,
        img_size=img_size,
        extra_summary={
            "method": spec["label"],
            "eval_command": command,
            "prediction_labels_dir": str(labels_dir),
        },
    )
    print(f"summary={args.out / 'summary.json'}")
    print(f"plot_index={summary['plot_index']}")
    print(f"worst_windows={summary['worst_windows_csv']}")
    print(f"low_accuracy_segments={summary['low_accuracy_segments_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
