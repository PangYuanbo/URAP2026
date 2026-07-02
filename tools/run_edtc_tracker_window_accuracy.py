from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.evaluation.window_accuracy import run_window_accuracy


def _abs(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).absolute()


def _read_antiuav_sequence_names(dataset_root: Path) -> list[str]:
    list_path = dataset_root / "list.txt"
    return [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_antiuav_sequence(dataset_root: Path, sequence: str | None) -> tuple[str | None, str | None]:
    if sequence is None:
        return None, None

    names = _read_antiuav_sequence_names(dataset_root)
    if sequence in names:
        # EDTC's test.py parses strings like 20190926_130341_1_8 with int(),
        # because underscores are valid digit separators. Pass the index to
        # avoid accidentally selecting an impossible integer sequence id.
        return sequence, str(names.index(sequence))

    try:
        sequence_index = int(sequence)
    except ValueError:
        return sequence, sequence
    if sequence_index < 0 or sequence_index >= len(names):
        raise SystemExit(f"AntiUAV sequence index out of range: {sequence} (0..{len(names) - 1})")
    return names[sequence_index], sequence


def _write_antiuav_score_subset(dataset_root: Path, out_dir: Path, sequence_name: str | None) -> Path:
    if sequence_name is None:
        return dataset_root

    source_label = dataset_root / sequence_name / "IR_label.json"
    if not source_label.is_file():
        raise SystemExit(f"AntiUAV sequence label not found: {source_label}")

    subset_root = out_dir / "_score_dataset"
    subset_seq = subset_root / sequence_name
    subset_seq.mkdir(parents=True, exist_ok=True)
    (subset_root / "list.txt").write_text(sequence_name + "\n", encoding="utf-8")
    shutil.copyfile(source_label, subset_seq / "IR_label.json")
    return subset_root


def _write_local_py(repo: Path, antiuav_root: Path, results_root: Path, show_result: bool) -> Path:
    local_py = repo / "lib" / "test" / "evaluation" / "local.py"
    local_py.parent.mkdir(parents=True, exist_ok=True)
    text = f"""from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = {str(repo)!r}
    settings.save_dir = {str(results_root.parent)!r}
    settings.results_path = {str(results_root)!r}
    settings.network_path = {str(repo / "test" / "networks")!r}
    settings.antiuav_path = {str(antiuav_root)!r}
    settings.show_result = {bool(show_result)!r}
    return settings
"""
    local_py.write_text(text, encoding="utf-8")
    return local_py


def _write_tracker_config(
    repo: Path,
    tracker_name: str,
    template_config: Path,
    config_name: str,
    yolo_weights: Path,
    yolo_data: Path,
    device: str,
) -> Path:
    target_dir = repo / "experiments" / tracker_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{config_name}.yaml"
    data = yaml.safe_load(template_config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{template_config}: expected YAML mapping")
    yolo_cfg = data.setdefault("YOLO", {})
    if not isinstance(yolo_cfg, dict):
        raise TypeError(f"{template_config}: YOLO must be a mapping")
    yolo_cfg["WEIGHTS"] = str(yolo_weights)
    yolo_cfg["DATA"] = str(yolo_data)
    yolo_cfg["DEVICE"] = str(device)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def _build_tracker_command(args: argparse.Namespace, repo: Path, config_name: str, sequence_arg: str | None) -> list[str]:
    tracker_model = str(_abs(args.tracker_model)) if args.tracker_model else "<UAVTrackEH.pth.tar>"
    cmd = [
        str(_abs(args.python)),
        str(repo / "tracking" / "test.py"),
        "--tracker_name",
        args.tracker_name,
        "--tracker_param",
        config_name,
        "--dataset_name",
        "antiuav",
        "--threads",
        str(args.threads),
        "--num_gpus",
        str(args.num_gpus),
        "--params__model",
        tracker_model,
        "--params__search_area_scale",
        str(args.search_area_scale),
    ]
    if sequence_arg:
        cmd.extend(["--sequence", sequence_arg])
    return cmd


def _run_tracker(cmd: list[str], repo: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    paths = [str(repo), str(repo / "yolov5")]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
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
        raise SystemExit(f"EDTC tracker failed with exit code {returncode}. Log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDTC tracker on AntiUAV, then render per-video +/-3s window accuracy curves."
    )
    parser.add_argument("--repo", type=Path, default=ROOT / "papers" / "EDTC")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset-root", type=Path, required=True, help="AntiUAV root containing list.txt and */IR_label.json.")
    parser.add_argument("--tracker-name", default="uavtrack_eh")
    parser.add_argument("--template-config", type=Path, default=None)
    parser.add_argument("--config-name", default="urap_window_accuracy")
    parser.add_argument("--tracker-model", type=Path, default=None, help="UAVTrackEH .pth.tar checkpoint.")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="EDTC YOLO detector weights.")
    parser.add_argument("--yolo-data", type=Path, default=None, help="YOLO data YAML used by DetectMultiBackend.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--sequence", default=None, help="Optional single AntiUAV sequence name/index for smoke runs.")
    parser.add_argument("--search-area-scale", type=float, default=4.55)
    parser.add_argument("--show-result", action="store_true")
    parser.add_argument("--skip-track", action="store_true", help="Only score an existing --results-dir.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--segment-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = _abs(args.repo)
    dataset_root = _abs(args.dataset_root)
    out_dir = _abs(args.out)
    results_root = _abs(args.results_root or (out_dir / "tracking_results"))
    results_dir = _abs(args.results_dir or (results_root / args.tracker_name / args.config_name))
    template_config = _abs(
        args.template_config or (repo / "experiments" / args.tracker_name / "baseline.yaml")
    )
    sequence_name: str | None = None
    tracker_sequence_arg: str | None = args.sequence
    score_dataset_root = dataset_root
    if dataset_root.is_dir() and (dataset_root / "list.txt").is_file():
        sequence_name, tracker_sequence_arg = _resolve_antiuav_sequence(dataset_root, args.sequence)
        score_dataset_root = (out_dir / "_score_dataset") if sequence_name is not None else dataset_root

    cmd = _build_tracker_command(args, repo=repo, config_name=args.config_name, sequence_arg=tracker_sequence_arg)
    dry_summary: dict[str, Any] = {
        "repo": str(repo),
        "dataset_root": str(dataset_root),
        "score_dataset_root": str(score_dataset_root),
        "sequence_name": sequence_name,
        "tracker_sequence_arg": tracker_sequence_arg,
        "local_py": str(repo / "lib" / "test" / "evaluation" / "local.py"),
        "template_config": str(template_config),
        "generated_config": str(repo / "experiments" / args.tracker_name / f"{args.config_name}.yaml"),
        "results_root": str(results_root),
        "results_dir": str(results_dir),
        "out": str(out_dir),
        "tracker_command": cmd,
        "skip_track": bool(args.skip_track),
    }
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2))
        return 0

    if not repo.is_dir():
        raise SystemExit(f"EDTC repo not found: {repo}")
    if not (dataset_root / "list.txt").is_file():
        raise SystemExit(f"AntiUAV list.txt not found: {dataset_root / 'list.txt'}")
    score_dataset_root = _write_antiuav_score_subset(dataset_root, out_dir, sequence_name)
    if not args.skip_track:
        if args.tracker_model is None or args.yolo_weights is None or args.yolo_data is None:
            raise SystemExit("--tracker-model, --yolo-weights, and --yolo-data are required unless --skip-track is set.")
        tracker_model = _abs(args.tracker_model)
        yolo_weights = _abs(args.yolo_weights)
        yolo_data = _abs(args.yolo_data)
        for required in (template_config, tracker_model, yolo_weights, yolo_data):
            if not required.exists():
                raise SystemExit(f"Required EDTC input not found: {required}")
        local_py = _write_local_py(repo, antiuav_root=dataset_root, results_root=results_root, show_result=args.show_result)
        config_path = _write_tracker_config(
            repo,
            tracker_name=args.tracker_name,
            template_config=template_config,
            config_name=args.config_name,
            yolo_weights=yolo_weights,
            yolo_data=yolo_data,
            device=args.device,
        )
        _run_tracker(cmd, repo=repo, log_path=out_dir / "edtc_tracker.log")
    else:
        local_py = repo / "lib" / "test" / "evaluation" / "local.py"
        config_path = repo / "experiments" / args.tracker_name / f"{args.config_name}.yaml"

    if not results_dir.is_dir() or not any(results_dir.glob("*.txt")):
        raise SystemExit(f"EDTC tracker results not found or empty: {results_dir}")

    summary = run_window_accuracy(
        gt=score_dataset_root,
        pred=results_dir,
        out_dir=out_dir,
        fps=args.fps,
        gt_format="antiuav-json",
        pred_format="xywh-file",
        frame_manifest=score_dataset_root,
        frame_manifest_format="antiuav-json",
        window_seconds=args.window_seconds,
        iou_threshold=args.iou,
        score_threshold=args.score_threshold,
        segment_threshold=args.segment_threshold,
        extra_summary={
            "method": "EDTC",
            "tracker_command": None if args.skip_track else cmd,
            "tracker_results_dir": str(results_dir),
            "edtc_local_py": str(local_py),
            "edtc_config": str(config_path),
        },
    )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"plot_index={summary['plot_index']}")
    print(f"worst_windows={summary['worst_windows_csv']}")
    print(f"low_accuracy_segments={summary['low_accuracy_segments_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
