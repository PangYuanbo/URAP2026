from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absoluteish(value: str) -> bool:
    return WINDOWS_ABS_RE.match(value) is not None or Path(value).expanduser().is_absolute()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if _is_absoluteish(value) else (base / path).resolve()


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _has_txt(path: Path) -> bool:
    return (path.is_file() and path.suffix == ".txt") or (path.is_dir() and any(path.rglob("*.txt")))


def _has_aot_json(path: Path) -> bool:
    if path.is_file() and path.name == "result.json":
        return True
    return path.is_dir() and ((path / "result.json").is_file() or any((child / "result.json").is_file() for child in path.iterdir() if child.is_dir()))


def _has_aot_groundtruth_json(path: Path) -> bool:
    if path.is_file() and path.name == "groundtruth.json":
        return True
    return path.is_dir() and ((path / "groundtruth.json").is_file() or (path / "ImageSets" / "groundtruth.json").is_file())


def _has_antiuav_json(path: Path) -> bool:
    return path.is_dir() and ((path / "IR_label.json").is_file() or (path / "list.txt").is_file() or any(path.glob("*/IR_label.json")))


def _has_tvd_pkl(path: Path) -> bool:
    return path.is_file() and path.suffix == ".pkl"


def _valid_gt(path: Path, fmt: str) -> bool:
    if fmt == "yolo-dir":
        return path.is_dir() and any(path.rglob("*.txt"))
    if fmt in {"li-tetc-txt", "xywh-file"}:
        return _has_txt(path)
    if fmt == "antiuav-json":
        return _has_antiuav_json(path)
    if fmt == "aot-json":
        return _has_aot_json(path)
    if fmt == "aot-gt-json":
        return _has_aot_groundtruth_json(path)
    if fmt in {"tvd-pkl-gt", "tvd-pkl-pred"}:
        return _has_tvd_pkl(path)
    if fmt == "csv":
        return path.is_file()
    return path.exists()


def _valid_pred(path: Path, fmt: str) -> bool:
    if fmt == "yolo-dir":
        return path.is_dir() and any(path.rglob("*.txt"))
    if fmt in {"li-tetc-txt", "xywh-file"}:
        return _has_txt(path)
    if fmt == "aot-json":
        return _has_aot_json(path)
    if fmt == "aot-gt-json":
        return _has_aot_groundtruth_json(path)
    if fmt == "antiuav-json":
        return _has_antiuav_json(path)
    if fmt in {"tvd-pkl-gt", "tvd-pkl-pred"}:
        return _has_tvd_pkl(path)
    if fmt == "csv":
        return path.is_file()
    return path.exists()


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "run"


def _li_tetc_video_id(pred_file: Path) -> str:
    stem = pred_file.stem
    for suffix in ("_dt", "_pred", "_result"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def _li_tetc_fps(base: Path, video_id: str) -> float | None:
    video_path = base / "papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking" / "Data" / "Videos" / f"Clip_{video_id}.mov"
    if not video_path.is_file():
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else None


def _tvd_dataset_slug(path: Path) -> str:
    parts = path.parts
    for dataset in ("NPS", "Fl", "FL", "AOT"):
        if dataset in parts:
            return _slug(dataset)
    return "unknown"


def _first_existing(paths: list[str], base: Path, fmt: str, kind: str) -> tuple[Path | None, list[str]]:
    checked = []
    for raw in paths:
        path = _resolve(raw, base)
        checked.append(str(path))
        ok = _valid_gt(path, fmt) if kind == "gt" else _valid_pred(path, fmt)
        if ok:
            return path, checked
    return None, checked


def _discover_yolo_runs(
    base: Path,
    method: str,
    gt_candidates: list[str],
    pred_globs: list[str],
    pred_root: str,
    extra: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt, checked_gt = _first_existing(gt_candidates, base, "yolo-dir", "gt")
    pred_dirs = []
    for pattern in pred_globs:
        pred_dirs.extend(sorted(base.glob(pattern)))
    pred_dirs = [p for p in pred_dirs if _valid_pred(p, "yolo-dir")]
    runs = []
    for pred in pred_dirs:
        name = f"{_slug(method)}_{_slug(pred.parent.name)}"
        run = {
            "name": name,
            "method": method,
            "gt": _rel(gt, base) if gt else gt_candidates[0],
            "gt_format": "yolo-dir",
            "pred": _rel(pred, base),
            "pred_format": "yolo-dir",
        }
        if extra:
            run.update(extra)
        runs.append(run)
    report = {
        "method": method,
        "gt_found": str(gt) if gt else None,
        "gt_checked": checked_gt,
        "pred_found": [str(p) for p in pred_dirs],
        "pred_root": str(_resolve(pred_root, base)),
        "run_count": len(runs),
    }
    return runs, report


def discover_runs(base_dir: str | Path = ROOT) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    runs: list[dict[str, Any]] = []
    discovery: list[dict[str, Any]] = []

    specs = [
        (
            "YOLOMG",
            [
                "runs/window_accuracy/yolomg_test_images_dataset/labels",
                "D:/URAP_datasets/ARD100_YOLOMG/labels/test",
                "datasets/ARD100_YOLOMG/labels/test",
            ],
            ["runs/window_accuracy/yolomg_test_images_eval/eval/**/labels", "papers/YOLOMG/runs/**/labels"],
            "papers/YOLOMG/runs",
            {},
        ),
        (
            "TransVisDrone",
            [
                "D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels",
                "datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels",
            ],
            ["papers/TransVisDrone/runs/**/labels"],
            "papers/TransVisDrone/runs",
            {"gt_frame_offset": 1},
        ),
        (
            "ESOD",
            [
                "runs/window_accuracy/yolomg_test_images_dataset/labels",
                "D:/URAP_datasets/VisDrone/VisDrone2019-DET-val/labels",
                "datasets/VisDrone/VisDrone2019-DET-val/labels",
            ],
            ["runs/window_accuracy/esod_test_images_eval/eval/**/labels", "papers/ESOD/runs/**/labels"],
            "papers/ESOD/runs",
            {},
        ),
        (
            "EDTC",
            [
                "runs/window_accuracy/yolomg_test_images_dataset/labels",
                "D:/URAP_datasets/AntiUAV600/labels",
                "datasets/AntiUAV600/labels",
            ],
            ["runs/window_accuracy/edtc_yolo_test_images_eval/eval/**/labels", "papers/EDTC/yolov5/runs/**/labels"],
            "papers/EDTC/yolov5/runs",
            {"score_threshold": 0.001},
        ),
    ]
    for method, gt_candidates, pred_globs, pred_root, extra in specs:
        found, report = _discover_yolo_runs(base, method, gt_candidates, pred_globs, pred_root, extra)
        runs.extend(found)
        discovery.append(report)

    tvd_pkls = sorted(base.glob("papers/TransVisDrone/runs/val/**/predictionsgt/predictionsgt_split_*.pkl"))
    tvd_extra_pred_pkls: list[Path] = []
    for pkl in tvd_pkls:
        run_parent = pkl.parent.parent
        experiment_root = run_parent.parent
        dataset_slug = _tvd_dataset_slug(run_parent)
        runs.append(
            {
                "name": f"transvisdrone_pkl_{dataset_slug}_{_slug(run_parent.name)}",
                "method": "TransVisDrone",
                "gt": _rel(pkl, base),
                "gt_format": "tvd-pkl-gt",
                "pred": _rel(pkl, base),
                "pred_format": "tvd-pkl-pred",
            }
        )
        for pred_pkl in sorted(list(experiment_root.glob("*/best_predictions.pkl")) + list(experiment_root.glob("*/last_predictions.pkl"))):
            if pred_pkl.parent == run_parent:
                continue
            tvd_extra_pred_pkls.append(pred_pkl)
            runs.append(
                {
                    "name": f"transvisdrone_pkl_{dataset_slug}_{_slug(pred_pkl.parent.name)}",
                    "method": "TransVisDrone",
                    "gt": _rel(pkl, base),
                    "gt_format": "tvd-pkl-gt",
                    "pred": _rel(pred_pkl, base),
                    "pred_format": "tvd-pkl-pred",
                }
            )
    discovery.append(
        {
            "method": "TransVisDrone_predictionsgt_pkl",
            "gt_found": "embedded in predictionsgt pkl" if tvd_pkls else None,
            "gt_checked": [str(base / "papers/TransVisDrone/runs/val/**/predictionsgt/predictionsgt_split_*.pkl")],
            "pred_found": [str(p) for p in tvd_pkls] + [str(p) for p in tvd_extra_pred_pkls],
            "pred_root": str(base / "papers/TransVisDrone/runs/val"),
            "run_count": len(tvd_pkls) + len(tvd_extra_pred_pkls),
        }
    )

    aot_gt, aot_gt_checked = _first_existing(
        ["D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels", "datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels"],
        base,
        "yolo-dir",
        "gt",
    )
    official_aot_gt, official_aot_gt_checked = _first_existing(
        ["D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json", "datasets/AOT/part1/ImageSets/groundtruth.json"],
        base,
        "aot-gt-json",
        "gt",
    )
    aot_preds = [
        p
        for p in sorted(base.glob("papers/AICrowd_AOT_Challenge_Winner/runs/**/results*"))
        if _valid_pred(p, "aot-json")
    ]
    for pred in aot_preds:
        pred_text = str(pred).lower()
        if "aot" in pred_text or "part1" in pred_text:
            run = {
                "name": f"aicrowd_winner_{_slug(pred.name)}",
                "method": "AICrowd_Winner_v022",
                "gt": _rel(official_aot_gt, base) if official_aot_gt else "D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json",
                "gt_format": "aot-gt-json",
                "pred": _rel(pred, base),
                "pred_format": "aot-json",
                "fps": 10,
            }
        else:
            run = {
                "name": f"aicrowd_winner_{_slug(pred.name)}",
                "method": "AICrowd_Winner_v022",
                "gt": _rel(aot_gt, base) if aot_gt else "D:/URAP_datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels",
                "gt_format": "yolo-dir",
                "gt_frame_offset": 1,
                "pred": _rel(pred, base),
                "pred_format": "aot-json",
                "img_width": 1280,
                "img_height": 960,
            }
        runs.append(run)
    discovery.append(
        {
            "method": "AICrowd_Winner_v022",
            "gt_found": str(aot_gt or official_aot_gt) if (aot_gt or official_aot_gt) else None,
            "gt_checked": aot_gt_checked + official_aot_gt_checked,
            "pred_found": [str(p) for p in aot_preds],
            "pred_root": str(base / "papers/AICrowd_AOT_Challenge_Winner/runs"),
            "run_count": len(aot_preds),
        }
    )

    edtc_gt, edtc_gt_checked = _first_existing(
        ["D:/URAP_datasets/AntiUAV600", "datasets/AntiUAV600", "papers/EDTC/data"],
        base,
        "antiuav-json",
        "gt",
    )
    edtc_preds = [
        p
        for p in sorted(base.glob("papers/EDTC/**/results*"))
        if p.is_dir() and _valid_pred(p, "xywh-file")
    ]
    for pred in edtc_preds:
        runs.append(
            {
                "name": f"edtc_{_slug(pred.name)}",
                "method": "EDTC",
                "gt": _rel(edtc_gt, base) if edtc_gt else "D:/URAP_datasets/AntiUAV600",
                "gt_format": "antiuav-json",
                "pred": _rel(pred, base),
                "pred_format": "xywh-file",
            }
        )
    discovery.append(
        {
            "method": "EDTC",
            "gt_found": str(edtc_gt) if edtc_gt else None,
            "gt_checked": edtc_gt_checked,
            "pred_found": [str(p) for p in edtc_preds],
            "pred_root": str(base / "papers/EDTC"),
            "run_count": len(edtc_preds),
        }
    )

    li_gt, li_gt_checked = _first_existing(
        ["papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925"],
        base,
        "li-tetc-txt",
        "gt",
    )
    li_pred = base / "papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Experiment_Results/Final/txt"
    li_pred_files = sorted(li_pred.glob("*.txt")) if li_pred.is_dir() else []
    if li_pred_files:
        for pred_file in li_pred_files:
            video_id = _li_tetc_video_id(pred_file)
            gt_file = li_gt / f"Video_{video_id}_gt.txt" if li_gt and li_gt.is_dir() else li_gt
            run = {
                "name": f"li_tetc_video_{_slug(video_id)}",
                "method": "Li_TETC_NPS",
                "gt": _rel(gt_file, base) if gt_file else "papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925",
                "gt_format": "li-tetc-txt",
                "pred": _rel(pred_file, base),
                "pred_format": "li-tetc-txt",
            }
            fps = _li_tetc_fps(base, video_id)
            if fps:
                run["fps"] = fps
            runs.append(run)
    elif _valid_pred(li_pred, "li-tetc-txt"):
        runs.append(
            {
                "name": "li_tetc_nps",
                "method": "Li_TETC_NPS",
                "gt": _rel(li_gt, base) if li_gt else "papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/Data/Annotation_update_180925",
                "gt_format": "li-tetc-txt",
                "pred": _rel(li_pred, base),
                "pred_format": "li-tetc-txt",
            }
        )
    discovery.append(
        {
            "method": "Li_TETC_NPS",
            "gt_found": str(li_gt) if li_gt else None,
            "gt_checked": li_gt_checked,
            "pred_found": [str(p) for p in li_pred_files] or ([str(li_pred)] if _valid_pred(li_pred, "li-tetc-txt") else []),
            "pred_root": str(li_pred),
            "run_count": len(li_pred_files) or (1 if _valid_pred(li_pred, "li-tetc-txt") else 0),
        }
    )

    manifest = {
        "out_root": "runs/window_accuracy/discovered",
        "defaults": {"fps": 30, "window_seconds": 3, "iou": 0.5, "score_threshold": 0.25},
        "runs": runs,
    }
    return {
        "base_dir": str(base),
        "manifest": manifest,
        "discovery": discovery,
        "run_count": len(runs),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Discovered Paper Window Accuracy Runs",
        "",
        f"- Base dir: `{report['base_dir']}`",
        f"- Discovered runs: `{report['run_count']}`",
        "",
        "| Method | Runs | GT found | Prediction outputs found |",
        "| --- | ---: | --- | ---: |",
    ]
    for item in report["discovery"]:
        lines.append(
            f"| {item['method']} | {item['run_count']} | {item['gt_found'] or '-'} | {len(item['pred_found'])} |"
        )
    if report["manifest"]["runs"]:
        lines.extend(["", "## Runs", ""])
        for run in report["manifest"]["runs"]:
            lines.append(f"- `{run['name']}`: `{run['method']}` -> `{run['pred']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover real paper prediction outputs and write a runnable window-accuracy manifest.")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--manifest-out", type=Path, default=ROOT / "runs" / "window_accuracy" / "discovered_manifest.json")
    parser.add_argument("--report-json", type=Path, default=ROOT / "runs" / "window_accuracy" / "discovery_report.json")
    parser.add_argument("--report-md", type=Path, default=ROOT / "runs" / "window_accuracy" / "discovery_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = discover_runs(args.base_dir)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(report["manifest"], indent=2), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.report_md, report)
    print(f"runs={report['run_count']}")
    print(f"manifest={args.manifest_out}")
    print(f"report={args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
