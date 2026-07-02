from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_paper_window_accuracy_batch import run_manifest


BOX_NORM = "0 0.500000 0.500000 0.200000 0.200000"
BOX_NORM_SHIFT = "0 0.700000 0.700000 0.120000 0.120000"
BOX_ABS = {"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "s": 0.9}
BOX_ABS_SHIFT = {"x": 70.0, "y": 70.0, "w": 12.0, "h": 12.0, "s": 0.8}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_yolo_case(root: Path, prefix: str, frame_offset_gt: int = 0) -> dict[str, Any]:
    gt_dir = root / prefix / "gt"
    pred_dir = root / prefix / "pred"
    frames_dir = root / prefix / "frames"
    gt_frames = [1, 5, 10, 15, 20]
    correct_pred_frames = [1, 5, 15]
    fp_frames = [8, 18]

    for frame in range(1, 21):
        _write(frames_dir / f"Clip_{prefix}_{frame:05d}.jpg", "")
    for frame in gt_frames:
        gt_frame = frame - frame_offset_gt
        _write(gt_dir / f"Clip_{prefix}_{gt_frame:05d}.txt", BOX_NORM + "\n")
    for frame in correct_pred_frames:
        _write(pred_dir / f"Clip_{prefix}_{frame:05d}.txt", BOX_NORM + " 0.90\n")
    for frame in fp_frames:
        _write(pred_dir / f"Clip_{prefix}_{frame:05d}.txt", BOX_NORM_SHIFT + " 0.70\n")
    return {
        "gt": str(gt_dir.relative_to(root.parent)),
        "pred": str(pred_dir.relative_to(root.parent)),
        "frame_manifest": str(frames_dir.relative_to(root.parent)),
    }


def _write_aot_case(root: Path) -> dict[str, Any]:
    gt_dir = root / "aicrowd" / "aot" / "ImageSets"
    pred_dir = root / "aicrowd" / "pred" / "Clip_AICrowd"
    frames_dir = root / "aicrowd" / "frames" / "Clip_AICrowd"
    entities = []
    for frame in [1, 5, 10, 15, 20]:
        entities.append(
            {
                "blob": {"frame": frame},
                "id": f"object-{frame}",
                "bb": [40.0, 40.0, 20.0, 20.0],
                "flight_id": "Clip_AICrowd",
                "img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png",
            }
        )
    _write(
        gt_dir / "groundtruth.json",
        json.dumps({"metadata": {"fps": 2}, "samples": {"Clip_AICrowd": {"entities": entities}}}, indent=2),
    )

    records = []
    for frame in [1, 5, 15]:
        records.append({"img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png", "frame": frame, "detections": [BOX_ABS]})
    for frame in [8, 18]:
        records.append({"img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png", "frame": frame, "detections": [BOX_ABS_SHIFT]})
    _write(pred_dir / "result.json", json.dumps(records, indent=2))
    for frame in range(1, 21):
        _write(frames_dir / f"Clip_AICrowd_{frame:05d}.png", "")
    return {
        "gt": str((root / "aicrowd" / "aot").relative_to(root.parent)),
        "pred": str((root / "aicrowd" / "pred").relative_to(root.parent)),
        "frame_manifest": str((root / "aicrowd" / "frames").relative_to(root.parent)),
    }


def _write_antiuav_case(root: Path) -> dict[str, Any]:
    dataset = root / "edtc" / "dataset"
    seq = dataset / "EDTC_seq"
    pred = root / "edtc" / "pred"
    gt_rect = []
    exist = []
    for frame in range(1, 21):
        if frame in {1, 5, 10, 15, 20}:
            gt_rect.append([40, 40, 20, 20])
            exist.append(1)
        else:
            gt_rect.append([0, 0, 0, 0])
            exist.append(0)
    _write(dataset / "list.txt", "EDTC_seq\n")
    _write(seq / "IR_label.json", json.dumps({"gt_rect": gt_rect, "exist": exist}, indent=2))

    pred_lines = []
    for frame in range(1, 21):
        if frame in {1, 5, 15}:
            pred_lines.append("40\t40\t20\t20")
        elif frame in {8, 18}:
            pred_lines.append("70\t70\t12\t12")
        else:
            pred_lines.append("0\t0\t0\t0")
    _write(pred / "EDTC_seq.txt", "\n".join(pred_lines) + "\n")
    return {"gt": str(dataset.relative_to(root.parent)), "pred": str(pred.relative_to(root.parent))}


def _li_box(y1: int, x1: int, y2: int, x2: int) -> str:
    return f"({y1}, {x1}, {y2}, {x2}), "


def _write_li_tetc_case(root: Path) -> dict[str, Any]:
    gt = root / "li_tetc" / "gt"
    pred = root / "li_tetc" / "pred"
    gt_lines = []
    pred_lines = []
    for frame in range(1, 21):
        gt_det = _li_box(40, 40, 60, 60) if frame in {1, 5, 10, 15, 20} else ""
        if frame in {1, 5, 15}:
            pred_det = _li_box(40, 40, 60, 60)
        elif frame in {8, 18}:
            pred_det = _li_box(70, 70, 82, 82)
        else:
            pred_det = ""
        gt_lines.append(f"time_layer: {frame} detections: {gt_det}")
        pred_lines.append(f"time_layer: {frame} detections: {pred_det}")
    _write(gt / "Video_1_gt.txt", "\n".join(gt_lines) + "\n")
    _write(pred / "Video_1_dt.txt", "\n".join(pred_lines) + "\n")
    return {"gt": str(gt.relative_to(root.parent)), "pred": str(pred.relative_to(root.parent))}


def build_smoke_fixture(out_root: Path) -> Path:
    inputs = out_root / "_inputs"
    cases = {
        "YOLOMG": _write_yolo_case(inputs, "YOLOMG"),
        "TransVisDrone": _write_yolo_case(inputs, "TransVisDrone", frame_offset_gt=1),
        "ESOD": _write_yolo_case(inputs, "ESOD"),
        "AICrowd_Winner_v022": _write_aot_case(inputs),
        "EDTC": _write_antiuav_case(inputs),
        "Li_TETC_NPS": _write_li_tetc_case(inputs),
    }
    manifest = {
        "out_root": "curves",
        "defaults": {"fps": 2, "window_seconds": 3, "iou": 0.5, "score_threshold": 0.25},
        "runs": [
            {
                "name": "smoke_yolomg",
                "method": "YOLOMG",
                "gt": cases["YOLOMG"]["gt"],
                "gt_format": "yolo-dir",
                "frame_manifest": cases["YOLOMG"]["frame_manifest"],
                "frame_manifest_format": "image-dir",
                "pred": cases["YOLOMG"]["pred"],
                "pred_format": "yolo-dir",
            },
            {
                "name": "smoke_transvisdrone",
                "method": "TransVisDrone",
                "gt": cases["TransVisDrone"]["gt"],
                "gt_format": "yolo-dir",
                "gt_frame_offset": 1,
                "frame_manifest": cases["TransVisDrone"]["frame_manifest"],
                "frame_manifest_format": "image-dir",
                "pred": cases["TransVisDrone"]["pred"],
                "pred_format": "yolo-dir",
            },
            {
                "name": "smoke_esod",
                "method": "ESOD",
                "gt": cases["ESOD"]["gt"],
                "gt_format": "yolo-dir",
                "frame_manifest": cases["ESOD"]["frame_manifest"],
                "frame_manifest_format": "image-dir",
                "pred": cases["ESOD"]["pred"],
                "pred_format": "yolo-dir",
            },
            {
                "name": "smoke_aicrowd_winner",
                "method": "AICrowd_Winner_v022",
                "gt": cases["AICrowd_Winner_v022"]["gt"],
                "gt_format": "aot-gt-json",
                "frame_manifest": cases["AICrowd_Winner_v022"]["frame_manifest"],
                "frame_manifest_format": "image-dir",
                "pred": cases["AICrowd_Winner_v022"]["pred"],
                "pred_format": "aot-json",
            },
            {
                "name": "smoke_edtc",
                "method": "EDTC",
                "gt": cases["EDTC"]["gt"],
                "gt_format": "antiuav-json",
                "frame_manifest": cases["EDTC"]["gt"],
                "frame_manifest_format": "antiuav-json",
                "pred": cases["EDTC"]["pred"],
                "pred_format": "xywh-file",
            },
            {
                "name": "smoke_li_tetc",
                "method": "Li_TETC_NPS",
                "gt": cases["Li_TETC_NPS"]["gt"],
                "gt_format": "li-tetc-txt",
                "frame_manifest": cases["Li_TETC_NPS"]["gt"],
                "frame_manifest_format": "li-tetc-txt",
                "pred": cases["Li_TETC_NPS"]["pred"],
                "pred_format": "li-tetc-txt",
            },
        ],
    }
    manifest_path = out_root / "smoke_manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2))
    return manifest_path


def write_top_index(out_root: Path, batch_summary: dict[str, Any]) -> Path:
    index_path = out_root / "index.html"
    rows = []
    for run in batch_summary["runs"]:
        if run.get("status") != "complete":
            continue
        plot_index = Path(run["plot_index"])
        rel = plot_index.relative_to(out_root)
        worst = Path(run["worst_windows_csv"]).relative_to(out_root)
        rows.append(
            "<li>"
            f"<a href='{rel.as_posix()}'>{run['method']} / {run['name']}</a>"
            f" - videos={run['videos']}, frames={run['frames']}"
            f" - <a href='{worst.as_posix()}'>worst windows</a>"
            "</li>"
        )
    html = "<!doctype html><meta charset='utf-8'><title>Paper Window Accuracy Smoke</title>"
    html += "<h1>Paper Window Accuracy Smoke</h1><ul>" + "\n".join(rows) + "</ul>\n"
    _write(index_path, html)
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small multi-paper fixture and render actual +/-3s curve outputs.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "window_accuracy" / "smoke")
    parser.add_argument("--no-run", action="store_true", help="Only write fixture inputs and manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = build_smoke_fixture(out_root)
    print(f"manifest={manifest}")
    if args.no_run:
        return 0
    summary = run_manifest(manifest_path=manifest, base_dir=out_root)
    index = write_top_index(out_root, summary)
    print(f"complete={summary['complete']}")
    print(f"batch_summary={summary['batch_summary']}")
    print(f"index={index}")
    return 0 if summary["complete"] == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
