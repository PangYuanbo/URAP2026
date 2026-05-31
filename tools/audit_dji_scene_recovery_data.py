from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class VideoAnnotationSummary:
    video_name: str
    video_path: str
    rows: int
    unique_frames: int
    classes: dict[str, int]
    split_hint: str
    role_note: str


def _match_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(pattern.lower() in low for pattern in patterns)


def _split_hint(video_name: str, args: argparse.Namespace) -> tuple[str, str]:
    if _match_any(video_name, args.heldout_patterns):
        return "heldout", "Do not use for gate training or calibration."
    if _match_any(video_name, args.calibration_patterns):
        return "calibration", "Use for non-held-out model selection only."
    if _match_any(video_name, args.train_patterns):
        return "train", "Usable for non-held-out scene-recovery mining."
    return "candidate_new_train_or_calibration", "New candidate; assign to train or calibration before mining."


def _summarize_csv(path: Path, args: argparse.Namespace) -> list[VideoAnnotationSummary]:
    by_video: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row.get("video_path") or row.get("frame_path") or ""
            video_name = Path(video_path).name or "__missing_video_path__"
            item = by_video.setdefault(
                video_name,
                {
                    "video_path": video_path,
                    "rows": 0,
                    "frames": set(),
                    "classes": {},
                },
            )
            item["rows"] += 1
            try:
                item["frames"].add(int(float(row.get("frame_id", -1))))
            except (TypeError, ValueError):
                item["frames"].add(row.get("frame_id", ""))
            cls = row.get("class") or row.get("label") or "unknown"
            item["classes"][cls] = item["classes"].get(cls, 0) + 1

    summaries: list[VideoAnnotationSummary] = []
    for video_name, item in sorted(by_video.items()):
        split, note = _split_hint(video_name, args)
        summaries.append(
            VideoAnnotationSummary(
                video_name=video_name,
                video_path=str(item["video_path"]),
                rows=int(item["rows"]),
                unique_frames=len(item["frames"]),
                classes=dict(sorted(item["classes"].items())),
                split_hint=split,
                role_note=note,
            )
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    annotation_dir = Path(args.annotations_dir)
    csv_paths = sorted(annotation_dir.glob(args.glob))
    summaries_by_csv: dict[str, list[VideoAnnotationSummary]] = {}
    for csv_path in csv_paths:
        summaries_by_csv[str(csv_path)] = _summarize_csv(csv_path, args)

    all_by_video: dict[str, VideoAnnotationSummary] = {}
    for summaries in summaries_by_csv.values():
        for item in summaries:
            current = all_by_video.get(item.video_name)
            if current is None or item.rows > current.rows:
                all_by_video[item.video_name] = item

    role_counts: dict[str, int] = {}
    frame_counts: dict[str, int] = {}
    for item in all_by_video.values():
        role_counts[item.split_hint] = role_counts.get(item.split_hint, 0) + 1
        frame_counts[item.split_hint] = frame_counts.get(item.split_hint, 0) + item.unique_frames

    recommendations: list[str] = []
    candidate_new = [v for v in all_by_video.values() if v.split_hint == "candidate_new_train_or_calibration"]
    if not candidate_new:
        recommendations.append("No new annotated DJI video is available beyond the known train/calibration/held-out clips.")
    if frame_counts.get("train", 0) < args.min_train_frames:
        recommendations.append(f"Train split has {frame_counts.get('train', 0)} annotated frames; target at least {args.min_train_frames}.")
    if frame_counts.get("calibration", 0) < args.min_calibration_frames:
        recommendations.append(
            f"Calibration split has {frame_counts.get('calibration', 0)} annotated frames; target at least {args.min_calibration_frames}."
        )
    recommendations.append("Do not move held-out clips into train/calibration; add new DJI clips instead.")

    result = {
        "annotations_dir": str(annotation_dir),
        "glob": args.glob,
        "known_patterns": {
            "train": args.train_patterns,
            "calibration": args.calibration_patterns,
            "heldout": args.heldout_patterns,
        },
        "role_counts": role_counts,
        "unique_annotated_frames_by_role": frame_counts,
        "videos": [asdict(v) for v in sorted(all_by_video.values(), key=lambda x: (x.split_hint, x.video_name))],
        "csv_files": {
            path: [asdict(v) for v in summaries]
            for path, summaries in summaries_by_csv.items()
        },
        "recommendations": recommendations,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dji_scene_recovery_data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(out_dir / "dji_scene_recovery_data_audit_videos.csv", result["videos"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DJI annotation coverage for QSTR scene-recovery gate training.")
    parser.add_argument("--annotations-dir", default=r"D:\datasets\my_video\final_annotations")
    parser.add_argument("--glob", default="*8col.csv")
    parser.add_argument("--out", default="reports/dji_scene_recovery_data_audit")
    parser.add_argument("--train-patterns", nargs="*", default=["121932", "122540"])
    parser.add_argument("--calibration-patterns", nargs="*", default=["20260522"])
    parser.add_argument("--heldout-patterns", nargs="*", default=["121806"])
    parser.add_argument("--min-train-frames", type=int, default=3000)
    parser.add_argument("--min-calibration-frames", type=int, default=500)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({k: result[k] for k in ("role_counts", "unique_annotated_frames_by_role", "recommendations")}, indent=2))


if __name__ == "__main__":
    main()
