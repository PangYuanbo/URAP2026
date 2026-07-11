from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.candidates.yolo_wrapper import candidates_from_yolo_tiled
from qstr_dronedet.pipelines.temporal_recovery import TemporalRecoveryConfig, run_temporal_recovery_frames
from qstr_dronedet.types import DetectionCandidate


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class FrameItem:
    frame: np.ndarray
    path: Path | None = None
    secondary_path: Path | None = None
    secondary_frame: np.ndarray | None = None


@dataclass
class FrameRecord:
    frame_id: int
    path: Path | None
    width: int
    height: int


def _remap_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    stale_prefixes = (
        ("D:\\URAP_datasets\\", "U:\\URAP_datasets\\"),
        ("D:/URAP_datasets/", "U:/URAP_datasets/"),
    )
    for old, new in stale_prefixes:
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return path


def _read_list(path: Path) -> list[Path]:
    return [_remap_path(Path(line.strip())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _secondary_from_primary(path: Path, secondary_frame_dir: Path | None = None) -> Path | None:
    if secondary_frame_dir is not None:
        return secondary_frame_dir / path.name
    text = str(path)
    for old, new in (("\\images\\", "\\images2\\"), ("/images/", "/images2/")):
        if old in text:
            return _remap_path(Path(text.replace(old, new)))
    return None


def iter_frame_items(
    video: Path | None,
    frame_dir: Path | None,
    image_list: Path | None,
    secondary_frame_dir: Path | None,
    max_frames: int = 0,
) -> Iterator[FrameItem]:
    count = 0
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {video}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield FrameItem(frame=frame)
                count += 1
                if max_frames and count >= max_frames:
                    break
        finally:
            cap.release()
        return
    if image_list is not None:
        paths = _read_list(image_list)
        for path in paths:
            if not path.exists():
                continue
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            secondary_path = _secondary_from_primary(path, secondary_frame_dir)
            secondary = cv2.imread(str(secondary_path), cv2.IMREAD_COLOR) if secondary_path is not None and secondary_path.exists() else None
            yield FrameItem(frame=frame, path=path, secondary_path=secondary_path, secondary_frame=secondary)
            count += 1
            if max_frames and count >= max_frames:
                break
        return
    if frame_dir is None:
        raise SystemExit("provide --video, --frame-dir, or --image-list")
    frame_paths = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        secondary_path = _secondary_from_primary(path, secondary_frame_dir)
        secondary = cv2.imread(str(secondary_path), cv2.IMREAD_COLOR) if secondary_path is not None and secondary_path.exists() else None
        yield FrameItem(frame=frame, path=path, secondary_path=secondary_path, secondary_frame=secondary)
        count += 1
        if max_frames and count >= max_frames:
            break


def iter_frames(video: Path | None, frame_dir: Path | None, max_frames: int = 0) -> Iterator[np.ndarray]:
    for item in iter_frame_items(video, frame_dir, None, None, max_frames):
        yield item.frame


class Yolov5DualDetector:
    def __init__(self, repo: Path, weights: Path, device: str = "0", img_size: int = 1280, half: bool = True, conf: float = 0.01, iou: float = 0.45):
        self.repo = repo.resolve()
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        import torch
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device

        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.device = select_device(device)
        self.model = attempt_load(str(weights), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = int(check_img_size(self.img_size, s=self.stride))
        self.half = bool(half and self.device.type != "cpu")
        if self.half:
            self.model.half()
        self.model.eval()
        if self.device.type != "cpu":
            z = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters()))
            self.model(z, z, augment=False)

    def _tensor(self, frame: np.ndarray):
        img = self.letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = self.torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __call__(self, frame: np.ndarray, secondary_frame: np.ndarray | None = None) -> list[DetectionCandidate]:
        second = secondary_frame if secondary_frame is not None else frame
        img1 = self._tensor(frame)
        img2 = self._tensor(second)
        with self.torch.no_grad():
            pred = self.model(img1, img2, augment=False)[0]
            pred = self.non_max_suppression(pred, self.conf, self.iou, classes=None, agnostic=False)
        det = pred[0]
        out: list[DetectionCandidate] = []
        if len(det):
            det = det.clone()
            self.scale_coords(img1.shape[2:], det[:, :4], frame.shape).round()
            for row in det.detach().cpu().numpy():
                x1, y1, x2, y2, score, cls = row[:6]
                out.append(DetectionCandidate((float(x1), float(y1), float(x2), float(y2)), float(score), "yolov5_dual", extra={"class_id": int(cls)}))
        return out


def write_outputs(rows, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trajectory.csv"
    json_path = out_dir / "trajectory.json"
    fieldnames = [
        "frame_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "source",
        "raw_objectness",
        "motion_memory_score",
        "memory_quality",
        "memory_write",
        "memory_write_reason",
        "memory_bank_size",
        "emit_detection",
        "emit_reason",
        "ncc_score",
        "hard_reset",
        "detector_candidates",
        "support_candidates",
    ]
    flat = []
    for row in rows:
        selected = row.selected
        if selected is None:
            item = {
                "frame_id": row.frame_id,
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "score": "",
                "source": "",
                "raw_objectness": "",
                "motion_memory_score": "",
                "memory_quality": row.diagnostics.get("memory_quality", ""),
                "memory_write": row.diagnostics.get("memory_write", False),
                "memory_write_reason": row.diagnostics.get("memory_write_reason", ""),
                "memory_bank_size": row.diagnostics.get("memory_bank_size", 0),
                "emit_detection": row.diagnostics.get("emit_detection", False),
                "emit_reason": row.diagnostics.get("emit_reason", ""),
                "ncc_score": "",
                "hard_reset": "",
            }
        else:
            x1, y1, x2, y2 = selected.bbox_xyxy
            item = {
                "frame_id": row.frame_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": selected.objectness,
                "source": selected.source,
                "raw_objectness": selected.extra.get("raw_objectness", selected.objectness),
                "motion_memory_score": selected.extra.get("motion_memory_score", selected.motion_score),
                "memory_quality": selected.extra.get("memory_quality", row.diagnostics.get("memory_quality", "")),
                "memory_write": selected.extra.get("memory_write", row.diagnostics.get("memory_write", False)),
                "memory_write_reason": selected.extra.get("memory_write_reason", row.diagnostics.get("memory_write_reason", "")),
                "memory_bank_size": row.diagnostics.get("memory_bank_size", 0),
                "emit_detection": selected.extra.get("emit_detection", row.diagnostics.get("emit_detection", True)),
                "emit_reason": selected.extra.get("emit_reason", row.diagnostics.get("emit_reason", "")),
                "ncc_score": selected.extra.get("ncc_score", ""),
                "hard_reset": selected.extra.get("hard_reset", False),
            }
        item.update(
            {
                "detector_candidates": row.diagnostics["detector_candidates"],
                "support_candidates": row.diagnostics["support_candidates"],
            }
        )
        flat.append(item)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    json_path.write_text(
        json.dumps(
            {
                "num_frames": len(rows),
                "detections": _json_safe(flat),
                "diagnostics": _json_safe([row.diagnostics for row in rows]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(0.0, float(x1)), float(width))
    x2 = min(max(0.0, float(x2)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    y2 = min(max(0.0, float(y2)), float(height))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return (x1 + bw / 2.0) / width, (y1 + bh / 2.0) / height, bw / width, bh / height


def _frame_stem(record: FrameRecord) -> str:
    return record.path.stem if record.path is not None else f"frame_{record.frame_id:06d}"


def write_candidate_outputs(
    rows,
    records: list[FrameRecord],
    out_dir: Path,
    class_id: int = 0,
    write_labels: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "candidate_predictions.jsonl"
    label_dir = out_dir / "candidate_labels"
    if write_labels:
        label_dir.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row, record in zip(rows, records):
            frame_candidates = []
            label_lines = []
            for rank, cand in enumerate(row.candidates):
                x1, y1, x2, y2 = cand.bbox_xyxy
                raw_score = float(cand.extra.get("raw_objectness", cand.objectness))
                motion_score = float(cand.extra.get("motion_memory_score", cand.motion_score))
                detector_bbox = cand.extra.get("detector_bbox_xyxy")
                if not (isinstance(detector_bbox, (list, tuple)) and len(detector_bbox) == 4):
                    detector_bbox = None
                item = {
                    "frame_id": int(row.frame_id),
                    "image_path": str(record.path) if record.path is not None else None,
                    "rank": rank,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(cand.objectness),
                    "raw_objectness": raw_score,
                    "motion_memory_score": motion_score,
                    "source": cand.source,
                    "ncc_score": cand.extra.get("ncc_score"),
                    "memory_quality": cand.extra.get("memory_quality"),
                    "num_merged": cand.extra.get("num_merged"),
                    "merged_sources": cand.extra.get("merged_sources"),
                    "has_detector_member": bool(cand.extra.get("has_detector_member", False)),
                    "detector_raw_objectness": cand.extra.get("detector_raw_objectness"),
                    "detector_source": cand.extra.get("detector_source"),
                    "detector_x1": float(detector_bbox[0]) if detector_bbox is not None else None,
                    "detector_y1": float(detector_bbox[1]) if detector_bbox is not None else None,
                    "detector_x2": float(detector_bbox[2]) if detector_bbox is not None else None,
                    "detector_y2": float(detector_bbox[3]) if detector_bbox is not None else None,
                    "hard_reset": cand.extra.get("hard_reset"),
                    "extra": _json_safe(cand.extra),
                }
                frame_candidates.append(item)
                if write_labels:
                    cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, record.width, record.height)
                    if bw > 0.0 and bh > 0.0:
                        label_lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {float(cand.objectness):.8f}\n")
            total_candidates += len(frame_candidates)
            f.write(
                json.dumps(
                    {
                        "frame_id": int(row.frame_id),
                        "image_path": str(record.path) if record.path is not None else None,
                        "width": record.width,
                        "height": record.height,
                        "selected_rank": 0 if row.selected is not None and row.candidates else None,
                        "candidates": frame_candidates,
                    }
                )
                + "\n"
            )
            if write_labels:
                (label_dir / f"{_frame_stem(record)}.txt").write_text("".join(label_lines), encoding="utf-8")

    summary = {
        "candidate_jsonl": str(jsonl_path),
        "candidate_label_dir": str(label_dir) if write_labels else None,
        "frames": len(rows),
        "records": len(records),
        "total_candidates": total_candidates,
        "write_labels": write_labels,
    }
    (out_dir / "candidate_predictions_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_annotated(frames: list[np.ndarray], rows, out_path: Path, fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame, row in zip(frames, rows):
            canvas = frame.copy()
            if row.selected is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in row.selected.bbox_xyxy]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"{row.selected.source} {row.selected.objectness:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            writer.write(canvas)
    finally:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detector-first temporal recovery pipeline.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--frame-dir", type=Path)
    src.add_argument("--image-list", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--crop-yolo-weights", type=Path)
    parser.add_argument("--detector-backend", choices=["ultralytics-tiled", "yolov5-dual"], default="ultralytics-tiled")
    parser.add_argument("--yolov5-repo", type=Path, default=ROOT / "URAP-UAV-to-UAV-Detection-and-Tracking" / "papers" / "YOLOMG")
    parser.add_argument("--secondary-frame-dir", type=Path)
    parser.add_argument("--profile", choices=["default", "dji-tiny"], default="default")
    parser.add_argument("--final-selection-score", choices=["temporal", "raw"], default="temporal")
    parser.add_argument("--final-output-score", choices=["temporal", "raw"], default="temporal")
    parser.add_argument("--memory-update-selection", choices=["final", "temporal"], default="final")
    parser.add_argument("--apply-output-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--crop-conf", type=float, default=0.01)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-stride", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--ncc-min-score", type=float, default=0.62)
    parser.add_argument("--ncc-score", type=float, default=0.34)
    parser.add_argument("--memory-quality-min", type=float, default=0.38)
    parser.add_argument("--memory-detector-min", type=float, default=0.05)
    parser.add_argument("--memory-motion-min", type=float, default=0.08)
    parser.add_argument("--allow-support-only-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support-only-output-min-quality", type=float, default=0.72)
    parser.add_argument("--support-only-min-detector-updates", type=int, default=2)
    parser.add_argument("--support-only-max-misses", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--write-candidate-labels", action="store_true")
    parser.add_argument("--candidate-class-id", type=int, default=0)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = TemporalRecoveryConfig(
        profile=args.profile,
        final_selection_score=args.final_selection_score,
        final_output_score=args.final_output_score,
        memory_update_selection=args.memory_update_selection,
        apply_output_gate=args.apply_output_gate,
        top_k=args.top_k,
        ncc_min_score=args.ncc_min_score,
        ncc_score=args.ncc_score,
        memory_quality_min=args.memory_quality_min,
        memory_detector_min=args.memory_detector_min,
        memory_motion_min=args.memory_motion_min,
        allow_support_only_output=args.allow_support_only_output,
        support_only_output_min_quality=args.support_only_output_min_quality,
        support_only_min_detector_updates=args.support_only_min_detector_updates,
        support_only_max_misses=args.support_only_max_misses,
    )
    current_secondary: dict[str, np.ndarray | None] = {"frame": None}
    annotated_frames: list[np.ndarray] = []
    frame_records: list[FrameRecord] = []

    def frame_stream():
        count = 0
        for item in iter_frame_items(args.video, args.frame_dir, args.image_list, args.secondary_frame_dir, args.max_frames):
            current_secondary["frame"] = item.secondary_frame
            frame_records.append(FrameRecord(frame_id=count, path=item.path, width=int(item.frame.shape[1]), height=int(item.frame.shape[0])))
            if args.save_annotated:
                annotated_frames.append(item.frame.copy())
            count += 1
            if count == 1 or (args.progress_every > 0 and count % args.progress_every == 0):
                print(
                    json.dumps(
                        {
                            "kind": "temporal_recovery_progress",
                            "frames_read": count,
                            "path": str(item.path) if item.path is not None else None,
                            "has_secondary": item.secondary_frame is not None,
                        }
                    ),
                    flush=True,
                )
            yield item.frame

    if args.detector_backend == "yolov5-dual":
        dual = Yolov5DualDetector(
            args.yolov5_repo,
            args.yolo_weights,
            device=args.device or "0",
            img_size=args.img_size,
            conf=args.conf,
            iou=args.iou_thres,
        )

        def detector(frame: np.ndarray):
            return dual(frame, current_secondary["frame"])

    else:
        def detector(frame: np.ndarray):
            return candidates_from_yolo_tiled(
                frame,
                str(args.yolo_weights),
                tile_size=args.tile_size,
                stride=args.tile_stride,
                conf=args.conf,
                device=args.device,
                max_det=args.top_k,
            )

    def crop_detector(crop: np.ndarray):
        if args.crop_yolo_weights is None:
            return []
        return candidates_from_yolo_tiled(
            crop,
            str(args.crop_yolo_weights),
            tile_size=min(args.tile_size, max(crop.shape[:2])),
            stride=max(32, min(args.tile_stride, max(crop.shape[:2]) // 2)),
            conf=args.crop_conf,
            device=args.device,
            max_det=args.top_k,
        )

    rows = run_temporal_recovery_frames(frame_stream(), detector, crop_detector, cfg)
    write_outputs(rows, args.out_dir)
    write_candidate_outputs(rows, frame_records, args.out_dir, class_id=args.candidate_class_id, write_labels=args.write_candidate_labels)
    if args.save_annotated:
        write_annotated(annotated_frames, rows, args.out_dir / "annotated.mp4", args.fps)
    print(json.dumps({"out_dir": str(args.out_dir), "num_frames": len(rows), "detections": sum(1 for row in rows if row.selected is not None)}, indent=2))


if __name__ == "__main__":
    main()
