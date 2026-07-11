from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def image_id(value: object) -> str:
    return Path(str(value)).stem


def load_labels(path: Path, width: int, height: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    labels: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        category, cx, cy, box_width, box_height = map(float, fields[:5])
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        labels.append({"bbox": [x1, y1, x2, y2], "category_id": int(category)})
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge TransVisDrone AOT candidates with ARD100 YOLO labels.")
    parser.add_argument("--aot-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--out-pkl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--sequence-size-json", type=Path)
    args = parser.parse_args()

    import cv2

    with args.aot_pkl.open("rb") as handle:
        candidates = pickle.load(handle)
    candidate_map = {image_id(key): value for key, value in candidates.items()}
    size_map = json.loads(args.sequence_size_json.read_text(encoding="utf-8-sig")) if args.sequence_size_json else {}
    output: dict[str, dict[str, list[dict[str, object]]]] = {}
    detections_total = labels_total = missing_predictions = 0
    sizes: dict[str, list[int]] = {}
    frames = sorted(
        frame
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for frame in args.frame_root.glob(pattern)
    )
    for frame_path in frames:
        key = frame_path.stem
        raw_detections = candidate_map.get(key, [])
        if key not in candidate_map:
            missing_predictions += 1
        detections = [
            {"bbox": [float(value) for value in row[:4]], "score": float(row[4]), "category_id": 0}
            for row in raw_detections
            if len(row) >= 5
        ]
        sequence = key.rsplit("_", 1)[0]
        known_size = size_map.get(sequence)
        if isinstance(known_size, list) and len(known_size) >= 2:
            width, height = int(known_size[0]), int(known_size[1])
        else:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read frame: {frame_path}")
            height, width = image.shape[:2]
        sizes.setdefault(sequence, [int(width), int(height)])
        labels = load_labels(args.label_root / f"{key}.txt", width, height)
        output[key] = {"detections": detections, "labels": labels}
        detections_total += len(detections)
        labels_total += len(labels)

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "images": len(output),
        "labels": labels_total,
        "detections": detections_total,
        "missing_prediction_frames": missing_predictions,
        "sequence_sizes": sizes,
        "aot_pkl": str(args.aot_pkl),
        "out_pkl": str(args.out_pkl),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
