from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert TransVisDrone AOT list-format predictions into row-aligned predictionsgt input.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("rb") as source:
        records = pickle.load(source)
    if not isinstance(records, list):
        raise TypeError(f"expected AOT list predictions, got {type(records)}")
    output = {}
    for record in records:
        image_id = Path(str(record["img_name"])).stem
        detections = []
        for detection in record.get("detections") or []:
            x = float(detection["x"])
            y = float(detection["y"])
            width = float(detection["w"])
            height = float(detection["h"])
            detections.append({
                "bbox": [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
                "score": float(detection.get("s", 0.0)),
                "track_id": detection.get("track_id"),
            })
        output[image_id] = {"detections": detections, "labels": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as target:
        pickle.dump(output, target, protocol=pickle.HIGHEST_PROTOCOL)
    print({"images": len(output), "detections": sum(len(item["detections"]) for item in output.values()), "out": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
