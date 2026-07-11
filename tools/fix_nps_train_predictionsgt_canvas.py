from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

DEFAULT_SOURCE = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl")
DEFAULT_OUTPUT = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl")
DEFAULT_SUMMARY = Path(r"D:\URAP_vatd_rank_results\tvd_train_label_fix_v91\summary.json")
SOURCE_1920_SEQUENCES = set(range(1, 14)) | {16, 21, 22, 23, 24, 25, 29, 30, 31, 35, 36}


def main() -> int:
    parser = argparse.ArgumentParser(description="Correct NPS train GT coordinates to the detector output canvas.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    with args.source.open("rb") as handle:
        data = pickle.load(handle)

    fixed_labels = 0
    fixed_images = 0
    for image_id, item in data.items():
        sequence = str(image_id).rsplit("_", 1)[0]
        sequence_number = int(sequence.split("_")[1])
        if sequence_number not in SOURCE_1920_SEQUENCES:
            continue
        touched = False
        for row in item.get("labels", []):
            bbox = row.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                row["bbox"] = [
                    float(bbox[0]) * 2.0 / 3.0,
                    float(bbox[1]) * 8.0 / 9.0,
                    float(bbox[2]) * 2.0 / 3.0,
                    float(bbox[3]) * 8.0 / 9.0,
                ]
                fixed_labels += 1
                touched = True
        fixed_images += int(touched)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    summary = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "images": len(data),
        "fixed_images": fixed_images,
        "fixed_labels": fixed_labels,
        "transformed_sequences": sorted(SOURCE_1920_SEQUENCES),
        "transform": "1920x1080 GT -> 1280x960 detector canvas: x*2/3, y*8/9",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
