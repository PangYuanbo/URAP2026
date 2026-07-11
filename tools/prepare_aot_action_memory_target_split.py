from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


WIDTH = 2448
HEIGHT = 2048


def labels_for(label_root: Path, image_id: str) -> list[dict[str, object]]:
    path = label_root / f"{image_id}.txt"
    labels: list[dict[str, object]] = []
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cx, cy, width, height = map(float, fields[1:5])
        labels.append({
            "bbox": [(cx - width / 2) * WIDTH, (cy - height / 2) * HEIGHT, (cx + width / 2) * WIDTH, (cy + height / 2) * HEIGHT],
            "category_id": int(float(fields[0])),
        })
    return labels


def filter_aux(source: Path, destination: Path, image_ids: set[str]) -> int:
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            if str((item.get("meta") or {}).get("image_id") or "") in image_ids:
                dst.write(json.dumps(item, separators=(",", ":")) + "\n")
                rows += len(item.get("rows") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-predictionsgt", type=Path, required=True)
    parser.add_argument("--part0-source", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--full-aux", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-sequences", default="Clip_693,Clip_694,Clip_695,Clip_696")
    args = parser.parse_args()

    full = pickle.loads(args.full_predictionsgt.read_bytes())
    part0 = pickle.loads(args.part0_source.read_bytes())
    selected_ids = {Path(str(record["img_name"])).stem for record in part0}
    val_sequences = {value.strip() for value in args.val_sequences.split(",") if value.strip()}
    train: dict[str, object] = {}
    val: dict[str, object] = {}
    for image_id in sorted(selected_ids):
        if image_id not in full:
            continue
        payload = dict(full[image_id])
        payload["labels"] = labels_for(args.label_root, image_id)
        sequence = image_id.rsplit("_", 1)[0]
        (val if sequence in val_sequences else train)[image_id] = payload

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = args.out_dir / "train_predictionsgt.pkl"
    val_pkl = args.out_dir / "val_predictionsgt.pkl"
    train_pkl.write_bytes(pickle.dumps(train, protocol=pickle.HIGHEST_PROTOCOL))
    val_pkl.write_bytes(pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL))
    train_aux = args.out_dir / "train_aux.jsonl"
    val_aux = args.out_dir / "val_aux.jsonl"
    train_aux_rows = filter_aux(args.full_aux, train_aux, set(train))
    val_aux_rows = filter_aux(args.full_aux, val_aux, set(val))
    summary = {
        "train_images": len(train), "train_labels": sum(len(item["labels"]) for item in train.values()),
        "train_detections": sum(len(item["detections"]) for item in train.values()), "train_aux_rows": train_aux_rows,
        "val_images": len(val), "val_labels": sum(len(item["labels"]) for item in val.values()),
        "val_detections": sum(len(item["detections"]) for item in val.values()), "val_aux_rows": val_aux_rows,
        "val_sequences": sorted(val_sequences), "full_test_labels_used": False,
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
