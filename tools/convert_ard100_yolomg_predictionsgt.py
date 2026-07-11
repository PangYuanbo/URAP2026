from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def normalized_box(fields: list[str], width: int, height: int) -> list[float]:
    center_x, center_y, box_width, box_height = map(float, fields)
    return [
        (center_x - box_width / 2) * width,
        (center_y - box_height / 2) * height,
        (center_x + box_width / 2) * width,
        (center_y + box_height / 2) * height,
    ]


def convert_image(image_path: Path, prediction_root: Path, width: int, height: int):
    stem = image_path.stem
    prefix, frame_text = stem.rsplit('_', 1)
    sequence = f"Clip_{int(prefix.removeprefix('phantom'))}"
    image_id = f'{sequence}_{int(frame_text):05d}'
    label_path = Path(str(image_path).replace('\\images\\', '\\labels\\')).with_suffix('.txt')
    labels = []
    if label_path.is_file():
        for line in label_path.read_text(encoding='utf-8-sig').splitlines():
            fields = line.split()
            if len(fields) >= 5:
                labels.append({'bbox': normalized_box(fields[1:5], width, height), 'category_id': int(float(fields[0]))})
    prediction_path = prediction_root / f'{stem}.txt'
    detections = []
    if prediction_path.is_file():
        for line in prediction_path.read_text(encoding='utf-8-sig').splitlines():
            fields = line.split()
            if len(fields) >= 6:
                detections.append({'bbox': normalized_box(fields[1:5], width, height), 'score': float(fields[5]), 'category_id': int(float(fields[0]))})
    return image_id, {'detections': detections, 'labels': labels}, len(detections), len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert YOLOMG ARD100 saved labels to predictionsgt format.')
    parser.add_argument('--image-list', type=Path, required=True)
    parser.add_argument('--prediction-label-root', type=Path, required=True)
    parser.add_argument('--out-pkl', type=Path, required=True)
    parser.add_argument('--out-summary', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    images = [Path(line.strip()) for line in args.image_list.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    output = {}
    detections_total = 0
    labels_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        converted = executor.map(lambda path: convert_image(path, args.prediction_label_root, args.width, args.height), images)
        for index, (image_id, payload, detection_count, label_count) in enumerate(converted, 1):
            output[image_id] = payload
            detections_total += detection_count
            labels_total += label_count
            if index % 5000 == 0 or index == len(images):
                print(f'converted {index}/{len(images)}', flush=True)

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open('wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        'images': len(output),
        'labels': labels_total,
        'detections': detections_total,
        'image_list': str(args.image_list),
        'prediction_label_root': str(args.prediction_label_root),
        'out_pkl': str(args.out_pkl),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
