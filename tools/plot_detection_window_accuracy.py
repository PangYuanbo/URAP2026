from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.evaluation.window_accuracy import (
    run_window_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame detection metrics using a +/-N second window "
            "and render one curve per video."
        )
    )
    formats = [
        "csv",
        "jsonl",
        "yolo-dir",
        "aot-json",
        "aot-gt-json",
        "xywh-file",
        "antiuav-json",
        "li-tetc-txt",
        "tvd-pkl-gt",
        "tvd-pkl-pred",
    ]
    frame_formats = formats + ["image-dir"]
    parser.add_argument("--gt", type=Path, required=True, help=f"Ground-truth boxes in one of: {', '.join(formats)}.")
    parser.add_argument("--pred", type=Path, required=True, help=f"Prediction boxes in one of: {', '.join(formats)}.")
    parser.add_argument("--gt-format", choices=formats, default="csv")
    parser.add_argument("--pred-format", choices=formats, default="csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--fps", type=float, required=True, help="Video FPS used to convert seconds to frames.")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Half-window duration on each side.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--segment-threshold", type=float, default=0.5, help="Accuracy threshold used to group contiguous low-accuracy center-frame segments.")
    parser.add_argument("--gt-labels", default="", help="Comma-separated ground-truth labels/classes to keep. Empty means all.")
    parser.add_argument("--pred-labels", default="", help="Comma-separated prediction labels/classes to keep. Empty means all.")
    parser.add_argument("--gt-frame-offset", type=int, default=0, help="Add this offset to parsed GT frame ids.")
    parser.add_argument("--pred-frame-offset", type=int, default=0, help="Add this offset to parsed prediction frame ids.")
    parser.add_argument("--frame-manifest", type=Path, default=None, help="Optional frame list/image dir to score every listed center frame, including empty frames.")
    parser.add_argument("--frame-manifest-format", choices=frame_formats, default=None, help="Format for --frame-manifest. Defaults to image-dir.")
    parser.add_argument("--frame-manifest-offset", type=int, default=0, help="Add this offset to parsed frame-manifest frame ids.")
    parser.add_argument("--img-width", type=float, default=None, help="Image width for normalized YOLO txt conversion.")
    parser.add_argument("--img-height", type=float, default=None, help="Image height for normalized YOLO txt conversion.")
    parser.add_argument("--sparse-centers", action="store_true", help="Only score frames that have gt or predictions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.img_width is None) != (args.img_height is None):
        raise SystemExit("--img-width and --img-height must be provided together.")
    img_size = (args.img_width, args.img_height) if args.img_width is not None else None
    gt_labels = [x.strip() for x in args.gt_labels.split(",") if x.strip()]
    pred_labels = [x.strip() for x in args.pred_labels.split(",") if x.strip()]
    summary = run_window_accuracy(
        args.gt,
        args.pred,
        args.out,
        fps=args.fps,
        gt_format=args.gt_format,
        pred_format=args.pred_format,
        window_seconds=args.window_seconds,
        iou_threshold=args.iou,
        score_threshold=args.score_threshold,
        segment_threshold=args.segment_threshold,
        gt_labels=gt_labels or None,
        pred_labels=pred_labels or None,
        gt_frame_offset=args.gt_frame_offset,
        pred_frame_offset=args.pred_frame_offset,
        frame_manifest=args.frame_manifest,
        frame_manifest_format=args.frame_manifest_format,
        frame_manifest_offset=args.frame_manifest_offset,
        img_size=img_size,
        sparse_centers=args.sparse_centers,
    )

    print(f"frames={summary['frames']}")
    print(f"csv={summary['csv']}")
    print(f"summary={args.out / 'summary.json'}")
    print(f"low_accuracy_segments={summary['low_accuracy_segments_csv']}")
    print(f"plot_index={summary['plot_index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
