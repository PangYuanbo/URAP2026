from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.nps_motion_interventions import (
    INTERVENTIONS,
    DISFrameInterpolator,
    build_clip,
    discover_clips,
    validate_intervention,
    write_dataset_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-preserving NPS motion intervention datasets.")
    parser.add_argument("--source-root", type=Path, default=Path(r"U:\URAP_datasets\TransVisDrone\NPS"))
    parser.add_argument("--out-root", type=Path, default=Path(r"U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["test"])
    parser.add_argument("--interventions", nargs="+", choices=INTERVENTIONS, default=list(INTERVENTIONS))
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--flow-median-threshold", type=float, default=3.0)
    parser.add_argument("--flow-bad-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--motion-threshold", type=int, default=16)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def ensure_safe_roots(source_root: Path, out_root: Path) -> tuple[Path, Path]:
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    protected = {source_root / "AllFrames", source_root / "NPSvisdroneStyle", source_root / "Videos"}
    if out_root == source_root or out_root in protected:
        raise ValueError(f"Unsafe output root: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    return source_root, out_root


def main() -> None:
    args = parse_args()
    source_root, out_root = ensure_safe_roots(args.source_root, args.out_root)
    interpolator = DISFrameInterpolator(args.flow_scale, args.flow_median_threshold, args.flow_bad_ratio_threshold)
    units = []
    for split in args.splits:
        frames_dir = source_root / "AllFrames" / split
        available = discover_clips(frames_dir)
        requested = args.clips or available
        clips = [clip for clip in requested if clip in available]
        if not clips:
            raise ValueError(f"No requested clips found for split {split}")
        for intervention in args.interventions:
            if split != "test" and intervention != "original":
                continue
            units.extend((intervention, split, clip) for clip in clips)
    progress_path = out_root / "progress.json"
    completed = 0
    summaries = []
    progress_path.write_text(json.dumps({"done": 0, "total": len(units), "last_completed_unit": None, "last_output": None}, indent=2), encoding="utf-8")
    for intervention, split, clip_name in units:
        summary = build_clip(
            source_root / "AllFrames" / split,
            source_root / "NPSvisdroneStyle" / split / "labels",
            out_root / intervention,
            split,
            clip_name,
            intervention,
            interpolator,
            motion_threshold=args.motion_threshold,
            max_frames=args.max_frames,
            seed=args.seed,
        )
        summaries.append(summary)
        completed += 1
        progress = {
            "done": completed,
            "total": len(units),
            "last_completed_unit": f"{intervention}/{split}/{clip_name}",
            "last_output": summary["last_output"],
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"[{completed}/{len(units)}] {progress['last_completed_unit']}", flush=True)
    overall = {}
    for intervention in args.interventions:
        selected = [summary for summary in summaries if summary["intervention"] == intervention]
        if not selected:
            continue
        split_lengths: dict[str, dict[int, int]] = {}
        for summary in selected:
            split_lengths.setdefault(summary["split"], {})[int(summary["clip"].split("_")[-1])] = int(summary["output_frames"])
        write_dataset_metadata(out_root / intervention, intervention, split_lengths)
        overall[intervention] = validate_intervention(out_root / intervention, intervention, split_lengths)
    result = {"out_root": str(out_root), "units": len(units), "interventions": overall}
    (out_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(item["valid"] for item in overall.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
