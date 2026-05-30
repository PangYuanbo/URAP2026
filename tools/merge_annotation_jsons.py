#!/usr/bin/env python3
"""Merge exported annotation JSON files from the browser tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = None
    by_frame = {}
    conflicts = []

    for filename in args.json_files:
        path = Path(filename)
        data = json.loads(path.read_text())
        if merged is None:
            merged = {
                "version": data.get("version", 1),
                "video_name": data["video_name"],
                "fps": data["fps"],
                "frame_step": data.get("frame_step"),
                "total_frames": data["total_frames"],
                "width": data["width"],
                "height": data["height"],
                "merged_from": [],
                "skipped_frames": [],
                "annotations": [],
            }
        elif data["video_name"] != merged["video_name"]:
            raise ValueError(f"Video mismatch in {path}: {data['video_name']} != {merged['video_name']}")

        merged["merged_from"].append(
            {
                "file": str(path),
                "assignment": data.get("assignment", {}),
                "annotations": len(data.get("annotations", [])),
            }
        )

        for frame in data.get("skipped_frames", []):
            merged["skipped_frames"].append(int(frame))

        for annotation in data.get("annotations", []):
            frame_id = int(annotation["frame_id"])
            if frame_id in by_frame and not args.allow_overwrite:
                conflicts.append((frame_id, by_frame[frame_id]["source"], str(path)))
                continue
            annotation = dict(annotation)
            annotation["source_file"] = str(path)
            by_frame[frame_id] = {"source": str(path), "annotation": annotation}

    if merged is None:
        raise ValueError("No input files")

    if conflicts:
        print("Conflicting duplicate frames found:")
        for frame_id, first, second in conflicts[:25]:
            print(f"  frame {frame_id}: {first} and {second}")
        if len(conflicts) > 25:
            print(f"  ... {len(conflicts) - 25} more")
        raise SystemExit("Use --allow-overwrite if you intentionally want later files to win.")

    merged["annotations"] = [item["annotation"] for _, item in sorted(by_frame.items())]
    merged["skipped_frames"] = sorted(set(merged["skipped_frames"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"{out}: {len(merged['annotations'])} annotations")


if __name__ == "__main__":
    main()
