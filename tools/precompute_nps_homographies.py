from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_frame(path: Path) -> tuple[str, int]:
    sequence, frame = path.stem.rsplit("_", 1)
    return sequence, int(frame)


def compute_sequence(frame_root: str, sequence: str, frame_ids: list[int], partial_path: str, max_size: int) -> dict[str, object]:
    import cv2

    from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache

    cv2.setNumThreads(1)
    output = Path(partial_path)
    if output.is_file() and output.stat().st_size > 0:
        with output.open("rb") as handle:
            values = pickle.load(handle)
        return {"sequence": sequence, "pairs": len(values), "valid": sum(bool(item.get("valid")) for item in values.values()), "resumed": True}
    cache = ActionChunkCameraMotionCache(Path(frame_root), None, max_size)
    for source, target in zip(frame_ids, frame_ids[1:]):
        if target == source + 1:
            cache.adjacent(sequence, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(cache.values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    return {"sequence": sequence, "pairs": len(cache.values), "valid": sum(bool(item.get("valid")) for item in cache.values.values()), "resumed": False}


def write_progress(path: Path | None, *, done: int, total: int, pairs: int, valid: int, sequence: str) -> None:
    payload = {"stage": "precompute_homographies", "done": done, "total": total, "pairs": pairs, "valid": valid, "last_sequence": sequence}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute adjacent camera homographies in parallel by sequence.")
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--max-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    grouped: dict[str, list[int]] = {}
    frame_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        frame_paths.extend(args.frame_root.glob(pattern))
    for path in sorted(frame_paths):
        sequence, frame_id = parse_frame(path)
        grouped.setdefault(sequence, []).append(frame_id)
    for frame_ids in grouped.values():
        frame_ids.sort()
    if not grouped:
        raise RuntimeError(f"no image frames found in {args.frame_root}")

    args.partial_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    pairs = valid = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                compute_sequence,
                str(args.frame_root),
                sequence,
                frame_ids,
                str(args.partial_dir / f"{sequence}.pkl"),
                args.max_size,
            ): sequence
            for sequence, frame_ids in sorted(grouped.items())
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            pairs += int(result["pairs"])
            valid += int(result["valid"])
            write_progress(args.progress_json, done=done, total=len(futures), pairs=pairs, valid=valid, sequence=str(result["sequence"]))

    merged: dict[tuple[str, int], dict[str, object]] = {}
    for sequence in sorted(grouped):
        with (args.partial_dir / f"{sequence}.pkl").open("rb") as handle:
            merged.update(pickle.load(handle))
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_cache.with_suffix(args.output_cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output_cache)
    summary = {
        "kind": "homography_precompute_done",
        "frame_root": str(args.frame_root),
        "output_cache": str(args.output_cache),
        "sequences": len(grouped),
        "pairs": len(merged),
        "valid": sum(bool(item.get("valid")) for item in merged.values()),
        "workers": max(1, args.workers),
        "results": sorted(results, key=lambda item: str(item["sequence"])),
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
