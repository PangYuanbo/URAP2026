import argparse
import json
import math
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm


@dataclass(frozen=True)
class SplitPart:
    part_id: int
    flight_ids: List[str]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink to avoid duplication; fallback to copy."""
    if dst.exists():
        return
    _ensure_dir(dst.parent)
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass
    shutil.copy2(str(src), str(dst))


def _load_json_list(p: Path) -> List[str]:
    return list(json.loads(p.read_text(encoding="utf-8")))


def _split_list(xs: List[str], part_size: int) -> List[SplitPart]:
    if part_size <= 0:
        raise ValueError("part_size must be > 0")
    parts: List[SplitPart] = []
    n = math.ceil(len(xs) / float(part_size))
    for pid in range(n):
        start = pid * part_size
        end = min((pid + 1) * part_size, len(xs))
        parts.append(SplitPart(pid, xs[start:end]))
    return parts


def _is_nan(x: Optional[float]) -> bool:
    try:
        return x is None or math.isnan(float(x))
    except Exception:
        return True


def _check_passes_criterion(object_location, distance_threshold_m: int) -> bool:
    # Mirrors `papers/TransVisDrone/conversion_scripts/aot_to_visdrone.py`
    if getattr(object_location, "planned", False):
        d = getattr(object_location, "range_distance_m", None)
        if not _is_nan(d) and int(d) <= int(distance_threshold_m):
            return True
    return False


_CLASSES_THAT_MATTER = {"Airplane", "Helicopter", "Drone"}


def _get_class_str(name: str) -> str:
    return "".join([c for c in name if not c.isdigit()])


def _decide_class_index(class_name: str) -> int:
    class_name_wo_digits = _get_class_str(class_name)
    return 0 if class_name_wo_digits in _CLASSES_THAT_MATTER else 1


def _to_cxcywhn_xyxy(bb_xyxy: Tuple[float, float, float, float], h: float, w: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return cx / w, cy / h, bw / w, bh / h


def _write_yolo_label(
    frame,
    clip_id: int,
    fid: int,
    labels_dir: Path,
    distance_threshold_m: int,
    img_h: float = 2048.0,
    img_w: float = 2448.0,
) -> None:
    # AOT images are 2448x2048 grayscale in the paper.
    object_locations = list(frame.detected_object_locations.values())
    object_locations = [ol for ol in object_locations if _check_passes_criterion(ol, distance_threshold_m)]
    if not object_locations:
        return

    out_path = labels_dir / f"Clip_{clip_id}_{fid:05d}.txt"
    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for ol in object_locations:
            class_idx = _decide_class_index(ol.object.id)
            x1, y1, x2, y2 = ol.bb.get_bbox_traditional()
            cxn, cyn, wn, hn = _to_cxcywhn_xyxy((x1, y1, x2, y2), h=img_h, w=img_w)
            f.write(f"{class_idx} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n")


def _import_aotcore(transvisdrone_repo: Path) -> None:
    # Make `from aotcore.dataset import Dataset` work when running from URAP root.
    sys.path.insert(0, str(transvisdrone_repo.resolve()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aot-root", required=True, help="AOT part1 root (contains ImageSets/ and Images/).")
    ap.add_argument("--out-root", required=True, help="Output root for YOLO-style data (AOT).")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--part-size", type=int, default=10, help="Flights per part (for split into parts).")
    ap.add_argument("--max-flights", type=int, default=0, help="For debugging: limit number of flights processed (0 = no limit).")
    ap.add_argument("--distance-threshold-m", type=int, default=700)
    ap.add_argument(
        "--transvisdrone-repo",
        default=str(Path("papers/TransVisDrone")),
        help="Path to TransVisDrone repo root (for aotcore + flight id lists).",
    )
    ap.add_argument(
        "--download-parallel",
        type=int,
        default=8,
        help="Parallelism per-flight when downloading images from S3 via aotcore.",
    )
    args = ap.parse_args()

    aot_root = Path(args.aot_root).resolve()
    out_root = Path(args.out_root).resolve()
    trans_repo = Path(args.transvisdrone_repo).resolve()

    _import_aotcore(trans_repo)
    from aotcore.dataset import Dataset as AOTDataset  # noqa

    flight_ids_path = trans_repo / "aot_flight_ids" / f"{args.split}flightidsfull1.json"
    if not flight_ids_path.exists():
        raise FileNotFoundError(f"Missing flight id list: {flight_ids_path}")
    flight_ids = _load_json_list(flight_ids_path)
    if args.max_flights and args.max_flights > 0:
        flight_ids = flight_ids[: int(args.max_flights)]

    # Mapping used by TransVisDrone evaluation scripts.
    flight_to_clip_pkl = trans_repo / "aot_flight_ids" / "aot_flight_id_to_clip_id.pkl"
    if not flight_to_clip_pkl.exists():
        raise FileNotFoundError(f"Missing mapping: {flight_to_clip_pkl}")
    flight_id_to_clip_id: Dict[str, int] = pickle.load(open(flight_to_clip_pkl, "rb"))

    ds = AOTDataset(
        local_path=str(aot_root),
        s3_path="s3://airborne-obj-detection-challenge-training/part1/",
        download_if_required=True,
        partial=False,
    )

    parts = _split_list(flight_ids, part_size=int(args.part_size))
    print(f"Split={args.split} flights={len(flight_ids)} -> parts={len(parts)} (part_size={args.part_size})")

    # Create per-part output and YAMLs compatible with TransVisDrone's `val.py`.
    yaml_dir = trans_repo / "data" / "AOTTestSplits_URAP"
    _ensure_dir(yaml_dir)

    for part in parts:
        part_name = f"part{part.part_id}"
        frames_dir = out_root / args.split / part_name / "frames"
        labels_dir = out_root / args.split / part_name / "labels"
        videos_dir = out_root / args.split / part_name / "videos"
        _ensure_dir(frames_dir)
        _ensure_dir(labels_dir)
        _ensure_dir(videos_dir)

        clip_id_to_len: Dict[int, int] = {}

        for flight_id in tqdm(part.flight_ids, desc=f"{args.split}:{part_name}", unit="flight"):
            if flight_id not in flight_id_to_clip_id:
                raise KeyError(f"flight_id {flight_id} not found in aot_flight_id_to_clip_id.pkl")
            clip_id = int(flight_id_to_clip_id[flight_id])

            flight = ds.get_flight(flight_id)
            # Ensure all images exist locally for linking.
            flight.download(parallel=int(args.download_parallel))

            frame_ids = list(flight.frames.keys())
            clip_id_to_len[clip_id] = len(frame_ids)
            for fid, frame_id in enumerate(frame_ids):
                frame = flight.frames[frame_id]
                src = aot_root / frame.image_path()
                dst = frames_dir / f"Clip_{clip_id}_{fid:05d}.png"
                if not src.exists():
                    raise FileNotFoundError(f"Missing AOT image after download: {src}")
                _safe_link_or_copy(src, dst)

                # Labels (sparse): only write label file for positive frames.
                _write_yolo_label(
                    frame,
                    clip_id=clip_id,
                    fid=fid,
                    labels_dir=labels_dir,
                    distance_threshold_m=int(args.distance_threshold_m),
                )

        # video_length_dict.pkl for this part
        with open(videos_dir / "video_length_dict.pkl", "wb") as f:
            pickle.dump(dict(clip_id_to_len), f)

        # data yaml for this part
        yaml_dict = {
            "path": str(out_root).replace("\\", "/"),
            "train": "train/frames",
            "val": "val/full/frames",
            "test": f"{args.split}/{part_name}/frames",
            "annotation_path": str(out_root).replace("\\", "/"),
            "annotation_train": "train/labels",
            "annotation_val": "val/full/labels",
            "annotation_test": f"{args.split}/{part_name}/labels",
            "video_root_path": str(out_root).replace("\\", "/"),
            "video_root_path_train": "train/videos",
            "video_root_path_val": "val/full/videos",
            "video_root_path_test": f"{args.split}/{part_name}/videos",
            "nc": 2,
            "names": ["drone", "airborne"],
        }
        yaml_path = yaml_dir / f"AOT{args.split.capitalize()}_{part.part_id}.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_dict, sort_keys=False), encoding="utf-8")

    print(f"Wrote {len(parts)} yaml files under: {yaml_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

