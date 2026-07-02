from __future__ import annotations

import csv
import html
import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BoxRow:
    video: str
    frame_id: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0
    label: str = "drone"


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-6, area_a + area_b - inter))


_VIDEO_FRAME_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")
_LI_TETC_FRAME_RE = re.compile(r"time_layer:\s*(?P<frame>\d+)\s+detections:\s*(?P<detections>.*)")
_LI_TETC_BOX_RE = re.compile(r"\(([^)]+)\)")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _normalized_path_parts(value: str) -> list[str]:
    return [part for part in str(value).replace("\\", "/").split("/") if part]


def _stem(value: str) -> str:
    name = _normalized_path_parts(value)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _parse_video_frame_name(value: str, fallback_video: str | None = None, fallback_frame: int | None = None) -> tuple[str, int]:
    parts = _normalized_path_parts(value)
    stem = _stem(value)
    match = _VIDEO_FRAME_RE.match(stem)
    if match:
        return match.group("video"), int(match.group("frame"))

    video = fallback_video
    if video is None and len(parts) >= 2:
        video = parts[-2]
    if video is None:
        video = "video"

    if stem.isdigit() and len(stem) <= 8:
        return video, int(stem)
    trailing = re.search(r"(\d{1,8})$", stem)
    if trailing:
        return video, int(trailing.group(1))
    if fallback_frame is not None:
        return video, fallback_frame
    raise ValueError(f"Could not parse frame id from name: {value}")


def _coerce_short_frame_id(value: Any) -> int | None:
    if isinstance(value, bool) or value in ("", None):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if re.fullmatch(r"\d{1,8}", text):
        return int(text)
    return None


def _aot_item_frame_id(item: dict[str, Any]) -> int | None:
    for key in ("frame_id", "frame", "img_id"):
        frame_id = _coerce_short_frame_id(item.get(key))
        if frame_id is not None:
            return frame_id
    blob = item.get("blob")
    if isinstance(blob, dict):
        return _coerce_short_frame_id(blob.get("frame"))
    return None


def _parse_aot_img_name(value: str, fallback_video: str | None = None, fallback_frame: int | None = None) -> tuple[str, int]:
    parts = _normalized_path_parts(value)
    stem = _stem(value) if value else ""
    match = _VIDEO_FRAME_RE.match(stem)
    if match:
        return match.group("video"), int(match.group("frame"))

    video = fallback_video
    if video is None and len(parts) >= 2:
        video = parts[-2]
    if video is None:
        video = "video"

    if fallback_frame is not None:
        return video, fallback_frame
    if stem.isdigit() and len(stem) <= 8:
        return video, int(stem)
    trailing = re.search(r"(\d{1,8})$", stem)
    if trailing:
        return video, int(trailing.group(1))
    raise ValueError(f"Could not parse AOT frame id from name: {value}")


def _aot_lookup_keys(img_name: str) -> list[str]:
    keys = []
    for value in (img_name, _normalized_path_parts(img_name)[-1] if img_name else "", _stem(img_name) if img_name else ""):
        if value and value not in keys:
            keys.append(value)
    return keys


def _xywh_to_xyxy(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def _tlwh_to_xyxy(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return x, y, x + width, y + height


def _clean_video_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_gt", "_dt", "_pred", "_result", "_all_boxes", "_all_scores", "_time"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.isdigit():
        return f"Video_{stem}"
    return stem


def _as_box(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        if "bbox_xyxy" in value:
            value = value["bbox_xyxy"]
        elif "bbox" in value:
            raw = value["bbox"]
            mode = value.get("bbox_mode", "xyxy")
            if mode == "xywh":
                x, y, w, h = raw
                value = [x, y, x + w, y + h]
            else:
                value = raw
    vals = [float(x) for x in value]
    if len(vals) != 4:
        raise ValueError(f"Expected 4 bbox values, got {value!r}")
    return vals[0], vals[1], vals[2], vals[3]


def _row_video(row: dict[str, Any]) -> str:
    for key in ("video", "seq", "sequence", "flight_id", "clip", "video_id"):
        if row.get(key):
            return str(row[key])
    if row.get("video_path"):
        return Path(str(row["video_path"])).stem
    if row.get("image_path"):
        return Path(str(row["image_path"])).parent.name
    return "video"


def _row_frame(row: dict[str, Any]) -> int:
    for key in ("frame_id", "frame", "img_id", "image_id"):
        if key in row and row[key] != "":
            return int(float(row[key]))
    raise ValueError(f"Missing frame id in row: {row}")


def _row_score(row: dict[str, Any]) -> float:
    for key in ("score", "confidence", "conf", "final_drone_score", "objectness"):
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return 1.0


def _row_label(row: dict[str, Any]) -> str:
    for key in ("label", "class", "class_name", "predicted_class", "category"):
        if row.get(key):
            return str(row[key])
    return "drone"


def _row_bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox_xyxy" in row:
        return _as_box(row["bbox_xyxy"])
    if "bbox" in row:
        mode = str(row.get("bbox_mode", "xyxy"))
        raw = row["bbox"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        if mode == "xywh":
            x, y, w, h = [float(x) for x in raw]
            return x, y, x + w, y + h
        return _as_box(raw)
    keys = ("x1", "y1", "x2", "y2")
    if all(k in row and row[k] != "" for k in keys):
        return tuple(float(row[k]) for k in keys)  # type: ignore[return-value]
    keys = ("xmin", "ymin", "xmax", "ymax")
    if all(k in row and row[k] != "" for k in keys):
        return tuple(float(row[k]) for k in keys)  # type: ignore[return-value]
    if all(k in row and row[k] != "" for k in ("x", "y", "w", "h")):
        x, y, w, h = (float(row[k]) for k in ("x", "y", "w", "h"))
        return x, y, x + w, y + h
    raise ValueError(f"Missing bbox in row: {row}")


def _row_to_box(row: dict[str, Any], score_threshold: float = 0.0, labels: set[str] | None = None) -> BoxRow | None:
    label = _row_label(row)
    if labels is not None and label not in labels:
        return None
    score = _row_score(row)
    if score < score_threshold:
        return None
    return BoxRow(video=_row_video(row), frame_id=_row_frame(row), bbox=_row_bbox(row), score=score, label=label)


def load_boxes_csv(path: str | Path, score_threshold: float = 0.0, labels: set[str] | None = None) -> list[BoxRow]:
    rows: list[BoxRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            item = _row_to_box(raw, score_threshold=score_threshold, labels=labels)
            if item is not None:
                rows.append(item)
    return rows


def load_boxes_jsonl(path: str | Path, score_threshold: float = 0.0, labels: set[str] | None = None) -> list[BoxRow]:
    rows: list[BoxRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = _row_to_box(json.loads(line), score_threshold=score_threshold, labels=labels)
            if item is not None:
                rows.append(item)
    return rows


def load_boxes_yolo_dir(
    path: str | Path,
    score_threshold: float = 0.0,
    labels: set[str] | None = None,
    frame_offset: int = 0,
    img_size: tuple[float, float] | None = None,
) -> list[BoxRow]:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"YOLO label directory not found: {root}")

    scale_x, scale_y = img_size if img_size else (1.0, 1.0)
    rows: list[BoxRow] = []
    for txt in sorted(root.rglob("*.txt")):
        video, frame_id = _parse_video_frame_name(str(txt.relative_to(root)), fallback_video=txt.parent.name)
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    raise ValueError(f"{txt}: expected YOLO line 'class cx cy w h [score]', got {line!r}")
                label = parts[0]
                if labels is not None and label not in labels:
                    continue
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                if score < score_threshold:
                    continue
                cx, cy, width, height = (float(x) for x in parts[1:5])
                bbox = _xywh_to_xyxy(cx * scale_x, cy * scale_y, width * scale_x, height * scale_y)
                rows.append(BoxRow(video=video, frame_id=frame_id + frame_offset, bbox=bbox, score=score, label=label))
    return rows


def _iter_aot_items(path: Path) -> Iterable[tuple[dict[str, Any], str | None, int]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"{path}: expected a JSON list")
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield item, None, idx
        return

    if not path.is_dir():
        raise FileNotFoundError(f"AOT result path not found: {path}")

    child_files = [child / "result.json" for child in sorted(path.iterdir()) if child.is_dir() and (child / "result.json").is_file()]
    files = child_files or ([path / "result.json"] if (path / "result.json").is_file() else [])
    if not files:
        raise FileNotFoundError(f"No result.json found under: {path}")
    for file_path in files:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"{file_path}: expected a JSON list")
        fallback_video = file_path.parent.name if file_path.parent != path else None
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield item, fallback_video, idx


def load_boxes_aot_json(
    path: str | Path,
    score_threshold: float = 0.0,
    frame_offset: int = 0,
    frame_lookup: dict[str, tuple[str, int]] | None = None,
) -> list[BoxRow]:
    rows: list[BoxRow] = []
    for item, fallback_video, fallback_frame in _iter_aot_items(Path(path)):
        img_name = str(item.get("img_name") or item.get("image_path") or item.get("file_name") or "")
        item_frame = _aot_item_frame_id(item)
        lookup_hit = None
        if frame_lookup is not None:
            for key in _aot_lookup_keys(img_name):
                if key in frame_lookup:
                    lookup_hit = frame_lookup[key]
                    break
        if lookup_hit is not None:
            video, frame_id = lookup_hit
        else:
            video, frame_id = _parse_aot_img_name(
                img_name,
                fallback_video=fallback_video,
                fallback_frame=item_frame if item_frame is not None else fallback_frame,
            )
        detections = item.get("detections") or []
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            score = float(det.get("s", det.get("score", det.get("confidence", 0.0))))
            if score < score_threshold:
                continue
            if "bbox" in det:
                if det.get("bbox_mode") == "xywh":
                    x, y, width, height = [float(v) for v in det["bbox"]]
                    bbox = _tlwh_to_xyxy(x, y, width, height)
                else:
                    bbox = _as_box(det["bbox"])
            else:
                bbox = _xywh_to_xyxy(float(det.get("x", 0.0)), float(det.get("y", 0.0)), float(det.get("w", 0.0)), float(det.get("h", 0.0)))
            rows.append(BoxRow(video=video, frame_id=frame_id + frame_offset, bbox=bbox, score=score, label="drone"))
    return rows


def _resolve_aot_groundtruth_json(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"AOT groundtruth path not found: {path}")
    candidates = [path / "groundtruth.json", path / "ImageSets" / "groundtruth.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No groundtruth.json found under: {path}")


def _iter_aot_groundtruth_entities(path: str | Path) -> Iterable[tuple[str, dict[str, Any]]]:
    gt_path = _resolve_aot_groundtruth_json(Path(path))
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    samples = data.get("samples", data) if isinstance(data, dict) else data
    if isinstance(samples, dict):
        sample_iter = samples.items()
    elif isinstance(samples, list):
        sample_iter = ((str(idx), sample) for idx, sample in enumerate(samples))
    else:
        raise TypeError(f"{gt_path}: expected AOT samples object or list")

    for sample_key, sample in sample_iter:
        if not isinstance(sample, dict):
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        fallback_video = str(sample.get("flight_id") or metadata.get("flight_id") or sample_key)
        entities = sample.get("entities") or []
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, dict):
                yield fallback_video, entity


def load_aot_groundtruth_frame_lookup(path: str | Path) -> dict[str, tuple[str, int]]:
    lookup: dict[str, tuple[str, int]] = {}
    for fallback_video, entity in _iter_aot_groundtruth_entities(path):
        frame_id = _aot_item_frame_id(entity)
        if frame_id is None:
            continue
        video = str(entity.get("flight_id") or fallback_video)
        img_name = str(entity.get("img_name") or entity.get("image_path") or entity.get("file_name") or "")
        for key in _aot_lookup_keys(img_name):
            lookup.setdefault(key, (video, frame_id))
    return lookup


def load_boxes_aot_groundtruth_json(
    path: str | Path,
    labels: set[str] | None = None,
    frame_offset: int = 0,
) -> list[BoxRow]:
    if labels is not None and not ({"drone", "airborne"} & labels):
        return []

    rows: list[BoxRow] = []
    for idx, (fallback_video, entity) in enumerate(_iter_aot_groundtruth_entities(path)):
        if "id" not in entity:
            continue
        raw_bbox = entity.get("bb") or entity.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
            continue
        x, y, width, height = (float(v) for v in raw_bbox[:4])
        if width <= 0 or height <= 0:
            continue
        video = str(entity.get("flight_id") or fallback_video)
        frame_id = _aot_item_frame_id(entity)
        if frame_id is None:
            img_name = str(entity.get("img_name") or entity.get("image_path") or entity.get("file_name") or "")
            video, frame_id = _parse_aot_img_name(img_name, fallback_video=video, fallback_frame=idx)
        rows.append(
            BoxRow(
                video=video,
                frame_id=frame_id + frame_offset,
                bbox=_tlwh_to_xyxy(x, y, width, height),
                score=1.0,
                label="drone",
            )
        )
    return rows


def load_boxes_xywh_file(
    path: str | Path,
    score_threshold: float = 0.0,
    labels: set[str] | None = None,
    frame_offset: int = 0,
) -> list[BoxRow]:
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [
            item
            for item in sorted(root.rglob("*.txt"))
            if not item.stem.endswith(("_time", "_all_boxes", "_all_scores"))
        ]
    else:
        raise FileNotFoundError(f"xywh file path not found: {root}")

    if labels is not None and "drone" not in labels:
        return []

    rows: list[BoxRow] = []
    for file_path in files:
        video = _clean_video_stem(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = [p for p in re.split(r"[,\t ]+", line) if p]
                if len(parts) < 4:
                    raise ValueError(f"{file_path}:{idx}: expected x y w h [score], got {line!r}")
                x, y, width, height = (float(v) for v in parts[:4])
                if width <= 0 or height <= 0:
                    continue
                score = float(parts[4]) if len(parts) >= 5 else 1.0
                if score < score_threshold:
                    continue
                rows.append(
                    BoxRow(
                        video=video,
                        frame_id=idx + frame_offset,
                        bbox=_tlwh_to_xyxy(x, y, width, height),
                        score=score,
                        label="drone",
                    )
                )
    return rows


def _iter_antiuav_label_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"AntiUAV label path not found: {path}")
    if (path / "IR_label.json").is_file():
        yield path / "IR_label.json"
        return
    if (path / "list.txt").is_file():
        for name in path.joinpath("list.txt").read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if name and (path / name / "IR_label.json").is_file():
                yield path / name / "IR_label.json"
        return
    yield from sorted(path.glob("*/IR_label.json"))


def load_boxes_antiuav_json(path: str | Path, frame_offset: int = 0) -> list[BoxRow]:
    rows: list[BoxRow] = []
    for file_path in _iter_antiuav_label_files(Path(path)):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        gt_rect = data.get("gt_rect") or []
        exist = data.get("exist")
        video = file_path.parent.name if file_path.name == "IR_label.json" else file_path.stem
        for idx, rect in enumerate(gt_rect, start=1):
            if exist is not None and idx - 1 < len(exist) and not bool(exist[idx - 1]):
                continue
            if not isinstance(rect, list) or len(rect) < 4:
                continue
            x, y, width, height = (float(v) for v in rect[:4])
            if width <= 0 or height <= 0:
                continue
            rows.append(
                BoxRow(
                    video=video,
                    frame_id=idx + frame_offset,
                    bbox=_tlwh_to_xyxy(x, y, width, height),
                    score=1.0,
                    label="drone",
                )
            )
    return rows


def load_boxes_li_tetc_txt(
    path: str | Path,
    labels: set[str] | None = None,
    frame_offset: int = 0,
) -> list[BoxRow]:
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.rglob("*.txt"))
    else:
        raise FileNotFoundError(f"Li-TETC txt path not found: {root}")

    if labels is not None and "drone" not in labels:
        return []

    rows: list[BoxRow] = []
    for file_path in files:
        video = _clean_video_stem(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                match = _LI_TETC_FRAME_RE.search(line.strip())
                if not match:
                    continue
                frame_id = int(match.group("frame")) + frame_offset
                for raw_box in _LI_TETC_BOX_RE.findall(match.group("detections")):
                    values = [float(part.strip()) for part in raw_box.split(",") if part.strip()]
                    if len(values) != 4:
                        continue
                    y1, x1, y2, x2 = values
                    if x2 <= x1 or y2 <= y1:
                        continue
                    rows.append(BoxRow(video=video, frame_id=frame_id, bbox=(x1, y1, x2, y2), score=1.0, label="drone"))
    return rows


def load_boxes_tvd_pkl(
    path: str | Path,
    kind: str,
    score_threshold: float = 0.0,
    labels: set[str] | None = None,
    frame_offset: int = 0,
) -> list[BoxRow]:
    """Load TransVisDrone validation pickle outputs.

    `save-json-gt` pickles are dictionaries keyed by image id and contain both
    native-pixel `labels` and `detections` in xyxy. Regular prediction pickles
    are lists with COCO-style xywh boxes.
    """
    with Path(path).open("rb") as f:
        data = pickle.load(f)

    rows: list[BoxRow] = []
    if isinstance(data, dict):
        field = "labels" if kind == "gt" else "detections"
        for image_id, item in data.items():
            if not isinstance(item, dict):
                continue
            video, frame_id = _parse_video_frame_name(str(image_id))
            for raw in item.get(field, []) or []:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("category_id", raw.get("label", "0")))
                if labels is not None and label not in labels:
                    continue
                score = float(raw.get("score", 1.0))
                if score < score_threshold:
                    continue
                rows.append(
                    BoxRow(
                        video=video,
                        frame_id=frame_id + frame_offset,
                        bbox=_as_box(raw.get("bbox", raw.get("bbox_xyxy"))),
                        score=score,
                        label=label,
                    )
                )
        return rows

    if kind == "gt":
        raise ValueError(f"{path}: TransVisDrone prediction-only pickle does not contain GT labels")
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a dict or list pickle, got {type(data)}")
    for raw in data:
        if not isinstance(raw, dict):
            continue
        image_id = raw.get("image_id") or raw.get("img_name") or raw.get("image_path")
        if image_id is None:
            continue
        video, frame_id = _parse_video_frame_name(str(image_id))
        label = str(raw.get("category_id", raw.get("label", "0")))
        if labels is not None and label not in labels:
            continue
        score = float(raw.get("score", raw.get("confidence", 1.0)))
        if score < score_threshold:
            continue
        bbox_raw = raw.get("bbox")
        if bbox_raw is None:
            continue
        x, y, width, height = [float(v) for v in bbox_raw]
        rows.append(
            BoxRow(
                video=video,
                frame_id=frame_id + frame_offset,
                bbox=_tlwh_to_xyxy(x, y, width, height),
                score=score,
                label=label,
            )
        )
    return rows


FrameManifest = dict[str, set[int]]
FrameLookup = dict[str, tuple[str, int]]


def _add_manifest_frame(frames: FrameManifest, video: str, frame_id: int) -> None:
    frames[str(video)].add(int(frame_id))


def _add_frame_lookup(lookup: FrameLookup, image_name: str, video: str, frame_id: int) -> None:
    for key in _aot_lookup_keys(image_name):
        lookup.setdefault(key, (str(video), int(frame_id)))


def load_frame_manifest_csv(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    frames: FrameManifest = defaultdict(set)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("start_frame") not in (None, "") and row.get("end_frame") not in (None, ""):
                video = _row_video(row)
                start = int(float(row["start_frame"])) + frame_offset
                end = int(float(row["end_frame"])) + frame_offset
                for frame_id in range(min(start, end), max(start, end) + 1):
                    _add_manifest_frame(frames, video, frame_id)
                continue
            if row.get("frame_id") not in (None, "") or row.get("frame") not in (None, ""):
                _add_manifest_frame(frames, _row_video(row), _row_frame(row) + frame_offset)
                continue
            image_name = row.get("image_path") or row.get("img_name") or row.get("file_name")
            if image_name:
                video, frame_id = _parse_video_frame_name(str(image_name))
                _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_manifest_image_dir(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    root = Path(path)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        base = root
    else:
        raise FileNotFoundError(f"Frame image path not found: {root}")

    frames: FrameManifest = defaultdict(set)
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for image_path in files:
        by_parent[image_path.parent].append(image_path)
    for parent, parent_files in sorted(by_parent.items(), key=lambda item: str(item[0])):
        for idx, image_path in enumerate(sorted(parent_files, key=lambda p: p.name)):
            rel = image_path.relative_to(base) if image_path.is_relative_to(base) else image_path
            video, frame_id = _parse_video_frame_name(
                str(rel),
                fallback_video=image_path.parent.name,
                fallback_frame=idx,
            )
            _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_lookup_image_dir(path: str | Path, frame_offset: int = 0) -> FrameLookup:
    root = Path(path)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        base = root
    else:
        raise FileNotFoundError(f"Frame image path not found: {root}")

    lookup: FrameLookup = {}
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for image_path in files:
        by_parent[image_path.parent].append(image_path)
    for parent, parent_files in sorted(by_parent.items(), key=lambda item: str(item[0])):
        for idx, image_path in enumerate(sorted(parent_files, key=lambda p: p.name)):
            rel = image_path.relative_to(base) if image_path.is_relative_to(base) else image_path
            video, frame_id = _parse_video_frame_name(
                str(rel),
                fallback_video=image_path.parent.name,
                fallback_frame=idx,
            )
            _add_frame_lookup(lookup, str(rel), video, frame_id + frame_offset)
            _add_frame_lookup(lookup, image_path.name, video, frame_id + frame_offset)
    return lookup


def load_frame_manifest_yolo_dir(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"YOLO frame manifest directory not found: {root}")
    frames: FrameManifest = defaultdict(set)
    for txt in sorted(root.rglob("*.txt")):
        video, frame_id = _parse_video_frame_name(str(txt.relative_to(root)), fallback_video=txt.parent.name)
        _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_manifest_xywh_file(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [
            item
            for item in sorted(root.rglob("*.txt"))
            if not item.stem.endswith(("_time", "_all_boxes", "_all_scores"))
        ]
    else:
        raise FileNotFoundError(f"xywh frame manifest path not found: {root}")

    frames: FrameManifest = defaultdict(set)
    for file_path in files:
        video = _clean_video_stem(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if line.strip():
                    _add_manifest_frame(frames, video, idx + frame_offset)
    return frames


def load_frame_manifest_antiuav_json(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    frames: FrameManifest = defaultdict(set)
    for file_path in _iter_antiuav_label_files(Path(path)):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        gt_rect = data.get("gt_rect") or []
        exist = data.get("exist") or []
        n = max(len(gt_rect), len(exist))
        video = file_path.parent.name if file_path.name == "IR_label.json" else file_path.stem
        for frame_id in range(1, n + 1):
            _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_manifest_li_tetc_txt(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.rglob("*.txt"))
    else:
        raise FileNotFoundError(f"Li-TETC frame manifest path not found: {root}")
    frames: FrameManifest = defaultdict(set)
    for file_path in files:
        video = _clean_video_stem(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                match = _LI_TETC_FRAME_RE.search(line.strip())
                if match:
                    _add_manifest_frame(frames, video, int(match.group("frame")) + frame_offset)
    return frames


def load_frame_manifest_tvd_pkl(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    frames: FrameManifest = defaultdict(set)
    if isinstance(data, dict):
        for image_id in data:
            video, frame_id = _parse_video_frame_name(str(image_id))
            _add_manifest_frame(frames, video, frame_id + frame_offset)
        return frames
    if isinstance(data, list):
        for raw in data:
            if not isinstance(raw, dict):
                continue
            image_id = raw.get("image_id") or raw.get("img_name") or raw.get("image_path")
            if image_id is None:
                continue
            video, frame_id = _parse_video_frame_name(str(image_id))
            _add_manifest_frame(frames, video, frame_id + frame_offset)
        return frames
    raise TypeError(f"{path}: expected a dict or list pickle, got {type(data)}")


def load_frame_manifest_aot_json(
    path: str | Path,
    frame_offset: int = 0,
    frame_lookup: dict[str, tuple[str, int]] | None = None,
) -> FrameManifest:
    frames: FrameManifest = defaultdict(set)
    for item, fallback_video, fallback_frame in _iter_aot_items(Path(path)):
        img_name = str(item.get("img_name") or item.get("image_path") or item.get("file_name") or "")
        item_frame = _aot_item_frame_id(item)
        lookup_hit = None
        if frame_lookup is not None:
            for key in _aot_lookup_keys(img_name):
                if key in frame_lookup:
                    lookup_hit = frame_lookup[key]
                    break
        if lookup_hit is not None:
            video, frame_id = lookup_hit
        else:
            video, frame_id = _parse_aot_img_name(
                img_name,
                fallback_video=fallback_video,
                fallback_frame=item_frame if item_frame is not None else fallback_frame,
            )
        _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_manifest_aot_groundtruth_json(path: str | Path, frame_offset: int = 0) -> FrameManifest:
    frames: FrameManifest = defaultdict(set)
    for idx, (fallback_video, entity) in enumerate(_iter_aot_groundtruth_entities(path)):
        video = str(entity.get("flight_id") or fallback_video)
        frame_id = _aot_item_frame_id(entity)
        if frame_id is None:
            img_name = str(entity.get("img_name") or entity.get("image_path") or entity.get("file_name") or "")
            video, frame_id = _parse_aot_img_name(img_name, fallback_video=video, fallback_frame=idx)
        _add_manifest_frame(frames, video, frame_id + frame_offset)
    return frames


def load_frame_manifest(
    path: str | Path,
    fmt: str,
    frame_offset: int = 0,
    frame_lookup: dict[str, tuple[str, int]] | None = None,
) -> FrameManifest:
    if fmt == "csv":
        return load_frame_manifest_csv(path, frame_offset=frame_offset)
    if fmt == "image-dir":
        return load_frame_manifest_image_dir(path, frame_offset=frame_offset)
    if fmt == "yolo-dir":
        return load_frame_manifest_yolo_dir(path, frame_offset=frame_offset)
    if fmt == "xywh-file":
        return load_frame_manifest_xywh_file(path, frame_offset=frame_offset)
    if fmt == "antiuav-json":
        return load_frame_manifest_antiuav_json(path, frame_offset=frame_offset)
    if fmt == "li-tetc-txt":
        return load_frame_manifest_li_tetc_txt(path, frame_offset=frame_offset)
    if fmt in {"tvd-pkl-gt", "tvd-pkl-pred"}:
        return load_frame_manifest_tvd_pkl(path, frame_offset=frame_offset)
    if fmt == "aot-json":
        return load_frame_manifest_aot_json(path, frame_offset=frame_offset, frame_lookup=frame_lookup)
    if fmt == "aot-gt-json":
        return load_frame_manifest_aot_groundtruth_json(path, frame_offset=frame_offset)
    if fmt == "jsonl":
        rows = load_boxes_jsonl(path)
        frames: FrameManifest = defaultdict(set)
        for row in rows:
            _add_manifest_frame(frames, row.video, row.frame_id + frame_offset)
        return frames
    raise ValueError(f"Unsupported frame manifest format: {fmt}")


def load_boxes(
    path: str | Path,
    fmt: str,
    score_threshold: float = 0.0,
    labels: Iterable[str] | None = None,
    frame_offset: int = 0,
    img_size: tuple[float, float] | None = None,
) -> list[BoxRow]:
    label_set = set(labels) if labels else None
    if fmt == "csv":
        return load_boxes_csv(path, score_threshold=score_threshold, labels=label_set)
    if fmt == "jsonl":
        return load_boxes_jsonl(path, score_threshold=score_threshold, labels=label_set)
    if fmt == "yolo-dir":
        return load_boxes_yolo_dir(path, score_threshold=score_threshold, labels=label_set, frame_offset=frame_offset, img_size=img_size)
    if fmt == "aot-json":
        return load_boxes_aot_json(path, score_threshold=score_threshold, frame_offset=frame_offset)
    if fmt == "aot-gt-json":
        return load_boxes_aot_groundtruth_json(path, labels=label_set, frame_offset=frame_offset)
    if fmt == "xywh-file":
        return load_boxes_xywh_file(path, score_threshold=score_threshold, labels=label_set, frame_offset=frame_offset)
    if fmt == "antiuav-json":
        return load_boxes_antiuav_json(path, frame_offset=frame_offset)
    if fmt == "li-tetc-txt":
        return load_boxes_li_tetc_txt(path, labels=label_set, frame_offset=frame_offset)
    if fmt == "tvd-pkl-gt":
        return load_boxes_tvd_pkl(path, kind="gt", labels=label_set, frame_offset=frame_offset)
    if fmt == "tvd-pkl-pred":
        return load_boxes_tvd_pkl(path, kind="pred", score_threshold=score_threshold, labels=label_set, frame_offset=frame_offset)
    raise ValueError(f"Unsupported box format: {fmt}")


def _group_by_video_frame(rows: Iterable[BoxRow]) -> dict[str, dict[int, list[BoxRow]]]:
    out: dict[str, dict[int, list[BoxRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        out[row.video][row.frame_id].append(row)
    return out


def _match_frame(gt_rows: list[BoxRow], pred_rows: list[BoxRow], iou_threshold: float) -> tuple[int, int, int]:
    preds = sorted(pred_rows, key=lambda x: x.score, reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_idx, pred in enumerate(preds):
        best_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_rows):
            if gt_idx in matched_gt:
                continue
            ov = _bbox_iou(pred.bbox, gt.bbox)
            if ov > best_iou:
                best_iou = ov
                best_idx = gt_idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matched_pred.add(pred_idx)
    tp = len(matched_gt)
    fp = len(preds) - len(matched_pred)
    fn = len(gt_rows) - tp
    return tp, fp, fn


def frame_counts(gt_rows: list[BoxRow], pred_rows: list[BoxRow], iou_threshold: float) -> dict[tuple[str, int], dict[str, int]]:
    gt_by = _group_by_video_frame(gt_rows)
    pred_by = _group_by_video_frame(pred_rows)
    keys = {
        (video, frame_id)
        for video, frames in gt_by.items()
        for frame_id in frames
    } | {
        (video, frame_id)
        for video, frames in pred_by.items()
        for frame_id in frames
    }
    out: dict[tuple[str, int], dict[str, int]] = {}
    for video, frame_id in sorted(keys):
        tp, fp, fn = _match_frame(gt_by.get(video, {}).get(frame_id, []), pred_by.get(video, {}).get(frame_id, []), iou_threshold)
        out[(video, frame_id)] = {
            "gt": len(gt_by.get(video, {}).get(frame_id, [])),
            "pred": len(pred_by.get(video, {}).get(frame_id, [])),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def _metric_row(video: str, frame_id: int, start_frame: int, end_frame: int, counts: list[dict[str, int]]) -> dict[str, Any]:
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    gt = sum(c["gt"] for c in counts)
    pred = sum(c["pred"] for c in counts)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "video": video,
        "frame_id": frame_id,
        "window_start_frame": start_frame,
        "window_end_frame": end_frame,
        "gt": gt,
        "pred": pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def sliding_window_metrics(
    gt_rows: list[BoxRow],
    pred_rows: list[BoxRow],
    fps: float,
    seconds: float = 3.0,
    iou_threshold: float = 0.5,
    include_empty_between_minmax: bool = True,
    center_frames: FrameManifest | None = None,
) -> list[dict[str, Any]]:
    per_frame = frame_counts(gt_rows, pred_rows, iou_threshold)
    frames_by_video: dict[str, set[int]] = defaultdict(set)
    for video, frame_id in per_frame:
        frames_by_video[video].add(frame_id)
    if center_frames is not None:
        for video, frames in center_frames.items():
            frames_by_video[str(video)].update(int(frame_id) for frame_id in frames)

    window = max(0, int(round(float(fps) * float(seconds))))
    rows: list[dict[str, Any]] = []
    for video in sorted(frames_by_video):
        frames = sorted(frames_by_video[video])
        if not frames:
            continue
        if center_frames is not None and video in center_frames:
            centers = sorted(center_frames[video])
        else:
            centers = range(frames[0], frames[-1] + 1) if include_empty_between_minmax else frames
        for center in centers:
            start = center - window
            end = center + window
            counts = [per_frame.get((video, frame_id), {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}) for frame_id in range(start, end + 1)]
            rows.append(_metric_row(video, center, start, end, counts))
    return rows


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "video",
        "frame_id",
        "window_start_frame",
        "window_end_frame",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            for key in ("precision", "recall", "f1", "accuracy"):
                clean[key] = f"{float(clean[key]):.6f}"
            writer.writerow(clean)


def write_worst_windows_csv(path: str | Path, rows: list[dict[str, Any]], per_video: int = 20) -> None:
    fields = [
        "video",
        "rank",
        "frame_id",
        "window_start_frame",
        "window_end_frame",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for video in sorted({str(r["video"]) for r in rows}):
            vr = [r for r in rows if r["video"] == video]
            ranked = sorted(
                vr,
                key=lambda r: (
                    float(r["accuracy"]),
                    float(r["recall"]),
                    float(r["precision"]),
                    int(r["frame_id"]),
                ),
            )[:per_video]
            for rank, row in enumerate(ranked, start=1):
                clean = {key: row[key] for key in fields if key != "rank"}
                clean["rank"] = rank
                for key in ("precision", "recall", "f1", "accuracy"):
                    clean[key] = f"{float(clean[key]):.6f}"
                writer.writerow(clean)


def _low_accuracy_segment_row(video: str, rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    start_frame = int(rows[0]["frame_id"])
    end_frame = int(rows[-1]["frame_id"])
    acc = [float(r["accuracy"]) for r in rows]
    precision = [float(r["precision"]) for r in rows]
    recall = [float(r["recall"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    worst = min(rows, key=lambda r: (float(r["accuracy"]), float(r["recall"]), int(r["frame_id"])))
    return {
        "video": video,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_frame / fps if fps > 0 else 0.0,
        "end_time_sec": end_frame / fps if fps > 0 else 0.0,
        "center_frames": len(rows),
        "window_start_frame": min(int(r["window_start_frame"]) for r in rows),
        "window_end_frame": max(int(r["window_end_frame"]) for r in rows),
        "min_accuracy": min(acc),
        "mean_accuracy": sum(acc) / max(1, len(acc)),
        "mean_precision": sum(precision) / max(1, len(precision)),
        "mean_recall": sum(recall) / max(1, len(recall)),
        "mean_f1": sum(f1) / max(1, len(f1)),
        "worst_frame": int(worst["frame_id"]),
        "gt": sum(int(r["gt"]) for r in rows),
        "pred": sum(int(r["pred"]) for r in rows),
        "tp": sum(int(r["tp"]) for r in rows),
        "fp": sum(int(r["fp"]) for r in rows),
        "fn": sum(int(r["fn"]) for r in rows),
    }


def low_accuracy_segments(rows: list[dict[str, Any]], fps: float, threshold: float = 0.5) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for video in sorted({str(r["video"]) for r in rows}):
        current: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for row in sorted((r for r in rows if str(r["video"]) == video), key=lambda r: int(r["frame_id"])):
            frame = int(row["frame_id"])
            is_low = float(row["accuracy"]) <= threshold
            if is_low and (previous_frame is None or frame == previous_frame + 1 or not current):
                current.append(row)
            else:
                if current:
                    segments.append(_low_accuracy_segment_row(video, current, fps=fps))
                    current = []
                if is_low:
                    current.append(row)
            previous_frame = frame
        if current:
            segments.append(_low_accuracy_segment_row(video, current, fps=fps))

    ranked: list[dict[str, Any]] = []
    for video in sorted({str(s["video"]) for s in segments}):
        video_segments = [s for s in segments if str(s["video"]) == video]
        video_segments.sort(
            key=lambda s: (
                float(s["min_accuracy"]),
                float(s["mean_accuracy"]),
                -int(s["center_frames"]),
                int(s["start_frame"]),
            )
        )
        for rank, segment in enumerate(video_segments, start=1):
            ranked.append({"rank": rank, **segment})
    return ranked


def write_low_accuracy_segments_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fps: float,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    fields = [
        "video",
        "rank",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "center_frames",
        "window_start_frame",
        "window_end_frame",
        "min_accuracy",
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "worst_frame",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = low_accuracy_segments(rows, fps=fps, threshold=threshold)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for segment in segments:
            clean = dict(segment)
            for key in ("start_time_sec", "end_time_sec", "min_accuracy", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1"):
                clean[key] = f"{float(clean[key]):.6f}"
            writer.writerow(clean)
    return segments


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, dict[str, Any]] = {}
    for video in sorted({str(r["video"]) for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        by_video[video] = {
            "frames": len(vr),
            "min_accuracy": min((float(r["accuracy"]) for r in vr), default=0.0),
            "mean_accuracy": sum(float(r["accuracy"]) for r in vr) / max(1, len(vr)),
            "min_recall": min((float(r["recall"]) for r in vr), default=0.0),
            "mean_recall": sum(float(r["recall"]) for r in vr) / max(1, len(vr)),
            "worst_frame_by_accuracy": min(vr, key=lambda r: float(r["accuracy"]))["frame_id"] if vr else None,
        }
    return {
        "videos": len(by_video),
        "frames": len(rows),
        "by_video": by_video,
    }


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_video_svg(path: str | Path, video: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 430
    left, right, top, bottom = 70, 24, 44, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    frames = [int(r["frame_id"]) for r in rows]
    min_f, max_f = min(frames), max(frames)
    span = max(1, max_f - min_f)

    def xy(row: dict[str, Any], key: str) -> tuple[float, float]:
        x = left + (int(row["frame_id"]) - min_f) / span * plot_w
        y = top + (1.0 - float(row[key])) * plot_h
        return x, y

    series = [
        ("accuracy", "#111827"),
        ("precision", "#2563eb"),
        ("recall", "#dc2626"),
        ("f1", "#059669"),
    ]
    lines = []
    for name, color in series:
        pts = [xy(r, name) for r in rows]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{_polyline(pts)}" />')

    grid = []
    for i in range(6):
        y = top + i / 5 * plot_h
        val = 1.0 - i / 5
        grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />')
        grid.append(f'<text x="18" y="{y+4:.2f}" font-size="12" fill="#4b5563">{val:.1f}</text>')
    for i in range(0, 11):
        x = left + i / 10 * plot_w
        frame = round(min_f + i / 10 * span)
        grid.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height-bottom}" stroke="#f3f4f6" />')
        grid.append(f'<text x="{x-18:.2f}" y="{height-24}" font-size="11" fill="#4b5563">{frame}</text>')

    legend_x = left
    legend = []
    for name, color in series:
        legend.append(f'<line x1="{legend_x}" x2="{legend_x+22}" y1="{height-13}" y2="{height-13}" stroke="{color}" stroke-width="3" />')
        legend.append(f'<text x="{legend_x+28}" y="{height-9}" font-size="12" fill="#374151">{name}</text>')
        legend_x += 110

    title = html.escape(f"{video}: per-frame +/-3s detection window metrics")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="24" font-size="18" font-family="Arial, sans-serif" fill="#111827">{title}</text>
{''.join(grid)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9ca3af"/>
{''.join(lines)}
{''.join(legend)}
<text x="{width/2-45:.2f}" y="{height-5}" font-size="12" fill="#4b5563">frame id</text>
<text x="8" y="{top-10}" font-size="12" fill="#4b5563">metric</text>
</svg>
"""
    out.write_text(svg, encoding="utf-8")


def write_plots(out_dir: str | Path, rows: list[dict[str, Any]]) -> list[Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for video in sorted({str(r["video"]) for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        path = root / f"{video}_window_metrics.svg"
        write_video_svg(path, video, vr)
        paths.append(path)
    index = root / "index.html"
    body = "\n".join(
        f'<section><h2>{html.escape(p.stem.replace("_window_metrics", ""))}</h2><img src="{html.escape(p.name)}" /></section>'
        for p in paths
    )
    index.write_text(f"<!doctype html><meta charset='utf-8'><title>Window Metrics</title>{body}\n", encoding="utf-8")
    return paths


def run_window_accuracy(
    gt: str | Path,
    pred: str | Path,
    out_dir: str | Path,
    fps: float,
    gt_format: str = "csv",
    pred_format: str = "csv",
    window_seconds: float = 3.0,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    segment_threshold: float = 0.5,
    gt_labels: Iterable[str] | None = None,
    pred_labels: Iterable[str] | None = None,
    gt_frame_offset: int = 0,
    pred_frame_offset: int = 0,
    img_size: tuple[float, float] | None = None,
    frame_manifest: str | Path | None = None,
    frame_manifest_format: str | None = None,
    frame_manifest_offset: int = 0,
    sparse_centers: bool = False,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_frame_lookup = None
    if frame_manifest is not None and (frame_manifest_format or "image-dir") == "image-dir":
        manifest_frame_lookup = load_frame_lookup_image_dir(frame_manifest, frame_offset=frame_manifest_offset)

    gt_rows = load_boxes(
        gt,
        gt_format,
        labels=gt_labels,
        frame_offset=gt_frame_offset,
        img_size=img_size,
    )
    if pred_format == "aot-json":
        pred_label_set = set(pred_labels) if pred_labels else None
        if pred_label_set is not None and "drone" not in pred_label_set:
            pred_rows = []
        else:
            frame_lookup = manifest_frame_lookup.copy() if manifest_frame_lookup else {}
            if gt_format == "aot-gt-json":
                frame_lookup.update(load_aot_groundtruth_frame_lookup(gt))
            pred_rows = load_boxes_aot_json(
                pred,
                score_threshold=score_threshold,
                frame_offset=pred_frame_offset,
                frame_lookup=frame_lookup or None,
            )
    else:
        pred_rows = load_boxes(
            pred,
            pred_format,
            score_threshold=score_threshold,
            labels=pred_labels,
            frame_offset=pred_frame_offset,
            img_size=img_size,
        )
    center_frames = None
    if frame_manifest is not None:
        fmt = frame_manifest_format or "image-dir"
        lookup = None
        if fmt == "aot-json":
            lookup = manifest_frame_lookup.copy() if manifest_frame_lookup else {}
            if gt_format == "aot-gt-json":
                lookup.update(load_aot_groundtruth_frame_lookup(gt))
        center_frames = load_frame_manifest(
            frame_manifest,
            fmt,
            frame_offset=frame_manifest_offset,
            frame_lookup=lookup or None,
        )
    rows = sliding_window_metrics(
        gt_rows,
        pred_rows,
        fps=fps,
        seconds=window_seconds,
        iou_threshold=iou_threshold,
        include_empty_between_minmax=not sparse_centers,
        center_frames=center_frames,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "per_frame_window_metrics.csv"
    worst_path = out / "worst_windows.csv"
    segments_path = out / "low_accuracy_segments.csv"
    summary_path = out / "summary.json"
    plot_dir = out / "plots"
    write_metrics_csv(csv_path, rows)
    write_worst_windows_csv(worst_path, rows)
    segments = write_low_accuracy_segments_csv(segments_path, rows, fps=fps, threshold=segment_threshold)
    plot_paths = write_plots(plot_dir, rows)
    summary = summarize_metrics(rows)
    summary.update(
        {
            "gt": str(gt),
            "pred": str(pred),
            "gt_format": gt_format,
            "pred_format": pred_format,
            "gt_boxes": len(gt_rows),
            "pred_boxes": len(pred_rows),
            "fps": fps,
            "window_seconds_each_side": window_seconds,
            "window_frames_each_side": round(float(fps) * float(window_seconds)),
            "iou": iou_threshold,
            "score_threshold": score_threshold,
            "low_accuracy_segment_threshold": segment_threshold,
            "low_accuracy_segments": len(segments),
            "gt_frame_offset": gt_frame_offset,
            "pred_frame_offset": pred_frame_offset,
            "frame_manifest": str(frame_manifest) if frame_manifest is not None else None,
            "frame_manifest_format": frame_manifest_format,
            "frame_manifest_offset": frame_manifest_offset,
            "frame_manifest_videos": len(center_frames) if center_frames is not None else None,
            "frame_manifest_frames": sum(len(frames) for frames in center_frames.values()) if center_frames is not None else None,
            "img_size": list(img_size) if img_size else None,
            "csv": str(csv_path),
            "worst_windows_csv": str(worst_path),
            "low_accuracy_segments_csv": str(segments_path),
            "plots": [str(p) for p in plot_paths],
            "plot_index": str(plot_dir / "index.html"),
        }
    )
    if extra_summary:
        summary.update(extra_summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
