from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

NPS_HOME = "https://engineering.purdue.edu/~bouman/UAV_Dataset/"
NPS_VIDEOS_URL = NPS_HOME + "Videos.zip"
NPS_ANNOTATIONS_URL = NPS_HOME + "Video_Annotation.zip"
AOT_S3_PART1 = "s3://airborne-obj-detection-challenge-training/part1"
EDTC_MODELS_DRIVE_ID = "1qp4rZ-d-KnYNLAz80wmqVAl8UXOPGfO-"
EDTC_DATASET_DRIVE_ID = "1igFX0yISt4auaH7ijHde1pBfvna8nn3G"
TRANSVISDRONE_PRETRAINED_DRIVE_ID = "1zOy_zIxkrvmHBIPU72PB_o0Da-h0h5JA"
TRANSVISDRONE_NPS_BEST_DRIVE_ID = "1rWsy7QwWIMWUfiNZKdpICcRH6FBNnQxr"
YOLOMG_ARD100_URL = "https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z"


@dataclass(frozen=True)
class HttpMeta:
    url: str
    ok: bool
    status: int | None = None
    content_length: int | None = None
    content_type: str | None = None
    error: str | None = None


def _head(url: str, timeout: int = 30) -> HttpMeta:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "URAP2026-source-inventory/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpMeta(
                url=url,
                ok=True,
                status=response.status,
                content_length=int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None,
                content_type=response.headers.get("Content-Type"),
            )
    except urllib.error.HTTPError as exc:
        return HttpMeta(url=url, ok=False, status=exc.code, error=str(exc))
    except OSError as exc:
        return HttpMeta(url=url, ok=False, error=str(exc))


def _get_text(url: str, timeout: int = 30) -> tuple[str | None, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore"), None
    except OSError as exc:
        return None, str(exc)


def _drive_folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"


def extract_drive_items(html: str) -> list[dict[str, str]]:
    """Best-effort parser for public Drive folder HTML.

    Google Drive does not expose a stable unauthenticated listing API here. The
    public page embeds visible file names and file IDs in the initial data blob;
    this parser extracts enough metadata to decide whether a source is worth a
    controlled download.
    """
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in re.finditer(r"<strong[^>]*>(?P<name>[^<]+)</strong>", html):
        name = match.group("name")
        head = html[max(0, match.start() - 1200) : match.start()]
        tail = html[match.end() : match.end() + 2500]
        data_id_matches = list(re.finditer(r'data-id="(?P<id>[A-Za-z0-9_-]{20,})"', head[-1200:]))
        file_id = None
        for data_id_match in reversed(data_id_matches):
            if "</strong>" not in head[-1200:][data_id_match.end() :]:
                file_id = data_id_match.group("id")
                break
        if file_id is None:
            id_match = re.search(r'\[\[null,"(?P<id>[A-Za-z0-9_-]{20,})"\]', tail)
            file_id = id_match.group("id") if id_match else None
        if file_id is None:
            continue
        item = (name, file_id)
        if item in seen:
            continue
        seen.add(item)
        items.append(
            {
                "name": name,
                "id": file_id,
                "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
            }
        )
    return items


def inventory_drive_folder(name: str, folder_id: str) -> dict[str, Any]:
    url = _drive_folder_url(folder_id)
    html, error = _get_text(url)
    if error:
        return {"name": name, "url": url, "ok": False, "error": error, "items": []}
    items = extract_drive_items(html or "")
    return {"name": name, "url": url, "ok": True, "items": items}


def _aws_ls(prefix: str, max_lines: int = 200) -> dict[str, Any]:
    aws = shutil.which("aws")
    if not aws:
        return {"ok": False, "error": "aws CLI not found", "lines": []}
    proc = subprocess.run(
        [aws, "s3", "ls", prefix, "--no-sign-request"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    lines = proc.stdout.splitlines()
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "lines": lines[:max_lines], "truncated": len(lines) > max_lines}


def inventory_aot() -> dict[str, Any]:
    image_sets = _aws_ls(f"{AOT_S3_PART1}/ImageSets/")
    top_level = _aws_ls(f"{AOT_S3_PART1}/")
    return {
        "s3_prefix": AOT_S3_PART1,
        "top_level": top_level,
        "image_sets": image_sets,
        "note": "ImageSets metadata is small enough to inspect; Images/ contains multi-MB frames and must not be recursively synced on this 16GiB-free local disk.",
    }


def inventory_edtc_local(base: Path) -> dict[str, Any]:
    validation_zip = base / "datasets" / "AntiUAV600" / "raw" / "validation.zip"
    validation_root = base / "datasets" / "AntiUAV600" / "validation"
    sequence_names = []
    if (validation_root / "list.txt").is_file():
        sequence_names = [
            line.strip()
            for line in (validation_root / "list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    label_files = list(validation_root.glob("*/IR_label.json")) if validation_root.is_dir() else []
    image_files = list(validation_root.glob("*/*.jpg")) if validation_root.is_dir() else []
    tracker_model = base / "papers" / "EDTC" / "pretrained" / "UAVTrackEH.pth.tar"
    yolo_weights = base / "papers" / "EDTC" / "yolov5" / "weights" / "edtc_yolo_best.pt"
    yolo_data = base / "data_templates" / "edtc_antiuav.yaml"
    smoke_summary = base / "runs" / "window_accuracy" / "papers" / "edtc_antiuav600_smoke_sequence23" / "summary.json"
    smoke_plots = smoke_summary.parent / "plots" if smoke_summary.parent else None
    return {
        "validation_zip": {
            "path": str(validation_zip),
            "exists": validation_zip.is_file(),
            "bytes": validation_zip.stat().st_size if validation_zip.is_file() else None,
        },
        "validation_root": {
            "path": str(validation_root),
            "exists": validation_root.is_dir(),
            "sequences": len(sequence_names),
            "label_files": len(label_files),
            "jpg_files": len(image_files),
        },
        "tracker_model": {
            "path": str(tracker_model),
            "exists": tracker_model.is_file(),
            "bytes": tracker_model.stat().st_size if tracker_model.is_file() else None,
        },
        "yolo_weights": {
            "path": str(yolo_weights),
            "exists": yolo_weights.is_file(),
            "bytes": yolo_weights.stat().st_size if yolo_weights.is_file() else None,
        },
        "yolo_data": {
            "path": str(yolo_data),
            "exists": yolo_data.is_file(),
        },
        "cpu_smoke": {
            "summary": str(smoke_summary),
            "summary_exists": smoke_summary.is_file(),
            "svg_count": len(list(smoke_plots.glob("*_window_metrics.svg"))) if smoke_plots and smoke_plots.is_dir() else 0,
        },
    }


def inventory_transvisdrone_local(base: Path) -> dict[str, Any]:
    nps_best = (
        base
        / "papers"
        / "TransVisDrone"
        / "pretrained"
        / "TransVisDrone_weights"
        / "runs"
        / "train"
        / "NPS"
        / "image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0"
        / "weights"
        / "best.pt"
    )
    smoke_summary = base / "runs" / "window_accuracy" / "papers" / "transvisdrone_nps_val_smoke" / "summary.json"
    smoke_plots = smoke_summary.parent / "plots"
    return {
        "pretrained_drive_folder": _drive_folder_url(TRANSVISDRONE_PRETRAINED_DRIVE_ID),
        "nps_best_drive_file_id": TRANSVISDRONE_NPS_BEST_DRIVE_ID,
        "nps_best_weight": {
            "path": str(nps_best),
            "exists": nps_best.is_file(),
            "bytes": nps_best.stat().st_size if nps_best.is_file() else None,
        },
        "nps_cpu_smoke": {
            "summary": str(smoke_summary),
            "summary_exists": smoke_summary.is_file(),
            "svg_count": len(list(smoke_plots.glob("*_window_metrics.svg"))) if smoke_plots.is_dir() else 0,
        },
    }


def inventory_nps(base: Path) -> dict[str, Any]:
    annotations_root = base / "datasets" / "Drone-Detection" / "annotations" / "NPS-Drones-Dataset"
    raw_zip = base / "datasets" / "NPS" / "raw" / "Videos.zip"
    raw_videos = base / "datasets" / "NPS" / "raw" / "Videos"
    prepared_frames = base / "datasets" / "TransVisDrone" / "NPS" / "AllFrames" / "val"
    prepared_labels = base / "datasets" / "TransVisDrone" / "NPS" / "NPSvisdroneStyle" / "val" / "labels"
    prepared_video_lengths = base / "datasets" / "TransVisDrone" / "NPS" / "Videos" / "val" / "video_length_dict.pkl"
    aicrowd_prepared = (
        base
        / "papers"
        / "AICrowd_AOT_Challenge_Winner"
        / "runs"
        / "submission-v022"
        / "results_nps_val"
        / "_prepared_nps_val"
    )
    annotation_files = sorted(annotations_root.glob("Clip_*.txt"))
    rows = 0
    objects = 0
    frame_min: int | None = None
    frame_max: int | None = None
    val_clip_rows: dict[str, int] = {}
    for path in annotation_files:
        clip_rows = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                parts = [part.strip() for part in raw.strip().split(",") if part.strip()]
                if len(parts) < 2:
                    continue
                frame_id = int(parts[0])
                count = int(parts[1])
                frame_min = frame_id if frame_min is None else min(frame_min, frame_id)
                frame_max = frame_id if frame_max is None else max(frame_max, frame_id)
                rows += 1
                clip_rows += 1
                objects += count
        clip_id = int(path.stem.split("_")[1])
        if 37 <= clip_id <= 40:
            val_clip_rows[path.stem] = clip_rows

    return {
        "home": NPS_HOME,
        "videos_zip": _head(NPS_VIDEOS_URL).__dict__,
        "annotations_zip": _head(NPS_ANNOTATIONS_URL).__dict__,
        "dogfight_repo": {
            "path": str(base / "datasets" / "Drone-Detection"),
            "present": (base / "datasets" / "Drone-Detection" / ".git").is_dir(),
            "nps_annotation_files": len(annotation_files),
            "annotation_rows": rows,
            "objects": objects,
            "frame_range": [frame_min, frame_max],
            "val_clip_rows": val_clip_rows,
        },
        "local_video_zip": {
            "path": str(raw_zip),
            "exists": raw_zip.is_file(),
            "bytes": raw_zip.stat().st_size if raw_zip.is_file() else None,
        },
        "local_extracted_videos": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(raw_videos.glob("Clip_*.mov"))
        ] if raw_videos.is_dir() else [],
        "local_prepared_val": {
            "frames_dir": str(prepared_frames),
            "frames": len(list(prepared_frames.glob("*"))) if prepared_frames.is_dir() else 0,
            "labels_dir": str(prepared_labels),
            "labels": len(list(prepared_labels.glob("*.txt"))) if prepared_labels.is_dir() else 0,
            "video_length_dict": str(prepared_video_lengths),
            "video_length_dict_exists": prepared_video_lengths.is_file(),
            "aicrowd_prepared_dir": str(aicrowd_prepared),
            "aicrowd_prepared_frames": len(list(aicrowd_prepared.glob("*/*"))) if aicrowd_prepared.is_dir() else 0,
        },
        "note": "Videos.zip is required to extract every NPS frame for a model rerun; the annotation repo alone is not enough for YOLO-style inference.",
    }


def build_inventory(base_dir: str | Path = ROOT) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    return {
        "base_dir": str(base),
        "disk_free": shutil.disk_usage(base)._asdict(),
        "sources": {
            "nps_purdue_and_dogfight": inventory_nps(base),
            "aot_s3_part1": inventory_aot(),
            "edtc_models_raw_results_drive": inventory_drive_folder("pretrained_models", EDTC_MODELS_DRIVE_ID),
            "edtc_antiuav600_dataset_drive": inventory_drive_folder("AntiUAV600", EDTC_DATASET_DRIVE_ID),
            "edtc_local": inventory_edtc_local(base),
            "transvisdrone_local": inventory_transvisdrone_local(base),
            "yolomg_ard100_baidu": {
                "url": YOLOMG_ARD100_URL,
                "access": "BaiduYun share from YOLOMG README; not automatically downloaded by this tool.",
            },
        },
    }


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    sources = report["sources"]
    nps = sources["nps_purdue_and_dogfight"]
    aot = sources["aot_s3_part1"]
    edtc_models = sources["edtc_models_raw_results_drive"]
    edtc_data = sources["edtc_antiuav600_dataset_drive"]
    edtc_local = sources["edtc_local"]
    transvisdrone_local = sources["transvisdrone_local"]
    lines = [
        "# External Window-Accuracy Source Inventory",
        "",
        f"- Base dir: `{report['base_dir']}`",
        f"- Disk free: `{_human_bytes(report['disk_free']['free'])}`",
        "",
        "## NPS / Dogfight",
        "",
        f"- Videos zip: `{_human_bytes(nps['videos_zip'].get('content_length'))}` from `{nps['videos_zip']['url']}`",
        f"- Annotations zip: `{_human_bytes(nps['annotations_zip'].get('content_length'))}` from `{nps['annotations_zip']['url']}`",
        f"- Local Dogfight annotations: `{nps['dogfight_repo']['nps_annotation_files']}` files, `{nps['dogfight_repo']['annotation_rows']}` rows, `{nps['dogfight_repo']['objects']}` boxes",
        f"- Val clips in local annotations: `{json.dumps(nps['dogfight_repo']['val_clip_rows'], sort_keys=True)}`",
        f"- Local Videos.zip: `{_human_bytes(nps['local_video_zip'].get('bytes'))}` at `{nps['local_video_zip']['path']}`",
        f"- Extracted local videos: `{', '.join(item['name'] for item in nps['local_extracted_videos']) or 'none'}`",
        f"- Prepared local val frames/labels: `{nps['local_prepared_val']['frames']}` frames, `{nps['local_prepared_val']['labels']}` label files",
        f"- Prepared AICrowd NPS frame links: `{nps['local_prepared_val']['aicrowd_prepared_frames']}`",
        f"- TransVisDrone pretrained folder: `{transvisdrone_local['pretrained_drive_folder']}`",
        f"- Local TransVisDrone NPS best.pt: `{_human_bytes(transvisdrone_local['nps_best_weight'].get('bytes'))}` at `{transvisdrone_local['nps_best_weight']['path']}`",
        f"- Local TransVisDrone NPS CPU smoke curves: `{transvisdrone_local['nps_cpu_smoke']['svg_count']}` SVG, summary exists `{transvisdrone_local['nps_cpu_smoke']['summary_exists']}`",
        f"- Note: {nps['note']}",
        "",
        "## AOT Part1",
        "",
        f"- S3 prefix: `{aot['s3_prefix']}`",
        f"- Top-level entries: `{len(aot['top_level'].get('lines', []))}`",
        f"- ImageSets entries: `{len(aot['image_sets'].get('lines', []))}`",
        f"- Note: {aot['note']}",
        "",
        "## EDTC",
        "",
        f"- Models/raw-results folder: `{edtc_models['url']}`",
        f"- Listed items: `{', '.join(item['name'] for item in edtc_models.get('items', [])) or 'none'}`",
        f"- AntiUAV600 folder: `{edtc_data['url']}`",
        f"- Listed items: `{', '.join(item['name'] for item in edtc_data.get('items', [])) or 'none'}`",
        f"- Local validation zip: `{_human_bytes(edtc_local['validation_zip'].get('bytes'))}` at `{edtc_local['validation_zip']['path']}`",
        f"- Local validation extracted: `{edtc_local['validation_root']['sequences']}` sequences, `{edtc_local['validation_root']['label_files']}` label JSON files, `{edtc_local['validation_root']['jpg_files']}` JPG frames",
        f"- Local UAVTrackEH checkpoint: `{_human_bytes(edtc_local['tracker_model'].get('bytes'))}` at `{edtc_local['tracker_model']['path']}`",
        f"- Local EDTC YOLO weights: `{_human_bytes(edtc_local['yolo_weights'].get('bytes'))}` at `{edtc_local['yolo_weights']['path']}`",
        f"- Local CPU smoke curves: `{edtc_local['cpu_smoke']['svg_count']}` SVG, summary exists `{edtc_local['cpu_smoke']['summary_exists']}`",
        "",
        "## YOLOMG ARD100",
        "",
        f"- Source: `{sources['yolomg_ard100_baidu']['url']}`",
        f"- Access: {sources['yolomg_ard100_baidu']['access']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory external data/model sources needed for paper window-accuracy curves.")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "external_source_inventory.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "external_source_inventory.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_inventory(args.base_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.markdown, report)
    print(f"json={args.json}")
    print(f"markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
