from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1] / 'URAP-UAV-to-UAV-Detection-and-Tracking' / 'papers' / 'YOLOMG'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from utils.metrics import ap_per_class
from utils.torch_utils import select_device
from val import process_batch

VIDEO_STEM_RE = re.compile(r'^(?P<video>.+)_(?P<frame>\d+)$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate YOLOMG over time and optionally render chart-augmented videos.')
    parser.add_argument('--weights', type=Path, required=True, help='Path to YOLOMG weights (.pt).')
    parser.add_argument('--images-list', type=Path, default=Path(r'D:\URAP_datasets\ARD100_YOLOMG\test.txt'))
    parser.add_argument('--images2-list', type=Path, default=Path(r'D:\URAP_datasets\ARD100_YOLOMG\test2.txt'))
    parser.add_argument('--video-root', type=Path, default=Path(r'D:\URAP_datasets\ARD100\test_videos'))
    parser.add_argument('--output-dir', type=Path, default=Path(r'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\timeline_eval\test_timeline'))
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.4)
    parser.add_argument('--match-iou', type=float, default=0.5, help='IoU threshold used for TP/FP/FN accounting.')
    parser.add_argument('--window-seconds', type=float, default=1.0)
    parser.add_argument('--metric', choices=['matched_confidence', 'ap50', 'f1', 'precision', 'recall'], default='matched_confidence')
    parser.add_argument('--video-filter', nargs='*', default=None, help='Only process specific video names, e.g. phantom02 phantom03')
    parser.add_argument('--start-frame', type=int, default=None, help='Optional lower bound on evaluated frame index, inclusive.')
    parser.add_argument('--end-frame', type=int, default=None, help='Optional upper bound on evaluated frame index, inclusive.')
    parser.add_argument('--render-overlay', action='store_true', help='Render a video with the timeline panel appended at the bottom.')
    parser.add_argument('--overlay-alpha', type=float, default=0.88, help='Alpha for the appended chart panel, between 0 and 1.')
    parser.add_argument('--panel-height', type=int, default=220)
    parser.add_argument('--line-width', type=int, default=3)
    parser.add_argument('--save-pred-labels', action='store_true', help='Save per-image YOLO txt predictions with confidence.')
    parser.add_argument('--save-pred-jsonl', action='store_true', help='Save per-box predictions as JSONL for Route-B/action rescoring.')
    parser.add_argument('--exist-ok', action='store_true')
    return parser.parse_args()


def ensure_path(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f'{kind} not found: {path}')
    return path.resolve()


def read_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def should_keep_frame(image_path: str, requested_videos: set[str], start_frame: int | None, end_frame: int | None) -> bool:
    video_name, frame_index = parse_frame_identity(image_path)
    if requested_videos and video_name not in requested_videos:
        return False
    if start_frame is not None and frame_index < start_frame:
        return False
    if end_frame is not None and frame_index > end_frame:
        return False
    return True


def align_motion_pairs(
    image_candidates: Sequence[str],
    paired_candidates: Sequence[str],
    requested_videos: set[str],
    start_frame: int | None,
    end_frame: int | None,
) -> List[Tuple[str, str]]:
    paired_by_video: Dict[str, List[str]] = defaultdict(list)
    for path in paired_candidates:
        video_name, _ = parse_frame_identity(path)
        paired_by_video[video_name].append(path)

    ordinal_by_video: Dict[str, int] = defaultdict(int)
    aligned: List[Tuple[str, str]] = []
    for image_path in image_candidates:
        video_name, _ = parse_frame_identity(image_path)
        ordinal = ordinal_by_video[video_name]
        ordinal_by_video[video_name] += 1
        if not should_keep_frame(image_path, requested_videos, start_frame, end_frame):
            continue

        paired_options = paired_by_video.get(video_name)
        if not paired_options:
            raise ValueError(f'No motion image candidates found for video {video_name}')
        paired_path = paired_options[ordinal % len(paired_options)]
        aligned.append((image_path, paired_path))
    return aligned


def parse_frame_identity(image_path: str) -> Tuple[str, int]:
    stem = Path(image_path).stem
    match = VIDEO_STEM_RE.match(stem)
    if not match:
        raise ValueError(f'Unable to parse video/frame from image stem: {stem}')
    return match.group('video'), int(match.group('frame'))


def normalize_weights_path(weights: Path) -> Path:
    return weights if weights.is_absolute() else (ROOT / weights)


def discover_video_path(video_root: Path, video_name: str) -> Path:
    candidates = [video_root / f'{video_name}.mp4', video_root / f'{video_name}.avi', video_root / f'{video_name}.mov']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Video file not found for {video_name} under {video_root}')


def get_video_metadata(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return {'fps': float(fps), 'frame_count': frame_count, 'width': width, 'height': height}


def metric_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_ap50(correct: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray) -> float:
    if target_cls.size == 0 or conf.size == 0:
        return 0.0
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(correct, conf, pred_cls, target_cls, plot=False, names={0: 'target'})
    if ap.size == 0:
        return 0.0
    return float(ap[:, 0].mean())


def aggregate_windows(records: Sequence[Dict[str, object]], fps: float, window_seconds: float) -> List[Dict[str, float]]:
    windows: Dict[int, Dict[str, float]] = {}
    for record in records:
        timestamp = (record['frame_index'] - 1) / fps
        window_index = int(math.floor(timestamp / window_seconds))
        window = windows.setdefault(window_index, {
            'window_index': window_index,
            'start_time_sec': window_index * window_seconds,
            'end_time_sec': (window_index + 1) * window_seconds,
            'frame_count': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'gt_count': 0,
            'pred_count': 0,
            'frame_start': record['frame_index'],
            'frame_end': record['frame_index'],
            'matched_confidence_sum': 0.0,
            'correct_parts': [],
            'conf_parts': [],
            'pred_cls_parts': [],
            'target_cls_parts': [],
        })
        window['frame_count'] += 1
        window['tp'] += record['tp']
        window['fp'] += record['fp']
        window['fn'] += record['fn']
        window['gt_count'] += record['gt_count']
        window['pred_count'] += record['pred_count']
        window['frame_start'] = min(window['frame_start'], record['frame_index'])
        window['frame_end'] = max(window['frame_end'], record['frame_index'])
        window['matched_confidence_sum'] += float(record['matched_confidence'])
        if record['correct'].size:
            window['correct_parts'].append(record['correct'])
            window['conf_parts'].append(record['conf'])
            window['pred_cls_parts'].append(record['pred_cls'])
        if record['target_cls'].size:
            window['target_cls_parts'].append(record['target_cls'])
    results: List[Dict[str, float]] = []
    for window_index in sorted(windows):
        window = windows[window_index]
        metrics = metric_from_counts(window['tp'], window['fp'], window['fn'])
        window.update(metrics)
        correct = np.concatenate(window.pop('correct_parts'), axis=0) if window['correct_parts'] else np.zeros((0, 1), dtype=bool)
        conf = np.concatenate(window.pop('conf_parts'), axis=0) if window['conf_parts'] else np.zeros((0,), dtype=np.float32)
        pred_cls = np.concatenate(window.pop('pred_cls_parts'), axis=0) if window['pred_cls_parts'] else np.zeros((0,), dtype=np.float32)
        target_cls = np.concatenate(window.pop('target_cls_parts'), axis=0) if window['target_cls_parts'] else np.zeros((0,), dtype=np.float32)
        window['ap50'] = compute_ap50(correct, conf, pred_cls, target_cls)
        window['matched_confidence_mean'] = window.pop('matched_confidence_sum') / max(window['frame_count'], 1)
        results.append(window)
    return results


def write_csv(path: Path, rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clip_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def xyxy_to_yolo_line(box: Sequence[float], conf: float, cls: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = clip_xyxy(box, width, height)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f'{cls} {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f} {conf:.8f}'


def render_plot_png(windows: Sequence[Dict[str, float]], metric_name: str, title: str, out_path: Path) -> None:
    if not windows:
        return
    metric_key = 'matched_confidence_mean' if metric_name == 'matched_confidence' else metric_name
    x = [0.5 * (row['start_time_sec'] + row['end_time_sec']) for row in windows]
    y = [row[metric_key] for row in windows]
    precision = [row['precision'] for row in windows]
    recall = [row['recall'] for row in windows]

    plt.figure(figsize=(12, 3.6), dpi=180)
    ax = plt.gca()
    ax.plot(x, y, color='#155eef', linewidth=2.6, label=metric_key.upper())
    ax.fill_between(x, y, color='#155eef', alpha=0.14)
    ax.plot(x, precision, color='#12b76a', linewidth=1.2, alpha=0.6, label='Precision')
    ax.plot(x, recall, color='#f79009', linewidth=1.2, alpha=0.6, label='Recall')
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(min(x), max(x) if len(x) > 1 else max(x) + 1e-6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.7)
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def render_frame_score_plot(rows: Sequence[Dict[str, object]], score_key: str, title: str, out_path: Path) -> None:
    if not rows:
        return
    x = [row['timestamp_sec'] for row in rows]
    y = [row[score_key] for row in rows]

    plt.figure(figsize=(12, 3.6), dpi=180)
    ax = plt.gca()
    ax.plot(x, y, color='#155eef', linewidth=1.8, alpha=0.92, label=score_key)
    ax.fill_between(x, y, color='#155eef', alpha=0.12)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(min(x), max(x) if len(x) > 1 else max(x) + 1e-6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(score_key.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.7)
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def write_frame_score_text(path: Path, rows: Sequence[Dict[str, object]], score_key: str) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(f"frame {row['frame_index']} at {row['timestamp_sec']:.4f}s -> {float(row[score_key]):.4f}\n")


def build_panel_image(windows: Sequence[Dict[str, float]], metric_name: str, video_name: str,
                      x_start_sec: float, x_end_sec: float, panel_width: int, panel_height: int,
                      alpha: float) -> np.ndarray:
    if not windows:
        raise ValueError(f'No windows available for {video_name}')
    metric_key = 'matched_confidence_mean' if metric_name == 'matched_confidence' else metric_name

    times = [0.5 * (row['start_time_sec'] + row['end_time_sec']) for row in windows]
    metric_values = [row[metric_key] for row in windows]
    precision = [row['precision'] for row in windows]
    recall = [row['recall'] for row in windows]

    fig = plt.figure(figsize=(panel_width / 100.0, panel_height / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.plot(times, metric_values, color='#155eef', linewidth=2.8, label=metric_key.upper())
    ax.fill_between(times, metric_values, color='#155eef', alpha=0.16)
    ax.plot(times, precision, color='#12b76a', linewidth=1.1, alpha=0.55, label='Precision')
    ax.plot(times, recall, color='#f79009', linewidth=1.1, alpha=0.55, label='Recall')
    ax.set_xlim(x_start_sec, max(x_end_sec, times[-1]))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_title(f'{video_name} timeline ({metric_key.upper()})', fontsize=11)
    ax.grid(alpha=0.24, linestyle='--', linewidth=0.7)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    fig.tight_layout(pad=1.1)

    canvas = fig.canvas
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    plt.close(fig)

    rgba = cv2.resize(buf, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    white = np.full_like(bgr, 255)
    blended = cv2.addWeighted(bgr, alpha, white, 1.0 - alpha, 0.0)
    return blended


def draw_cursor(panel: np.ndarray, current_time_sec: float, x_start_sec: float, x_end_sec: float, line_width: int) -> np.ndarray:
    frame = panel.copy()
    h, w = frame.shape[:2]
    left_margin = 64
    right_margin = 26
    top_margin = 22
    bottom_margin = 42
    usable_width = max(1, w - left_margin - right_margin)
    span = max(x_end_sec - x_start_sec, 1e-6)
    ratio = min(max((current_time_sec - x_start_sec) / span, 0.0), 1.0)
    x = int(round(left_margin + ratio * usable_width))
    cv2.line(frame, (x, top_margin), (x, h - bottom_margin), (37, 99, 235), line_width, cv2.LINE_AA)
    label = f't={current_time_sec:5.1f}s'
    cv2.putText(frame, label, (max(8, x - 35), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
    return frame


def render_overlay_video(video_path: Path, out_path: Path, windows: Sequence[Dict[str, float]], metric_name: str,
                         panel_height: int, overlay_alpha: float, line_width: int,
                         clip_start_frame: int | None = None, clip_end_frame: int | None = None) -> Dict[str, float]:
    meta = get_video_metadata(video_path)
    fps = meta['fps']
    frame_count = meta['frame_count']
    width = meta['width']
    height = meta['height']
    start_frame = max(1, clip_start_frame or 1)
    end_frame = min(frame_count, clip_end_frame or frame_count)
    selected_frame_count = max(0, end_frame - start_frame + 1)
    x_start_sec = (start_frame - 1) / fps if fps > 0 else 0.0
    x_end_sec = end_frame / fps if fps > 0 else 0.0

    panel = build_panel_image(windows, metric_name, video_path.stem, x_start_sec, x_end_sec, width, panel_height, overlay_alpha)

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height + panel_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to open video writer: {out_path}')

    frame_idx = 0
    written_frames = 0
    progress = tqdm(total=selected_frame_count, desc=f'render {video_path.stem}', leave=False)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < start_frame:
            continue
        if frame_idx > end_frame:
            break
        current_time_sec = (frame_idx - 1) / fps
        panel_with_cursor = draw_cursor(panel, current_time_sec, x_start_sec, x_end_sec, line_width)
        stacked = np.vstack((frame, panel_with_cursor))
        writer.write(stacked)
        written_frames += 1
        progress.update(1)
    progress.close()
    cap.release()
    writer.release()

    return {
        'fps': fps,
        'frame_count': written_frames,
        'width': width,
        'height': height,
        'panel_height': panel_height,
        'duration_sec': selected_frame_count / fps if fps > 0 else 0.0,
        'clip_start_frame': start_frame,
        'clip_end_frame': end_frame,
        'video_path': str(video_path),
        'rendered_path': str(out_path),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    images_list = ensure_path(args.images_list, 'images list')
    images2_list = ensure_path(args.images2_list, 'images2 list')
    weights = ensure_path(normalize_weights_path(args.weights), 'weights')
    video_root = ensure_path(args.video_root, 'video root')

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f'Output directory already exists and is not empty: {args.output_dir}. Use --exist-ok to reuse it.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = args.output_dir / 'per_frame'
    per_window_dir = args.output_dir / 'per_window'
    plots_dir = args.output_dir / 'plots'
    frame_scores_dir = args.output_dir / 'frame_scores'
    overlay_dir = args.output_dir / 'overlay_videos'
    pred_labels_dir = args.output_dir / 'pred_labels'
    pred_jsonl_path = args.output_dir / 'predictions.jsonl'
    for path in (per_frame_dir, per_window_dir, plots_dir, frame_scores_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.render_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred_labels:
        pred_labels_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl_handle = pred_jsonl_path.open('w', encoding='utf-8') if args.save_pred_jsonl else None

    requested_videos = set(args.video_filter or [])
    image_candidates = read_list(images_list)
    paired_candidates = read_list(images2_list)

    if requested_videos or args.start_frame is not None or args.end_frame is not None:
        filtered_pairs = align_motion_pairs(
            image_candidates=image_candidates,
            paired_candidates=paired_candidates,
            requested_videos=requested_videos,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        if not filtered_pairs:
            raise ValueError('No frames matched the requested filters.')
        tmp_list = args.output_dir / 'filtered_images.txt'
        tmp_list.write_text('\n'.join(img1 for img1, _ in filtered_pairs) + '\n', encoding='utf-8')
        tmp_list2 = args.output_dir / 'filtered_images2.txt'
        tmp_list2.write_text('\n'.join(img2 for _, img2 in filtered_pairs) + '\n', encoding='utf-8')
        images_list = tmp_list
        images2_list = tmp_list2
    elif len(image_candidates) != len(paired_candidates):
        aligned_pairs = align_motion_pairs(
            image_candidates=image_candidates,
            paired_candidates=paired_candidates,
            requested_videos=requested_videos,
            start_frame=None,
            end_frame=None,
        )
        tmp_list = args.output_dir / 'aligned_images.txt'
        tmp_list.write_text('\n'.join(img1 for img1, _ in aligned_pairs) + '\n', encoding='utf-8')
        tmp_list2 = args.output_dir / 'aligned_images2.txt'
        tmp_list2.write_text('\n'.join(img2 for _, img2 in aligned_pairs) + '\n', encoding='utf-8')
        images_list = tmp_list
        images2_list = tmp_list2

    device = select_device(args.device, batch_size=args.batch_size)
    model = DetectMultiBackend(str(weights), device=device, fp16=False)
    stride = int(model.stride)
    imgsz = check_img_size(args.imgsz, s=stride)
    dataloader, _ = create_dataloader(str(images_list), str(images2_list), imgsz, args.batch_size, stride,
                                      single_cls=False, pad=0.5, rect=model.pt, workers=args.workers,
                                      prefix='timeline: ')
    model.eval()
    model.warmup(imgsz=(1 if model.pt else args.batch_size, 3, imgsz, imgsz))
    iouv = torch.tensor([args.match_iou], device=device)

    per_video_records: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    video_meta_cache: Dict[str, Dict[str, float]] = {}
    total_frames = 0

    progress = tqdm(dataloader, desc='timeline-eval', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for im, im2, targets, paths, paths2, shapes, shapes2 in progress:
        im = im.to(device, non_blocking=True).float() / 255.0
        im2 = im2.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        nb, _, height, width = im.shape

        preds, _ = model(im, im2, augment=False, val=True)
        preds = non_max_suppression(preds, args.conf_thres, args.iou_thres, multi_label=True, agnostic=False)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        for sample_index, pred in enumerate(preds):
            labels = targets[targets[:, 0] == sample_index, 1:]
            path = paths[sample_index]
            video_name, frame_index = parse_frame_identity(path)
            if video_name not in video_meta_cache:
                video_meta_cache[video_name] = get_video_metadata(discover_video_path(video_root, video_name))
            shape = shapes[sample_index][0]
            orig_height, orig_width = int(shape[0]), int(shape[1])
            nl = len(labels)

            if len(pred):
                predn = pred.clone()
                scale_coords(im[sample_index].shape[1:], predn[:, :4], shape, shapes[sample_index][1])
            else:
                predn = pred

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(im[sample_index].shape[1:], tbox, shape, shapes[sample_index][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv) if len(predn) else torch.zeros((0, 1), dtype=torch.bool, device=device)
            else:
                correct = torch.zeros((len(predn), 1), dtype=torch.bool, device=device)

            tp = int(correct[:, 0].sum().item()) if len(predn) else 0
            fp = int(len(predn) - tp)
            fn = int(nl - tp)
            pred_count = int(len(predn))
            gt_count = int(nl)
            metrics = metric_from_counts(tp, fp, fn)
            fps = video_meta_cache[video_name]['fps']
            timestamp_sec = (frame_index - 1) / fps
            confidence_mean = float(predn[:, 4].mean().item()) if pred_count else 0.0
            correct_np = correct[:, :1].detach().cpu().numpy().astype(bool)
            conf_np = predn[:, 4].detach().cpu().numpy().astype(np.float32) if pred_count else np.zeros((0,), dtype=np.float32)
            pred_cls_np = predn[:, 5].detach().cpu().numpy().astype(np.float32) if pred_count else np.zeros((0,), dtype=np.float32)
            target_cls_np = labels[:, 0].detach().cpu().numpy().astype(np.float32) if gt_count else np.zeros((0,), dtype=np.float32)
            matched_confidence = float(conf_np[correct_np[:, 0]].max()) if correct_np.size and correct_np[:, 0].any() else 0.0
            frame_correct = float(1.0 if gt_count > 0 and fn == 0 and tp > 0 else 0.0)

            if args.save_pred_labels and pred_count:
                pred_label_path = pred_labels_dir / f'{Path(path).stem}.txt'
                pred_label_lines = [
                    xyxy_to_yolo_line(
                        row[:4].detach().cpu().tolist(),
                        float(row[4].item()),
                        int(row[5].item()),
                        orig_width,
                        orig_height,
                    )
                    for row in predn
                ]
                pred_label_path.write_text('\n'.join(pred_label_lines) + '\n', encoding='utf-8')

            if pred_jsonl_handle is not None and pred_count:
                pred_cpu = predn.detach().cpu().numpy()
                correct_flat = correct_np[:, 0] if correct_np.size else np.zeros((0,), dtype=bool)
                for pred_index, row in enumerate(pred_cpu):
                    x1, y1, x2, y2 = clip_xyxy(row[:4], orig_width, orig_height)
                    out_row = {
                        'video': video_name,
                        'seq': video_name,
                        'frame_index': frame_index,
                        'frame_id': frame_index,
                        'timestamp_sec': timestamp_sec,
                        'bbox': [x1, y1, x2, y2],
                        'conf': float(row[4]),
                        'objectness': float(row[4]),
                        'final_drone_score': float(row[4]),
                        'class_id': int(row[5]),
                        'prediction_index': pred_index,
                        'is_tp_match_iou': bool(correct_flat[pred_index]) if pred_index < len(correct_flat) else False,
                        'match_iou': args.match_iou,
                        'image_path': path,
                        'motion_image_path': paths2[sample_index],
                        'image_width': orig_width,
                        'image_height': orig_height,
                        'source': 'yolomg_lowconf',
                    }
                    pred_jsonl_handle.write(json.dumps(out_row, ensure_ascii=False) + '\n')

            per_video_records[video_name].append({
                'video': video_name,
                'frame_index': frame_index,
                'timestamp_sec': timestamp_sec,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gt_count': gt_count,
                'pred_count': pred_count,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'frame_correct': frame_correct,
                'matched_confidence': matched_confidence,
                'confidence_mean': confidence_mean,
                'image_path': path,
                'motion_image_path': paths2[sample_index],
                'correct': correct_np,
                'conf': conf_np,
                'pred_cls': pred_cls_np,
                'target_cls': target_cls_np,
            })
            total_frames += 1

    if pred_jsonl_handle is not None:
        pred_jsonl_handle.close()

    manifest = {
        'weights': str(weights),
        'images_list': str(images_list),
        'images2_list': str(images2_list),
        'video_root': str(video_root),
        'imgsz': imgsz,
        'batch_size': args.batch_size,
        'device': str(device),
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'match_iou': args.match_iou,
        'window_seconds': args.window_seconds,
        'metric': args.metric,
        'render_overlay': bool(args.render_overlay),
        'overlay_alpha': args.overlay_alpha,
        'panel_height': args.panel_height,
        'save_pred_labels': bool(args.save_pred_labels),
        'pred_labels_dir': str(pred_labels_dir) if args.save_pred_labels else None,
        'save_pred_jsonl': bool(args.save_pred_jsonl),
        'pred_jsonl': str(pred_jsonl_path) if args.save_pred_jsonl else None,
        'videos': {},
        'total_frames': total_frames,
    }

    for video_name in sorted(per_video_records):
        rows = sorted(per_video_records[video_name], key=lambda row: row['frame_index'])
        meta = video_meta_cache[video_name]
        frame_csv = per_frame_dir / f'{video_name}_per_frame.csv'
        frame_csv_rows = [
            {k: row[k] for k in (
                'video', 'frame_index', 'timestamp_sec', 'tp', 'fp', 'fn', 'gt_count', 'pred_count',
                'precision', 'recall', 'f1', 'frame_correct', 'matched_confidence', 'confidence_mean',
                'image_path', 'motion_image_path'
            )}
            for row in rows
        ]
        write_csv(frame_csv, frame_csv_rows, [
            'video', 'frame_index', 'timestamp_sec', 'tp', 'fp', 'fn', 'gt_count', 'pred_count',
            'precision', 'recall', 'f1', 'frame_correct', 'matched_confidence', 'confidence_mean',
            'image_path', 'motion_image_path'
        ])

        windows = aggregate_windows(rows, meta['fps'], args.window_seconds)
        window_csv = per_window_dir / f'{video_name}_per_window.csv'
        write_csv(window_csv, windows, [
            'window_index', 'start_time_sec', 'end_time_sec', 'frame_start', 'frame_end', 'frame_count',
            'tp', 'fp', 'fn', 'gt_count', 'pred_count', 'precision', 'recall', 'f1', 'ap50', 'matched_confidence_mean'
        ])

        matched_conf_plot_path = plots_dir / f'{video_name}_matched_confidence.png'
        render_frame_score_plot(rows, 'matched_confidence', f'{video_name} frame-level matched confidence (miss=0)', matched_conf_plot_path)
        window_plot_path = plots_dir / f'{video_name}_window_ap50.png'
        render_plot_png(windows, 'ap50', f'{video_name} window-level AP50', window_plot_path)
        frame_score_text_path = frame_scores_dir / f'{video_name}_matched_confidence.txt'
        write_frame_score_text(frame_score_text_path, rows, 'matched_confidence')

        summary_counts = {
            'tp': int(sum(row['tp'] for row in rows)),
            'fp': int(sum(row['fp'] for row in rows)),
            'fn': int(sum(row['fn'] for row in rows)),
            'frame_count': len(rows),
        }
        summary_counts.update(metric_from_counts(summary_counts['tp'], summary_counts['fp'], summary_counts['fn']))
        summary_counts['ap50'] = compute_ap50(
            np.concatenate([row['correct'] for row in rows], axis=0) if rows else np.zeros((0, 1), dtype=bool),
            np.concatenate([row['conf'] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
            np.concatenate([row['pred_cls'] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
            np.concatenate([row['target_cls'] for row in rows], axis=0) if rows else np.zeros((0,), dtype=np.float32),
        )
        summary_counts['matched_confidence_mean'] = float(np.mean([row['matched_confidence'] for row in rows])) if rows else 0.0
        video_entry = {
            'video_path': str(discover_video_path(video_root, video_name)),
            'fps': meta['fps'],
            'frame_count_video': meta['frame_count'],
            'frame_count_evaluated': len(rows),
            'duration_sec_video': meta['frame_count'] / meta['fps'],
            'duration_sec_evaluated': rows[-1]['timestamp_sec'] if rows else 0.0,
            'per_frame_csv': str(frame_csv),
            'per_window_csv': str(window_csv),
            'matched_confidence_plot_png': str(matched_conf_plot_path),
            'window_ap50_plot_png': str(window_plot_path),
            'frame_score_txt': str(frame_score_text_path),
            'summary': summary_counts,
        }

        if args.render_overlay:
            overlay_path = overlay_dir / f'{video_name}_{args.metric}_timeline.mp4'
            overlay_meta = render_overlay_video(discover_video_path(video_root, video_name), overlay_path, windows,
                                                args.metric, args.panel_height, args.overlay_alpha, args.line_width,
                                                clip_start_frame=args.start_frame, clip_end_frame=args.end_frame)
            video_entry['overlay_video'] = overlay_meta

        manifest['videos'][video_name] = video_entry

    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
