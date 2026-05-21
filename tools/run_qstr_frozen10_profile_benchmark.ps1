param(
    [string]$HeldoutRoot = "D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq",
    [string]$AnnotationsCsv = "D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv",
    [string]$OutRoot = "runs\profiles\frozen10_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$Device = "0",
    [int]$MaxFrames = 0,
    [int]$MaxVideos = 0,
    [double]$ScoreThreshold = 0.20,
    [double]$IouThreshold = 0.30,
    [switch]$SkipStable,
    [switch]$SkipHardRecovery
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path $HeldoutRoot)) {
    throw "Missing held-out root: $HeldoutRoot"
}
if (-not (Test-Path $AnnotationsCsv)) {
    throw "Missing annotations CSV: $AnnotationsCsv"
}

$Videos = Get-ChildItem -Recurse -Filter visible.mp4 (Join-Path $HeldoutRoot "raw_videos") |
    Sort-Object FullName
if ($MaxVideos -gt 0) {
    $Videos = $Videos | Select-Object -First $MaxVideos
}
if (-not $Videos -or $Videos.Count -eq 0) {
    throw "No visible.mp4 files found under $HeldoutRoot"
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($VideoFile in $Videos) {
    $SeqName = Split-Path (Split-Path $VideoFile.FullName -Parent) -Leaf
    if (-not $SkipStable) {
        & (Join-Path $PSScriptRoot "run_qstr_stable_profile.ps1") `
            -Video $VideoFile.FullName `
            -Out (Join-Path $OutRoot "stable\$SeqName") `
            -Device $Device `
            -MaxFrames $MaxFrames
    }
    if (-not $SkipHardRecovery) {
        & (Join-Path $PSScriptRoot "run_qstr_hard_recovery_profile.ps1") `
            -Video $VideoFile.FullName `
            -Out (Join-Path $OutRoot "hard_recovery\$SeqName") `
            -Device $Device `
            -MaxFrames $MaxFrames
    }
}

$env:QSTR_BENCH_OUT = (Resolve-Path $OutRoot).Path
$env:QSTR_BENCH_GT = (Resolve-Path $AnnotationsCsv).Path
$env:QSTR_BENCH_SCORE = "$ScoreThreshold"
$env:QSTR_BENCH_IOU = "$IouThreshold"
$env:QSTR_BENCH_MAX_FRAMES = "$MaxFrames"

@'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ["QSTR_BENCH_OUT"])
gt_csv = Path(os.environ["QSTR_BENCH_GT"])
score_threshold = float(os.environ["QSTR_BENCH_SCORE"])
iou_threshold = float(os.environ["QSTR_BENCH_IOU"])
max_frames = int(os.environ.get("QSTR_BENCH_MAX_FRAMES", "0"))

def iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

gt_rows = []
with gt_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        frame_id = int(float(row["frame_id"]))
        if max_frames > 0 and frame_id >= max_frames:
            continue
        gt_rows.append({
            "video_path": row["video_path"],
            "seq": Path(row["video_path"]).parent.name,
            "frame_id": frame_id,
            "bbox": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
            "cls": row.get("class", ""),
            "tag": row.get("tag", ""),
        })

def load_preds(profile_dir):
    preds = []
    for pred_path in profile_dir.glob("*/predictions.jsonl"):
        seq = pred_path.parent.name
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                if max_frames > 0 and int(p.get("frame_id", -1)) >= max_frames:
                    continue
                if p.get("predicted_class") != "drone":
                    continue
                if float(p.get("final_drone_score", 0.0)) < score_threshold:
                    continue
                preds.append({
                    "seq": seq,
                    "frame_id": int(p.get("frame_id", -1)),
                    "bbox": p.get("bbox", []),
                    "score": float(p.get("final_drone_score", 0.0)),
                    "source": p.get("source", ""),
                    "mode": p.get("mode", ""),
                })
    return preds

def summarize_profile(profile):
    profile_dir = out_root / profile
    processed_seqs = {p.parent.name for p in profile_dir.glob("*/predictions.jsonl")}
    profile_gt_rows = [gt for gt in gt_rows if gt["seq"] in processed_seqs]
    preds = load_preds(profile_dir)
    gt_by_key = {}
    for idx, gt in enumerate(profile_gt_rows):
        gt_by_key.setdefault((gt["seq"], gt["frame_id"]), []).append((idx, gt))

    matched_gt = set()
    matched_pred = set()
    for pred_idx, pred in enumerate(sorted(preds, key=lambda x: x["score"], reverse=True)):
        best = None
        best_iou = 0.0
        for gt_idx, gt in gt_by_key.get((pred["seq"], pred["frame_id"]), []):
            if gt_idx in matched_gt:
                continue
            ov = iou(pred["bbox"], gt["bbox"])
            if ov > best_iou:
                best_iou = ov
                best = gt_idx
        if best is not None and best_iou >= iou_threshold:
            matched_gt.add(best)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(profile_gt_rows) - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    per_seq = {}
    for gt_idx, gt in enumerate(profile_gt_rows):
        item = per_seq.setdefault(gt["seq"], {"gt": 0, "tp": 0, "fp": 0, "tags": {}})
        item["gt"] += 1
        if gt_idx in matched_gt:
            item["tp"] += 1
        tag = gt.get("tag") or "unknown"
        tag_item = item["tags"].setdefault(tag, {"gt": 0, "tp": 0})
        tag_item["gt"] += 1
        if gt_idx in matched_gt:
            tag_item["tp"] += 1
    for pred_idx, pred in enumerate(preds):
        if pred_idx not in matched_pred:
            per_seq.setdefault(pred["seq"], {"gt": 0, "tp": 0, "fp": 0, "tags": {}})["fp"] += 1
    for item in per_seq.values():
        item["precision"] = item["tp"] / max(1, item["tp"] + item["fp"])
        item["recall"] = item["tp"] / max(1, item["gt"])
        for tag_item in item["tags"].values():
            tag_item["recall"] = tag_item["tp"] / max(1, tag_item["gt"])

    return {
        "profile": profile,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_frames": max_frames,
        "gt": len(profile_gt_rows),
        "pred_drone": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "per_sequence": dict(sorted(per_seq.items())),
    }

profiles = [p.name for p in out_root.iterdir() if p.is_dir() and p.name in {"stable", "hard_recovery"}]
summary = {"profiles": [summarize_profile(p) for p in sorted(profiles)]}
(out_root / "profile_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

csv_rows = []
for prof in summary["profiles"]:
    csv_rows.append({
        "profile": prof["profile"],
        "gt": prof["gt"],
        "pred_drone": prof["pred_drone"],
        "tp": prof["tp"],
        "fp": prof["fp"],
        "fn": prof["fn"],
        "precision": f'{prof["precision"]:.6f}',
        "recall": f'{prof["recall"]:.6f}',
    })
with (out_root / "profile_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["profile"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(json.dumps(summary, indent=2))
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark aggregation failed with exit code $LASTEXITCODE"
}

Write-Host "Benchmark summary:"
Write-Host (Join-Path $OutRoot "profile_benchmark_summary.json")
