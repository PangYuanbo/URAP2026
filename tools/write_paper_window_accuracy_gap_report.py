from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_paper_window_accuracy_readiness import audit_manifest, _safe_rel, _valid_box_path


SEARCH_PATTERNS = {
    "image-dir": ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.tif", "**/*.tiff"],
    "yolo-dir": ["**/labels", "**/labels/test", "**/val/labels", "**/test/labels"],
    "aot-json": ["**/result.json", "**/results*", "**/results_*"],
    "aot-gt-json": ["**/groundtruth.json", "**/ImageSets/groundtruth.json"],
    "antiuav-json": ["**/list.txt", "**/IR_label.json"],
    "xywh-file": ["**/results*", "**/*_dt.txt", "**/*_pred.txt", "**/*.txt"],
    "li-tetc-txt": ["**/*_gt.txt", "**/*_dt.txt"],
    "tvd-pkl-gt": ["**/predictionsgt_split_*.pkl"],
    "tvd-pkl-pred": ["**/predictionsgt_split_*.pkl", "**/best_predictions.pkl", "**/last_predictions.pkl"],
    "csv": ["**/*.csv"],
    "jsonl": ["**/*.jsonl"],
}


NEXT_STEPS = {
    "YOLOMG": [
        "Provide full ARD100/NPS YOLO labels and run YOLOMG eval with --save-txt --save-conf.",
        "Then point the manifest pred field to the saved labels directory and rerun the batch command.",
    ],
    "TransVisDrone": [
        "Provide NPS/VisDrone-style YOLO labels or a TransVisDrone predictionsgt_split_*.pkl.",
        "Prefer --save-json-gt outputs when possible because one pkl contains both GT and predictions.",
    ],
    "ESOD": [
        "Provide full VisDrone/UAVDT/TinyPerson YOLO labels and run ESOD test.py through tools/run_yolo_eval_window_accuracy.py.",
        "Existing fixture curves only prove the post-processing path, not the full dataset curve.",
    ],
    "AICrowd_Winner_v022": [
        "Provide winner result.json outputs and, if using AOT, the official ImageSets/groundtruth.json.",
        "If weights are missing, run tools/download_aicrowd_lfs_weights.py with AICROWD_GITLAB_TOKEN or GITLAB_TOKEN first.",
    ],
    "EDTC": [
        "Run tools/run_edtc_tracker_window_accuracy.py to execute the EDTC tracker and render curves in one step.",
        "The local CPU smoke path can validate wiring, but the full AntiUAV600 tracker run should use the detached GPU/Windows runner because the tracker is slow on CPU.",
        "Detector-branch fixture curves do not replace the full EDTC tracker evaluation.",
    ],
    "Li_TETC_NPS": [
        "Run additional videos through the compat runner or original environment to add more *_dt.txt files.",
    ],
}


GENERATION_COMMANDS = {
    "yolomg_ard100_test": [
        r"""powershell -ExecutionPolicy Bypass -File tools\start_yolo_eval_window_accuracy_detached.ps1 `
  -Method yolomg `
  -PythonExe papers\YOLOMG\.venv\Scripts\python.exe `
  -Data D:\URAP_datasets\ARD100_YOLOMG\ard100.yaml `
  -Weights 'papers\YOLOMG\runs\train\<run_name>\weights\best.pt' `
  -Gt D:\URAP_datasets\ARD100_YOLOMG\labels\test `
  -FrameManifest D:\URAP_datasets\ARD100_YOLOMG\images\test `
  -FrameManifestFormat image-dir `
  -Out runs\window_accuracy\papers\yolomg_ard100_test `
  -Name yolomg_ard100_test `
  -Fps 30 `
  -RunId yolomg_ard100_test""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_yolo_eval_window_accuracy.ps1 `
  -Out runs\window_accuracy\papers\yolomg_ard100_test `
  -RunId yolomg_ard100_test""",
    ],
    "transvisdrone_nps_val": [
        r"""bash tools/start_yolo_eval_window_accuracy_detached.sh \
  --method transvisdrone \
  --paper-repo papers/TransVisDrone \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/papers/transvisdrone_nps_local.yaml \
  --weights papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
  --gt datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --frame-manifest datasets/TransVisDrone/NPS/AllFrames/val \
  --frame-manifest-format image-dir \
  --out runs/window_accuracy/papers/transvisdrone_nps_val \
  --name transvisdrone_nps_val \
  --task val \
  --img 1280 \
  --batch-size 1 \
  --device mps \
  --num-frames 5 \
  --fps 30 \
  --run-id transvisdrone_nps_val""",
        r"""bash tools/monitor_yolo_eval_window_accuracy.sh \
  --out runs/window_accuracy/papers/transvisdrone_nps_val \
  --run-id transvisdrone_nps_val""",
        r"""powershell -ExecutionPolicy Bypass -File tools\start_yolo_eval_window_accuracy_detached.ps1 `
  -Method transvisdrone `
  -PythonExe papers\TransVisDrone\.venv\Scripts\python.exe `
  -Data runs\window_accuracy\papers\transvisdrone_nps_local.yaml `
  -Weights papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt `
  -Gt datasets\TransVisDrone\NPS\NPSvisdroneStyle\val\labels `
  -GtFrameOffset 1 `
  -FrameManifest datasets\TransVisDrone\NPS\AllFrames\val `
  -FrameManifestFormat image-dir `
  -Out runs\window_accuracy\papers\transvisdrone_nps_val `
  -Name transvisdrone_nps_val `
  -Img 1280 `
  -BatchSize 2 `
  -NumFrames 5 `
  -Half `
  -Fps 30 `
  -RunId transvisdrone_nps_val""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_yolo_eval_window_accuracy.ps1 `
  -Out runs\window_accuracy\papers\transvisdrone_nps_val `
  -RunId transvisdrone_nps_val""",
        r""".venv/paper-cv/bin/python tools/run_yolo_eval_window_accuracy.py \
  --method transvisdrone \
  --repo papers/TransVisDrone \
  --python .venv/paper-cv/bin/python \
  --data runs/window_accuracy/papers/transvisdrone_nps_val_smoke_dataset.yaml \
  --weights papers/TransVisDrone/pretrained/TransVisDrone_weights/runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
  --gt runs/window_accuracy/papers/transvisdrone_nps_val_smoke_dataset/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --frame-manifest runs/window_accuracy/papers/transvisdrone_nps_val_smoke_dataset/AllFrames/val \
  --frame-manifest-format image-dir \
  --out runs/window_accuracy/papers/transvisdrone_nps_val_smoke \
  --name transvisdrone_nps_val_smoke \
  --task val \
  --img 1280 \
  --batch-size 1 \
  --device cpu \
  --num-frames 5 \
  --fps 30""",
    ],
    "esod_visdrone_val": [
        r"""powershell -ExecutionPolicy Bypass -File tools\start_yolo_eval_window_accuracy_detached.ps1 `
  -Method esod `
  -PythonExe papers\ESOD\.venv\Scripts\python.exe `
  -Data papers\ESOD\data\visdrone.yaml `
  -Weights papers\ESOD\weights\esod_yolov5m.pt `
  -Gt D:\URAP_datasets\VisDrone\VisDrone2019-DET-val\labels `
  -FrameManifest D:\URAP_datasets\VisDrone\VisDrone2019-DET-val\images `
  -FrameManifestFormat image-dir `
  -Out runs\window_accuracy\papers\esod_visdrone_val `
  -Name esod_visdrone_val `
  -Img 1280 `
  -BatchSize 1 `
  -Fps 30 `
  -RunId esod_visdrone_val""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_yolo_eval_window_accuracy.ps1 `
  -Out runs\window_accuracy\papers\esod_visdrone_val `
  -RunId esod_visdrone_val""",
    ],
    "aicrowd_winner_nps_val": [
        r"""powershell -ExecutionPolicy Bypass -File tools\start_winner_v022_nps_val_detached.ps1 `
  -DatasetPath papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val\_prepared_nps_val `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_winner_v022_nps_val.ps1 `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val `
  -RunId nps_val""",
        r"""python3 tools/plot_detection_window_accuracy.py \
  --gt datasets/TransVisDrone/NPS/NPSvisdroneStyle/val/labels \
  --gt-format yolo-dir \
  --gt-frame-offset 1 \
  --pred papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_nps_val/nps_val \
  --pred-format aot-json \
  --frame-manifest datasets/TransVisDrone/NPS/AllFrames/val \
  --frame-manifest-format image-dir \
  --img-width 1280 \
  --img-height 960 \
  --fps 30 \
  --window-seconds 3 \
  --out runs/window_accuracy/papers/aicrowd_winner_nps_val""",
    ],
    "aicrowd_winner_aot_part1": [
        r"""powershell -ExecutionPolicy Bypass -File tools\start_winner_v022_fulltest_detached.ps1 `
  -DatasetPath D:\URAP_datasets\AOT\part1\Images `
  -FlightIdsJson papers\TransVisDrone\aot_flight_ids\testflightidsfull1.json `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_aot_part1 `
  -RunId part1""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_winner_v022_fulltest.ps1 `
  -OutputRoot papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_aot_part1 `
  -RunId part1""",
        r"""python3 tools/plot_detection_window_accuracy.py \
  --gt D:/URAP_datasets/AOT/part1/ImageSets/groundtruth.json \
  --gt-format aot-gt-json \
  --pred papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_aot_part1/part1 \
  --pred-format aot-json \
  --frame-manifest D:/URAP_datasets/AOT/part1/Images \
  --frame-manifest-format image-dir \
  --fps 10 \
  --window-seconds 3 \
  --out runs/window_accuracy/papers/aicrowd_winner_aot_part1""",
    ],
    "edtc_antiuav600": [
        r"""bash tools/start_edtc_tracker_window_accuracy_detached.sh \
  --python .venv/edtc-window/bin/python \
  --dataset-root datasets/AntiUAV600/validation \
  --tracker-model papers/EDTC/pretrained/UAVTrackEH.pth.tar \
  --yolo-weights papers/EDTC/yolov5/weights/edtc_yolo_best.pt \
  --yolo-data data_templates/edtc_antiuav.yaml \
  --out runs/window_accuracy/papers/edtc_antiuav600 \
  --threads 0 \
  --num-gpus 1 \
  --device cpu \
  --fps 30 \
  --window-seconds 3 \
  --run-id edtc_antiuav600""",
        r"""bash tools/monitor_edtc_tracker_window_accuracy.sh \
  --out runs/window_accuracy/papers/edtc_antiuav600 \
  --run-id edtc_antiuav600""",
        r"""powershell -ExecutionPolicy Bypass -File tools\start_edtc_tracker_window_accuracy_detached.ps1 `
  -DatasetRoot datasets\AntiUAV600\validation `
  -TrackerModel papers\EDTC\pretrained\UAVTrackEH.pth.tar `
  -YoloWeights papers\EDTC\yolov5\weights\edtc_yolo_best.pt `
  -YoloData data_templates\edtc_antiuav.yaml `
  -Out runs\window_accuracy\papers\edtc_antiuav600 `
  -Fps 30 `
  -WindowSeconds 3 `
  -RunId edtc_antiuav600""",
        r"""powershell -ExecutionPolicy Bypass -File tools\monitor_edtc_tracker_window_accuracy.ps1 `
  -Out runs\window_accuracy\papers\edtc_antiuav600 `
  -RunId edtc_antiuav600""",
        r"""python3 tools/run_edtc_tracker_window_accuracy.py \
  --dataset-root datasets/AntiUAV600/validation \
  --skip-track \
  --results-dir runs/window_accuracy/papers/edtc_antiuav600/tracking_results/uavtrack_eh/urap_window_accuracy \
  --out runs/window_accuracy/papers/edtc_antiuav600 \
  --fps 30 \
  --window-seconds 3""",
        r""".venv/edtc-window/bin/python tools/run_edtc_tracker_window_accuracy.py \
  --python .venv/edtc-window/bin/python \
  --dataset-root datasets/AntiUAV600/validation \
  --tracker-model papers/EDTC/pretrained/UAVTrackEH.pth.tar \
  --yolo-weights papers/EDTC/yolov5/weights/edtc_yolo_best.pt \
  --yolo-data data_templates/edtc_antiuav.yaml \
  --out runs/window_accuracy/papers/edtc_antiuav600_smoke_sequence23 \
  --threads 0 \
  --num-gpus 1 \
  --device cpu \
  --sequence 23""",
    ],
}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _candidate_paths(match: Path, fmt: str) -> list[Path]:
    if fmt == "image-dir":
        return [match.parent] if match.is_file() else [match]
    if fmt == "yolo-dir":
        return [match] if match.is_dir() else [match.parent]
    if fmt == "aot-json":
        return [match, match.parent] if match.is_file() else [match]
    if fmt == "aot-gt-json":
        return [match, match.parent, match.parent.parent] if match.is_file() else [match]
    if fmt == "antiuav-json":
        if match.name == "IR_label.json":
            return [match, match.parent, match.parent.parent]
        return [match, match.parent] if match.is_file() else [match]
    if fmt in {"xywh-file", "li-tetc-txt", "tvd-pkl-gt", "tvd-pkl-pred", "csv", "jsonl"}:
        return [match, match.parent] if match.is_file() else [match]
    return [match]


def _is_fixture_candidate(path: Path) -> bool:
    normalized = path.as_posix()
    return (
        "/runs/window_accuracy/" in normalized
        or "/_inputs/" in normalized
        or "/yolomg_test_images_" in normalized
        or "/esod_test_images_" in normalized
        or "/edtc_yolo_test_images_" in normalized
    )


def _looks_like_xywh_line(line: str) -> bool:
    parts = line.replace(",", " ").replace("\t", " ").split()
    if len(parts) < 4:
        return False
    try:
        values = [float(part) for part in parts[:4]]
    except ValueError:
        return False
    if len(parts) >= 5:
        try:
            extra = float(parts[4])
        except ValueError:
            return True
        # YOLO labels often look like: class cx cy w h. Reject normalized
        # class-first rows so label files are not suggested as tracker output.
        if float(int(values[0])) == values[0] and 0.0 <= values[1] <= 1.0 and 0.0 <= values[2] <= 1.0 and 0.0 <= values[3] <= 1.0 and 0.0 <= extra <= 1.0:
            return False
    return True


def _looks_like_xywh_file(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".txt":
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    return False
                if not line.strip():
                    continue
                return _looks_like_xywh_line(line)
    except OSError:
        return False
    return False


def _valid_candidate_path(path: Path, fmt: str) -> bool:
    if fmt == "xywh-file":
        if path.is_file():
            return _looks_like_xywh_file(path)
        if path.is_dir():
            return any(_looks_like_xywh_file(child) for child in path.rglob("*.txt"))
        return False
    return _valid_box_path(path, fmt)


PRIMARY_DATASET_TOKENS = ("ard100", "nps", "aot", "antiuav", "uavdt", "tinyperson")


def _matches_expected_dataset(candidate: Path, expected_path: str | None) -> bool:
    if not expected_path:
        return True
    expected = str(expected_path).lower().replace("\\", "/")
    candidate_text = candidate.as_posix().lower()
    expected_primary = [token for token in PRIMARY_DATASET_TOKENS if token in expected]
    if expected_primary:
        return any(token in candidate_text for token in expected_primary)
    if "visdrone" in expected:
        return "visdrone" in candidate_text
    return True


def find_candidates(
    fmt: str,
    search_roots: list[Path],
    base_dir: Path,
    max_candidates: int = 20,
    expected_path: str | None = None,
) -> list[str]:
    patterns = SEARCH_PATTERNS.get(fmt, ["**/*"])
    candidates: list[Path] = []
    for root in _dedupe_paths([p.resolve() for p in search_roots if p.exists()]):
        for pattern in patterns:
            for match in root.glob(pattern):
                for candidate in _candidate_paths(match, fmt):
                    if _is_fixture_candidate(candidate.resolve()):
                        continue
                    if not _matches_expected_dataset(candidate.resolve(), expected_path):
                        continue
                    if candidate.exists() and _valid_candidate_path(candidate, fmt):
                        candidates.append(candidate.resolve())
                        if len(_dedupe_paths(candidates)) >= max_candidates:
                            return [_safe_rel(p, base_dir) for p in _dedupe_paths(candidates)[:max_candidates]]
    return [_safe_rel(p, base_dir) for p in _dedupe_paths(candidates)[:max_candidates]]


def _input_gap(run: dict[str, Any], key: str, search_roots: list[Path], base_dir: Path, max_candidates: int) -> dict[str, Any]:
    info = run[key]
    fmt = str(run[f"{key}_format"])
    missing = key in run.get("missing", [])
    roots = search_roots
    if key == "pred":
        roots = [root for root in search_roots if root.name != "datasets"]
    candidates = find_candidates(
        fmt,
        roots,
        base_dir=base_dir,
        max_candidates=max_candidates,
        expected_path=info["path"],
    ) if missing else []
    return {
        "missing": missing,
        "format": fmt,
        "expected_path": info["path"],
        "placeholder": info["placeholder"],
        "valid_now": info["exists"],
        "candidates": candidates,
    }


def _frame_manifest_gap(run: dict[str, Any], search_roots: list[Path], base_dir: Path, max_candidates: int) -> dict[str, Any] | None:
    info = run.get("frame_manifest")
    if not info:
        return None
    fmt = str(run.get("frame_manifest_format") or "image-dir")
    missing = "frame_manifest" in run.get("missing", [])
    roots = search_roots
    if fmt == "image-dir":
        roots = [root for root in search_roots if root.name != "papers"]
    candidates = find_candidates(
        fmt,
        roots,
        base_dir=base_dir,
        max_candidates=max_candidates,
        expected_path=info["path"],
    ) if missing else []
    return {
        "missing": missing,
        "format": fmt,
        "expected_path": info["path"],
        "placeholder": info["placeholder"],
        "valid_now": info["exists"],
        "candidates": candidates,
    }


def build_gap_report(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    search_roots: list[str | Path] | None = None,
    max_candidates: int = 20,
) -> dict[str, Any]:
    base = Path(base_dir).resolve() if base_dir else ROOT
    roots = [Path(p).expanduser() for p in (search_roots or [])]
    if not roots:
        roots = [base / "datasets", base / "runs", base / "papers"]
    audit = audit_manifest(manifest_path, base_dir=base, out_root=out_root)

    gaps = []
    for run in audit["runs"]:
        if run["status"] == "complete_curves":
            continue
        gt_gap = _input_gap(run, "gt", roots, base_dir=base, max_candidates=max_candidates)
        pred_gap = _input_gap(run, "pred", roots, base_dir=base, max_candidates=max_candidates)
        frame_gap = _frame_manifest_gap(run, roots, base_dir=base, max_candidates=max_candidates)
        gaps.append(
            {
                "method": run["method"],
                "name": run["name"],
                "status": run["status"],
                "missing": run["missing"],
                "repo": run["repo"],
                "gt": gt_gap,
                "pred": pred_gap,
                "frame_manifest": frame_gap,
                "next_command": run["next_command"],
                "next_steps": NEXT_STEPS.get(run["method"], []),
                "generation_commands": GENERATION_COMMANDS.get(run["name"], []),
            }
        )

    return {
        "manifest": audit["manifest"],
        "base_dir": audit["base_dir"],
        "out_root": audit["out_root"],
        "search_roots": [str(p.resolve()) for p in roots],
        "counts": audit["counts"],
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Paper Window Accuracy Gap Report",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Output root: `{report['out_root']}`",
        f"- Counts: `{json.dumps(report['counts'], sort_keys=True)}`",
        f"- Search roots: `{', '.join(report['search_roots'])}`",
        "",
    ]
    for gap in report["gaps"]:
        lines.extend(
            [
                f"## {gap['method']} / {gap['name']}",
                "",
                f"- Status: `{gap['status']}`",
                f"- Missing: `{', '.join(gap['missing']) if gap['missing'] else '-'}`",
                f"- Repo: `{gap['repo'].get('path')}` ({gap['repo'].get('kind') or 'missing'})",
                f"- GT: `{gap['gt']['expected_path']}` (`{gap['gt']['format']}`)",
                f"- Pred: `{gap['pred']['expected_path']}` (`{gap['pred']['format']}`)",
                f"- Frame manifest: `{gap['frame_manifest']['expected_path']}` (`{gap['frame_manifest']['format']}`)" if gap.get("frame_manifest") else "- Frame manifest: `not configured`",
                f"- Run when ready: `{gap['next_command']}`",
                "",
            ]
        )
        for label in ("gt", "pred", "frame_manifest"):
            if not gap.get(label):
                continue
            candidates = gap[label]["candidates"]
            lines.append(f"### {label.upper()} candidates")
            if candidates:
                lines.extend(f"- `{candidate}`" for candidate in candidates)
            else:
                lines.append("- none found in search roots")
            lines.append("")
        if gap["next_steps"]:
            lines.append("### Next steps")
            lines.extend(f"- {step}" for step in gap["next_steps"])
            lines.append("")
        if gap["generation_commands"]:
            lines.append("### Generation commands")
            for command in gap["generation_commands"]:
                lines.extend(["", "```bash", command, "```"])
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a missing-input report for paper +/-3s window accuracy runs.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_templates" / "paper_window_accuracy_runs.example.json")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--search-root", action="append", default=[], help="Root to search for candidate GT/prediction inputs. Defaults to datasets, runs, and papers under base-dir.")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--json", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "gap_report.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "runs" / "window_accuracy" / "papers" / "gap_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_gap_report(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        out_root=args.out_root,
        search_roots=args.search_root or None,
        max_candidates=args.max_candidates,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.markdown, report)
    print("counts=" + json.dumps(report["counts"], sort_keys=True))
    print(f"gaps={report['gap_count']}")
    print(f"json={args.json}")
    print(f"markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
