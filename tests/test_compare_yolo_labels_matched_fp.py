from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_compare_yolo_labels_selects_recall_gain_under_baseline_fp(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    baseline_dir = tmp_path / "pred_baseline"
    samurai_dir = tmp_path / "pred_samurai"
    images = [image_dir / f"clip_000{i}.jpg" for i in range(1, 4)]
    for image in images:
        _write(image)
    images_list = tmp_path / "images.txt"
    images_list.write_text("\n".join(str(path) for path in images), encoding="utf-8")

    gt_box = "0 0.50000000 0.50000000 0.20000000 0.20000000\n"
    for image in images:
        _write(label_dir / f"{image.stem}.txt", gt_box)

    _write(baseline_dir / "clip_0001.txt", "0 0.50000000 0.50000000 0.20000000 0.20000000 0.90\n")
    _write(baseline_dir / "clip_0002.txt", "0 0.10000000 0.10000000 0.10000000 0.10000000 0.80\n")
    _write(baseline_dir / "clip_0003.txt", "")

    _write(samurai_dir / "clip_0001.txt", "0 0.50000000 0.50000000 0.20000000 0.20000000 0.90\n")
    _write(samurai_dir / "clip_0002.txt", "0 0.50000000 0.50000000 0.20000000 0.20000000 0.70\n")
    _write(samurai_dir / "clip_0003.txt", "0 0.50000000 0.50000000 0.20000000 0.20000000 0.60\n")

    out_json = tmp_path / "compare.json"
    out_csv = tmp_path / "compare.csv"
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "compare_yolo_labels_matched_fp.py"),
            "--images-list",
            str(images_list),
            "--method",
            f"baseline={baseline_dir}",
            "--method",
            f"samurai={samurai_dir}",
            "--baseline-method",
            "baseline",
            "--baseline-threshold",
            "0.0",
            "--thresholds",
            "0.0",
            "0.5",
            "0.75",
            "--image-width",
            "100",
            "--image-height",
            "100",
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
        ],
        cwd=repo,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["fp_budget"] == 1
    selected = {row["method"]: row for row in payload["selected_under_fp_budget"]}
    assert selected["baseline"]["recall"] == 1 / 3
    assert selected["samurai"]["fp"] == 0
    assert selected["samurai"]["recall"] == 1.0
    assert selected["samurai"]["delta_recall_vs_baseline"] > 0.0
    assert out_csv.exists()
